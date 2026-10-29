"""Read-only: build the 'who is waiting for whom' graph at the end of a combined-town run and look for a cycle.

Uses the last record of actor_trace.jsonl. A vehicle in `vehicle_headway`/`emergency_gap` waits for the nearest same-direction
body ahead in its lane corridor (same test as india_combined_traffic.py); a vehicle in `junction_wait` waits for the junction it is
approaching. Prints the chains and any cycle.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


def last_record(path):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size, block, data = f.tell(), 1 << 20, b""
        while True:
            f.seek(max(0, size - block))
            data = f.read()
            lines = [l for l in data.split(b"\n") if l.strip()]
            if len(lines) >= 2 or block >= size:
                return json.loads(lines[-1])
            block *= 4


def leader(v, vehicles):
    best = None
    for o in vehicles:
        if o is v or math.cos(o["yaw"] - v["yaw"]) < .35:
            continue
        if o["road_id"] == v["road_id"] and o["lane"] != v["lane"]:
            continue
        dx, dy = o["x"] - v["x"], o["y"] - v["y"]
        lon = dx * math.cos(v["yaw"]) + dy * math.sin(v["yaw"])
        lat = abs(-dx * math.sin(v["yaw"]) + dy * math.cos(v["yaw"]))
        if lon <= 0 or lat > v["ey"] + o["ey"] + .18:
            continue
        gap = lon - v["ex"] - o["ex"]
        if best is None or gap < best[1]:
            best = (o, gap)
    return best


def main(run):
    rec = last_record(Path(run) / "actor_trace.jsonl")
    vehicles = [a for a in rec["actors"] if "route_s" in a]
    print("sim_s %.1f, vehicles %d, stopped %d" % (rec["sim_s"], len(vehicles), sum(a["speed"] < .1 for a in vehicles)))
    print("reasons", dict(Counter(a["reason"] for a in vehicles)))
    waits = {}
    for v in vehicles:
        if v["reason"] in ("vehicle_headway", "emergency_gap"):
            found = leader(v, vehicles)
            waits[v["name"]] = ("vehicle", found[0]["name"], round(found[1], 1)) if found else ("vehicle", None, None)
        elif v["reason"] == "junction_wait":
            waits[v["name"]] = ("junction", v["junction"] or "next junction", None)
        else:
            waits[v["name"]] = (v["reason"], None, None)
    for v in sorted(vehicles, key=lambda a: (a["road_id"] or 0, a["name"])):
        w = waits[v["name"]]
        print("  %-16s road %-3s lane %-2s node %-5s %-16s -> %s %s" % (v["name"], v["road_id"], v["lane"], v["junction"], v["reason"], w[1], "" if w[2] is None else "(gap %.1f m)" % w[2]))
    # follow vehicle->vehicle edges to find chains ending at junction waiters and any pure cycle
    cycles = set()
    for start in waits:
        seen, cur = [], start
        while cur in waits and waits[cur][0] == "vehicle" and waits[cur][1] and cur not in seen:
            seen.append(cur)
            cur = waits[cur][1]
        if cur in seen:
            cycles.add(tuple(seen[seen.index(cur):]))
    print("vehicle-only waiting cycles:", [list(c) for c in cycles] or "none")
    heads = Counter()
    for start in waits:
        cur, steps = start, 0
        while waits.get(cur, ("",))[0] == "vehicle" and waits[cur][1] and steps < 30:
            cur, steps = waits[cur][1], steps + 1
        heads[(cur, waits.get(cur, ("?",))[0])] += 1
    print("queue heads (vehicle at the front of each waiting chain, and why it waits):", dict(heads))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    main(p.parse_args().run)
