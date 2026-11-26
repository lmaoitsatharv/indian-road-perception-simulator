"""Junction movement times from camera-driving town runs (for learned routing).

For every junction the camera car went through with a destination: which movement (junction, street in, street
out), how long it waited for its turn there (decision reason "waiting_turn"), and how long it took from 20 m before
the junction to its exit. Reads each run's camera_driving/navigation.jsonl (route legs, junction entry/exit) and
decisions.jsonl (reasons, 10 Hz).

  .venv-carla/Scripts/python.exe rl/junction_times.py [town folders...]     (default: every town run with a route)

Writes rl/data/junction_times.json: one record per traversal, and per movement count / mean / median.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def traversals(town: Path):
    nav_file = town / "camera_driving" / "navigation.jsonl"
    network = json.loads((town / "network.json").read_text())
    connectors = {c["id"]: c for c in network["connectors"]}
    events = [json.loads(l) for l in nav_file.open(encoding="utf-8")]
    reasons = []
    for line in (town / "camera_driving" / "decisions.jsonl").open(encoding="utf-8"):
        k = line.find('"sim_s": ')
        t = float(line[k + 9:line.find(',', k)])
        k = line.find('"previous_reason": "')
        reasons.append((t, line[k + 20:line.find('"', k + 20)] if k > 0 else ""))
    times = np.array([t for t, _ in reasons])
    waiting = np.array([r == "waiting_turn" for _, r in reasons])
    learned = any(r in ("speed_limit", "learned_slow") for _, r in reasons)
    out = []
    legs, entered = None, None
    for e in events:
        if e["event"] in ("route_planned", "replanned"):
            legs = [leg for leg in e["legs"] if leg[2]]          # junction connectors on the route, in order
        elif e["event"] == "junction_entered" and legs:
            entered = e
        elif e["event"] == "junction_left" and entered is not None and legs:
            node = entered["node"]
            leg = next((leg for leg in legs if leg[2] == node), None)
            if leg is None:
                entered = None
                continue
            legs = legs[legs.index(leg) + 1:]
            c = connectors.get(leg[0])
            t0, t1 = entered["sim_s"], e["sim_s"]
            # waiting for its turn: from 30 s before the 20 m mark to the exit (the hold starts up to 35 m out)
            span = (times >= t0 - 30.) & (times <= t1)
            out.append({"run": town.name, "learned_speed": learned, "node": node, "turn": leg[3], "connector": leg[0],
                        "from": None if c is None else c["incoming"], "to": None if c is None else c["outgoing"],
                        "wait_s": round(float(waiting[span].sum() * np.median(np.diff(times))) if span.sum() > 1 else 0., 1),
                        "through_s": round(t1 - t0, 1), "entered_s": t0})
            entered = None
    return out


def main(folders):
    if not folders:
        folders = sorted({str(Path(p).parents[1]) for p in glob.glob(str(ROOT / "runs/carla_indian_worlds_*/camera_driving/navigation.jsonl"))})
    rows = []
    for f in folders:
        try:
            rows += traversals(Path(f))
        except (OSError, json.JSONDecodeError) as error:
            print("skipped", f, error)
    per = {}
    for r in rows:
        per.setdefault(f"{r['node']} {r['from']}->{r['to']} ({r['turn']})", []).append(r["wait_s"])
    summary = {k: {"n": len(v), "mean_wait_s": round(float(np.mean(v)), 1), "median_wait_s": round(float(np.median(v)), 1)}
               for k, v in sorted(per.items())}
    out = ROOT / "rl" / "data" / "junction_times.json"
    out.write_text(json.dumps({"traversals": rows, "movements": summary}, indent=1))
    for k, v in summary.items():
        print(f"{k:28s} n={v['n']:3d} mean wait {v['mean_wait_s']:5.1f} s  median {v['median_wait_s']:5.1f} s")
    print(len(rows), "traversals from", len(folders), "runs")


if __name__ == "__main__":
    main(sys.argv[1:])
