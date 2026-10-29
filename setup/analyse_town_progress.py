"""Read-only: per-vehicle progress in a combined-town run (stalls, movement after a time, junction tokens).

Reads actor_trace.jsonl and result.json from one run folder. The bus is expected to stop for good at its service stop.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main(run, after):
    run = Path(run)
    result = json.loads((run / "result.json").read_text())
    last, moved, since, spans, reasons = {}, defaultdict(float), {}, defaultdict(list), defaultdict(Counter)
    end = 0.
    with open(run / "actor_trace.jsonl") as f:
        for line in f:
            record = json.loads(line)
            t = end = record["sim_s"]
            for a in record["actors"]:
                name = a["name"]
                if t >= after and name in last:
                    moved[name] += max(0., a["distance_m"] - last[name])
                last[name] = a["distance_m"]
                if a["speed"] < .1:
                    since.setdefault(name, t)
                    reasons[name][a["reason"]] += 1
                elif name in since:
                    spans[name].append(t - since.pop(name))
    for name, t in since.items():
        spans[name].append(end - t)
    longest = {name: max(spans[name], default=0.) for name in last}
    summary = {
        "run": run.name, "passed": result["passed"], "sim_seconds": round(end, 1),
        "contacts": len(result.get("collisions", [])), "overlaps": len(result.get("vehicle_overlaps", [])),
        "departures": len(result.get("vehicle_centre_departures", [])),
        "wrong_direction": len(result.get("wrong_direction_events", [])),
        "min_gap_m": result.get("sampled_min_vehicle_gap_m"),
        "longest_all_vehicles_stopped_s": result.get("longest_all_vehicles_stopped_s"),
        "longest_ego_stall_s": result.get("longest_ego_stall_s"),
        "junction_token_grants": result.get("junction_token_grants"),
        "watchdog_releases": len(result.get("junction_watchdog_releases", [])),
        "ego_m": round(result["travelled_m"]["ego"]), "all_vehicles_m": round(sum(result["travelled_m"].values())),
        "moved_over_5m_after_s": {"after_s": after, "vehicles": sum(v > 5 for v in moved.values()), "of": len(last)},
        "longest_stops_s": [{"actor": n, "s": round(s), "main_reason": reasons[n].most_common(1)[0][0] if reasons[n] else None}
                            for n, s in sorted(longest.items(), key=lambda p: -p[1])[:5]],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--after", type=float, default=786.)
    args = p.parse_args()
    main(args.run, args.after)
