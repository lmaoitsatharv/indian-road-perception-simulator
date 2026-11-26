"""Side-by-side table of town drives scored by rl/analyse_town_run.py (markdown).

  .venv-carla/Scripts/python.exe rl/compare_town_pairs.py runs/camera_driving_A runs/camera_driving_B [...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
from analyse_town_run import score  # noqa: E402


def town_of(folder: Path) -> Path:
    if (folder / "result.json").exists():
        return folder
    return Path(json.loads((folder / "status.json").read_text())["town_output"])


def main(folders):
    rows = []
    for f in folders:
        town = town_of(Path(f))
        cached = town / "camera_driving" / "rl_speed_score.json"
        rows.append((Path(f).name, json.loads(cached.read_text()) if cached.exists() else score(town)))
    names = [n for n, _ in rows]
    print("| Measure | " + " | ".join(names) + " |")
    print("| --- |" + " --- |" * len(rows))

    def line(label, fn):
        print(f"| {label} | " + " | ".join(str(fn(r)) for _, r in rows) + " |")
    line("Driver", lambda r: "learned speed" if r["learned_speed"] else "rule")
    line("Destinations reached", lambda r: f"{r['reached']} of {len(r['legs'])}")
    for i in range(max(len(r["legs"]) for _, r in rows)):
        line(f"Leg {i + 1}", lambda r, i=i: "-" if i >= len(r["legs"]) else
             f"{r['legs'][i]['name']}: {r['legs'][i]['leg_s'] if r['legs'][i]['leg_s'] is not None else 'not reached'} s")
    line("Last arrival (s)", lambda r: max([l["arrived_s"] for l in r["legs"] if l["arrived_s"] is not None], default="-"))
    line("Stopped waiting for a turn, all legs (s)", lambda r: round(sum(l.get("turn_wait_s", 0.) for l in r["legs"]), 1))
    line("Reached legs without turn waits (s)", lambda r: round(sum(l["leg_without_turn_waits_s"] for l in r["legs"]
                                                                      if l.get("leg_without_turn_waits_s") is not None), 1))
    line("Collisions", lambda r: r["collisions"])
    line("Resets", lambda r: r["resets"])
    line("Over the posted limit (> 0.5 m/s, s)", lambda r: r["over_limit_s"])
    line("Over the posted limit at all (> 0.05 m/s, s)", lambda r: r.get("over_limit_strict_s", "-"))
    line("Largest excess over a limit (m/s)", lambda r: r["max_over_m_s"])
    line("Hard braking (< -3 m/s², s)", lambda r: r["hard_brake_s"])
    line("Emergency braking (s)", lambda r: r["emergency_s"])
    line("Off road / wrong side (s)", lambda r: f"{round(r['off_road_s'], 1)} / {round(r['wrong_side_s'], 1)}")
    line("Mean speed while moving (m/s)", lambda r: r["mean_speed_moving"])
    line("Policy below its shield (share of frames)", lambda r: r.get("policy_below_shield_share") if r["learned_speed"] else "-")
    line("Limit held with a raised set-point (share)", lambda r: r.get("limit_hold_share") if r["learned_speed"] else "-")
    line("Waiting for a turn (s)", lambda r: r["reason_s"].get("waiting_turn", 0.))
    line("Town's own pass check", lambda r: r["town_passed"])


if __name__ == "__main__":
    main(sys.argv[1:])
