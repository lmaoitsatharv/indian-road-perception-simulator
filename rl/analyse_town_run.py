"""Score a camera-driving town run for the learned-speed comparison (simulator ground truth, scoring only).

  .venv-carla/Scripts/python.exe rl/analyse_town_run.py runs/carla_indian_worlds_smoke_<...>  [more folders]

For each run: destinations reached and the time of every leg (destination set -> arrived), distance,
collisions, resets, time over the posted limit (each street's district speed; inside a junction the lower of
its two streets'; more than 0.5 m/s over), hard braking (below -3 m/s^2), time per decision reason, and how
long the learned speed chose to go below its shield. Writes <run>/camera_driving/rl_speed_score.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DT = .05


def ego_rows(trace: Path):
    rows = []
    with trace.open(encoding="utf-8") as f:
        for line in f:
            i = line.find('"actors": [')
            j = line.find('}, {', i)
            a = json.loads(line[i + 11:j + 1] if j > 0 else line[i + 11:line.rfind("]")])
            if a.get("name") != "ego":
                continue
            k = line.find('"sim_s": ')
            rows.append((float(line[k + 9:line.find(',', k)]), a["speed"], a.get("road_id"), a.get("lane"), a.get("junction"),
                         a.get("reason"), a.get("target")))
    return rows


def limits(network):
    streets = {s["id"]: s for s in network["streets"]}
    connectors = {c["id"]: c for c in network["connectors"]}

    def limit(road_id):
        if road_id in streets:
            return float(streets[road_id]["cruise"])
        c = connectors.get(road_id)
        if c is None:
            return None
        return float(min(streets[c["incoming"]]["cruise"], streets[c["outgoing"]]["cruise"]))
    return limit


def score(town: Path):
    result = json.loads((town / "result.json").read_text())
    network = json.loads((town / "network.json").read_text())
    drive = result.get("camera_driving") or {}
    limit = limits(network)
    rows = ego_rows(town / "actor_trace.jsonl")
    t = np.array([r[0] for r in rows]); v = np.array([r[1] for r in rows])
    lim = np.array([np.nan if limit(r[2]) is None else limit(r[2]) for r in rows])
    over = v > lim + .5
    accel = np.r_[0., np.diff(v) / DT]
    nav = drive.get("navigation") or {}
    legs = []
    reason_t = np.array([r[5] or "" for r in rows])
    for d in nav.get("destinations", []):
        end = d["arrived_s"] if d["arrived_s"] is not None else t[-1]
        inside = (t >= d["set_s"]) & (t <= end)
        # waiting for its turn at a junction depends on when the car happens to arrive there (the town's
        # turn-taking), not on how it drives; shown separately so pairs can be compared without that luck
        waits = float((inside & (reason_t == "waiting_turn") & (v < .5)).sum() * DT)
        legs.append({"name": d["name"], "set_s": d["set_s"], "arrived_s": d["arrived_s"],
                     "leg_s": None if d["arrived_s"] is None else round(d["arrived_s"] - d["set_s"], 1),
                     "turn_wait_s": round(waits, 1),
                     "leg_without_turn_waits_s": None if d["arrived_s"] is None else round(d["arrived_s"] - d["set_s"] - waits, 1)})
    decisions = town / "camera_driving" / "decisions.jsonl"
    reasons = drive.get("reason_s", {})
    rl_slow = reasons.get("learned_slow", 0.)
    fractions, raised = [], 0
    for line in decisions.open(encoding="utf-8"):
        k = line.find('"rl": {')
        if k < 0:
            continue
        rec = json.loads(line[k + 6:line.find('}', k) + 1])
        fractions.append(rec.get("fraction", 1.))
        raised += "set_point" in rec
    fractions = np.array(fractions)
    session = json.loads((town / "camera_driving" / "session.json").read_text())
    # Speed on open road, by posted limit: where only the cruise / the posted limit held the car back
    reason = np.array([r[5] or "" for r in rows])
    open_road = np.isin(reason, ("cruise", "speed_limit")) & (v > 1.)
    by_limit = {}
    for value in sorted(set(lim[open_road & np.isfinite(lim)].tolist())):
        m = open_road & (lim == value)
        if m.sum() >= 20:
            by_limit[f"{value:.1f}"] = {"seconds": round(float(m.sum() * DT), 1), "median_m_s": round(float(np.median(v[m])), 2),
                                        "p95_m_s": round(float(np.percentile(v[m], 95)), 2)}
    out = {"town": town.name, "learned_speed": bool((session.get("driver") or {}).get("rl_speed")),
           "policy": session.get("rl_speed_policy"), "sim_s": result.get("sim_seconds"),
           "distance_m": drive.get("distance_m"), "reached": nav.get("reached"), "legs": legs,
           "collisions": drive.get("collision_count"), "collision_with": [c.get("with") for c in drive.get("collisions", [])],
           "resets": drive.get("reset_count"), "off_road_s": drive.get("off_road_s"), "wrong_side_s": drive.get("wrong_side_s"),
           "over_limit_s": round(float(over.sum() * DT), 2),
           "over_limit_strict_s": round(float((v > lim + .05).sum() * DT), 2),
           "over_limit_share_moving": round(float(over.sum() / max(1, (v > .5).sum())), 4),
           "max_over_m_s": round(float(np.nanmax(np.where(np.isnan(lim), 0., v - lim))), 2),
           "hard_brake_s": round(float((accel < -3.).sum() * DT), 2), "emergency_s": round(reasons.get("emergency_vehicle", 0.) +
                                                                                       reasons.get("emergency_person", 0.), 2),
           "learned_slow_s": round(rl_slow, 2),
           "policy_frames": int(len(fractions)), "policy_below_shield_share": round(float((fractions < 1.).mean()), 4) if len(fractions) else None,
           "policy_mean_fraction": round(float(fractions.mean()), 4) if len(fractions) else None,
           "limit_hold_share": round(raised / len(fractions), 4) if len(fractions) else None,
           "stopped_s": drive.get("stopped_s"), "mean_speed_moving": round(float(v[v > .5].mean()), 2),
           "reason_s": {k: round(x, 1) for k, x in sorted(reasons.items(), key=lambda kv: -kv[1])},
           "open_road_speed_by_limit": by_limit, "town_passed": result.get("passed")}
    (town / "camera_driving" / "rl_speed_score.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        folder = Path(arg)
        if (folder / "status.json").exists() and not (folder / "result.json").exists():
            folder = Path(json.loads((folder / "status.json").read_text())["town_output"])
        print(json.dumps(score(folder), indent=1))
