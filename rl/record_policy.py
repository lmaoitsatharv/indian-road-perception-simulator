"""Install a trained speed policy as the camera car's and record what it is (models/rl_speed_policy.json).

  .venv-carla/Scripts/python.exe rl/record_policy.py runs/rl_speed_<run>

Copies <run>/policy.npz to models/rl_speed/policy.npz (binaries stay local, like every other model here) and writes
the tracked metadata: its SHA-256, the training run, iteration and decisions, the stand-in evaluation next to the
baselines, and the training settings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(run: Path):
    run = run.resolve()
    source = run / "policy.npz"
    target = ROOT / "models" / "rl_speed" / "policy.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    meta = json.loads(str(np.load(target)["meta"]))
    keep = ("time_s", "collisions", "near_miss_s", "person_near_miss_s", "hard_brake_s", "emergency_s", "over_limit_s",
            "max_over_m_s", "violations", "timeouts", "episodes")
    record = {"schema": "rl_speed_policy_v1", "path": "models/rl_speed/policy.npz",
              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "training_run": str(run.relative_to(ROOT)).replace("\\", "/"),
              "iteration": meta.get("iteration"), "decisions_trained": meta.get("samples"),
              "standin_eval": {k: meta["eval"].get(k) for k in keep},
              "standin_baselines": {m: {k: v.get(k) for k in keep} for m, v in meta.get("baselines", {}).items()},
              "settings": {k: meta["config"].get(k) for k in ("envs", "steps", "iterations", "lr", "gamma", "lam", "hidden", "wait_median",
                                                               "limit_hold", "entry_ease", "stress", "seed")},
              "note": "Simulation only: trained and evaluated in rl/speed_env.py; see docs/RL_SPEED_RESULTS.md for town drives."}
    (ROOT / "models" / "rl_speed_policy.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
