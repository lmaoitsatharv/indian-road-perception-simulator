"""Disjoint declared training/development/withheld captures for a model candidate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from verify_run import audit

ROOT = Path(__file__).resolve().parents[2]
PLAN = [("training", "mixed", 11, "overcast", 110),
        ("development", "clear_overtake", 13, "dusk", 60),
        ("withheld", "mixed", 91, "wet", 110)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Keep evidence in a new runs/ directory")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "episodes": records,
              "capture_hz": 1, "same_town": True, "completed": False,
              "limits": "Weather/case/seed separation within one town, not independent real-world scenes"}
    for role, case, seed, weather, duration in PLAN:
        argv = [sys.executable, "-u", "setup/carla/run_session.py", "--india", "--smoke", "--observe-road",
                "--road-camera-hz", "1", "--quality", "Low", "--surface-style", "asphalt", "--weather", weather,
                "--case", case, "--seed", str(seed), "--speed", "3", "--sim-seconds", str(duration), "--seconds", "420"]
        previous = json.loads((ROOT / "setup/carla/server_owner.json").read_text())["output"]
        launched = subprocess.run(argv, cwd=ROOT)
        owner = json.loads((ROOT / "setup/carla/server_owner.json").read_text())
        current = owner["output"] if owner["output"] != previous else None
        record = {"role": role, "case": case, "seed": seed, "weather": weather, "output": current,
                  "argv": argv, "exit_code": launched.returncode}
        records.append(record)
        try:
            if not current or launched.returncode or owner.get("status") != "completed" or not owner.get("owned_job_handles_closed"):
                raise RuntimeError("Owned capture did not finish cleanly")
            episode = next(Path(current).glob("episode_*"))
            result = audit(episode)
            record.update(episode=str(episode), audit_passed=result["passed"], audit=result)
            summary = json.loads((episode / "road_camera/capture_summary.json").read_text())
            record["capture_summary"] = summary
            if not result["passed"] or any(summary["dropped"].values()):
                raise RuntimeError("Capture/traffic checks failed; preserve and inspect evidence")
        except Exception as error:
            record["error"] = str(error)
        (output / "captures.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k: record.get(k) for k in ("role", "episode", "audit_passed", "error")}), flush=True)
        if "error" in record:
            return 1
    report["completed"] = True
    (output / "captures.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
