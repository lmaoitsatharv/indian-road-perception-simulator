"""Predeclare one fresh synthetic confirmation episode without scoring pixels."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from verify_run import audit

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = ROOT / "runs/retention_holdout_capture_20261021_01"
    output.mkdir(exist_ok=False)
    argv = [sys.executable, "-u", "setup/carla/run_session.py", "--india", "--smoke", "--observe-road",
            "--road-camera-hz", "1", "--quality", "Low", "--surface-style", "asphalt", "--weather", "clear",
            "--case", "mixed", "--seed", "137", "--speed", "3", "--sim-seconds", "110", "--seconds", "420"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "role": "withheld_after_selection",
              "plan": {"case": "mixed", "seed": 137, "weather": "clear", "capture_hz": 1, "duration_s": 110},
              "argv": argv, "state": "predeclared_before_capture", "pixels_scored": False,
              "limitations": "Fresh seed in the same town and scene family; synthetic confirmation, not real-road accuracy"}

    def save():
        temporary = output / "capture.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "capture.json")

    save()
    previous = json.loads((ROOT / "setup/carla/server_owner.json").read_text(encoding="utf-8"))
    process = subprocess.run(argv, cwd=ROOT)
    owner = json.loads((ROOT / "setup/carla/server_owner.json").read_text(encoding="utf-8"))
    report.update(exit_code=process.returncode, launcher=owner)
    if (owner["output"] == previous["output"] or process.returncode or owner.get("status") != "completed"
            or not owner.get("owned_job_handles_closed")):
        report["state"] = "failed_preserved"
        save()
        raise RuntimeError("Owned capture did not finish cleanly; preserved for inspection")
    episode = next(Path(owner["output"]).glob("episode_*"))
    summary = json.loads((episode / "road_camera/capture_summary.json").read_text(encoding="utf-8"))
    report.update(episode=str(episode), trace_audit=audit(episode), capture_summary=summary)
    report["state"] = "captured_unscored" if report["trace_audit"]["passed"] and not any(summary["dropped"].values()) else "capture_audit_failed"
    save()
    print(json.dumps({"state": report["state"], "episode": str(episode), "paired_frames": summary["paired_frames"]}), flush=True)
    return 0 if report["state"] == "captured_unscored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
