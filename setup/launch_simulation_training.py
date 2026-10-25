"""Launch one fresh simulation run in its own persistent Linux process session."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setup.prepare_simulation_dataset import save


def launch(name, verification):
    if sys.platform != "linux" or Path(name).name != name or not name.startswith("simulation_"):
        raise ValueError("Launch from the verified WSL runtime with a simulation_* run name")
    run_dir = ROOT / "runs" / name
    owner_dir = ROOT / "runs" / (name + "_launcher")
    if run_dir.exists() or owner_dir.exists():
        raise FileExistsError("This run was already attempted; inspect its saved status")
    if json.loads(verification.read_text())["status"] != "passed":
        raise ValueError("A passed verification report is required")
    owner_dir.mkdir(parents=True)
    environment = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2", PYTHONUNBUFFERED="1",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command = [sys.executable, "-u", str(ROOT / "train_simulation.py"),
               "--run-dir", str(run_dir), "--verification", str(verification.resolve())]
    with (owner_dir / "training.log").open("xb", buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
    record = {"status": "launched", "pid": process.pid, "process_group": process.pid,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "run_directory": str(run_dir), "log": str(owner_dir / "training.log"),
              "command": command, "test_model_evaluation_started": False}
    process_info = Path(f"/proc/{process.pid}/stat")
    if process_info.exists():
        record["linux_process_start_ticks"] = process_info.read_text().split(") ", 1)[1].split()[19]
    save(owner_dir / "owner.json", record)
    time.sleep(.25)
    if process.poll() is not None:
        raise RuntimeError(f"Trainer exited immediately: inspect {owner_dir / 'training.log'}")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    launch(args.run_name, args.verification)
