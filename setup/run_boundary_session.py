"""One finite session; fresh subprocesses prevent stale GPU/optimizer state."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from review_boundary_masks import atomic
ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "runs/boundary_20261003"


def main():
    state = dict(status="running", controller_pid=os.getpid(), active_stage=None, test_images_used=False)
    atomic(SESSION / "session_status.json", state)
    steps = [("control", ["setup/train_boundary_idd.py", "--name", "control"]),
             ("boundary_weight4", ["setup/train_boundary_idd.py", "--name", "boundary_weight4"]),
             ("evaluation", ["setup/evaluate_boundary_gpu.py"])]
    child = None
    try:
        for name, args in steps:
            if (SESSION / "STOP.json").exists():
                raise RuntimeError("Session stop requested")
            state.update(active_stage=name)
            with (SESSION / f"{name}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                state.update(worker_pid=child.pid, worker_process_group=child.pid)
                atomic(SESSION / "session_status.json", state)
                print(f"Started {name}, owned PID {child.pid}", flush=True)
                start = time.monotonic()
                while child.poll() is None:
                    if (SESSION / "STOP.json").exists() or time.monotonic()-start > 3000:
                        # Only the subprocess group just created by this controller.
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        raise RuntimeError("Owned stage stopped: requested stop or 50 minute timeout")
                    time.sleep(2)
                if child.returncode != 0:
                    raise RuntimeError(f"{name} failed; exit {child.returncode}; see its saved log")
                child = None
            state[name] = "completed"
            atomic(SESSION / "session_status.json", state)
        state.update(status="completed", active_stage=None, worker_pid=None, worker_process_group=None)
    except BaseException as error:
        state.update(status="failed_or_interrupted", error=repr(error))
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        raise
    finally:
        atomic(SESSION / "session_status.json", state)


if __name__ == "__main__":
    main()
