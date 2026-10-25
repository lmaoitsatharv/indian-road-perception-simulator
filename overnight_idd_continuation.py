"""Historical continuation of a fixed, lease/deadline-bound experiment session.

Requires that session's prior state and preserves all stop/deadline checks.
This is not a general-purpose training entry point.
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import overnight_idd_controller as controller


def main(queue="10_12"):
    # The independent guard can also terminate this controller's own process group.
    if os.getpgrp() != os.getpid():
        os.setsid()
    if os.environ.get("IDD_OVERNIGHT_SESSION") != controller.TAG:
        raise RuntimeError("Missing overnight ownership tag")
    with (controller.SESSION / "continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = controller.read(controller.SESSION / "session_status.json", {})
        if previous.get("status") not in {"completed", "stopped"} or previous.get("active_worker_pid"):
            raise RuntimeError("Previous queue has not finished")
        if controller.stop_reason():
            raise RuntimeError(controller.stop_reason())
        if controller.DEADLINE - time.time() < 600:
            raise RuntimeError("Insufficient time for a new useful trial")
        plans = ([
            (10, "shadow_fixed_bn", "shadow", 1e-5, 12, False, 13, True),
            (11, "light_fixed_bn", "light", 1e-5, 12, False, 13, True),
            (12, "refinement_fixed_bn", "none", 3e-6, 10, True, 13, True),
        ] if queue == "10_12" else [
            (13, "shadow_fixed_bn_low_lr", "shadow", 3e-6, 8, False, 13, True),
            (14, "no_lighting_fixed_bn_low_lr", "none", 3e-6, 8, False, 13, True),
            (15, "frozen_shadow_low_lr", "shadow", 1e-5, 8, False, None, False),
        ])
        for index, label, *_ in plans:
            if (controller.RUNS / f"trial_{index:02d}_{label}").exists():
                raise FileExistsError(f"Trial {index} already exists")
        archive = "initial_queue_completed_status.json" if queue == "10_12" else "continuation_10_12_completed_status.json"
        controller.write(controller.SESSION / archive, previous)
        state = {"status": "running", "controller_pid": os.getpid(), "active_worker_pid": None,
                 "deadline_utc": "2026-10-03T01:30:00+00:00",
                 "started_at_utc": previous["started_at_utc"],
                 "continuation_started_at_utc": datetime.now(timezone.utc).isoformat(),
                 "test_set_evaluated": False, "backup": previous["backup"], "queue": f"continuation_{queue}"}
        controller.write(controller.SESSION / "session_status.json", state)
        with (controller.SESSION / "watchdog_console.log").open("a") as log:
            watchdog = subprocess.Popen([sys.executable, str(controller.ROOT / "overnight_idd_controller.py"), "--guard"],
                                        cwd=controller.ROOT, env=os.environ.copy(), stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        state["watchdog_pid"] = watchdog.pid
        controller.write(controller.SESSION / "session_status.json", state)
        try:
            controller.report(state)
            for index, label, augmentation, lr, epochs, refinement, first_block, freeze_bn in plans:
                if controller.stop_reason() or time.time() >= controller.DEADLINE - 600:
                    break
                initial = "runs/frozen_01/best.keras"
                if refinement:
                    initial = controller.read(controller.RUNS / "comparison.json", {}).get("best_verified_checkpoint", initial)
                relative = f"runs/overnight_20261003/trial_{index:02d}_{label}"
                run = controller.ROOT / relative
                if run.exists():
                    raise FileExistsError(run)
                spec = {"name": label, "run_dir": relative, "augmentation": augmentation,
                        "first_encoder_block": first_block, "learning_rate": lr, "batch_size": 16,
                        "seed": 42, "epochs": epochs, "initial_checkpoint": initial,
                        "freeze_decoder_batchnorm": freeze_bn, "deadline_epoch": controller.DEADLINE,
                        "rationale": ("Test gentler updates from the original baseline with matched batch/augmentation; isolate decoder BatchNorm or encoder trainability. Paired labels and normalization are preserved." if queue == "13_15" else "Hold decoder BatchNorm statistics fixed to test whether their drift obscures small fine-tuning gains; paired labels and normalization are preserved.")}
                spec_path = controller.SESSION / f"trial_{index:02d}_spec.json"
                controller.write(spec_path, spec)
                code = controller.run_process([sys.executable, "overnight_idd_worker.py", "--session", str(controller.SESSION),
                                               "--spec", str(spec_path)], f"trial_{index:02d}_training", state)
                if code:
                    failed = controller.read(run / "run_status.json", {})
                    failed.update(status="interrupted_or_failed", error=controller.stop_reason() or f"worker exited {code}")
                    if run.exists():
                        controller.write(run / "run_status.json", failed)
                controller.report(state)
                if controller.stop_reason():
                    break
                if code == 0:
                    review_code = controller.run_process([sys.executable, "review_idd_validation.py", "--run-dir", str(run)],
                                                         f"trial_{index:02d}_review", state)
                    if review_code:
                        controller.write(run / "review_error.json", {"exit_code": review_code,
                                         "log": f"setup/overnight_20261003/trial_{index:02d}_review.log"})
                    elif not controller.stop_reason():
                        plot_python = os.environ.get("PROJECT_PLOT_PYTHON", sys.executable)
                        controller.run_process([plot_python, str(controller.ROOT / "plot_idd_training.py"), "--run-dir",
                                                str(run)],
                                               f"trial_{index:02d}_plots", state)
                    controller.report(state)
            state.update(status="stopped" if controller.stop_reason() else "completed",
                         stop_reason=controller.stop_reason() or "continuation_trials_complete",
                         active_worker_pid=None, finished_at_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            state.update(status="failed", stop_reason=f"{type(error).__name__}: {error}")
            controller.write(controller.SESSION / "STOP.json", {"reason": state["stop_reason"]})
            controller.terminate_owned(state.get("active_worker_pid"))
            state["active_worker_pid"] = None
        finally:
            controller.write(controller.SESSION / "session_status.json", state)
            controller.report(state)
            try:
                watchdog.wait(timeout=15)
            except subprocess.TimeoutExpired:
                controller.terminate_owned(watchdog.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", choices=["10_12", "13_15"], default="10_12")
    main(parser.parse_args().queue)
