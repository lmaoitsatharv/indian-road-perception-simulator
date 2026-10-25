"""Historical session-bound overnight queue, watchdog and saved report.

No TensorFlow import here: the watchdog can stop a stalled GPU worker independently.
The recorded deadline and monitor lease remain enforced; this is not a general
training launcher for a new installation.
"""
import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSION = ROOT / "setup/overnight_20261003"
RUNS = ROOT / "runs/overnight_20261003"
DEADLINE = datetime.fromisoformat("2026-10-03T01:30:00+00:00").timestamp()
TAG = "tesla-overnight-20261003"


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def stop_reason():
    stop = read(SESSION / "STOP.json")
    if stop is not None:
        return stop.get("reason", "user_or_monitor_stop")
    if time.time() >= DEADLINE:
        return "7_am_ist_deadline"
    lease = read(SESSION / "monitor_lease.json", {})
    try:
        expires = datetime.fromisoformat(lease["expires_at_utc"]).timestamp()
    except (KeyError, ValueError):
        return "monitor_lease_unavailable"
    return "monitor_lease_expired" if time.time() >= expires else None


def owned(pid):
    try:
        environment = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        return f"IDD_OVERNIGHT_SESSION={TAG}".encode() in environment and os.getpgid(pid) == pid
    except (OSError, ProcessLookupError):
        return False


def terminate_owned(pid):
    if not pid or not owned(pid):
        return
    os.killpg(pid, signal.SIGTERM)
    until = time.time() + 8
    while time.time() < until and owned(pid):
        time.sleep(.5)
    if owned(pid):
        os.killpg(pid, signal.SIGKILL)


def report(state):
    entries = []
    for folder in sorted(RUNS.glob("trial_*")):
        config = read(folder / "run_config.json", {})
        status = read(folder / "run_status.json", {})
        verified = read(folder / "validation_review/validation_metrics.json", {})
        best = read(folder / "best_checkpoint.json", {})
        entries.append({"run": str(folder.relative_to(ROOT)), "status": status.get("status", "preparing"),
                        "epochs": status.get("epochs_completed"), "config": config,
                        "logged_best_iou": best.get("road_iou"),
                        "verified_iou": verified.get("global_road_iou") if verified.get("saved_checkpoint_matches_logged_iou") and verified.get("graph_predictions_match_keras_evaluation") else None,
                        "dice": verified.get("global_road_dice"), "error": status.get("error"),
                        "review_error": read(folder / "review_error.json")})
    baseline = read(ROOT / "runs/frozen_01/validation_review/validation_metrics.json", {})
    candidates = [(baseline.get("global_road_iou", 0), "runs/frozen_01/best.keras")]
    candidates.extend((entry["verified_iou"], entry["run"] + "/best.keras") for entry in entries if entry["verified_iou"] is not None)
    best_score, best_path = max(candidates)
    summary = {"session": state, "baseline_iou": baseline.get("global_road_iou"),
               "best_verified_iou": best_score, "best_verified_checkpoint": best_path,
               "trials": entries, "test_set_evaluated": False}
    write(RUNS / "comparison.json", summary)
    lines = ["# Overnight road segmentation report", "",
             f"Updated UTC: {datetime.now(timezone.utc).isoformat()}",
             f"Session: **{state.get('status')}**. Stop reason: {state.get('stop_reason', 'not stopped')}.",
             "Deadline: **3 October 2026, 7:00 AM IST**. Monitoring stops early at 95% usage or an expired lease.", "",
             "## Preserved baseline", "",
             "Verified backup: `backups/trained_baseline_20261003_0239/best.keras`.",
             "Original baseline validation IoU **92.0345%**, Dice **95.8521%**, epoch 14.",
             "Original data and split assignments remain unchanged: 5,507 train / 687 validation / 689 test.",
             "Masks remain 0/1; RGB normalization happens once. The test set has not been evaluated.", "",
             "## Experiments", "",
             "All trials start from the same backed-up baseline except explicitly labelled refinement trials.",
             "Trials 10 onward continue after the initial nine completed, under the same deadline and monitoring lease.",
             "Mixed precision, 512 x 512, GPU-only, seeded augmentation, float32 probability head/loss, loss scaling.",
             "Lighting changes apply only to training RGB; encoder BatchNorm remains frozen.", "",
             "| Trial | Status | Completed epochs | Verified validation IoU | Change |",
             "|---|---|---:|---:|---|"]
    for entry in entries:
        config = entry["config"]
        score = f"{entry['verified_iou']:.2%}" if entry["verified_iou"] is not None else "pending / unverified"
        change = f"aug={config.get('augmentation')}; blocks={config.get('first_encoder_block')}; LR={config.get('learning_rate')}; batch={config.get('batch_size')}; seed={config.get('seed')}; decoder BN={'frozen' if config.get('freeze_decoder_batchnorm') else 'trainable'}"
        lines.append(f"| {Path(entry['run']).name} | {entry['status']} | {entry['epochs'] or '-'} | {score} | {change} |")
        if entry["error"] or entry["review_error"]:
            lines.extend(["", f"Failure in {entry['run']}: {entry['error'] or entry['review_error']}", ""])
    lines.extend(["", "## Best verified result so far", "",
                  f"Checkpoint: `{best_path}`. Validation road IoU: **{best_score:.4%}**.",
                  f"Difference versus baseline: **{(best_score - baseline.get('global_road_iou', 0)) * 100:.4f} percentage points**.",
                  "Selection uses unchanged global validation IoU at threshold 0.5; no accuracy improvement is assumed.",
                  "Only reloaded checkpoints matching their logged result enter this comparison.", "",
                  "## Evidence to inspect", "",
                  "Each finished trial keeps exact source snapshots, configuration, epoch history, progress, memory-by-epoch",
                  "and atomically saved best checkpoint. Verified reviews keep all 687 per-image scores, 0/1 predicted",
                  "masks, illustrative overlays and weakest cases. Console/review logs are in setup/overnight_20261003.",
                  "The original baseline files and backup are never replaced. New trials use their own folders.", "",
                  "## Interpretation and remaining limits", "",
                  "Review full validation scores and difficult examples, especially ID 4160 and false road near vehicles.",
                  "Global overlap can hide weak individual scenes. Original mask-class definitions remain unconfirmed",
                  "in ambiguous examples. This small adaptive search repeatedly uses validation, so improvements",
                  "need later reserved-test and external-image checks. Sequence metadata is absent; related frames",
                  "may cross splits. The 110 exclusions remain a limitation. No labels were deleted or relabelled.",
                  "GPU allocator readings are TensorFlow-used memory, not whole-card reserved memory or other apps.",
                  "An interrupted trial may retain a valid earlier checkpoint, but is marked unverified until reloaded.",
                  "No RL, MATLAB simulation, metric projection, or final test evaluation occurred overnight.", "",
                  "## Cleanup", "",
                  f"Controller PID: {state.get('controller_pid')}; active worker PID: {state.get('active_worker_pid')}.",
                  "The watchdog terminates only tagged process groups created for this session. It does not stop",
                  "unrelated Jupyter, WSL distributions, games, or other Python tasks. Temporary keep-awake requests",
                  "are released by the Windows helper at session completion/deadline.", ""])
    temporary = RUNS / "MORNING_REPORT.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(RUNS / "MORNING_REPORT.md")


def guard():
    with (SESSION / "watchdog.log").open("a", buffering=1) as log:
        while True:
            state = read(SESSION / "session_status.json", {})
            if state.get("status") in {"completed", "stopped", "failed"}:
                break
            reason = stop_reason()
            if reason:
                print(f"{datetime.now(timezone.utc).isoformat()} stop: {reason}", file=log)
                write(SESSION / "STOP.json", {"reason": reason})
                terminate_owned(state.get("active_worker_pid"))
                # Allow the controller to write final records; only kill it if stalled.
                time.sleep(12)
                final = read(SESSION / "session_status.json", {})
                if final.get("status") not in {"completed", "stopped", "failed"}:
                    terminate_owned(state.get("controller_pid"))
                    final.update(status="stopped", stop_reason=reason, active_worker_pid=None,
                                 finished_at_utc=datetime.now(timezone.utc).isoformat())
                    write(SESSION / "session_status.json", final)
                    report(final)
                break
            time.sleep(2)


def run_process(arguments, name, state):
    log_path = SESSION / f"{name}.log"
    environment = {**os.environ, "IDD_OVERNIGHT_SESSION": TAG, "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONUNBUFFERED": "1"}
    with log_path.open("a") as log:
        child = subprocess.Popen(arguments, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(active_worker_pid=child.pid, active_task=name, updated_at_utc=datetime.now(timezone.utc).isoformat())
        write(SESSION / "session_status.json", state)
        while child.poll() is None:
            reason = stop_reason()
            if reason:
                terminate_owned(child.pid)
                break
            time.sleep(2)
        code = child.wait()
    state.update(active_worker_pid=None, last_exit_code=code)
    write(SESSION / "session_status.json", state)
    return code


def main():
    os.environ["IDD_OVERNIGHT_SESSION"] = TAG
    if not read(SESSION / "smoke_result.json", {}).get("status") == "passed":
        raise RuntimeError("Preparation was not verified")
    state = {"status": "running", "controller_pid": os.getpid(), "active_worker_pid": None,
             "deadline_utc": "2026-10-03T01:30:00+00:00", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "test_set_evaluated": False, "backup": "backups/trained_baseline_20261003_0239"}
    write(SESSION / "session_status.json", state)
    environment = {**os.environ, "IDD_OVERNIGHT_SESSION": TAG}
    with (SESSION / "watchdog_console.log").open("a") as log:
        watchdog = subprocess.Popen([sys.executable, __file__, "--guard"], cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["watchdog_pid"] = watchdog.pid
    write(SESSION / "session_status.json", state)
    specifications = [
        ("fine_only", "none", 13, 1e-5, 8, 42, 15),
        ("lighting_only", "shadow", None, 1e-4, 8, 42, 12),
        ("fine_lighting", "shadow", 13, 1e-5, 8, 42, 15),
        ("fine_light", "light", 13, 1e-5, 8, 42, 12),
        ("deeper_lighting", "shadow", 6, 5e-6, 8, 42, 12),
        ("fine_lighting_lr", "shadow", 13, 3e-5, 8, 42, 12),
        ("fine_lighting_b16", "shadow", 13, 1e-5, 16, 42, 12),
        ("replication", "shadow", 13, 1e-5, 8, 73, 12),
        ("refinement", "shadow", 13, 3e-6, 8, 42, 10),
    ]
    try:
        report(state)
        for index, (label, augmentation, first_block, lr, batch, seed, epochs) in enumerate(specifications, 1):
            if stop_reason() or time.time() > DEADLINE - 240:
                break
            run_relative = f"runs/overnight_20261003/trial_{index:02d}_{label}"
            initial = "runs/frozen_01/best.keras"
            if label in {"replication", "refinement"}:
                comparison = read(RUNS / "comparison.json", {})
                chosen = comparison.get("best_verified_checkpoint", initial)
                chosen_config = read((ROOT / chosen).parent / "run_config.json", {})
                # Replication repeats the selected configuration from the same baseline.
                if chosen != initial:
                    augmentation = chosen_config.get("augmentation", augmentation)
                    first_block = chosen_config.get("first_encoder_block", first_block)
                    lr = chosen_config.get("learning_rate", lr) if label == "replication" else 3e-6
                    batch = chosen_config.get("batch_size", batch)
                if label == "refinement":
                    initial = chosen
            spec = {"name": label, "run_dir": run_relative, "augmentation": augmentation,
                    "first_encoder_block": first_block, "learning_rate": lr,
                    "batch_size": batch, "seed": seed, "epochs": epochs,
                    "initial_checkpoint": initial, "deadline_epoch": DEADLINE}
            spec_path = SESSION / f"trial_{index:02d}_spec.json"
            write(spec_path, spec)
            code = run_process([sys.executable, "overnight_idd_worker.py", "--session", str(SESSION), "--spec", str(spec_path)], f"trial_{index:02d}_training", state)
            run = ROOT / run_relative
            if code and (run / "run_status.json").is_file():
                failed = read(run / "run_status.json", {})
                if failed.get("status") == "running":
                    failed.update(status="interrupted_or_failed", error=stop_reason() or f"worker exited {code}")
                    write(run / "run_status.json", failed)
            report(state)
            if stop_reason():
                break
            if code == 0:
                review_code = run_process([sys.executable, "review_idd_validation.py", "--run-dir", str(run)], f"trial_{index:02d}_review", state)
                if review_code:
                    write(run / "review_error.json", {"exit_code": review_code, "log": f"setup/overnight_20261003/trial_{index:02d}_review.log"})
                elif not stop_reason():
                    # Plot in the current Linux runtime, or an explicit plotting environment.
                    plot_python = os.environ.get("PROJECT_PLOT_PYTHON", sys.executable)
                    run_process([plot_python, str(ROOT / "plot_idd_training.py"), "--run-dir",
                                 str(run)], f"trial_{index:02d}_plots", state)
                report(state)
        state.update(status="stopped" if stop_reason() else "completed",
                     stop_reason=stop_reason() or "planned_controlled_trials_complete",
                     finished_at_utc=datetime.now(timezone.utc).isoformat(), active_worker_pid=None)
    except BaseException as error:
        state.update(status="failed", stop_reason=f"{type(error).__name__}: {error}")
        write(SESSION / "STOP.json", {"reason": state["stop_reason"]})
        terminate_owned(state.get("active_worker_pid"))
        state["active_worker_pid"] = None
    finally:
        write(SESSION / "session_status.json", state)
        report(state)
        try:
            watchdog.wait(timeout=15)
        except subprocess.TimeoutExpired:
            terminate_owned(watchdog.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard", action="store_true")
    args = parser.parse_args()
    guard() if args.guard else main()
