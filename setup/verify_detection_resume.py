"""Verify real optimizer execution after restoring the interrupted detector run."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import file_sha, save_json
from setup.verify_detection_runtime import protection


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(name):
    if Path(name).name!=name or not name.startswith("detection_train_"):
        raise ValueError("Expected a project detector run")
    run=ROOT/"runs"/name
    active=read(run/"active_recovery.json")
    # Evidence records use Linux paths; resolve the saved run-local suffix on either OS.
    suffix=active["directory"].split("/recovery/",1)[1]
    if Path(suffix).name!=suffix:
        raise ValueError("Invalid recovery evidence directory")
    evidence=run/"recovery"/suffix
    restored=read(evidence/"resume_verification.json")
    framework=read(evidence/"framework_resume.json")
    first=read(evidence/"first_training_batches.json")
    readiness=read(evidence/"readiness.json")
    original=read(evidence/"status.json")
    old_batches=read(evidence/"batch_progress.json")
    current=read(run/"batch_progress.json")
    owner=run.with_name(name+"_launcher")
    awake=read(owner/"keep_awake.json")
    resume_request=read(owner/"user_resume_requested.json") if (owner/"user_resume_requested.json").is_file() else {}
    sources={p.relative_to(evidence/"source").as_posix():file_sha(p)==file_sha(ROOT/p.relative_to(evidence/"source"))
        for p in (evidence/"source").rglob("*") if p.is_file()}
    checks={
        "checkpoint_backup_intact":file_sha(evidence/"weights"/active.get("checkpoint_file","last.pt"))==restored["checkpoint_sha256"]==active["checkpoint_sha256"],
        "full_dataset_file_audit_passed":readiness.get("ready_for_explicit_training_start") is True and readiness.get("full_file_check_performed") is True,
        "original_recipe_preserved":readiness["recipe_sha256"]==file_sha(ROOT/"detection_refinement_clean_recipe.json"),
        "epoch_restored":framework["restored_start_epoch"]==restored["completed_epochs_restored"]==original["epochs_completed"],
        "optimizer_recovery_mode_correct":(framework["optimizer_state_entries"]==0 if restored.get("optimizer_reset") else framework["optimizer_state_entries"]>0) and framework["restored_ema_updates"]==restored["ema_updates"],
        "new_backward_batches_and_optimizer_steps":first["passed"] is True and first["session_batches_completed"]>=10 and first["session_optimizer_steps"]>0,
        "first_resumed_epoch_correct":first["epoch"]==restored["completed_epochs_restored"]+1,
        "fresh_progress_written":datetime.fromisoformat(current["checked_utc"])>datetime.fromisoformat(old_batches["checked_utc"]),
        "losses_finite":first["sampled_losses_finite"] is True and current["sampled_losses_finite"] is True,
        "classification_loss_stable":first["losses"]["cls_loss"]<20 and current["losses"]["cls_loss"]<20,
        "multiple_updates_stable":current["optimizer_steps_completed"]>=first["optimizer_steps_completed"]+3 and (current["epoch"]>first["epoch"] or current["batches_completed_in_epoch"]>=64),
        "protected_files_unchanged":protection()==read(owner/"protected_before.json")==original["protected_sha256"],
        "executed_source_snapshots_match":bool(sources) and all(sources.values()),
        "training_scoped_keep_awake_active":awake["status"]=="active" and awake["worker_pid"]==read(owner/"owner.json")["pid"],
    }
    report={"passed":all(checks.values()),"checked_utc":datetime.now(timezone.utc).isoformat(),"run":run.relative_to(ROOT).as_posix(),
        "checks":checks,"resume":restored,"framework":framework,"first_resumed_batches":first,
        "progress_at_verification":current,"source_checks":sources,
        "root_cause":resume_request.get("previous_stop_cause", "Previous intentional stop requested by the user for an electricity cut" if resume_request else "Environment was no longer running; initiating cause not established"),
        "resume_request":resume_request,
        "sampler_augmentation_rng_exact_replay":False,"model_control":False,"official_test_used":False}
    save_json(evidence/"recovery_verification.json",report)
    local_date=datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime("%Y%m%d")
    save_json(ROOT/"docs/results"/("detection_resume_"+local_date+".json"),report)
    print(json.dumps({"passed":report["passed"],"checks":checks,"epoch":current["epoch"],
        "batches":current["batches_completed_in_epoch"],"session_optimizer_steps":first["session_optimizer_steps"]},indent=2))
    return report["passed"]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",required=True)
    raise SystemExit(0 if verify(parser.parse_args().run_name) else 1)
