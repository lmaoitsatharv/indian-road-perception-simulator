"""Continue verified extraction through indexing, preprocessing and readiness only."""
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import prepare_dataset, save_json
from setup.index_idd117k_detection import main as index_sources
from setup.audit_prepared_idd_detection import main as audit_dataset
from train_detection import readiness


def main():
    run = ROOT/"runs/idd_detection_import_20261023_01"
    source = ROOT/"dataset/idd117k_extracted"
    manifest = run/"source_manifest.csv"
    dataset = ROOT/"dataset/idd117k_detection_v1"
    result_path = run/"pipeline.json"
    if result_path.exists():
        raise FileExistsError("Preserve previous import pipeline attempts")
    state = {"status": "waiting_for_extraction", "training_started": False, "vehicle_control_allowed": False}
    save_json(result_path, state)
    print(json.dumps(state), flush=True)
    try:
        while not (run/"extraction.json").exists():
            time.sleep(10)
        state["status"] = "indexing_provider_splits"
        save_json(result_path, state)
        index = index_sources(source, manifest, source/"IDD117K_Detection/IDD_Detection", "xyxy_edges", "xyxy_edges",
            run/"archive_integrity.json", run/"extraction.json")
        if not index["passed"]:
            raise ValueError("Review actual provider counts or recording/split conflicts before preprocessing")
        state["status"] = "preprocessing_labelled_pairs"
        save_json(result_path, state)
        preparation = prepare_dataset(source, manifest, dataset, workers=6,
            source_index=manifest.with_suffix(".index.json"), download_record=ROOT/"dataset/idd117k_raw/download_verification.json")
        state["status"] = "independent_saved_label_audit"
        save_json(result_path, state)
        audit_dataset(dataset, source, manifest, run/"actual_data_audit")
        checked = readiness(dataset)
        save_json(run/"training_readiness.json", checked)
        state.update(status="completed" if preparation["passed"] else "quality_review_required",
            counts=preparation["counts"], preparation_passed=preparation["passed"],
            ready_for_explicit_training_start=checked["ready_for_explicit_training_start"])
    except BaseException as error:
        state.update(status="failed_preserved", error=type(error).__name__+": "+str(error))
        (run/"pipeline_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        save_json(result_path, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
