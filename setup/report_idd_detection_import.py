"""Record final detector-data evidence; never trains or changes a road model."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import file_sha,save_json
from setup.verify_detection_runtime import protection


def main():
    run=ROOT/"runs/idd_detection_import_20261023_01"
    dataset=ROOT/"dataset/idd117k_detection_v2"
    def read(path):return json.loads(Path(path).read_text())
    preparation=read(dataset/"preparation.json")
    audit=read(run/"actual_data_audit/verification.json")
    readiness=read(run/"training_readiness_windows.json")
    framework=read(run/"framework_check/verification.json")
    archive=read(run/"archive_integrity.json")
    extraction=read(run/"extraction.json")
    source_index=read(run/"source_manifest_quality.index.json")
    hardlinks=read(run/"hardlink_recovery.json")
    preprocessing_checks={}
    for folder in ("converter_checks_bound","archive_checks","index_checks",
                   "legacy_index_checks","curation_checks","quality_checks_parallel"):
        path=run/folder/"verification.json"
        checked=read(path)
        if not checked.get("passed") or checked.get("training_started") is not False:
            raise ValueError("Preprocessing verification failed: "+folder)
        preprocessing_checks[folder]={"passed":True,"check_groups":len(checked["checks"]),"sha256":file_sha(path)}
    before=read(run/"protected_before.json")
    after=protection()
    save_json(run/"protected_after.json",after)
    if not all((preparation["passed"],audit["passed"],readiness["ready_for_explicit_training_start"],
                archive["passed"],extraction["passed"],source_index["passed"],framework["passed"],
                not framework["training_started"],framework["optimizer_steps"]==0,
                hardlinks["status"]=="completed",before==after)):
        raise ValueError("Final data/protection checks must all pass")
    if (framework["metadata_readiness"]["recipe_sha256"] != readiness["recipe_sha256"] or
        readiness["recipe_sha256"] != file_sha(ROOT/"detection_training_recipe.json")):
        raise ValueError("Readiness evidence must match the final training recipe")
    pixel_groups={s:Counter() for s in ("train","val")}
    annotations=defaultdict(set)
    for line in (dataset/"pairs.jsonl").open(encoding="utf-8"):
        row=json.loads(line)
        pixel_groups[row["split"]][row["pixels_sha256"]]+=1
        annotations[(row["split"],row["pixels_sha256"])].add(row["label_sha256"])
    if set(pixel_groups["train"]) & set(pixel_groups["val"]):
        raise ValueError("Final pixel hashes overlap across labelled splits")
    repeated={s:sum(n-1 for n in v.values()) for s,v in pixel_groups.items()}
    source=run/"source_final"
    source.mkdir(exist_ok=False)
    files=["idd_detection_data.py","train_detection.py","detection_policy.json","detection_training_recipe.json",
        "requirements-detection-wsl.txt","setup/import_idd117k_archive.py","setup/index_idd95k_detection.py",
        "setup/index_idd117k_detection.py","setup/prepare_idd_detection.py","setup/curate_idd_detection_manifest.py",
        "setup/finalize_idd_detection_quality.py","setup/relink_idd_prepared_images.py",
        "setup/audit_prepared_idd_detection.py","setup/verify_idd117k_import.py",
        "setup/verify_detection_data.py","setup/verify_idd_detection_quality.py",
        "setup/check_idd_detection_framework.py","setup/report_idd_detection_import.py"]
    hashes={}
    for name in files:
        target=source/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
        hashes[name]=file_sha(target)
    save_json(run/"source_final_hashes.json",hashes)
    result={"schema":"idd_detection_import_result_v1","completed_utc":datetime.now(timezone.utc).isoformat(),
        "passed":True,"training_started":False,"epochs_completed":0,"optimizer_steps":0,
        "training_deferred_by_user":True,"scope":"real_indian_object_detection_dataset",
        "dataset":"dataset/idd117k_detection_v2","raw_extraction":"dataset/idd117k_extracted",
        "evidence":"runs/idd_detection_import_20261023_01","counts":preparation["counts"],
        "provider_counts":preparation["provider_counts"],"names":preparation["names"],"objects":preparation["objects"],
        "negatives":preparation["negatives"],"native_sizes":audit["native_sizes"],
        "archive_regular_files":archive["regular_files"],"archive_unpacked_bytes":archive["regular_file_bytes"],
        "combined_gzip_crc_and_all_five_part_hashes_verified":True,"extraction_matches_inventory":True,
        "native_image_candidates_fully_decoded":sum(read(ROOT/"dataset/idd117k_detection_v1/preparation.json")["curated_manifest_counts"][s] for s in ("train","val")),
        "unlisted_raw_images_excluded":len(source_index["releases"]["IDD_Detection"]["unassigned_raw_images_excluded"]),
        "recording_exclusions":source_index["curation"],"quality_exclusions":source_index["quality_curation"],
        "cross_split_pixel_duplicates_remaining":0,"recording_groups_disjoint_in_selected_manifest":True,
        "unique_pixel_images_by_split":{s:len(v) for s,v in pixel_groups.items()},
        "within_split_repeated_pixel_frames_retained":repeated,
        "within_split_pixel_groups_with_multiple_label_serializations":sum(len(v)>1 for v in annotations.values()),
        "all_final_saved_labels_independently_reopened":True,
        "final_saved_labels_independently_reopened":sum(audit["counts"].values()),
        "source_images_and_annotations_independently_rechecked":audit["source_images_and_annotations_rechecked"],
        "maximum_sample_box_error_pixels":audit["maximum_sample_box_error_pixels"],
        "source_vs_saved_previews_visually_inspected":["03_IDD_95kDetection_val.jpg","09_IDD_Detection_val.jpg"],
        "native_images_resized_offline":False,"prepared_images_hardlinked":preparation["link_count"],
        "prepared_images_copied":preparation["copy_count"],"duplicate_bytes_reclaimed":hardlinks["duplicate_bytes_reclaimed"],
        "clipped_boxes":preparation["clipped_boxes"],"duplicate_boxes_removed":preparation["duplicate_boxes_removed"],
        "official_test_pixels_opened":False,"official_test_used_for_training_or_local_accuracy":False,
        "ready_for_explicit_training_start":readiness["ready_for_explicit_training_start"],
        "framework_configuration_checked":True,
        "preprocessing_check_reports":preprocessing_checks,
        "preprocessing_check_groups_passed":sum(v["check_groups"] for v in preprocessing_checks.values()),
        "framework_check":"runs/idd_detection_import_20261023_01/framework_check/verification.json",
        "wsl_image_timestamp_and_size_samples":framework["wsl_image_timestamp_and_size_samples"],
        "training_backward_memory_checked":False,
        "protected_files_unchanged":True,"protected_fingerprints":len(before),"source_snapshot_files":len(files),
        "gallery":"runs/idd_detection_import_20261023_01/actual_data_audit/index.html",
        "limits":["Custom 12-class mapping and filtered validation; not the official IDD benchmark protocol",
            "Within-split repeated frames and source annotation variations are retained; source labels may contain human noise",
            "Exact hashes and supplied/normalized recording identities do not establish absence of every near-duplicate across releases",
            "No model was trained, accuracy-scored, or connected to vehicle control"]}
    save_json(run/"results.json",result)
    save_json(ROOT/"docs/results/idd_detection_import.json",result)
    (dataset/"README.md").write_text(
        "# Prepared IDD object-detection data\n\n"
        "Preprocessing and readiness checks passed on 23 October 2026. The user requested "
        "doing training later. Training has not started; zero epochs and zero optimizer steps.\n\n"
        "- Training: 67,159 entries (59,245 distinct pixel images).\n"
        "- Validation: 17,675 entries, all distinct by decoded-pixel hash.\n"
        "- Official test: 24,644 unlabelled images, metadata only and outside this training view.\n"
        "- Classes: person, rider, car, motorcycle, bicycle, autorickshaw, bus, truck, animal, "
        "traffic_light, traffic_sign, other_vehicle.\n\n"
        "Images are under images/train and images/val; normalized YOLO box labels are under "
        "labels/train and labels/val. data.yaml is the training input configuration. Native "
        "images are hard links to the raw extraction, without offline resizing or recompression.\n\n"
        "The raw provider lists and every exclusion are retained in "
        "../../runs/idd_detection_import_20261023_01/. Within-training repeated frames remain; "
        "no exact decoded-pixel duplicate crosses training and validation. The custom class "
        "mapping and filtered validation are not the official IDD benchmark protocol.\n\n"
        "Read [the import guide](../../docs/IDD_DETECTION_IMPORT.md) and "
        "[the verified results](../../docs/results/idd_detection_import.json). "
        "The v1 folder preserves the earlier audit and rejected pairs; use v2 for later training.\n",
        encoding="utf-8")
    save_json(run/"import_state.json",{"stage":"completed","passed":True,"counts":result["counts"],
        "ready_for_explicit_training_start":True,"training_started":False,"training_deferred_by_user":True,
        "epochs_completed":0,"optimizer_steps":0})
    print(json.dumps({k:result[k] for k in ("passed","counts","ready_for_explicit_training_start",
        "within_split_repeated_pixel_frames_retained","protected_files_unchanged")},indent=2))


if __name__=="__main__":
    main()
