"""Exclude rejected pairs/leaked train duplicates and reuse the complete pixel audit."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import (MANIFEST_FIELDS, convert_annotation, file_sha, load_policy,
    read_manifest, save_json, source_path, yolo_lines)


def identity(row):
    return row["family"], row["source_id"]


def reuse_pair(row, audited, source, output, policy):
    old_image, old_label = source_path(audited, row["prepared_image"]), source_path(audited, row["prepared_label"])
    info = old_image.stat()
    if info.st_size != row["image_bytes"] or info.st_mtime_ns//1000000 != row["image_mtime_ms"]:
        raise ValueError("Audited native image changed before reuse")
    label_data = old_label.read_bytes()
    if hashlib.sha256(label_data).hexdigest() != row["label_sha256"]:
        raise ValueError("Audited labels changed before reuse")
    annotation_data = source_path(source, row["annotation"]).read_bytes()
    if hashlib.sha256(annotation_data).hexdigest() != row["annotation_sha256"]:
        raise ValueError("Source annotation changed after full audit")
    converted = convert_annotation(annotation_data.decode("utf-8-sig"), row["format"], row["native_width"],
        row["native_height"], row["box_convention"], policy)
    if yolo_lines(converted) != label_data.decode("utf-8").replace("\r\n", "\n"):
        raise ValueError("Retained annotation differs from previously audited labels")
    image_target, label_target = output/row["prepared_image"], output/row["prepared_label"]
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    linked = True
    try:
        os.link(old_image, image_target)
    except OSError:
        if output.is_relative_to(ROOT/"dataset"):
            raise RuntimeError("Hard links are required for this final real dataset; retry with project file-link access instead of duplicating all images")
        shutil.copy2(old_image, image_target)
        linked = False
    label_target.write_bytes(label_data)
    final_row = {**row, "image_mtime_ms": image_target.stat().st_mtime_ns//1000000,
        "clipped_boxes": converted["clipped_boxes"], "duplicate_boxes_removed": converted["duplicate_boxes_removed"]}
    return final_row, converted, linked


def reused_pairs(pairs, audited, source, output, policy):
    iterator = iter(pairs)
    with ThreadPoolExecutor(max_workers=6) as executor:
        while batch := list(itertools.islice(iterator, 96)):
            yield from executor.map(lambda row: reuse_pair(row, audited, source, output, policy), batch)


def main(audited, source, manifest, output, final_manifest):
    audited, source, output = Path(audited).resolve(), Path(source).resolve(), Path(output).resolve()
    manifest, final_manifest = Path(manifest).resolve(), Path(final_manifest).resolve()
    if output.exists() or final_manifest.exists():
        raise FileExistsError("Preserve previous preprocessing versions")
    report = json.loads((audited/"preparation.json").read_text())
    progress = json.loads((audited/"progress.json").read_text())
    fixture = report.get("scope") == "fixture_only_not_for_training"
    if fixture:
        if not output.is_relative_to(ROOT/"runs"):
            raise ValueError("Fixtures must remain under runs and may never become training data")
    elif report.get("scope") != "real_idd_detection" or not output.is_relative_to(ROOT/"dataset"):
        raise ValueError("Use a real audited dataset and a fresh project dataset output")
    if progress.get("status") != "completed":
        raise ValueError("The complete real-image audit must finish before quality selection")
    if file_sha(manifest) != report["source_manifest_sha256"] or file_sha(audited/"pairs.jsonl") != report["manifest_sha256"]:
        raise ValueError("Source or audited manifest changed")
    if report["policy_sha256"] != file_sha(ROOT/"detection_policy.json"):
        raise ValueError("Do not change class policy after the complete native-image audit")
    original = read_manifest(manifest)
    pairs = [json.loads(line) for line in (audited/"pairs.jsonl").open(encoding="utf-8")]
    by_identity = {identity(row): row for row in pairs}
    rejected = {identity(row): row for row in report["quarantine"]}
    if len(by_identity) != len(pairs) or set(by_identity) & set(rejected):
        raise ValueError("Ambiguous audit identities")
    candidate_ids = {identity(r) for r in original if r["split"] != "test"}
    if candidate_ids != set(by_identity) | set(rejected):
        raise ValueError("Every labelled candidate must have an audited pair or a recorded rejection")
    # File-system failures must be repaired, not hidden as annotation exclusions.
    known_rejections = ("Degenerate or reversed box", "Foreground box is entirely outside the image",
        "XML dimensions differ from actual image dimensions", "Unsupported orientation/colour mode; coordinate review required",
        "cannot identify image file", "broken data stream", "image file is truncated")
    if any(not any(row["reason"].startswith(prefix) for prefix in known_rejections) for row in rejected.values()):
        raise ValueError("Review unexpected source errors before excluding them")
    hashes = {kind: defaultdict(list) for kind in ("image_sha256", "pixels_sha256")}
    for row in pairs:
        for kind in hashes:
            hashes[kind][row[kind]].append(row)
    duplicate_exclusions = {}
    for kind, groups in hashes.items():
        for digest, group in groups.items():
            if {r["split"] for r in group} == {"train", "val"}:
                for row in group:
                    if row["split"] == "train":
                        duplicate_exclusions[identity(row)] = {**row, "exclusion_reason": "exact_image_also_in_validation",
                            "duplicate_kind": kind, "duplicate_sha256": digest}
    excluded_ids = set(rejected) | set(duplicate_exclusions)
    retained = [r for r in original if r["split"] == "test" or identity(r) not in excluded_ids]
    with final_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader(); writer.writerows(retained)
    read_manifest(final_manifest)
    exclusion_path = final_manifest.with_name(final_manifest.stem+"_exclusions.jsonl")
    with exclusion_path.open("w", encoding="utf-8") as stream:
        for row in rejected.values():
            stream.write(json.dumps({**row, "exclusion_reason": "invalid_native_image_or_annotation_pair"})+"\n")
        for row in duplicate_exclusions.values():
            stream.write(json.dumps(row)+"\n")
    counts = dict(Counter(r["split"] for r in retained))
    selected_pairs = [by_identity[identity(r)] for r in retained if r["split"] != "test"]
    # Verify all retained encoded/pixel digest groups are train/val-disjoint.
    for kind in hashes:
        seen = {}
        for row in selected_pairs:
            if row[kind] in seen and seen[row[kind]] != row["split"]:
                raise ValueError("A duplicate remains across labelled splits")
            seen[row[kind]] = row["split"]
    index = json.loads(manifest.with_suffix(".index.json").read_text())
    quality = {"source_audit_directory": str(audited), "source_preparation_sha256": file_sha(audited/"preparation.json"),
        "source_audited_pairs_sha256": file_sha(audited/"pairs.jsonl"),
        "invalid_pairs_excluded": len(rejected), "invalid_pair_reasons": dict(Counter(r["reason"] for r in rejected.values())),
        "training_duplicates_of_validation_excluded": len(duplicate_exclusions),
        "exclusions_file": str(exclusion_path), "exclusions_sha256": file_sha(exclusion_path),
        "retained_counts": counts, "source_split_membership_changed": False,
        "raw_files_deleted": False, "image_pixels_decoded_again": False, "training_started": False}
    index.update(counts=counts, manifest_sha256=file_sha(final_manifest), quality_curation=quality, passed=True)
    final_index = final_manifest.with_suffix(".index.json")
    save_json(final_index, index)
    output.mkdir(parents=True)
    (output/"provenance").mkdir()
    evidence = {"source_index": final_index, "archive_integrity": Path(index["archive_integrity_file"]),
        "extraction": Path(index["extraction_report_file"])}
    provenance = {}
    for key, path in evidence.items():
        if key != "source_index" and file_sha(path) != index["archive_integrity_sha256" if key=="archive_integrity" else "extraction_report_sha256"]:
            raise ValueError("Archive/extraction evidence changed")
        shutil.copy2(path, output/"provenance"/(key+".json"))
        provenance[key] = {"file": "provenance/"+key+".json", "sha256": file_sha(path)}
    save_json(output/"provenance/quality_curation.json", quality)
    policy = load_policy()
    result = {**report, "passed": False, "training_allowed": False, "training_started": False,
        "source_manifest_sha256": file_sha(final_manifest), "archive_provenance": provenance,
        "counts": {"train": 0, "val": 0, "test_metadata_only": counts["test"]},
        "curated_manifest_counts": {"test_metadata_only" if s=="test" else s:n for s,n in counts.items()},
        "objects": {s: {n:0 for n in policy["names"]} for s in ("train", "val")},
        "negatives": {"train": 0, "val": 0}, "quarantine": [], "cross_split_duplicates": [],
        "clipped_boxes": 0, "duplicate_boxes_removed": 0, "link_count": 0, "copy_count": 0,
        "quality_curation": quality, "native_image_audit_reused": True}
    save_json(output/"preparation.json", result)
    started = last = time.monotonic()
    with (output/"pairs.jsonl").open("w", encoding="utf-8") as stream:
        for i, (row, converted, linked) in enumerate(reused_pairs(selected_pairs, audited, source, output, policy), 1):
            result["link_count"] += linked
            result["copy_count"] += not linked
            stream.write(json.dumps(row)+"\n")
            split = row["split"]
            result["counts"][split] += 1
            result["negatives"][split] += converted["negative"]
            result["clipped_boxes"] += converted["clipped_boxes"]
            result["duplicate_boxes_removed"] += converted["duplicate_boxes_removed"]
            for box in converted["objects"]:
                result["objects"][split][box["class_name"]] += 1
            if time.monotonic()-last >= 20:
                status = {"processed_labelled_images": i, "total_labelled_images": len(selected_pairs),
                    "elapsed_seconds": round(time.monotonic()-started, 1), "stage": "materialising_quality_filtered_dataset", "training_started": False}
                save_json(output/"progress.json", status)
                print(json.dumps(status), flush=True)
                last = time.monotonic()
    shutil.copy2(audited/"data.yaml", output/"data.yaml")
    missing = [n for n in policy["names"] if not result["objects"]["train"][n] or not result["objects"]["val"][n]]
    result.update(missing_train_or_validation_classes=missing, manifest_sha256=file_sha(output/"pairs.jsonl"),
        data_yaml_sha256=file_sha(output/"data.yaml"), elapsed_seconds=round(time.monotonic()-started, 1))
    result["passed"] = not missing and result["counts"] == result["curated_manifest_counts"]
    result["training_allowed"] = result["passed"] and not fixture
    save_json(output/"preparation.json", result)
    save_json(output/"progress.json", {"status": "completed", "passed": result["passed"], "counts": result["counts"], "training_started": False})
    print(json.dumps({"passed": result["passed"], "counts": result["counts"], "quality_curation": quality,
        "missing_classes": missing}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audited", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--wait-for-audit", action="store_true")
    args = parser.parse_args()
    if args.wait_for_audit:
        print("Waiting for complete native-image quality audit", flush=True)
        while True:
            try:
                status = json.loads((args.audited/"progress.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                status = {}
            if status.get("status") == "completed":
                break
            time.sleep(10)
    main(args.audited, args.source, args.manifest, args.output, args.final_manifest)
