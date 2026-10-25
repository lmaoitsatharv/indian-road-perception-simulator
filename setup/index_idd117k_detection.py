"""Index both inspected IDD releases, keeping provider lists and recording drives."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import MANIFEST_FIELDS, file_sha, read_manifest, save_json
from setup.index_idd95k_detection import index_release

EXPECTED = {"train": 96897, "val": 20202, "test": 24644}
LEGACY_EXPECTED = {"train": 31569, "val": 10225, "test": 4794}
RECORDING_ALIASES = {key: "BLR-"+key for key in ("2018-05-31_11-21-58", "2018-05-31_13-01-28", "2018-05-31_11-10-16")}


def legacy_drive(key):
    parts = PurePosixPath(key).parts
    if len(parts) != 3:
        raise ValueError("Legacy drive identity must come from its capture/drive/frame folders: " + key)
    if parts[1] in RECORDING_ALIASES:
        return RECORDING_ALIASES[parts[1]]
    if parts[1] == "15-07-18-upload":
        # This source folder mixes named and numeric frames without a recording
        # index. Keep the entire upload batch together, without inventing drives.
        return "unresolved_upload_batch_15-07-18"
    # The archive also has side/rear cameras and a highquality_16k category.
    # Its recording directory explicitly contains city, date and start time.
    match = re.fullmatch(r"([A-Za-z]+-\d{4}-\d{2}-\d{2})[_-](\d{2}-\d{2}-\d{2})(?:_[A-Za-z0-9_-]+)?", parts[1])
    if match is None:
        raise ValueError("Review unrecognised legacy recording directory: "+parts[1])
    return match.group(1)+"_"+match.group(2)


def index_legacy(source_root, split_directory, convention):
    source_root, split_directory = Path(source_root).resolve(), Path(split_directory).resolve()
    base = source_root/"IDD117K_Detection/IDD_Detection"
    image_root, annotation_root = base/"JPEGImages", base/"Annotations"
    if not image_root.is_dir() or not annotation_root.is_dir() or not split_directory.is_relative_to(base):
        raise ValueError("Use the inspected legacy image/annotation roots and its provider split directory")
    images, annotations, basename_keys = {}, {}, {}
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        key = path.relative_to(image_root).with_suffix("").as_posix()
        legacy_drive(key)
        if key in images:
            raise ValueError("Ambiguous legacy image key")
        images[key] = path
        basename_keys.setdefault(PurePosixPath(key).name, []).append(key)
    for path in sorted(annotation_root.rglob("*.xml")):
        key = path.relative_to(annotation_root).with_suffix("").as_posix()
        if key in annotations:
            raise ValueError("Ambiguous legacy annotation key")
        annotations[key] = path
    rows, identities, lists = [], set(), {}
    counts = Counter()
    for split in ("train", "val", "test"):
        path = split_directory/(split+".txt")
        if not path.is_file():
            raise ValueError("Missing explicit provider split list: " + str(path))
        lists[split] = {"path": path.relative_to(source_root).as_posix(), "sha256": file_sha(path)}
        for text in path.read_text(encoding="utf-8-sig").splitlines():
            identifier = text.strip()
            if not identifier:
                continue
            if len(identifier.split()) != 1:
                raise ValueError("Review provider list schema instead of guessing fields")
            key = PurePosixPath(identifier)
            if key.is_absolute() or ".." in key.parts or ":" in identifier or "\\" in identifier:
                raise ValueError("Unsafe provider image identifier")
            if key.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                key = key.with_suffix("")
            key = key.as_posix()
            if key.startswith("JPEGImages/"):
                key = key[len("JPEGImages/"):]
            if key not in images and "/" not in key:
                matches = basename_keys.get(key, [])
                if len(matches) != 1:
                    raise ValueError("Provider basename must match exactly one actual drive/frame")
                key = matches[0]
            if key not in images or key in identities:
                raise ValueError("Missing image or repeated provider source identity: " + identifier)
            identities.add(key)
            if split != "test" and key not in annotations:
                raise ValueError("Missing labelled legacy annotation: " + key)
            rows.append({"image": images[key].relative_to(source_root).as_posix(),
                "annotation": annotations[key].relative_to(source_root).as_posix() if split != "test" else "-",
                "format": "voc_xml", "split": split, "family": "IDD_Detection",
                "drive_id": legacy_drive(key), "source_id": key, "box_convention": convention})
            counts[split] += 1
    unassigned = sorted(set(images)-identities)
    trainval = {r["source_id"] for r in rows if r["split"] != "test"}
    if trainval-set(annotations):
        raise ValueError("Missing labelled legacy annotations")
    return rows, {"counts": dict(counts), "expected_counts": LEGACY_EXPECTED,
        "passed": dict(counts) == LEGACY_EXPECTED, "provider_lists": lists,
        "test_annotation_files_ignored": len(set(annotations) & (identities-trainval)),
        "unassigned_raw_images_excluded": unassigned,
        "unassigned_raw_annotation_files": sorted(set(annotations)-identities),
        "raw_images": len(images), "raw_annotation_files": len(annotations),
        "unassigned_policy": "retain raw files, record exclusions, never invent provider membership or negatives",
        "reviewed_recording_aliases": RECORDING_ALIASES,
        "unresolved_recording_policy": "15-07-18-upload is one conservative indivisible upload group",
        "drive_id_policy": "explicit city/date/start-time recording directory; group all cameras from the same recording"}


def main(source, output, split_directory, legacy_convention, new_convention, integrity, extraction):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve source manifests")
    archive_report = json.loads(Path(integrity).read_text())
    extraction_report = json.loads(Path(extraction).read_text())
    if not archive_report.get("passed") or not extraction_report.get("passed"):
        raise ValueError("Full archive integrity and extraction must pass before indexing")
    if (Path(extraction_report["destination"]).resolve() != source or
        archive_report["part_records"] != extraction_report["part_records"]):
        raise ValueError("Source directory or archive identity differs from extraction evidence")
    new_manifest = output.with_name(output.stem+"_95k.csv")
    new_index = index_release(source, new_manifest, new_convention)
    new_rows = read_manifest(new_manifest)
    old_rows, old_index = index_legacy(source, split_directory, legacy_convention)
    rows = new_rows + old_rows
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    try:
        read_manifest(output)
        conflict = None
    except ValueError as error:
        conflict = str(error)
    counts = dict(Counter(r["split"] for r in rows))
    result = {"schema": "idd117k_source_index_v1", "passed": new_index["passed"] and old_index["passed"] and not conflict and counts == EXPECTED,
        "counts": counts, "expected_counts": EXPECTED, "releases": {"IDD_95kDetection": new_index, "IDD_Detection": old_index},
        "drive_split_conflict": conflict, "image_pixels_opened": False, "training_started": False,
        "manifest_sha256": file_sha(output), "archive_integrity_sha256": file_sha(integrity),
        "extraction_report_sha256": file_sha(extraction), "test_images_metadata_only": True,
        "source_root": str(source), "archive_integrity_file": str(Path(integrity).resolve()),
        "extraction_report_file": str(Path(extraction).resolve())}
    save_json(output.with_suffix(".index.json"), result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-split-directory", type=Path, required=True)
    parser.add_argument("--legacy-box-convention", choices=("xyxy_edges", "voc_1based_inclusive"), required=True)
    parser.add_argument("--95k-box-convention", dest="new_convention", choices=("xyxy_edges", "voc_1based_inclusive"), required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    args = parser.parse_args()
    result = main(args.source, args.output, args.legacy_split_directory, args.legacy_box_convention,
        args.new_convention, args.integrity, args.extraction)
    raise SystemExit(0 if result["passed"] else 2)
