"""Remove recording overlaps from a provider view without moving any source split."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import MANIFEST_FIELDS, file_sha, read_manifest, save_json
from setup.index_idd117k_detection import EXPECTED


def main(provider_manifest, output):
    provider_manifest, output = Path(provider_manifest).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix(".index.json").exists():
        raise FileExistsError("Keep previous curated manifests")
    raw_index_path = provider_manifest.with_suffix(".index.json")
    index = json.loads(raw_index_path.read_text())
    if index.get("schema") != "idd117k_source_index_v1" or index["manifest_sha256"] != file_sha(provider_manifest):
        raise ValueError("The provider manifest must match its actual archive index")
    if index["counts"] != EXPECTED or not all(r["passed"] for r in index["releases"].values()):
        raise ValueError("Review incomplete provider data before recording-level curation")
    rows = read_manifest(provider_manifest, allow_drive_conflicts=True)
    groups = defaultdict(set)
    for row in rows:
        groups[(row["family"], row["drive_id"])].add(row["split"])
    retained, excluded = [], []
    for row in rows:
        splits = groups[(row["family"], row["drive_id"])]
        reason = None
        if row["split"] != "test" and row["drive_id"].startswith("unresolved_upload_batch_"):
            reason = "unresolved_mixed_upload_recording_identity"
        elif row["split"] != "test" and "test" in splits:
            reason = "recording_also_in_reserved_provider_test"
        elif row["split"] == "train" and "val" in splits:
            reason = "recording_also_in_provider_validation"
        if reason:
            excluded.append({**row, "exclusion_reason": reason})
        else:
            retained.append(row)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader(); writer.writerows(retained)
    read_manifest(output)  # Every retained recording group must now be disjoint.
    exclusions = output.with_name(output.stem+"_exclusions.jsonl")
    with exclusions.open("w", encoding="utf-8") as stream:
        for row in excluded:
            stream.write(json.dumps(row)+"\n")
    counts = dict(Counter(r["split"] for r in retained))
    summary = {"method": "Keep original split labels; test recording groups take priority over val, val over train; exclude unresolved mixed uploads",
        "source_split_membership_changed": False, "raw_files_deleted": False,
        "provider_counts": index["counts"], "retained_counts": counts,
        "excluded_labelled_images": len(excluded),
        "exclusion_reasons": dict(Counter(r["exclusion_reason"] for r in excluded)),
        "excluded_by_original_split": dict(Counter(r["split"] for r in excluded)),
        "provider_recording_conflicts": sum(len(v)>1 for v in groups.values()),
        "exclusions_file": str(exclusions), "exclusions_sha256": file_sha(exclusions),
        "provider_manifest": str(provider_manifest), "provider_manifest_sha256": file_sha(provider_manifest),
        "provider_index_sha256": file_sha(raw_index_path),
        "official_test_images_retained_as_metadata": counts["test"] == EXPECTED["test"],
        "training_started": False}
    result = {**index, "passed": True, "provider_counts": index["counts"], "counts": counts,
        "drive_split_conflict": None, "curation": summary, "manifest_sha256": file_sha(output)}
    save_json(output.with_suffix(".index.json"), result)
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.provider_manifest, args.output)
