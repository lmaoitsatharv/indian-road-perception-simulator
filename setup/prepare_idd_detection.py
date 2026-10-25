"""Export an explicitly reviewed provider manifest to a separate YOLO dataset."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import prepare_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-record", type=Path, default=ROOT / "dataset/idd117k_raw/download_verification.json")
    parser.add_argument("--policy", type=Path, default=ROOT / "detection_policy.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--source-index", type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "dataset"):
        parser.error("Use a fresh project dataset/ output folder")
    if not args.download_record.is_file():
        parser.error("Download verification is not complete; no dataset was exported")
    result = prepare_dataset(args.source_root, args.manifest, args.output,
        policy_path=args.policy, download_record=args.download_record, workers=args.workers,
        source_index=args.source_index or args.manifest.with_suffix(".index.json"))
    print(json.dumps({k: result[k] for k in ("passed", "counts", "training_allowed", "missing_train_or_validation_classes")}, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
