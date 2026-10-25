"""Validate the existing human review and freeze an evaluation-only guard set."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_inference import file_sha256, write_json_atomic
from road_video_labels import score_pack


def main():
    output = ROOT / "runs/real_retention_guard_20261021_01"
    output.mkdir(exist_ok=False)
    pack = ROOT / "runs/real_video_labels_20261019_01"
    annotations = ROOT / "runs/real_video_reviewed_20261019_01/annotations_source.json"
    historical = json.loads((ROOT / "runs/real_video_reviewed_20261019_01/scores.json").read_text(encoding="utf-8"))
    if file_sha256(annotations) != historical["annotation_file_sha256"]:
        raise ValueError("The reviewed human annotation source changed")
    result = score_pack(pack, annotations, output / "validated_label_import")
    if (result["state"] != "completed_development_frame_score"
            or len(result["per_frame"]) != 18
            or result["primary_score"] != historical["primary_score"]
            or result["diagnostic_only_score"] != historical["diagnostic_only_score"]):
        raise ValueError("Human review or original stored-prediction recount changed")
    data = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for frame in data["frames"]:
        image = pack / frame["image"]
        mask = output / "validated_label_import/reviewed_masks" / (frame["id"] + ".png")
        records.append({"id": frame["id"], "group": frame["group"],
                        "image": str(image), "image_sha256": file_sha256(image),
                        "mask": str(mask), "mask_sha256": file_sha256(mask)})
    manifest = {"schema": "real_retention_guard_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Evaluation only; never include these frames in training, teacher replay or augmentation",
                "labels": [0, 1, 255], "shape": [512, 512], "records": records,
                "annotation_source": str(annotations), "annotation_sha256": file_sha256(annotations),
                "pack_manifest_sha256": file_sha256(pack / "manifest.json"),
                "primary_frames": 12, "diagnostic_frames": 6,
                "evaluation_scope": data["evaluation_scope"],
                "selection_limit": "Repeated development guard, not an independent test or a guarantee on unseen real roads",
                "reserved_test_images_read": False,
                "historical_baseline_recount": result["primary_score"],
                "historical_diagnostic_recount": result["diagnostic_only_score"]}
    write_json_atomic(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "manifest.json"), "frames": len(records),
                      "primary_road_iou": result["primary_score"]["metrics"]["road_iou"],
                      "diagnostic_road_iou": result["diagnostic_only_score"]["metrics"]["road_iou"]}), flush=True)


if __name__ == "__main__":
    main()
