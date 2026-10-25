"""Prepare blind real-video labels and score only reviewed human annotations.

This is a development review set from already inspected clips, not a new
independent test set. Model predictions are never used to initialize labels.
"""
import argparse
import base64
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from PIL import Image
from idd_inference import ROOT, file_sha256, write_json_atomic

RUBRIC = "visible_road_v1"
DIAGNOSTIC = {"03_070": [22, 450], "12_096": [912, 1120], "17_030": [412, 1341]}
SIZE = 512 * 512


def render_label_editor(pack, target):
    """Publish a working standalone editor; source template is never a user entry."""
    pack, target = Path(pack).resolve(), Path(target).resolve()
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    template = (ROOT / "setup/road_label_editor_template.html").read_text(encoding="utf-8")
    if template.count("__PACK_DATA__") != 1:
        raise ValueError("Editor template must contain exactly one pack-data placeholder.")
    images = {}
    for frame in manifest["frames"]:
        path = (pack / frame["image"]).resolve(strict=True)
        if not path.is_relative_to(pack):
            raise ValueError("A review image escapes the label pack.")
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != frame["image_sha256"]:
            raise ValueError("Review image hash changed while rendering the editor.")
        images[frame["id"]] = "data:image/png;base64,"+base64.b64encode(encoded).decode("ascii")
    payload = json.dumps({"manifest": manifest, "manifest_sha256": file_sha256(manifest_path),
                          "images": images}, separators=(",", ":")).replace("<", "\\u003c")
    rendered = template.replace("__PACK_DATA__", payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                     prefix=".road-editor-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"file": str(target), "embedded_images": len(images), "manifest_sha256": file_sha256(manifest_path)}


def decode_rle(values):
    if not isinstance(values, list) or not values or len(values) % 2 or len(values) > SIZE*2:
        raise ValueError("Mask RLE must contain value/count pairs.")
    total = 0
    for value, count in zip(values[::2], values[1::2]):
        if type(value) is not int or value not in (0, 1, 255) or type(count) is not int or count <= 0:
            raise ValueError("Only integer labels 0/1/255 and positive run lengths are allowed.")
        total += count
        if total > SIZE:
            raise ValueError("Mask RLE exceeds 512x512.")
    if total != SIZE:
        raise ValueError("Mask RLE must decode to exactly 512x512.")
    return np.repeat(np.array(values[::2], np.uint8), values[1::2]).reshape(512, 512)


def counts(prediction, labels):
    if prediction.dtype != np.uint8 or prediction.shape != (512, 512) or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Prediction must be an unchanged uint8 0/1 mask.")
    valid = labels != 255
    road = labels == 1
    return {"tp": int(np.count_nonzero(valid & road & (prediction == 1))),
            "fp": int(np.count_nonzero(valid & ~road & (prediction == 1))),
            "fn": int(np.count_nonzero(valid & road & (prediction == 0))),
            "tn": int(np.count_nonzero(valid & ~road & (prediction == 0))),
            "valid_pixels": int(valid.sum()), "ignored_pixels": int((~valid).sum())}


def metrics(c):
    def divide(a, b):
        return a/b if b else None
    tp, fp, fn, tn = (c[k] for k in ("tp", "fp", "fn", "tn"))
    return {"road_iou": divide(tp, tp+fp+fn), "road_dice": divide(2*tp, 2*tp+fp+fn),
            "road_precision": divide(tp, tp+fp), "road_recall": divide(tp, tp+fn),
            "pixel_accuracy": divide(tp+tn, c["valid_pixels"]),
            "label_coverage": divide(c["valid_pixels"], c["valid_pixels"]+c["ignored_pixels"])}


def validate_annotation(record, frame):
    if not isinstance(record, dict) or record.get("id") != frame["id"]:
        raise ValueError("Annotation frame ID differs from the locked pack.")
    if record.get("image_sha256") != frame["image_sha256"] or type(record.get("reviewed")) is not bool:
        raise ValueError("Annotation image hash/review flag is invalid.")
    mask = decode_rle(record.get("mask_rle"))
    coverage = float(np.mean(mask != 255))
    if record["reviewed"] and coverage < .8:
        raise ValueError("A reviewed frame must have at least 80% explicitly labelled pixels.")
    return mask


def create_pack(output):
    import av
    from predict_road_video import resize_input
    base = ROOT / "runs/real_video_20261019_02"
    previous = json.loads((base / "run_status.json").read_text(encoding="utf-8"))
    source_manifest_path = ROOT / "media/real_driving_20261017/clips.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if previous["state"] != "completed":
        raise ValueError("Use the completed real-video diagnostic run.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    (output / "evaluation_predictions").mkdir()
    frames = []
    for clip, source in zip(previous["clips"], source_manifest["clips"]):
        if clip["clip_id"] != source["clip_id"]:
            raise ValueError("Source order changed.")
        source_path = source_manifest_path.parent / source["file"]
        if file_sha256(source_path) != source["sha256"]:
            raise ValueError("Source bytes changed.")
        n = clip["processed_frames"]
        scoring = {int(n*(2*i+1)/8) for i in range(4)}
        diagnostic = set(DIAGNOSTIC[clip["clip_id"]])
        wanted = scoring | diagnostic
        rate = Fraction(clip["source_video"]["fps_fraction"])
        with av.open(str(source_path)) as video:
            for index, frame in enumerate(video.decode(video=0)):
                if index not in wanted:
                    continue
                rgb = resize_input(frame.to_ndarray(format="rgb24"), allow_resize=True)
                sample_id = f"{clip['clip_id']}_{index:06d}"
                image = output / "frames" / f"{sample_id}.png"
                Image.fromarray(rgb).save(image)
                pred = output / "evaluation_predictions" / f"{sample_id}.png"
                reference = base / clip["folder"] / "masks_01" / f"frame_{index:06d}.png"
                shutil.copy2(reference, pred)
                with Image.open(pred) as opened:
                    mask = np.array(opened)
                if mask.dtype != np.uint8 or mask.shape != (512, 512) or not np.isin(mask, [0, 1]).all():
                    raise ValueError("Reference prediction binary contract changed.")
                frames.append({"id": sample_id, "clip_id": clip["clip_id"], "frame_index": index,
                               "timestamp_s": float(Fraction(index) / rate),
                               "group": "time_selected_scoring" if index in scoring else "diagnostic_only",
                               "image": f"frames/{sample_id}.png", "image_sha256": file_sha256(image),
                               "prediction": f"evaluation_predictions/{sample_id}.png",
                               "prediction_sha256": file_sha256(pred),
                               "source_sha256": source["sha256"]})
        if len([f for f in frames if f["clip_id"] == clip["clip_id"]]) != len(wanted):
            raise ValueError("Not all selected source frames were decoded.")
        if file_sha256(source_path) != source["sha256"]:
            raise ValueError("Source changed during label-pack preparation.")
    frames.sort(key=lambda f: (f["group"] != "time_selected_scoring", f["clip_id"], f["frame_index"]))
    manifest = {"schema": "road_label_pack_v1", "pack_id": output.name, "rubric_id": RUBRIC,
                "created_utc": datetime.now(timezone.utc).isoformat(), "image_shape": [512, 512],
                "labels": {"nonroad": 0, "road": 1, "ignore": 255},
                "minimum_reviewed_coverage": .8, "scoring_frame_count": 12, "diagnostic_frame_count": 6,
                "selection": "Four equal-time-bin midpoints per clip, fixed before annotation; two separately reported diagnostic frames per clip.",
                "evaluation_scope": "Development frames from already inspected clips, not a new independent test set. Labels must be human reviewed; ignore ambiguous pixels. Do not train/tune on these if reporting their fixed-protocol result.",
                "rubric": "Visible carriageway pavement, including lane paint, crossing paint, shadows and worn patches, is road. Sky, buildings, vehicle/person bodies, raised footpaths and kerbs are nonroad. Do not hallucinate road behind occluders. Ignore uncertain shoulder/kerb pixels rather than guessing. This proposed real-video rubric does not rewrite IDD labels.",
                "checkpoint_sha256": previous["runtime"]["checkpoint_sha256"],
                "source_attribution": source_manifest["attribution"], "source_license": source_manifest["license"],
                "source_url": source_manifest["source_repository"], "frames": frames}
    write_json_atomic(output / "manifest.json", manifest)
    manifest_sha = file_sha256(output / "manifest.json")
    render_label_editor(output, output / "index.html")
    write_json_atomic(output / "label_status.json", {"state": "awaiting_human_labels", "reviewed_scoring_frames": 0,
                      "scoring_frames_required": 12, "diagnostic_frames_available": 6, "accuracy": None,
                      "annotations_initialized_from_model": False, "pack_manifest_sha256": manifest_sha})
    (output / "README.md").write_text("""# Human road-label pack

Open index.html locally. The page shows only original RGB frames; no model mask
is used as a starting label. All pixels initially mean **ignore/unlabelled**.
Draw road and non-road polygons; add smaller non-road polygons over vehicles.
Paint is available for corrections. Uncertain kerb/shoulder pixels can stay
ignored. The display restores the source 16:9 appearance while labels stay on
the exact 512x512 model grid. No calibration or training data is changed.

After checking a frame, tick Reviewed. At least 80% must be labelled. Export
the annotation JSON; it contains the exact image hashes and a lossless mask.
The editor also autosaves in the browser when local storage is available.
Keep exported JSON as the durable copy. Labels are not uploaded anywhere.

There are 12 time-selected scoring frames and 6 separate diagnostic frames.
These come from already reviewed development clips; they cannot establish
independent whole-video or real-road accuracy. Diagnostic frames do not enter
the primary score. All 12 scoring frames must be reviewed before a primary
aggregate result is published. No score exists until reviewed labels are imported.

Import the exported JSON from PowerShell in the project folder, choosing a new
result directory (replace the JSON path with the file you exported):

```powershell
.\\.venv\\Scripts\\python.exe road_video_labels.py score `
  --pack runs/real_video_labels_20261019_01 `
  --annotations 'C:\\path\\to\\road_annotations.json' `
  --output-dir runs/my_reviewed_video_scores_01
```

The scorer validates pack/image/prediction hashes, labels, review flags and
coverage. It ignores uncertain pixels and reports both coverage and metrics.
Original/baseline predictions stay unchanged. See manifest.json for the fixed
rubric, sample selection, source attribution and CC BY-NC 4.0 restriction.
""", encoding="utf-8")
    print(json.dumps({"state": "awaiting_human_labels", "output": str(output), "frames": len(frames), "accuracy": None}), flush=True)


def score_pack(pack, annotations, output):
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "road_label_pack_v1" or manifest.get("rubric_id") != RUBRIC
            or manifest.get("image_shape") != [512, 512]
            or manifest.get("minimum_reviewed_coverage") != .8
            or manifest.get("scoring_frame_count") != 12 or manifest.get("diagnostic_frame_count") != 6
            or manifest.get("pack_id") != pack.name or len(manifest.get("frames", [])) != 18):
        raise ValueError("The fixed 12-scoring/6-diagnostic label-pack protocol changed.")
    for frame in manifest["frames"]:
        if (not isinstance(frame, dict) or not isinstance(frame.get("id"), str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", frame["id"])
                or frame.get("group") not in ("time_selected_scoring", "diagnostic_only")):
            raise ValueError("Invalid frame ID or scoring group in pack.")
    if (len({f["id"] for f in manifest["frames"]}) != 18
            or sum(f["group"] == "time_selected_scoring" for f in manifest["frames"]) != 12):
        raise ValueError("Pack has repeated frames or changed scoring membership.")
    if annotations.stat().st_size > 30_000_000:
        raise ValueError("Annotation JSON exceeds the bounded 30 MB format.")
    data = json.loads(annotations.read_text(encoding="utf-8-sig"))
    if (data.get("schema") != "road_annotations_v1" or data.get("pack_id") != manifest["pack_id"]
            or data.get("pack_manifest_sha256") != file_sha256(pack / "manifest.json")
            or data.get("rubric_id") != RUBRIC or manifest.get("rubric_id") != RUBRIC):
        raise ValueError("Annotation pack/hash/rubric does not match.")
    if not isinstance(data.get("reviewer"), str) or not data["reviewer"].strip():
        raise ValueError("Record a reviewer name or alias.")
    records = data.get("frames")
    if not isinstance(records, list) or len(records) > len(manifest["frames"]):
        raise ValueError("Annotation records are invalid.")
    expected = {f["id"]: f for f in manifest["frames"]}
    if any(not isinstance(r, dict) or r.get("id") not in expected for r in records):
        raise ValueError("Unknown frame in annotation export.")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate annotation frame.")
    prepared = []
    for record in records:
        frame = expected[record["id"]]
        for key, sha_key in [("image", "image_sha256"), ("prediction", "prediction_sha256")]:
            path = (pack / frame[key]).resolve(strict=True)
            if not path.is_relative_to(pack.resolve()) or file_sha256(path) != frame[sha_key]:
                raise ValueError("A fixed pack image or prediction changed.")
        mask = validate_annotation(record, frame)
        if record["reviewed"]:
            prepared.append((frame, mask))
    output.mkdir(parents=True, exist_ok=False)
    (output / "reviewed_masks").mkdir()
    results = []
    for frame, mask in prepared:
        with Image.open(pack / frame["prediction"]) as opened:
            prediction = np.array(opened)
        c = counts(prediction, mask)
        Image.fromarray(mask).save(output / "reviewed_masks" / f"{frame['id']}.png")
        results.append({"id": frame["id"], "group": frame["group"], "counts": c, "metrics": metrics(c)})
    scoring = [r for r in results if r["group"] == "time_selected_scoring"]
    complete = len(scoring) == manifest["scoring_frame_count"]
    def aggregate(group):
        if not group:
            return None
        c = {k: sum(r["counts"][k] for r in group) for k in group[0]["counts"]}
        return {"frames": len(group), "counts": c, "metrics": metrics(c)}
    result = {"state": "completed_development_frame_score" if complete else "awaiting_all_scoring_labels",
              "pack_id": manifest["pack_id"], "pack_manifest_sha256": file_sha256(pack / "manifest.json"),
              "annotation_file_sha256": file_sha256(annotations), "reviewer": data["reviewer"],
              "reviewed_scoring_frames": len(scoring), "scoring_frames_required": manifest["scoring_frame_count"],
              "primary_score": aggregate(scoring) if complete else None,
              "diagnostic_only_score": aggregate([r for r in results if r["group"] == "diagnostic_only"]),
              "per_frame": results, "whole_video_accuracy_established": False,
              "evaluation_scope": manifest["evaluation_scope"], "ignore_value": 255,
              "training_performed": False, "reserved_dataset_images_read": False}
    write_json_atomic(output / "scores.json", result)
    def percent(value):
        return f"{100*value:.2f}%" if value is not None else "undefined (empty class)"
    if complete:
        primary = result["primary_score"]["metrics"]
        heading = (f"Road IoU: **{percent(primary['road_iou'])}**. "
                   f"Dice: **{percent(primary['road_dice'])}**. "
                   f"Pixel accuracy: **{percent(primary['pixel_accuracy'])}**. "
                   f"Label coverage: **{percent(primary['label_coverage'])}**.")
    else:
        heading = f"**No primary accuracy score yet.** Reviewed scoring frames: {len(scoring)}/12."
    lines = ["# Human-reviewed development-frame scores", "", heading, "",
             manifest["evaluation_scope"], "", "Diagnostic frames are excluded from the primary score. "
             "Ignored pixels (255) are excluded; coverage is always reported. "
             "These are reviewed-frame results, not whole-video accuracy.", "",
             "| Frame | Group | Road IoU | Dice | Label coverage |", "|---|---|---:|---:|---:|"]
    for row in results:
        m = row["metrics"]
        lines.append(f"| {row['id']} | {row['group']} | {percent(m['road_iou'])} | {percent(m['road_dice'])} | {percent(m['label_coverage'])} |")
    lines += ["", "Exact counts, source hashes and review provenance are in scores.json. "
              "Reviewed reference masks are in reviewed_masks/. The original dataset, "
              "model and baseline predictions were not rewritten."]
    (output / "SCORES_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    shutil.copy2(Path(__file__), output / "score_source_snapshot.py")
    print(json.dumps({"state": result["state"], "reviewed_scoring_frames": len(scoring), "primary_score": result["primary_score"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--output-dir", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--pack", type=Path, required=True)
    score.add_argument("--annotations", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        parser.error("Choose a new result directory inside runs/.")
    if args.command == "create":
        create_pack(output)
    else:
        score_pack(args.pack.resolve(strict=True), args.annotations.resolve(strict=True), output)


if __name__ == "__main__":
    main()
