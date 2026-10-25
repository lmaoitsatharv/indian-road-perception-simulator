"""Audit supplied human labels and save comparisons; no model inference/training."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from idd_inference import ROOT, file_sha256, write_json_atomic


def main():
    out = ROOT / "runs/real_video_reviewed_20261019_01"
    pack = ROOT / "runs/real_video_labels_20261019_01"
    source = ROOT / "real_video_labels_20261019_01_annotations.json"
    scores = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    frames = {f["id"]: f for f in manifest["frames"]}
    source_before = file_sha256(source)
    assert source_before == scores["annotation_file_sha256"]
    assert file_sha256(pack / "manifest.json") == scores["pack_manifest_sha256"]
    assert scores["reviewed_scoring_frames"] == 12
    shutil.copy2(source, out / "annotations_source.json")
    gallery = out / "comparisons"
    gallery.mkdir(exist_ok=False)
    aggregate = {g: np.zeros(4, dtype=np.int64) for g in ("time_selected_scoring", "diagnostic_only")}
    checked = []
    font = ImageFont.load_default(size=16)
    for row in scores["per_frame"]:
        f = frames[row["id"]]
        assert file_sha256(pack / f["image"]) == f["image_sha256"]
        assert file_sha256(pack / f["prediction"]) == f["prediction_sha256"]
        with Image.open(pack / f["image"]) as opened:
            rgb = np.array(opened.convert("RGB"))
        with Image.open(pack / f["prediction"]) as opened:
            pred = np.array(opened)
        with Image.open(out / "reviewed_masks" / (f["id"]+".png")) as opened:
            label = np.array(opened)
        assert label.dtype == pred.dtype == np.uint8 and label.shape == pred.shape == (512, 512)
        assert np.isin(label, [0, 1, 255]).all() and np.isin(pred, [0, 1]).all()
        valid = label != 255
        # Independent 2x2 confusion matrix: rows=reference, columns=prediction.
        matrix = np.bincount(2*label[valid].astype(np.int64)+pred[valid], minlength=4)
        tn, fp, fn, tp = map(int, matrix)
        assert (tp, fp, fn, tn) == tuple(row["counts"][k] for k in ("tp", "fp", "fn", "tn"))
        aggregate[f["group"]] += matrix
        extra, missed = valid & (pred == 1) & (label == 0), valid & (pred == 0) & (label == 1)
        difference = rgb.copy()
        difference[extra] = (.4*rgb[extra]+.6*np.array([40, 120, 255])).astype(np.uint8)
        difference[missed] = (.4*rgb[missed]+.6*np.array([255, 45, 55])).astype(np.uint8)
        reference_display = np.where(label == 255, 127, label*255).astype(np.uint8)
        images = [Image.fromarray(rgb), Image.fromarray(reference_display).convert("RGB"),
                  Image.fromarray(pred*255).convert("RGB"), Image.fromarray(difference)]
        panel = Image.new("RGB", (2048, 344), (18, 25, 35))
        draw = ImageDraw.Draw(panel)
        titles = [f"Original | {f['clip_id']} | {f['timestamp_s']:.2f}s", "Your reviewed road label",
                  f"Locked model | road IoU {100*row['metrics']['road_iou']:.2f}%", "Red: missed label road / Blue: extra road"]
        for i, (image, title) in enumerate(zip(images, titles)):
            panel.paste(image.resize((512, 288), Image.Resampling.NEAREST if i in (1, 2) else Image.Resampling.BILINEAR), (i*512, 28))
            draw.text((i*512+8, 5), title, fill="white", font=font)
        draw.text((8, 321), f"{f['group']} | Compared against your annotations; label uncertainty can affect disagreement. No source labels or model weights changed.", fill="white", font=font)
        panel.save(gallery / (f["id"]+".jpg"), quality=93)
        checked.append(f["id"])
    for group, target in [("time_selected_scoring", scores["primary_score"]), ("diagnostic_only", scores["diagnostic_only_score"])]:
        tn, fp, fn, tp = map(int, aggregate[group])
        assert {"tp":tp,"fp":fp,"fn":fn,"tn":tn} == {k:target["counts"][k] for k in ("tp","fp","fn","tn")}
        assert abs(tp/(tp+fp+fn)-target["metrics"]["road_iou"]) < 1e-12
        assert abs((tp+tn)/(tp+tn+fp+fn)-target["metrics"]["pixel_accuracy"]) < 1e-12
    catalog = json.loads((ROOT / "models/retained_models.json").read_text(encoding="utf-8"))
    hashes = {m["name"]: file_sha256(ROOT / m["checkpoint"]) for m in catalog["models"]}
    assert all(hashes[m["name"]] == m["checkpoint_sha256"] for m in catalog["models"])
    assert file_sha256(source) == source_before == file_sha256(out / "annotations_source.json")
    m, d = scores["primary_score"]["metrics"], scores["diagnostic_only_score"]["metrics"]
    cards = []
    for row in sorted(scores["per_frame"], key=lambda x:(x["group"]!="time_selected_scoring", x["metrics"]["road_iou"])):
        f = frames[row["id"]]
        cards.append(f'<section><h2>{html.escape(f["id"])} · {f["timestamp_s"]:.2f}s · IoU {100*row["metrics"]["road_iou"]:.2f}%</h2><p>{html.escape(row["group"])}</p><a href="comparisons/{f["id"]}.jpg"><img loading="lazy" src="comparisons/{f["id"]}.jpg" alt="Original, human label, model mask and disagreement"></a></section>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Road predictions versus your labels</title><style>body{{background:#101720;color:#e8eff7;font:16px/1.6 system-ui;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}section{{background:#192432;border:1px solid #34465b;border-radius:10px;padding:18px;margin:22px 0}}img{{width:100%;height:auto}}a{{color:#8bdfbe}}h2{{font-size:20px}}.score{{font-size:28px;color:#91e2b9}}</style><main><h1>Road predictions compared with your labels</h1><p class="score">12 scoring frames: road IoU {100*m["road_iou"]:.2f}% · pixel accuracy {100*m["pixel_accuracy"]:.2f}%</p><p>Dice {100*m["road_dice"]:.2f}% · road precision {100*m["road_precision"]:.2f}% · recall {100*m["road_recall"]:.2f}% · label coverage {100*m["label_coverage"]:.0f}%.</p><p>6 diagnostic frames, reported separately: road IoU {100*d["road_iou"]:.2f}%. These are results against your reviewed annotations on 18 selected development frames, not accuracy for every frame in the videos or a new independent test set. The model was not retrained.</p><p><a href="SCORES_REPORT.md">Score report</a> · <a href="scores.json">Exact counts</a> · <a href="independent_verification.json">Independent recount</a></p>{''.join(cards)}</main></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    verification = {"state":"passed", "reviewed_frames_recounted":len(checked), "primary_frames":12,
                    "diagnostic_frames":6, "independent_confusion_matrices_match":True,
                    "annotation_source_sha256":source_before, "annotation_source_unchanged":True,
                    "all_three_checkpoint_hashes_match":hashes, "training_or_inference_run":False,
                    "reserved_dataset_images_read":False, "whole_video_accuracy_established":False}
    write_json_atomic(out / "independent_verification.json", verification)
    shutil.copy2(Path(__file__), out / "review_source_snapshot.py")
    before = out / "label_status_before.json"
    shutil.copy2(pack / "label_status.json", before)
    write_json_atomic(pack / "label_status.json", {"state":"scored_development_frames",
                      "reviewed_scoring_frames":12, "scoring_frames_required":12, "reviewed_diagnostic_frames":6,
                      "label_coverage":m["label_coverage"], "selected_frame_scores":m,
                      "result":"runs/real_video_reviewed_20261019_01/SCORES_REPORT.md",
                      "scope":"12 reviewed time-selected development frames; diagnostic frames separate; not whole-video accuracy",
                      "annotations_initialized_from_model":False, "pack_manifest_sha256":scores["pack_manifest_sha256"]})
    print(json.dumps({"state":"passed", "scoring_metrics":m, "diagnostic_metrics":d,
                      "gallery":str(out / "index.html")}), flush=True)


if __name__ == "__main__":
    main()
