"""Build the final report and gallery from the single saved test evaluation.

No TensorFlow import, model loading, new inference, relabeling or threshold tuning.
"""
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/test_evaluation_20261003_morning_best"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def comparison(row):
    i = row["id"]
    with Image.open(ROOT / f"dataset/splits/test/images/Image_{i}.png") as im:
        rgb = np.asarray(im.convert("RGB"))
    with Image.open(ROOT / f"dataset/splits/test/masks/Mask_{i}.png") as im:
        truth = np.asarray(im)
    path = OUT / f"predicted_masks/Prediction_{i}.png"
    with Image.open(path) as im:
        pred = np.asarray(im)
    assert truth.shape == pred.shape == (512, 512)
    assert np.isin(truth, [0, 1]).all() and np.isin(pred, [0, 1]).all()
    assert sha(path) == row["prediction_sha256"]
    overlay = rgb.copy()
    missed, extra = (truth == 1) & (pred == 0), (truth == 0) & (pred == 1)
    overlay[missed] = (.4*rgb[missed]+.6*np.array([255, 40, 55])).astype(np.uint8)
    overlay[extra] = (.4*rgb[extra]+.6*np.array([30, 135, 255])).astype(np.uint8)
    panels = [rgb, truth*255, pred*255, overlay]
    titles = [f"Test Image {i}: RGB", "Supplied road mask", "Locked morning-best model", "Label disagreement"]
    subtitles = ["Unaugmented 512 x 512", "Unchanged 0/1 labels; white is display",
                 f"Road IoU {row['road_iou']*100:.2f}% | B-IoU {row['boundary_iou_5px']*100:.2f}%",
                 "Red: labelled road missed | Blue: extra"]
    sheet = Image.new("RGB", (2048, 612), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (panel, title, subtitle) in enumerate(zip(panels, titles, subtitles)):
        x = j*512
        draw.text((x+10, 7), title, fill="#15243a", font=font(23))
        draw.text((x+10, 39), subtitle, fill="#40516c", font=font(16))
        sheet.paste(Image.fromarray(panel).convert("RGB"), (x, 70))
    draw.text((12, 587), "Held-out test reporting only. Disagreement can reflect annotation/class-definition issues. Stored prediction masks stay 0/1.", fill="#34495e", font=font(17))
    return sheet


def contact_sheet(rows, path):
    images = [comparison(row).resize((1280, 382), Image.Resampling.LANCZOS) for row in rows]
    sheet = Image.new("RGB", (1280, 382*len(images)), "white")
    for i, im in enumerate(images):
        sheet.paste(im, (0, i*382))
    sheet.save(path)


def main():
    result = read(OUT / "test_metrics.json")
    lock = read(OUT / "EVALUATION_LOCK.json")
    verification = read(OUT / "metric_verification.json")
    status = read(OUT / "session_status.json")
    assert result["status"] == status["status"] == "completed" and result["images"] == 689
    assert verification["status"] == "passed"
    assert sha(ROOT / result["checkpoint"]) == result["checkpoint_sha256"]
    with (OUT / "per_image_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for k, v in list(row.items()):
            if k == "id":
                row[k] = int(v)
            elif k not in ("split", "prediction_sha256"):
                row[k] = float(v)
    ids = {r["id"] for r in rows}
    assert len(rows) == len(ids) == 689 and ids == set(lock["test_ids"])
    by_id = {r["id"]: r for r in rows}
    metrics = result["global"]
    counts = result["counts"]
    validation = read(ROOT / "runs/boundary_20261003/reference_trial13/metrics.json")["global"]
    source = OUT / "source/setup"
    source.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), source / Path(__file__).name)
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    axes[0].hist([100*r["road_iou"] for r in rows], bins=np.linspace(0, 100, 21), color="#186aa0", edgecolor="white")
    axes[0].axvline(metrics["road_iou"]*100, color="#dd7930", label="Global road IoU")
    axes[0].axvline(result["per_image_road_iou_distribution"]["median"]*100, color="#39917b", linestyle="--", label="Median image IoU")
    axes[0].set(title="All 689 test images", xlabel="Per-image road IoU (%)", ylabel="Images")
    axes[0].legend(fontsize=9)
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    axes[1].imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for y in range(2):
        for x in range(2):
            axes[1].text(x, y, f"{matrix[y,x]:,}\n{normalized[y,x]*100:.2f}% of label row", ha="center", va="center",
                         color="white" if normalized[y,x]>.5 else "#15243a", fontsize=10)
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Background", "Road"], yticklabels=["Background", "Road"],
                xlabel="Predicted label", ylabel="Supplied label", title="Pixel confusion matrix")
    keys = ["road_iou", "road_dice", "boundary_iou_5px", "visible_boundary_f1_3px"]
    x = np.arange(4)
    axes[2].bar(x-.18, [validation[k]*100 for k in keys], width=.36, label="Validation (687)", color="#7eb8bf")
    axes[2].bar(x+.18, [metrics[k]*100 for k in keys], width=.36, label="Test (689)", color="#25639f")
    axes[2].set(xticks=x, xticklabels=["Road IoU", "Dice", "Boundary\nIoU, 5px", "Contour F1\n3px"], ylim=[0, 100],
                ylabel="Global score (%)", title="Same locked model and threshold")
    axes[2].legend(fontsize=9)
    fig.suptitle("Morning-best trial 13: final held-out test evaluation", fontweight="bold")
    fig.savefig(plots / "test_summary.png", dpi=165)
    plt.close(fig)
    weakest = sorted(rows, key=lambda r: (r["road_iou"], r["id"]))[:12]
    contact_sheet(weakest[:4], plots / "test_worst_cases.png")
    illustrative = [by_id[i] for i in lock["illustrative_test_ids"]]
    contact_sheet(illustrative[:4], plots / "test_predetermined_examples.png")
    save(OUT / "case_selection.json", {"lowest_iou_ids": [r["id"] for r in weakest],
          "predetermined_illustrative_ids": lock["illustrative_test_ids"], "no_test_cases_excluded": True,
          "selection_does_not_change_model_or_threshold": True})
    gallery = OUT / "gallery"
    previews = gallery / "comparisons"
    previews.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for index, row in enumerate(rows):
        comparison(row).save(previews / f"Image_{row['id']}.jpg", quality=88, optimize=True)
        if (index+1) % 100 == 0:
            save(OUT / "artifact_status.json", {"status": "building_gallery", "completed": index+1, "total": 689})
            print(f"Test gallery: {index+1}/689", flush=True)
    cards = [{"id": r["id"], "iou": r["road_iou"], "dice": r["road_dice"], "boundary": r["boundary_iou_5px"]} for r in rows]
    summary = f"Road IoU {metrics['road_iou']*100:.2f}% · Dice {metrics['road_dice']*100:.2f}% · Boundary IoU {metrics['boundary_iou_5px']*100:.2f}%"
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IDD final test evaluation — 689 images</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0c1420;color:#edf3fa;font:16px system-ui,sans-serif}header{padding:28px 3vw 20px;background:#142237;border-bottom:1px solid #2b4058}h1{font-size:27px;margin:0 0 8px}.scores{color:#71d5c7;font-weight:650;margin:14px 0}.note{color:#bfccd9;font-size:14px;max-width:1000px;line-height:1.55}.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap;padding:18px 3vw;background:#101c2c;position:sticky;top:0;z-index:2}label{font-size:13px;color:#c4d1df;display:flex;flex-direction:column;gap:6px}input,select,button{background:#1e3046;color:#fff;border:1px solid #49647f;border-radius:7px;padding:9px 12px;font:inherit}input{width:140px}button{cursor:pointer}button:hover{background:#30536b}#status{padding:16px 3vw;color:#afc4d8}main{padding:0 2vw 30px}article{background:#142237;border:1px solid #304258;border-radius:10px;margin-bottom:20px;overflow:hidden}article h2{display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:17px;padding:12px 16px;margin:0}article small{color:#a9dcd2;font-weight:400}article img{display:block;width:100%;height:auto;background:#fff}a{color:#81d9dd}footer{padding:25px 3vw;color:#b6c6d8;font-size:14px}#empty{padding:35px;background:#1b2c43;border-radius:8px}a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #7bf2d3;outline-offset:3px}</style></head><body>
<header><h1>Final test evaluation: morning-best model</h1><div>All 689 test pairs · Locked threshold 0.5 · Supplied masks unchanged</div><div class="scores">__SUMMARY__</div><div class="note">This test set has now been used for final reporting. Keep model and inference settings fixed; do not use these cases to select another checkpoint or tune this test score. Red marks supplied road missed; blue marks predicted road outside the supplied label. Label disagreement can include annotation/class-definition problems.</div></header>
<div class="controls"><label>Image ID<input id="id" type="number" min="0" aria-label="Image ID" placeholder="e.g. 48"></label><button id="go">Find</button><button id="reset">Show all</button><label>Order<select id="sort" aria-label="Sort order"><option value="numeric">Image ID</option><option value="worst">Lowest IoU first</option><option value="best">Highest IoU first</option></select></label><button id="prev">Previous</button><label>Page<select id="page" aria-label="Page"></select></label><button id="next">Next</button></div>
<div id="status" role="status"></div><main id="cards"></main><footer>JPEGs are display comparisons. The separate predicted_masks folder contains all 689 binary 0/1 PNGs at original 512×512 resolution. The complete per-image CSV, lock, metrics, verification, latency and report are in the parent evaluation folder.</footer>
<script>const data=__DATA__;let page=0;const n=12,el=id=>document.getElementById(id);function filtered(){let a=data.slice();let id=el('id').value.trim();if(id!=='')a=a.filter(r=>r.id===Number(id));const s=el('sort').value;a.sort(s==='worst'?(x,y)=>x.iou-y.iou||x.id-y.id:s==='best'?(x,y)=>y.iou-x.iou||x.id-y.id:(x,y)=>x.id-y.id);return a}function draw(){const a=filtered(),pages=Math.max(1,Math.ceil(a.length/n));page=Math.max(0,Math.min(page,pages-1));el('page').innerHTML=Array.from({length:pages},(_,i)=>`<option value="${i}" ${i===page?'selected':''}>${i+1} / ${pages}</option>`).join('');const shown=a.slice(page*n,(page+1)*n);el('status').textContent=a.length?`Showing ${page*n+1}–${Math.min((page+1)*n,a.length)} of ${a.length} images. Click a comparison to zoom.`:'No matching test image. That ID may belong to training or validation.';el('cards').innerHTML=shown.length?shown.map(r=>`<article><h2>Test Image ${r.id}<small>IoU ${(r.iou*100).toFixed(2)}% · Dice ${(r.dice*100).toFixed(2)}% · Boundary ${(r.boundary*100).toFixed(2)}%</small><a href="comparisons/Image_${r.id}.jpg" target="_blank" rel="noopener">Zoom</a></h2><a href="comparisons/Image_${r.id}.jpg" target="_blank" rel="noopener"><img loading="lazy" src="comparisons/Image_${r.id}.jpg" alt="Test image ${r.id}: RGB, supplied label, locked prediction, label disagreement"></a></article>`).join(''):'<div id="empty">No matching test pair.</div>';el('prev').disabled=page===0;el('next').disabled=page>=pages-1}el('go').onclick=()=>{page=0;draw()};el('id').onkeydown=e=>{if(e.key==='Enter'){page=0;draw()}};el('reset').onclick=()=>{el('id').value='';page=0;draw()};el('sort').onchange=()=>{page=0;draw()};el('page').onchange=()=>{page=Number(el('page').value);draw()};el('prev').onclick=()=>{page--;draw()};el('next').onclick=()=>{page++;draw()};draw();</script></body></html>'''
    html = html.replace("__SUMMARY__", summary).replace("__DATA__", json.dumps(cards, separators=(",", ":")))
    (gallery / "index.html").write_text(html, encoding="utf-8")
    save(gallery / "manifest.json", {"images": 689, "ids": sorted(ids), "columns": ["RGB", "supplied binary label", "locked morning-best prediction", "label disagreement"],
         "checkpoint_sha256": result["checkpoint_sha256"], "threshold": .5, "stored_prediction_mask_values": [0, 1], "display_only_jpeg": True})
    latency = read(OUT / "latency.json")
    timing = latency["raw_rgb_cpu_ram_to_binary_mask_cpu_ram_ms"]
    gpu_samples = read(OUT / "gpu_samples.json")
    observed_device_memory_mib = max(float(s["values"].split(",")[2].split()[0])
                                     for s in gpu_samples if s["returncode"] == 0)
    ist = datetime.fromisoformat(result["finished_at_utc"]).astimezone(timezone(timedelta(hours=5, minutes=30)))
    ci = result["image_bootstrap_95pct"]
    names = {"road_iou": "Road IoU", "road_dice": "Road Dice / foreground F1", "road_precision": "Road precision", "road_recall": "Road recall",
             "boundary_iou_5px": "Strict Boundary IoU, 5px", "boundary_iou_14px": "Boundary IoU, 14px", "visible_boundary_f1_3px": "Visible contour F1, 3px tolerance"}
    lines = ["# Final road segmentation test evaluation", "", f"Completed **{ist.strftime('%d %B %Y, %I:%M %p')} IST** on the RTX 4060 through WSL2.", "",
             "## Result and fixed model", "", "All **689 test image-mask pairs** were evaluated with the locked morning-best trial-13 checkpoint. No test pair was excluded. The 44th batch contained the final single image.", "",
             f"Checkpoint: `{result['checkpoint']}` (selected epoch 1).", f"SHA-256: `{result['checkpoint_sha256']}`.", "",
             "The checkpoint and protocol were recorded in EVALUATION_LOCK.json before opening any test PNG contents. The choice was based on prior training/validation comparisons; neither other retained model was run on this test set.", "",
             "| Metric | Validation, 687 images | Test, 689 images | Test image-bootstrap 95% interval |", "|---|---:|---:|---:|"]
    for key, label in names.items():
        lines.append(f"| {label} | {validation[key]*100:.3f}% | **{metrics[key]*100:.3f}%** | {ci[key]['lower']*100:.3f}–{ci[key]['upper']*100:.3f}% |")
    delta = (metrics["road_iou"]-validation["road_iou"])*100
    lines += ["", f"Test minus validation road IoU: **{delta:+.3f} percentage points**. This compares different image subsets; it is not a gain from another training step or a direct paired comparison.", "",
              f"Supplementary pixel accuracy: {metrics['pixel_accuracy']*100:.3f}%. Mean per-image BCE + soft Dice loss: {metrics['bce_plus_soft_dice_loss']:.6f}.", "",
              "Global scores aggregate pixel counts over the entire test split. The mean and median of per-image scores are different quantities and are reported separately below. Pixel accuracy includes background pixels and is not the primary score.", "",
              "## Per-image behavior", "",
              f"- Mean per-image road IoU: **{result['mean_per_image']['road_iou']*100:.3f}%**.",
              f"- Median per-image road IoU: **{result['per_image_road_iou_distribution']['median']*100:.3f}%**.",
              f"- 5th percentile: {result['per_image_road_iou_distribution']['p05']*100:.3f}%; minimum: {result['per_image_road_iou_distribution']['minimum']*100:.3f}%.",
              f"- Images below 50% IoU: **{result['road_iou_below_50pct']} / 689**; below 75%: **{result['road_iou_below_75pct']} / 689**.",
              f"- Completely empty predicted road masks: **{result['empty_predictions']}**. Supplied masks were all nonempty, consistent with the previously cleaned subset.", "",
              "Lowest 12 test road-overlap cases, selected for honest reporting after scoring every pair:", "",
              "| Test image ID | Road IoU | Boundary IoU, 5px | Labelled road area | Predicted road area |", "|---|---:|---:|---:|---:|"]
    for row in weakest:
        lines.append(f"| {row['id']} | {row['road_iou']*100:.2f}% | {row['boundary_iou_5px']*100:.2f}% | {row['truth_road_fraction']*100:.2f}% | {row['predicted_road_fraction']*100:.2f}% |")
    lines += ["", "These scores measure agreement with supplied masks. Red/blue overlays may reflect model errors or annotation/class-definition issues. The report does not certify that every supplied mask matches physical road boundaries. No poor case was removed or relabelled.", "",
              "Visual review of the four lowest-scoring cases shows rough or unpaved surfaces in IDs 607, 806 and 5076, and a narrow urban scene with traffic and shadows in ID 2446. The supplied positive regions are small while the model predicts broader ground areas. This suggests that target-definition ambiguity may contribute alongside model errors; it does not establish that the model is correct or the annotations are wrong.", "",
              "![Test score distribution and confusion matrix](plots/test_summary.png)", "",
              "![Four lowest-scoring test cases](plots/test_worst_cases.png)", "",
              "[All 689 test comparisons](gallery/index.html) · [Predetermined test examples](plots/test_predetermined_examples.png) · [Every image's scores](per_image_metrics.csv)", "",
              "## Speed and GPU evidence", "",
              f"- Warm batch-one raw RGB in CPU RAM to binary mask back in CPU RAM: median **{timing['median_ms']:.2f} ms**, p95 **{timing['p95_ms']:.2f} ms**.",
              f"- Normalized GPU forward plus probability copy: median {latency['normalized_forward_and_probability_copy_ms']['median_ms']:.2f} ms.",
              f"- Peak TensorFlow-tracked GPU memory during this evaluation: **{result['peak_tensorflow_gpu_memory_mib']:.1f} MiB** (a 6,144 MiB allocation limit was configured).",
              f"- NVIDIA snapshots reached **{observed_device_memory_mib:,.0f} MiB** total device memory in use. This includes the reserved TensorFlow allocation pool and driver/context memory; the smaller TensorFlow counter tracks active allocations. Snapshots are not a continuous measurement of the absolute peak.",
              f"- Full worker elapsed time: {result['elapsed_seconds']:.1f} seconds, including file/hash checks, CPU boundary counts, export, independent recount, uncertainty summary and timing. Gallery/report creation is additional.",
              f"- Inference phase wall time: {result['inference_phase_wall_seconds_including_io_cpu_metrics_and_export']:.1f} seconds; sum of batch GPU forward/count/copy measurements: {result['summed_batch_gpu_forward_count_and_copy_seconds']:.1f} seconds.", "",
              "Timing used 15 warmups and 64 synchronized samples per path on eight predetermined validation images, keeping the test accuracy pass at one forward pass per test image. It includes normalization and host/device transfers in the RAM-to-mask path, but excludes camera capture, cold initialization, file loading, tracking, planning, display and vehicle control. Inverse latency is not demonstrated end-to-end vehicle FPS. Current device load/temperature can affect small timing differences.", "",
              "GPU:0 output tensors and NVIDIA samples confirm RTX 4060 execution. TensorFlow inference is in the saved mixed-precision layers with a float32 probability head. No optimizer, gradients, training-mode BatchNorm or network rebuilding was used.", "",
              "## Reproducibility and correctness checks", "",
              "- Locked threshold **0.5 (>=)**; 512×512 input, one existing MobileNetV2 normalization to [-1,1], original 0/1 labels.",
              "- training=False, no augmentation, no test-time augmentation, no morphology, connected-component removal or threshold sweep.",
              "- Test order follows numeric manifest IDs, batch 16, deterministic operations/seed 42, drop_remainder=False. All 689 unique IDs appeared once.",
              "- All test PNG hashes matched the original split manifest before and after evaluation. All masks were uint8 0/1, all inputs/labels 512×512. Existing split remains 5507/687/689.",
              "- Stored manifest image hashes show zero exact-image cross-split overlap. This is not a near-duplicate or sequence-independence guarantee.",
              "- CPU counts from predictions exactly matched the independent TensorFlow RoadIoU/RoadDice/5px BoundaryIoU accumulators. All 689 exported binary PNGs were independently reopened/recounted without another model inference pass.",
              "- Hand-computed shifted-square overlap and perfect-boundary fixtures passed. Source snapshots, lock, test file hashes, batch log, per-image counts and output hashes are saved.",
              "- All three retained checkpoint file hashes, split manifest and in-memory model/BatchNorm weights remained unchanged. Original dataset and preprocessing files were not rewritten.", "",
              "## Metric definitions", "",
              "TP = labelled road predicted as road; FP = labelled background predicted as road; FN = labelled road missed; TN = labelled background predicted as background.", "",
              "- Road IoU = TP / (TP + FP + FN).", "- Road Dice = 2 TP / (2 TP + FP + FN).",
              "- Road precision = TP / (TP + FP); road recall = TP / (TP + FN).",
              "- Supplementary pixel accuracy = (TP + TN) / all pixels.",
              "- Boundary IoU uses the intersection/union of inner foreground boundary bands, with square erosion and frame padding. We retained the predeclared strict 5px band and supplementary 14px band. Boundary construction follows the [authors' reference implementation](https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py).",
              "- Visible contour F1 is the previously defined approximate many-to-one contour score with 3px square tolerance and the frame excluded. It includes object/hole boundaries and is not an outer-white-line detection metric.",
              "- Entire empty/empty mask union scores 1. For a nonempty supplied road mask with an empty predicted road mask, per-image precision is defined as 0; IoU, Dice and recall are 0.",
              "- Image-bootstrap intervals use 3,000 resamples with replacement, seed 42 and percentile endpoints. The global ratio is recomputed per resample. These are descriptive image-resampling intervals and do not account for unknown within-sequence correlation.", "",
              "## Limits of the result", "",
              "1. Original driving-sequence IDs are unavailable. Correlated/adjacent scenes may cross splits, so this is an image-held-out result, not verified sequence-held-out generalization.",
              "2. Exact third-party positive-class mapping and ignore-label handling remain unrecovered. Scores assess the existing binary road-surface labels, not every drivable shoulder, legal driving region, obstacle detector or safety decision.",
              "3. The previous 110 all-zero-mask exclusions affect coverage. These test results apply to the retained 6,883-pair nonempty-mask dataset and do not evaluate true no-road scenes or the excluded pairs.",
              "4. The test split is now consumed for final reporting. Use new independent data/a separately reserved benchmark for future model selection after making test-informed changes; do not tune against these 689 cases and call them untouched again.",
              "5. This is an academic image-segmentation evaluation. Physical road-boundary validity, unfamiliar conditions and full closed-loop driving remain unverified.", "",
              "## Files", "",
              "EVALUATION_LOCK.json; test_metrics.json; metric_verification.json; data_integrity.json; test_file_manifest.csv; per_image_metrics.csv; inference_batches.csv; latency.json; gpu_samples.json; predicted_masks/ (689 original-value 0/1 PNGs); plots/; gallery/; source/; evaluation.log; power_request_released.json.", "",
              "## Evaluation summary", "",
              f'"We selected the morning model using validation data and fixed its threshold and preprocessing before opening the test set. We evaluated all 689 held-out image-mask pairs without training or changing labels. The global road IoU was {metrics["road_iou"]*100:.2f}% and Dice was {metrics["road_dice"]*100:.2f}%. We also measured boundaries, speed and weak cases, and independently verified the saved predictions. Missing sequence identities and uncertain supplied label definitions limit the generalization claim."', "",
              "## Next stage", "",
              "Keep this checkpoint and inference protocol fixed for the current project. Package reusable inference, check independent road images where appropriate, and resolve camera calibration/ground-plane-to-metric-map geometry before connecting to Hybrid A*. RL STOP / SLOW / PROCEED and MATLAB/Simulink car visualization remain pending."]
    (OUT / "TEST_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    save(OUT / "artifact_status.json", {"status": "completed", "gallery_images": 689, "report_written": True,
         "no_new_model_inference": True, "elapsed_seconds": time.perf_counter()-started})
    print(json.dumps({"report": str(OUT / "TEST_REPORT.md"), "gallery": str(gallery / "index.html"), "images": 689}, indent=2))


if __name__ == "__main__":
    main()
