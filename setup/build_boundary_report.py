"""Build an honest four-model report, curves, and all-validation review gallery."""
import csv
import hashlib
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from boundary_metrics import boundary
from review_boundary_masks import sha, atomic

ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "runs/boundary_20261003"
OLD = ROOT / "runs/model_comparisons/baseline_vs_trial13_validation"
LABELS = {"original": "Original model", "morning_best": "Morning best (trial 13)",
          "control": "Matched normal-loss control", "boundary_weight4": "Boundary-focused model"}
REVIEWS = {"original": SESSION / "reference_original", "morning_best": SESSION / "reference_trial13",
           "control": SESSION / "control/validation_review", "boundary_weight4": SESSION / "boundary_weight4/validation_review"}
MASKS = {"original": OLD / "predicted_masks/baseline", "morning_best": OLD / "predicted_masks/trial13",
         "control": SESSION / "control/predicted_masks", "boundary_weight4": SESSION / "boundary_weight4/predicted_masks"}
CHECKPOINTS = {"original": ROOT / "runs/frozen_01/best.keras",
               "morning_best": ROOT / "runs/overnight_20261003/trial_13_shadow_fixed_bn_low_lr/best.keras",
               "control": SESSION / "control/best.keras", "boundary_weight4": SESSION / "boundary_weight4/best.keras"}


def read_image(path):
    with Image.open(path) as im:
        return np.asarray(im).copy()


def bootstraps(rows, baseline="morning_best", repeats=2000):
    # Paired image bootstrap describes this validation sample, not independent test evidence.
    rng = np.random.default_rng(42)
    ids = sorted(rows[baseline])
    keys = ("tp", "fp", "fn", "boundary_intersection_5px", "boundary_union_5px")
    values = {name: np.array([[int(rows[name][i][key]) for key in keys] for i in ids]) for name in rows}
    changes = {name: [] for name in rows if name != baseline}
    for _ in range(repeats):
        indices = rng.integers(0, len(ids), len(ids))
        totals = {name: data[indices].sum(axis=0) for name, data in values.items()}
        scores = {name: np.array([t[0] / t[:3].sum(), t[3] / t[4]]) for name, t in totals.items()}
        for name in changes:
            changes[name].append((scores[name] - scores[baseline]) * 100)
    return {name: {metric: list(map(float, np.percentile(np.array(changes[name])[:, j], [2.5, 97.5])))
                  for j, metric in enumerate(("road_iou_delta_pp", "boundary_iou_5px_delta_pp"))} for name in changes}


def font(size):
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def comparison_image(i, rows, path):
    rgb = read_image(ROOT / f"dataset/splits/val/images/Image_{i}.png")
    truth = read_image(ROOT / f"dataset/splits/val/masks/Mask_{i}.png").astype(bool)
    predictions = {name: read_image(MASKS[name] / f"Prediction_{i}.png").astype(bool)
                   for name in ("original", "morning_best", "boundary_weight4")}
    rgb = np.asarray(Image.fromarray(rgb).convert("RGB"))
    overlay = rgb.copy()
    tb, pb = boundary(truth, 1, False), boundary(predictions["boundary_weight4"], 1, False)
    overlay[tb] = [255, 210, 0]
    overlay[pb] = [0, 230, 255]
    overlay[tb & pb] = [255, 255, 255]
    # Last panel is zoomable at original 512 resolution; no mask values rewritten.
    sheet = Image.new("RGB", (6 * 512, 512 + 100), "white")
    draw = ImageDraw.Draw(sheet)
    panels = [rgb, np.uint8(truth) * 255, *[np.uint8(predictions[name]) * 255 for name in predictions], overlay]
    titles = [f"Image {i}: RGB", "Supplied mask (unchanged)", "Original", "Morning best", "Boundary-focused", "Outline overlay"]
    subtitles = ["Validation only", "Display white=1; files stay 0/1"]
    for name in predictions:
        r = rows[name][i]
        subtitles.append(f"Road {100*float(r['road_iou']):.2f}% | Boundary {100*float(r['boundary_iou_5px']):.2f}%")
    subtitles.append("Yellow: supplied | Cyan: prediction")
    for col, (panel, title, sub) in enumerate(zip(panels, titles, subtitles)):
        x = col * 512
        draw.text((x+10, 7), title, fill="#162331", font=font(22))
        draw.text((x+10, 38), sub, fill="#34495e", font=font(18))
        sheet.paste(Image.fromarray(panel).convert("RGB"), (x, 70))
    draw.text((12, 585), "Boundaries include vehicles, holes and frame edges in Boundary IoU; these are supplied-label comparisons, not certified physical road edges.", fill="#34495e", font=font(17))
    sheet.save(path)
    return sheet


def main():
    if (ROOT / "models/segmentation_final_lock.json").exists():
        raise RuntimeError(
            "The final road model is locked and its test evaluation is complete. "
            "Use the saved boundary report; historical regeneration would replace "
            "selection metadata and final-test claims."
        )
    state = json.loads((SESSION / "session_status.json").read_text())
    assert state["status"] == "completed"
    metrics = {name: json.loads((path / "metrics.json").read_text()) for name, path in REVIEWS.items()}
    assert {m["validation_mask_digest"] for m in metrics.values()}.__len__() == 1
    assert {m["manifest_sha256"] for m in metrics.values()} == {"512cc054e2291ec5e197b0dca913ba94698cf055062b54511977816872005e90"}
    rows = {}
    for name, path in REVIEWS.items():
        with (path / "per_image_metrics.csv").open(newline="") as stream:
            rows[name] = {int(r["id"]): r for r in csv.DictReader(stream)}
        assert len(rows[name]) == 687
    ids = sorted(rows["original"])
    ci = bootstraps(rows)
    control_ci = bootstraps(rows, baseline="control")["boundary_weight4"]
    configs = {name: json.loads((SESSION / name / "run_config.json").read_text()) for name in ("control", "boundary_weight4")}
    for key in ("initial_checkpoint_sha256", "learning_rate", "batch_size", "seed", "epochs_maximum", "augmentation", "first_encoder_block", "encoder_and_decoder_batchnorm", "road_iou_guard_minimum"):
        assert configs["control"][key] == configs["boundary_weight4"][key]
    latency = json.loads((SESSION / "latency.json").read_text())
    assert latency["status"] == "completed"
    changes = {}
    for name in ("control", "boundary_weight4"):
        changes[name] = {}
        for ref in ("original", "morning_best"):
            delta = {key: (metrics[name]["global"][key] - metrics[ref]["global"][key]) * 100 for key in metrics[name]["global"]}
            differences = {i: float(rows[name][i]["boundary_iou_5px"]) - float(rows[ref][i]["boundary_iou_5px"]) for i in ids}
            changes[name][ref] = dict(global_delta_pp=delta, boundary_images_improved=sum(d>0 for d in differences.values()),
                                     boundary_images_worsened=sum(d<0 for d in differences.values()),
                                     boundary_images_equal=sum(d==0 for d in differences.values()))
    tails = {name: {"road_iou_5th_percentile": float(np.percentile([float(rows[name][i]["road_iou"]) for i in ids], 5)),
                   "boundary_iou_5px_5th_percentile": float(np.percentile([float(rows[name][i]["boundary_iou_5px"]) for i in ids], 5))}
             for name in LABELS}
    floor = configs["boundary_weight4"]["road_iou_guard_minimum"]
    eligible = [name for name in metrics if metrics[name]["global"]["road_iou"] >= floor]
    leader = max(eligible, key=lambda name: metrics[name]["global"]["boundary_iou_5px"])
    boundary_supported = control_ci["boundary_iou_5px_delta_pp"][0] > 0
    region_maintained = metrics["boundary_weight4"]["global"]["road_iou"] >= metrics["morning_best"]["global"]["road_iou"]
    recommendation = "boundary_weight4" if boundary_supported and region_maintained else "morning_best"
    # Control measurements remain valid evidence after its binary is removed.
    # Verify retained binaries; use the recorded reload audit for a deleted control.
    hashes = {}
    for name, path in CHECKPOINTS.items():
        if path.is_file():
            hashes[name] = sha(path)
        elif name == "control":
            audit = json.loads((SESSION / "control/reload_verification.json").read_text())
            assert audit["status"] == "passed"
            hashes[name] = audit["checkpoint_sha256"]
        else:
            raise FileNotFoundError(path)
    assert hashes["original"] == "652c3e537e6b944430304d103a80bdff21a67daddcba1d9590e49dfbff0765c3"
    assert hashes["morning_best"] == "93a8faf3d43b77f0ce74d5cdbaf94124728d371c3b68b2a3386f9f72fc106725"
    data = dict(status="completed", metrics=metrics, changes=changes, paired_bootstrap_95pct=ci,
                boundary_vs_matched_control_bootstrap_95pct=control_ci,
                validation_boundary_leader=leader, checkpoint_sha256=hashes, test_images_used=False,
                recommended_current_model=recommendation,
                lower_tail_metrics=tails,
                all_validation_ids=ids, configs_matched_except_loss_alpha=True,
                finished_at_utc=datetime.now(timezone.utc).isoformat())
    atomic(SESSION / "comparison_results.json", data)
    fields = ["id"] + [f"{name}_{key}" for name in LABELS for key in metrics[name]["global"]]
    with (SESSION / "comparison_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"id": i, **{f"{name}_{key}": rows[name][i][key] for name in LABELS for key in metrics[name]["global"]}} for i in ids)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(axes, ("val_boundary_iou_5px", "val_road_iou"), ("Validation Boundary IoU (5px)", "Validation road IoU")):
        for name, color in (("control", "#667788"), ("boundary_weight4", "#007a99")):
            history = json.loads((SESSION / name / "history.json").read_text())
            ax.plot(range(1, len(history[key])+1), np.array(history[key])*100, marker="o", label=LABELS[name], color=color)
        metric = key.removeprefix("val_")
        ax.axhline(metrics["morning_best"]["global"][metric]*100, ls="--", color="#dc9230", label="Morning best")
        ax.axhline(metrics["original"]["global"][metric]*100, ls=":", color="#955d87", label="Original")
        ax.set(title=title, xlabel="Epoch", ylabel="Percent")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SESSION / "training_curves.png", dpi=180)
    plt.close(fig)

    gallery = SESSION / "gallery"
    gallery.mkdir(exist_ok=False)
    (gallery / "comparisons").mkdir()
    (gallery / "previews").mkdir()
    cards = []
    for index, i in enumerate(ids):
        sheet = comparison_image(i, rows, gallery / f"comparisons/Image_{i}.png")
        sheet.resize((1536, 306), Image.Resampling.LANCZOS).save(gallery / f"previews/Image_{i}.jpg", quality=90)
        cards.append({"id": i, **{name: {key: float(rows[name][i][key]) for key in ("road_iou", "boundary_iou_5px", "visible_boundary_f1_3px")} for name in LABELS},
                      "delta": (float(rows["boundary_weight4"][i]["boundary_iou_5px"]) - float(rows["morning_best"][i]["boundary_iou_5px"])) * 100})
        if (index+1)%100==0:
            print(f"Rendered {index+1}/687 comparisons", flush=True)
    hard_ids = [48, 103, 4160, 606, 6514]
    montage = Image.new("RGB", (1536, 306 * len(hard_ids)), "white")
    for j, i in enumerate(hard_ids):
        with Image.open(gallery / f"comparisons/Image_{i}.png") as im:
            montage.paste(im.resize((1536, 306), Image.Resampling.LANCZOS), (0, j*306))
    montage.save(SESSION / "hard_cases.png")
    gains = sorted(ids, key=lambda i: float(rows["boundary_weight4"][i]["boundary_iou_5px"])-float(rows["morning_best"][i]["boundary_iou_5px"]))
    atomic(SESSION / "selected_cases.json", dict(user_discussed_ids=hard_ids, largest_boundary_regressions=gains[:10], largest_boundary_gains=gains[-10:][::-1]))

    def table_row(name):
        m = metrics[name]["global"]
        ms = latency["models"][name]["rgb_in_cpu_ram_to_binary_mask_in_cpu_ram_ms"]["median"]
        return f"<tr><td>{LABELS[name]}</td><td>{m['road_iou']*100:.3f}%</td><td>{m['boundary_iou_5px']*100:.3f}%</td><td>{m['visible_boundary_f1_3px']*100:.3f}%</td><td>{ms:.2f} ms</td></tr>"
    doc = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IDD boundary comparison</title>
<style>body{margin:0;background:#0d1520;color:#e1e9f1;font:16px system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:24px}h1{margin:0 0 10px;font-size:30px}p{color:#adbfd0;line-height:1.5}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:12px;text-align:left;border-bottom:1px solid #324253}th{color:#b7d3e6}button,select,input{font:inherit;background:#19293b;color:#edf4ff;border:1px solid #415269;border-radius:6px;padding:9px}button{cursor:pointer}nav{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:22px 0}article{background:#182333;border:1px solid #33495c;border-radius:9px;margin:24px 0;overflow:hidden}article header{padding:12px;display:flex;justify-content:space-between;align-items:center}article img{width:100%;display:block}a{color:#8cdce5}small{color:#b4c5d6}#count{min-width:200px}.good{color:#69d6b8}.bad{color:#ff9898}</style>
<main><h1>Road boundaries: original vs morning best vs boundary training</h1><p>All 687 validation images. Road IoU measures area; Boundary IoU measures outline bands. The control uses the usual loss with the same continuation settings. The supplied masks are unchanged; their outlines are not necessarily physical outer road edges or painted lines.</p>
<table><thead><tr><th>Checkpoint</th><th>Road IoU</th><th>Boundary IoU (5px)</th><th>Visible outline F1 (3px)</th><th>Warm RAM-to-mask median</th></tr></thead><tbody>__TABLE__</tbody></table>
<p>Panel order: RGB → supplied mask → original → morning best → boundary model → boundary overlay. Yellow = supplied outline; cyan = predicted outline; white = overlap. Open PNG for original resolution. Timing excludes capture, planning and display.</p>
<nav><label>Image ID <input id="search" placeholder="e.g. 48" size="12"></label><label>Order <select id="sort"><option value="id">Image ID</option><option value="worst">Lowest boundary IoU</option><option value="gain">Largest boundary gain</option><option value="regression">Largest boundary regression</option><option value="road">Lowest road IoU</option></select></label><button id="clear">Show all</button></nav>
<nav><button id="prev">Previous</button><span id="count"></span><button id="next">Next</button><label>Page <input id="page" type="number" min="1" style="width:70px"></label><button id="go">Go</button></nav><section id="cards"></section>
<p>Full numeric results, configurations, checkpoint verification, hard cases and curves are saved in BOUNDARY_REPORT.md and comparison_per_image.csv in the parent folder. The reserved test images were not used.</p></main>
<script>const data=__DATA__;let page=1;const $=s=>document.querySelector(s);function render(){let id=$('#search').value.match(/\\d+/);let rows=id?data.filter(r=>r.id===Number(id[0])):[...data];const order=$('#sort').value;rows.sort((a,b)=>order==='id'?a.id-b.id:order==='worst'?a.boundary_weight4.boundary_iou_5px-b.boundary_weight4.boundary_iou_5px:order==='road'?a.boundary_weight4.road_iou-b.boundary_weight4.road_iou:order==='gain'?b.delta-a.delta:a.delta-b.delta);let pages=Math.max(1,Math.ceil(rows.length/6));page=Math.min(Math.max(1,page),pages);$('#count').textContent=`Page ${page} / ${pages} • ${rows.length} images`;$('#page').value=page;$('#prev').disabled=page===1;$('#next').disabled=page===pages;$('#cards').innerHTML=rows.slice((page-1)*6,page*6).map(r=>`<article><header><strong>Image ${r.id}</strong><small class="${r.delta>=0?'good':'bad'}">Boundary change vs morning: ${r.delta>=0?'+':''}${r.delta.toFixed(3)} pp &nbsp; <a href="comparisons/Image_${r.id}.png" target="_blank">Open full PNG</a></small></header><a href="comparisons/Image_${r.id}.png" target="_blank"><img loading="lazy" src="previews/Image_${r.id}.jpg" alt="Image ${r.id}, unchanged mask, original, morning best, boundary prediction and outline overlay"></a></article>`).join('')||'<p>No matching validation image ID.</p>';}$('#search').oninput=()=>{page=1;render()};$('#sort').onchange=()=>{page=1;render()};$('#clear').onclick=()=>{$('#search').value='';$('#sort').value='id';page=1;render()};$('#prev').onclick=()=>{page--;render()};$('#next').onclick=()=>{page++;render()};function go(){page=Number($('#page').value)||1;render()}$('#go').onclick=go;$('#page').onkeydown=e=>{if(e.key==='Enter')go()};render();</script></html>'''
    (gallery / "index.html").write_text(doc.replace("__TABLE__", "".join(table_row(n) for n in LABELS)).replace("__DATA__", json.dumps(cards)), encoding="utf-8")
    atomic(gallery / "manifest.json", dict(images=687, ids=ids, columns=["RGB", "supplied_mask", "original", "morning_best", "boundary_weight4", "outline_overlay"], mask_values=[0,1], test_images_used=False))

    report = ["# Boundary-focused road segmentation evaluation and training", "", f"Completed: {data['finished_at_utc']}", "",
              "## What was done", "", "Both new runs start from the morning best (trial 13). The matched control uses ordinary BCE + soft Dice. The boundary model uses a custom normalized weighted BCE + the same soft Dice: pixels in a 3px morphological band around supplied-mask transitions receive weight 4; other pixels receive weight 1. Weighted BCE is divided by the sum of weights per image. This is not a reproduction of Kervadec's signed-distance boundary loss.", "",
              "Each run used six epochs, Adam 3e-6 (constant), batch 16, seed 42, mixed precision with float32 head/loss, the same lighting/shadow augmentation as trial 13, trainable encoder blocks 13–16 except BN, and fixed encoder/decoder BatchNorm. Optimizers and dataset seeds restart identically in each fresh process. Only loss alpha differs (control 0, boundary 3). Training-only morphology is absent from the forward network.", "",
              "Checkpoint rule, declared before training: maximize global validation Boundary IoU at 5px, provided road IoU is no more than 0.20 percentage points below the morning best. Both boundary-selected and road-selected checkpoints, final weights, per-epoch histories, exact configs and code snapshots are saved. The report compares boundary-selected checkpoints for both matched runs.", "",
              "## Results on the same 687 validation images", "",
              "| Model | Road IoU | Dice | Boundary IoU 5px | Boundary IoU 14px | Visible boundary F1 3px | Road precision | Road recall | Images below 50% road IoU |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in LABELS:
        m = metrics[name]["global"]
        report.append(f"| {LABELS[name]} | " + " | ".join(f"{m[k]*100:.4f}%" for k in ("road_iou", "road_dice", "boundary_iou_5px", "boundary_iou_14px", "visible_boundary_f1_3px", "road_precision", "road_recall")) + f" | {metrics[name]['road_iou_below_50pct']} |")
    report += ["", "Higher is better for overlap and F1 metrics. These are ratios of summed pixel counts over all images. Mean per-image scores are separately available in each metrics.json. A 55% boundary score and a 92% area score measure different regions and must not be interpreted as the same kind of accuracy.", "", "### Effect of boundary training", ""]
    for name in LABELS:
        report.append(f"- {LABELS[name]} lower tail: fifth-percentile per-image road IoU {tails[name]['road_iou_5th_percentile']*100:.2f}%, boundary IoU {tails[name]['boundary_iou_5px_5th_percentile']*100:.2f}%; {metrics[name]['road_iou_below_75pct']} images below 75% road IoU.")
    for ref in ("original", "morning_best", "control"):
        delta_b = (metrics["boundary_weight4"]["global"]["boundary_iou_5px"]-metrics[ref]["global"]["boundary_iou_5px"])*100
        delta_r = (metrics["boundary_weight4"]["global"]["road_iou"]-metrics[ref]["global"]["road_iou"])*100
        report.append(f"- Boundary model vs {LABELS[ref]}: Boundary IoU change {delta_b:+.4f} pp; road IoU change {delta_r:+.4f} pp.")
    for name in ("control", "boundary_weight4"):
        change = changes[name]["morning_best"]
        report.append(f"- {LABELS[name]} vs morning: {change['boundary_images_improved']} image boundaries improved, {change['boundary_images_worsened']} worsened, {change['boundary_images_equal']} tied.")
    report += ["", f"**Validation boundary leader under the declared region guard: {LABELS[leader]}.** This selection applies to these supplied labels and this validation sample; it does not prove reliable physical road edges or safe driving.", "", "### Paired image-bootstrap uncertainty", "", "2,000 resamples, seed 42; 95% percentile intervals for global-score changes versus morning best (percentage points). These describe variability within this validation sample. The same validation set selected checkpoints and prior trials, so the intervals are exploratory and are not an independent significance claim.", ""]
    for name, interval in ci.items():
        report.append(f"- {LABELS[name]}: road IoU [{interval['road_iou_delta_pp'][0]:+.4f}, {interval['road_iou_delta_pp'][1]:+.4f}], Boundary IoU 5px [{interval['boundary_iou_5px_delta_pp'][0]:+.4f}, {interval['boundary_iou_5px_delta_pp'][1]:+.4f}].")
    report.append(f"- Boundary model vs matched control: road IoU [{control_ci['road_iou_delta_pp'][0]:+.4f}, {control_ci['road_iou_delta_pp'][1]:+.4f}], Boundary IoU 5px [{control_ci['boundary_iou_5px_delta_pp'][0]:+.4f}, {control_ci['boundary_iou_5px_delta_pp'][1]:+.4f}].")
    report += ["", f"**Current project recommendation: {LABELS[recommendation]}.** The boundary leader above is the outcome of the predeclared boundary checkpoint criterion. Replacing the current project model is a separate judgment: examine the matched-control uncertainty and road-area tradeoff before adopting a new checkpoint."]
    if not boundary_supported:
        report += ["", "The boundary-versus-control interval includes or lies below zero. This single seed does not show a robust benefit attributable to the boundary loss. Retain the boundary checkpoint and evidence for review, rather than calling the loss an established improvement."]
    if not region_maintained:
        report += ["", "The boundary model's selected checkpoint has lower global road IoU than the morning model. A small outline gain therefore comes with an area-overlap tradeoff, within the declared guard."]
    report += ["", "## Difficult images and label issues", "", "| Image | Original road / boundary IoU | Morning road / boundary IoU | Boundary model road / boundary IoU |", "|---|---:|---:|---:|"]
    for i in hard_ids:
        report.append(f"| {i} | " + " | ".join(f"{float(rows[n][i]['road_iou'])*100:.2f}% / {float(rows[n][i]['boundary_iou_5px'])*100:.2f}%" for n in ("original", "morning_best", "boundary_weight4")) + " |")
    report += ["", "Review hard_cases.png and all 687 comparisons in gallery/index.html. selected_cases.json also lists the largest gains and regressions automatically; inferior cases are retained. Image 6514 has a visibly debatable supplied mask, and 606 has very little labelled foreground. Optimizing their mask outlines cannot establish that the masks match all visible paved/drivable areas. No examples were relabelled or excluded. Image 4160 is the previously discussed shadow case; 48 is the traffic/fragmented-road failure case. Numerical changes, not visual plausibility alone, determine the reported overlap scores.", "",
               "## Real-time cost measured on your RTX 4060", "", "| Model | Warm forward + probability copy: median / p95 | RGB in CPU RAM → binary mask in CPU RAM: median / p95 |", "|---|---:|---:|"]
    for name in LABELS:
        timing = latency["models"][name]
        a = timing["resident_normalized_forward_plus_probability_copy_ms"]
        b = timing["rgb_in_cpu_ram_to_binary_mask_in_cpu_ram_ms"]
        report.append(f"| {LABELS[name]} | {a['median']:.2f} / {a['p95']:.2f} ms | {b['median']:.2f} / {b['p95']:.2f} ms |")
    report += ["", "Batch 1 at 512×512, tf.function, 15 warmups per path, 64 synchronized samples per path, eight spaced validation RGB images. RAM-to-mask includes normalization, transfer, GPU forward pass, threshold and output copy. It excludes camera capture, cold startup, disk I/O, tracking, planning, display and vehicle control. Full-system FPS remains unmeasured. Original uses its saved float32 computation; the morning/control/boundary models use mixed precision, so original-to-new timing differences do not isolate the loss.", "",
               f"All four models have the same forward layer structure and {latency['models']['original']['parameters']:,} parameters. Boundary weighting adds no inference layer, white-line search, or second pass. Small differences between equally structured mixed-precision models can reflect thermal/load/timing variation rather than architectural speed gains.", "",
               "## Metric definitions and important limits", "",
               "- **Road IoU** = road intersection / road union; **Dice** = 2TP / (2TP + FP + FN). Probability threshold is fixed at 0.5 for every model; no threshold search was performed.",
               "- **Boundary IoU 5px** = intersection / union of the inner 5px road boundary bands of truth and prediction. Each band is mask minus square-kernel erosion. Frame is padded as background, matching the authors' published morphology implementation. This strict radius is a project choice declared before training.",
               "- **Boundary IoU 14px** uses round(0.02 × sqrt(512²+512²)) = 14px, the authors' common 2%-diagonal setting. It is secondary and more tolerant.",
               "- **Visible boundary F1 at 3px** uses one-pixel inner contours and approximate many-to-one matches within 3px Chebyshev distance (square dilation). Image-frame contours are excluded, because the frame is not an observed physical edge. This is explicitly an approximate contour score, not exact bipartite edge matching.",
               "- Binary road masks include outer road edges, obstacle contours and holes. We have no separate outer-edge-line labels, lane detector, or outer-boundary-only evaluation. This training encourages agreement with supplied road/nonroad outlines; it cannot demonstrate explicit use of white paint.",
               "- Supplied class conversion/ignore policy is still not audited against the original IDD annotations. Some blue/red disagreements may be annotation/class-definition issues. Mask-focused training may reinforce incorrect boundaries.",
               "- Sequence identities are absent from the existing split, so correlated frames could cross splits. Only one continuation seed was tested; the validation set has already been used for many choices. More independent data/seeds and final held-out evaluation are needed for stronger generalization claims.", "",
               "The Boundary IoU definition and morphology follow [Cheng et al., CVPR 2021](https://arxiv.org/html/2103.16562v1) and the [authors' implementation](https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py). The broader idea of combining regional and boundary objectives is discussed by [Kervadec et al.](https://arxiv.org/abs/1812.07032); our weighted-BCE implementation is different.", "",
               "## Preservation and actual completion checks", "",
               "- Training/validation/testing counts remain **5507 / 687 / 689**. Test contents were never read; read_pairs checks only their filenames and existence. No test dataset was constructed.",
               "- PNG originals, existing split, 0/1 masks and normalization-once input pipeline are preserved. Validation RGB/masks are unaugmented. All four reviews have the same manifest and validation-mask digest.",
               "- Original, morning-best and boundary-best checkpoints retain their verified SHA-256 hashes. Redundant backup copies and the control binary were removed at the user's request; control measurements and reload verification are preserved.",
               "- Each new selected checkpoint was reloaded and independently exported over all 687 images; CPU counts exactly matched its training validation road IoU/Dice and Boundary IoU. Binary predicted masks, per-image results and verification JSON are saved.",
               "- End-of-run assertions confirmed decoder weight updates and unchanged frozen BatchNorm weights/statistics. No running training job is required to use the saved artifacts. The old overnight automation remains paused.", "",
               "### Checkpoints", ""]
    for name in LABELS:
        availability = "retained" if CHECKPOINTS[name].is_file() else "binary removed during cleanup; recorded measurements retained"
        report.append(f"- {LABELS[name]}: `{CHECKPOINTS[name].relative_to(ROOT).as_posix()}` — {availability}; SHA-256 `{hashes[name]}`.")
    report += ["", "## Method explanation", "", "Our first loss treated every pixel equally, so a large correct road area could hide a less accurate outline. We measured boundary overlap separately, then gave mask-transition pixels extra weight during training. A matched normal-loss continuation helped separate the effect of extra training from the changed loss. The architecture stayed unchanged, so the boundary loss is not run during live prediction. We compare all models on the same validation images, keep the test set for final evaluation, and report failures and possible label problems along with the averages.", "",
               "## Next stage", "", "Review the boundary gains and regressions before freezing a final model. Audit questionable label/class conversion if it affects interpretation. Once model and inference settings are frozen, do one final reserved-test evaluation, then prepare the calibrated road/obstacle map for planning. RL STOP/SLOW/PROCEED and MATLAB/Simulink vehicle visualization remain pending and were not included in this training task."]
    (SESSION / "BOUNDARY_REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    atomic(ROOT / "models/boundary_20261003_selected_model.json", dict(status="validation_boundary_selected_test_pending", model=leader,
           checkpoint=str(CHECKPOINTS[leader].relative_to(ROOT)), checkpoint_sha256=hashes[leader],
           selection="highest verified global Boundary IoU at 5px subject to declared road-IoU guard",
           global_validation_metrics=metrics[leader]["global"], test_evaluated=False,
           recommended_current_model=recommendation,
           report="runs/boundary_20261003/BOUNDARY_REPORT.md"))
    atomic(SESSION / "artifact_status.json", dict(status="completed", gallery_images=687, report_written=True,
           original_and_morning_models_preserved=True, selected_model=leader, test_images_used=False))
    print(json.dumps({name: m["global"] for name, m in metrics.items()}, indent=2))
    print("Validation boundary leader:", leader)


if __name__ == "__main__":
    main()
