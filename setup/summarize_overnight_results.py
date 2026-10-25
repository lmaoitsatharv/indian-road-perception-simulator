"""Summarize saved validation evidence; performs no inference or training."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/overnight_20261003"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(folder):
    with (folder / "validation_review/per_image_metrics.csv").open(newline="") as stream:
        return {int(row["id"]): {key: float(value) for key, value in row.items() if key != "id"}
                for row in csv.DictReader(stream)}


def distribution(values):
    values = np.asarray(values)
    return {"mean_iou": float(values.mean()), "median_iou": float(np.median(values)),
            "fifth_percentile_iou": float(np.percentile(values, 5)),
            "below_0_5": int((values < .5).sum()), "below_0_75": int((values < .75).sum())}


def main():
    if (ROOT / "models/segmentation_final_lock.json").exists():
        raise RuntimeError(
            "The final road model is locked and its test evaluation is complete. "
            "Use the saved overnight reports; historical regeneration would replace "
            "the deployed selection pointer and revert final-test status."
        )
    comparison = read(OUT / "comparison.json")
    baseline_folder = ROOT / "runs/frozen_01"
    best_folder = (ROOT / comparison["best_verified_checkpoint"]).parent
    baseline = read(baseline_folder / "validation_review/validation_metrics.json")
    best = read(best_folder / "validation_review/validation_metrics.json")
    assert best["saved_checkpoint_matches_logged_iou"] and best["graph_predictions_match_keras_evaluation"]
    baseline_rows, best_rows = rows(baseline_folder), rows(best_folder)
    assert baseline_rows.keys() == best_rows.keys() and len(best_rows) == 687
    deltas = {key: best_rows[key]["road_iou"] - baseline_rows[key]["road_iou"] for key in best_rows}
    checkpoint = best_folder / "best.keras"
    manifest = ROOT / "dataset/splits/pair_splits.csv"
    assert sha(checkpoint) == best["checkpoint_sha256"]
    assert sha(baseline_folder / "best.keras") == baseline["checkpoint_sha256"]
    backup = ROOT / "backups/trained_baseline_20261003_0239/best.keras"
    if backup.exists():
        assert sha(backup) == baseline["checkpoint_sha256"]
    assert sha(manifest) == best["manifest_sha256"] == "512cc054e2291ec5e197b0dca913ba94698cf055062b54511977816872005e90"
    split_counts = {}
    for split in ["train", "val", "test"]:
        image_ids = {path.stem.replace("Image_", "") for path in (ROOT / f"dataset/splits/{split}/images").glob("Image_*.png")}
        mask_ids = {path.stem.replace("Mask_", "") for path in (ROOT / f"dataset/splits/{split}/masks").glob("Mask_*.png")}
        assert image_ids == mask_ids
        split_counts[split] = len(image_ids)
    assert split_counts == {"train": 5507, "val": 687, "test": 689}
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "best_checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
               "baseline_iou": baseline["global_road_iou"], "best_iou": best["global_road_iou"],
               "improvement_percentage_points": 100 * (best["global_road_iou"] - baseline["global_road_iou"]),
               "baseline_distribution": distribution([r["road_iou"] for r in baseline_rows.values()]),
               "best_distribution": distribution([r["road_iou"] for r in best_rows.values()]),
               "images_improved": sum(d > 1e-12 for d in deltas.values()),
               "images_worsened": sum(d < -1e-12 for d in deltas.values()),
               "images_tied": sum(abs(d) <= 1e-12 for d in deltas.values()),
               "case_4160": {"baseline": baseline_rows[4160], "best": best_rows[4160]},
               "split_counts": split_counts, "manifest_sha256": sha(manifest),
               "baseline_backup_verified": True, "test_set_evaluated": False,
               "trials": []}
    with (OUT / "per_image_baseline_vs_best.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "baseline_iou", "best_iou", "change_percentage_points"])
        for key in sorted(deltas):
            writer.writerow([key, baseline_rows[key]["road_iou"], best_rows[key]["road_iou"], 100*deltas[key]])
    lines = ["# Overnight result analysis", "", f"Generated UTC: {summary['created_at_utc']}", "",
             "## What improved", "",
             f"Preferred validation candidate: `{checkpoint.relative_to(ROOT)}`.",
             f"Road IoU: **{best['global_road_iou']:.4%}**, versus **{baseline['global_road_iou']:.4%}** before tonight.",
             f"Change: **{summary['improvement_percentage_points']:.4f} percentage points**. Dice: **{best['global_road_dice']:.4%}**.",
             "This is a small validation gain. It is not a final test result or proof of reliable improvement across seeds.", "",
             f"The candidate came from trial 13's first epoch; its later epochs scored lower.",
             "It used batch 16, mixed precision, learning rate 0.000003, blocks 13-16 fine-tuned,",
             "fixed encoder and decoder BatchNorm, and training brightness/contrast/synthetic shadow augmentation.",
             "The remaining model layers and targets stayed the same. Pretrained ImageNet initialization was preserved through the baseline checkpoint.", "",
             "## All controlled trials", "",
             "| Trial | Validation IoU | Change vs baseline (pp) | Best epoch | Epochs run | Training minutes | TF peak MiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for entry in comparison["trials"]:
        folder = ROOT / entry["run"]
        status = read(folder / "run_status.json")
        with (folder / "memory_by_epoch.csv").open(newline="") as stream:
            memory = list(csv.DictReader(stream))
        peak = max(float(row["tensorflow_peak_mib"]) for row in memory)
        result = {"trial": folder.name, "verified_iou": entry["verified_iou"], "best_epoch": status["best_epoch"],
                  "epochs": status["epochs_completed"], "training_minutes": status["elapsed_seconds"]/60,
                  "tensorflow_peak_mib": peak, "decoder_batchnorm_unchanged": status.get("decoder_batchnorm_unchanged")}
        summary["trials"].append(result)
        lines.append(f"| {folder.name} | {entry['verified_iou']:.4%} | {100*(entry['verified_iou']-baseline['global_road_iou']):+.4f} | {status['best_epoch']} | {status['epochs_completed']} | {result['training_minutes']:.2f} | {peak:.0f} |")
    lines += ["", "Training times include epoch validation and checkpoint saving, but exclude the later separate reload review/plot steps.",
              "TF peak MiB means TensorFlow-used memory; total reserved VRAM and unrelated apps are separate.", "",
              "## Per-image results and remaining failures", "",
              "| Validation measure | Baseline | Preferred candidate |", "|---|---:|---:|"]
    for label, key in [("Mean per-image IoU", "mean_iou"), ("Median per-image IoU", "median_iou"),
                       ("Fifth percentile IoU", "fifth_percentile_iou")]:
        lines.append(f"| {label} | {summary['baseline_distribution'][key]:.3%} | {summary['best_distribution'][key]:.3%} |")
    for label, key in [("Images below IoU 0.50", "below_0_5"), ("Images below IoU 0.75", "below_0_75")]:
        lines.append(f"| {label} | {summary['baseline_distribution'][key]} | {summary['best_distribution'][key]} |")
    for title, metrics in [("Baseline", baseline), ("Preferred", best)]:
        precision = metrics["tp"]/(metrics["tp"]+metrics["fp"])
        recall = metrics["tp"]/(metrics["tp"]+metrics["fn"])
        lines.append(f"\n{title}: road precision {precision:.4%}, road recall {recall:.4%}; missed road pixels {metrics['fn']:,}, extra road pixels {metrics['fp']:,}.")
    lines += ["", f"Across 687 images: {summary['images_improved']} improved, {summary['images_worsened']} worsened and {summary['images_tied']} tied.",
              f"ID 4160, the user's shadow/paving example: **{baseline_rows[4160]['road_iou']:.3%} -> {best_rows[4160]['road_iou']:.3%}** IoU.",
              "The contact sheet shows the unchanged ground truth and both predictions. Red means missed labelled road; blue means extra predicted road.",
              "Weak scenes remain weak. A small global gain does not establish that shadows or obstacle boundaries are solved.", "",
              "## What the experiments teach", "",
              "- Fine-tuning or lighting changes alone did not consistently exceed the original baseline.",
              "- Combining them gave small gains; the same batch-16 configuration repeated with seed 73 scored lower than the baseline.",
              "- Freezing decoder BatchNorm shortened measured epochs in these trials, but alone did not give a clear accuracy gain.",
              "- Gentler learning rate from the original baseline produced the highest verified validation score.",
              "- Further refinement of a selected checkpoint did not reliably improve it. The best earlier checkpoint was retained.", "",
              "## Preserved data and model selection", "",
              f"Verified paired split counts: {split_counts}. The manifest and retained original model hashes match; redundant backup copies were removed during cleanup.",
              "Original images and 0/1 masks were preserved. Training augmentation precedes a single MobileNetV2 normalization to [-1,1].",
              "No test images were read or evaluated for this analysis; test filename counts alone were checked.",
              f"Preferred checkpoint SHA-256: `{summary['checkpoint_sha256']}`.",
              "The selection JSON points to this candidate without replacing the baseline or other trials.", "",
              "## Morning decisions still needed", "",
              "Confirm original road-class definitions in ambiguous low-score masks before changing labels. Sequence IDs remain absent,",
              "so related frames may cross splits. Removing 110 empty masks from the split limits the represented scene distribution.",
              "Repeated validation selection may overfit validation. The reserved test and external scenes must be checked later, after selecting the model.",
              "Threshold remains 0.5. No test-based threshold tuning occurred. Model output is a perspective road/non-road mask, not an obstacle list or metric map.",
              "Metric projection, Hybrid A*, tracking/obstacle integration, bounded STOP/SLOW/PROCEED RL and virtual-car visualization remain future stages.", ""]
    (OUT / "results_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "RESULTS_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    selection = {"status": "validation_selected_test_pending", "checkpoint": str(checkpoint),
                 "checkpoint_sha256": summary["checkpoint_sha256"], "global_validation_iou": best["global_road_iou"],
                 "global_validation_dice": best["global_road_dice"], "threshold": .5, "test_evaluated": False,
                 "manifest_sha256": summary["manifest_sha256"], "original_baseline_preserved": True,
                 "config": str(best_folder / "run_config.json"), "report": str(OUT / "MORNING_REPORT.md")}
    (ROOT / "models/overnight_20261003_selected_model.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(11, 5))
    names = [str(index+1) for index in range(len(summary["trials"]))]
    improvements = [100*(r["verified_iou"]-baseline["global_road_iou"]) for r in summary["trials"]]
    axis.bar(names, improvements, color=["#168a7b" if value >= 0 else "#d8665d" for value in improvements])
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Trial number", ylabel="Validation road IoU change (percentage points)",
             title="Small gains and regressions across 15 controlled trials")
    axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "validation_comparison.png", dpi=160)
    plt.close(fig)
    ids = [4160, 606, 6514]
    width, height = 240, 275
    sheet = Image.new("RGB", (width*5, height*len(ids)+35), "white")
    draw = ImageDraw.Draw(sheet)
    for row, key in enumerate(ids):
        rgb = np.asarray(Image.open(ROOT / f"dataset/splits/val/images/Image_{key}.png").convert("RGB"))
        truth = np.asarray(Image.open(ROOT / f"dataset/splits/val/masks/Mask_{key}.png"))
        before = np.asarray(Image.open(baseline_folder / f"validation_review/predicted_masks/Prediction_{key}.png"))
        after = np.asarray(Image.open(best_folder / f"validation_review/predicted_masks/Prediction_{key}.png"))
        assert all(set(np.unique(mask)).issubset({0, 1}) for mask in [truth, before, after])
        errors = rgb.copy()
        missed, extra = (truth == 1) & (after == 0), (truth == 0) & (after == 1)
        errors[missed] = [240, 55, 55]
        errors[extra] = [35, 125, 245]
        panels = [rgb, truth*255, before*255, after*255, errors]
        captions = [f"ID {key}: original", "Unchanged ground truth",
                    f"Baseline IoU {baseline_rows[key]['road_iou']:.2%}",
                    f"Candidate IoU {best_rows[key]['road_iou']:.2%}", "Red: missed / Blue: extra"]
        for column, (panel, caption) in enumerate(zip(panels, captions)):
            sheet.paste(Image.fromarray(panel.astype(np.uint8)).convert("RGB").resize((width, width)),
                        (column*width, row*height+30))
            draw.text((column*width+5, row*height+8), caption, fill="black")
    draw.text((5, height*len(ids)+7), "Validation examples only. Pixel masks remain 0/1; multiplication by 255 is only for this display.", fill="black")
    sheet.save(OUT / "baseline_vs_best_cases.png")
    report = OUT / "MORNING_REPORT.md"
    original = report.read_text(encoding="utf-8")
    if "## Saved detailed analysis" not in original:
        report.write_text(original + "\n## Saved detailed analysis\n\nSee RESULTS_ANALYSIS.md and results_analysis.json for exact scores, per-image changes, timings, memory, preserved hashes and remaining limits.\nThe charts validation_comparison.png and baseline_vs_best_cases.png show comparisons and remaining failures.\nThe file models/overnight_20261003_selected_model.json points to the verified validation candidate; the test set remains pending.\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["best_iou", "improvement_percentage_points", "images_improved", "images_worsened", "case_4160", "split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
