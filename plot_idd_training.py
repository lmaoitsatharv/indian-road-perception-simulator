"""Create training curves and validation contact sheets from saved run outputs."""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image


def plot_run(run_dir):
    root = Path(__file__).resolve().parent
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    fine = config["phase"] == "fine"
    encoder_description = "partially fine-tuned MobileNetV2 encoder" if fine else "frozen MobileNetV2 encoder"
    with (run_dir / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise ValueError("No completed epoch is available")
    epochs = np.array([int(row["epoch"]) + 1 for row in history])
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, key, title in zip(axes, ("loss", "road_iou", "road_dice"),
                                ("BCE + Dice loss", "Global road IoU", "Global road Dice")):
        axis.plot(epochs, [float(row[key]) for row in history], label="Training", color="#2156a5", marker=".")
        axis.plot(epochs, [float(row[f"val_{key}"]) for row in history], label="Validation", color="#d26425", marker=".")
        axis.set(title=title, xlabel="Epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        if key != "loss":
            axis.set_ylim(0, 1)
    figure.suptitle(f"IDD road segmentation | {encoder_description} | batch {config['batch_size']} | seed {config['seed']}", fontsize=14)
    figure.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(figure)
    review = run_dir / "validation_review"
    if not (review / "validation_metrics.json").is_file():
        return
    report = json.loads((review / "validation_metrics.json").read_text())
    with (review / "per_image_metrics.csv").open(newline="") as stream:
        metrics = {int(row["id"]): row for row in csv.DictReader(stream)}
    for ids, filename, title in (
        (report["illustrative_ids"], "validation_examples.png", "Validation examples (evenly spaced IDs)"),
        (report["lowest_iou_ids"], "validation_lowest_iou.png", "Six lowest-IoU validation examples"),
    ):
        figure, axes = plt.subplots(len(ids), 4, figsize=(14, len(ids) * 2.9),
                                    squeeze=False, constrained_layout=True)
        for row_index, pair_id in enumerate(ids):
            image = np.asarray(Image.open(root / "dataset" / "splits" / "val" / "images" / f"Image_{pair_id}.png").convert("RGB"))
            target = np.asarray(Image.open(root / "dataset" / "splits" / "val" / "masks" / f"Mask_{pair_id}.png"))
            prediction = np.asarray(Image.open(review / "predicted_masks" / f"Prediction_{pair_id}.png"))
            axes[row_index, 0].imshow(image)
            axes[row_index, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 3].imshow(image)
            rgba = np.zeros((*prediction.shape, 4), dtype=float)
            rgba[prediction == 1] = [0.1, 0.9, 0.4, 0.35]
            axes[row_index, 3].imshow(rgba)
            for column, heading in enumerate(("Original image", "Provided mask", "Predicted mask", "Prediction overlay")):
                axes[row_index, column].set_title(heading if row_index == 0 else "", fontsize=11)
                axes[row_index, column].set_xticks([])
                axes[row_index, column].set_yticks([])
            axes[row_index, 0].set_ylabel(f"ID {pair_id}\nIoU {float(metrics[pair_id]['road_iou']):.3f}")
        figure.suptitle(f"{title} | best epoch {report['best_epoch']} | threshold 0.5", fontsize=14)
        figure.savefig(review / filename, dpi=140)
        plt.close(figure)
    print(f"Saved training curves and validation images in {run_dir}")
    status = json.loads((run_dir / "run_status.json").read_text())
    config = json.loads((run_dir / "run_config.json").read_text())
    ist = timezone(timedelta(hours=5, minutes=30))
    started = datetime.fromisoformat(status["started_at_utc"]).astimezone(ist)
    finished = datetime.fromisoformat(status["finished_at_utc"]).astimezone(ist)
    minutes = status["elapsed_seconds"] / 60
    gpu_names = ", ".join(device["device_name"] for device in config["gpu_details"])
    ious = np.array([float(row["road_iou"]) for row in metrics.values()])
    lines = [
        "# Road segmentation training result", "",
        f"Run: `{run_dir.name}`. Status: **{status['status']}**.", "",
        f"Started: {started:%d %B %Y, %I:%M %p} IST. Finished: {finished:%d %B %Y, %I:%M %p} IST.",
        f"Training duration: **{minutes:.1f} minutes**, excluding initial setup and the later validation review.", "",
        "## What was trained", "",
        f"U-Net with a {encoder_description} and a trainable decoder.",
        f"GPU: {gpu_names}, through WSL2. Resolution 512 x 512, batch {config['batch_size']}, seed {config['seed']}.",
        "Images used the existing [-1, 1] preprocessing once; target masks stayed 0/1.",
        f"Loss: binary cross-entropy plus soft Dice. Adam starting learning rate {config['learning_rate']}.", "",
        "## GPU monitoring scope", "",
        "The user reported playing Valorant during the original frozen_01 baseline run and observing about 2.5 GB used by training.",
        "Recorded nvidia-smi samples in setup/frozen_01_gpu_samples.csv are whole-device totals,",
        "including other applications. They do not establish TensorFlow-only VRAM or utilization.",
        "The training-only memory peak and gaming's effect on duration were not measured.",
        "TensorFlow GPU execution was verified independently of those totals.",
        "See the frozen_01/GPU_MONITORING_NOTES.md for the correction; newer trials record TensorFlow memory separately.", "",
        "## Validation result", "",
        f"Completed **{status['epochs_completed']} of at most {config['epochs_maximum']} epochs**.",
        f"Selected checkpoint: `best.keras`, from **epoch {report['best_epoch']}**.",
        f"Reloaded checkpoint global road IoU: **{report['global_road_iou']:.4f} ({report['global_road_iou']:.2%})**.",
        f"Reloaded checkpoint global road Dice: **{report['global_road_dice']:.4f} ({report['global_road_dice']:.2%})**.",
        "These are validation results on 687 examples at threshold 0.5, not final test results.",
        f"Reloaded checkpoint reproduced logged IoU: {report['saved_checkpoint_matches_logged_iou']}.", "",
        f"Pretrained encoder weights remained unchanged: {report['encoder_weights_unchanged']}.",
        f"Decoder output weights changed through training: {report['decoder_weights_changed']}.", "",
        "## Validation review observations", "",
        f"Median per-image IoU: {np.median(ious):.2%}. Mean per-image IoU: {np.mean(ious):.2%}.",
        f"Images with per-image IoU below 0.5: {int(np.count_nonzero(ious < 0.5))} of {len(ious)}.",
        "These per-image summaries differ from the global pixel-count IoU used to select the checkpoint.",
        "Illustrative paved-road examples generally match the provided target well. Inspected weak cases",
        "include rough/unpaved surfaces, strong shadows, and crowded/occluded scenes.",
        "Some provided masks have tiny or narrow positive regions despite broader visible ground,",
        "including IDs 606 and 864. Confirm the original class mapping before treating these as label errors.",
        "Existing images, masks, split assignments, and exclusions were preserved.",
        f"This run completed {status['epochs_completed']} epochs. Stop reason: {status.get('stop_reason', 'epoch budget or early stopping')}.",
        f"Fine-tuning performed in this run: {fine}.", "",
        "## Reload verification", "",
        "Compiled Keras evaluation and graph predictions were compared with the saved training score.",
        f"Reloaded-vs-logged IoU difference: {report['logged_iou_difference']}; graph-vs-evaluation difference: {report['graph_prediction_iou_difference']}.",
        "The original frozen_01 baseline also preserves its earlier eager diagnostic separately.",
        "The reported validation result and preview masks use the verified graph execution path.", "",
        "## Files to inspect", "",
        "- `training_curves.png`: training and validation loss, road IoU, and Dice.",
        "- `validation_review/validation_examples.png`: six illustrative validation examples.",
        "- `validation_review/validation_lowest_iou.png`: the six weakest validation examples by IoU.",
        "- `validation_review/per_image_metrics.csv`: all 687 validation example scores.",
        "- `history.csv`, `run_config.json`, `run_status.json`: settings and complete epoch logs.",
        "- `source/`: the exact training, model, and loader source files used.", "",
        "## Preserved inputs and remaining work", "",
        f"Split manifest unchanged: {status['manifest_unchanged']}.",
        f"Initial untrained checkpoint unchanged: {status['initial_checkpoint_unchanged']}.",
        "Split remains 5,507 train / 687 validation / 689 test. Test images were not evaluated.",
        "The 110 exclusions and absent sequence IDs remain dataset limitations; adjacent frames may cross splits.",
        "Next: review target-label definitions and weak validation cases, then choose any fine-tuning or threshold adjustment.",
        "Final test evaluation, external-photo checks, metric-map integration, MATLAB driving, and RL remain pending.",
        "RL remains planned for STOP / SLOW / PROCEED, with Hybrid A* route planning and car visualization.",
        "",
    ]
    (run_dir / "TRAINING_RESULT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    plot_run(parser.parse_args().run_dir)
