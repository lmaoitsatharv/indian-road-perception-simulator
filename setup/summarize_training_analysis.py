"""Export saved training measurements only to an explicitly selected new folder.

Import is read-only. Running without --output-dir shows the saved report location.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/frozen_01"


def find_console_log(root):
    root = Path(root)
    for path in (root / "setup/frozen_01_console.log",
                 root / "archive/completed_runtime_20261003/setup/frozen_01_console.log"):
        if path.is_file():
            return path
    raise FileNotFoundError("The recorded frozen_01 console log is unavailable in setup or its archive")


def summarize(output_dir):
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a new analysis folder; saved training evidence is preserved")
    history = list(csv.DictReader((RUN / "history.csv").open()))
    cases = list(csv.DictReader((RUN / "validation_review/per_image_metrics.csv").open()))
    validation = json.loads((RUN / "validation_review/validation_metrics.json").read_text())
    benchmarks = [json.loads(path.read_text()) for path in sorted((ROOT / "setup").glob("benchmark_*.json"))]
    epochs = np.arange(1, len(history) + 1)
    train = np.array([float(row["road_iou"]) for row in history])
    val = np.array([float(row["val_road_iou"]) for row in history])
    best = int(np.argmax(val))
    ious = np.array([float(row["road_iou"]) for row in cases])
    times = [float(value) for value in re.findall(r"Epoch \d+ completed in ([0-9.]+) seconds", find_console_log(ROOT).read_text(encoding="utf-8"))]
    if not history or not cases or not times:
        raise ValueError("Saved history, validation measurements and epoch timings must not be empty")
    summary = {"best_epoch": best + 1, "best_validation_iou": float(val[best]),
               "best_epoch_train_validation_gap_percentage_points": float((train[best] - val[best]) * 100),
               "final_train_validation_gap_percentage_points": float((train[-1] - val[-1]) * 100),
               "per_image_iou_percentiles": {str(p): float(np.percentile(ious, p)) for p in [5, 10, 25, 50]},
               "cases_below_0_5": int(np.count_nonzero(ious < .5)),
               "cases_below_0_75": int(np.count_nonzero(ious < .75)),
               "epoch_seconds_min_median_max": [min(times), float(np.median(times)), max(times)],
               "benchmarks": benchmarks}
    output.mkdir(parents=True, exist_ok=False)
    (output / "training_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    axes[0].plot(epochs, train * 100, label="Training", color="#2156a5", marker=".")
    axes[0].plot(epochs, val * 100, label="Validation", color="#d26425", marker=".")
    axes[0].scatter([best + 1], [val[best] * 100], s=65, color="#d26425", zorder=3)
    axes[0].annotate(f"Best validation: {val[best]:.2%}\nat epoch {best + 1}",
                     xy=(best + 1, val[best] * 100), xytext=(8, 89),
                     arrowprops={"arrowstyle": "->"}, fontsize=10)
    axes[0].set(xlabel="Epoch", ylabel="Road IoU (%)", title="Training improves; validation levels off", ylim=(86, 96))
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.2)
    axes[1].hist(ious, bins=np.linspace(0, 1, 21), color="#2156a5", edgecolor="white")
    axes[1].axvline(.5, color="#d26425", linestyle="--", label="IoU 0.5")
    axes[1].set(xlabel="Per-image road IoU", ylabel="Validation images", title=f"687 images: {sum(ious < .5)} below 0.5; {sum(ious < .75)} below 0.75")
    axes[1].legend(frameon=False)
    figure.suptitle("First U-Net run: where improvement is needed", fontsize=14)
    figure.savefig(output / "training_analysis.png", dpi=160)
    plt.close(figure)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New output folder; historical analysis is never overwritten")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        print(f"Saved analysis: {RUN / 'training_analysis.json'}")
        print("To export separately, pass --output-dir with a new folder. Historical files are preserved.")
        return
    print(json.dumps(summarize(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
