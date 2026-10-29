"""Matched old-vs-new detector comparison for the 29 Oct 2026 overnight v4 run (reads existing analysis files only; no GPU).

Old = epoch-3 checkpoint f7bb7e4c, new = runs/detection_train_20261029_01/weights/best.pt. Each pair uses the same images and settings.
Writes runs/detection_v4_comparison_20261029/comparison.json and per-class before/after charts into that same runs/ folder.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "detection_v4_comparison_20261029"
DOCS = ROOT / "docs"
OLD_SHA = "f7bb7e4c5959520d797dcde5547fdcded9ece1e9651a3e7bd6ffa2273749182c"

PAIRS = {
    "validation_1536_nms060": ("detection_setting_probe_20261028_1536_nms060", "detection_v4_eval_20261029_1536_nms060"),
    # detection_error_review_20261028_01 is the old checkpoint at 1280 / framework default NMS 0.7; its framework metrics equal
    # runs/detection_pilot_recovery_20261025_01/matched_final_1280.json exactly, and it also has the size bins.
    "validation_1280_nms070": ("detection_error_review_20261028_01", "detection_v4_eval_20261029_1280_nms070"),
    "dats_holdout_1536_nms060": ("detection_v4_holdout_20261029_old", "detection_v4_holdout_20261029_new"),
    "val_land_1536_nms060": ("detection_v4_valland_20261029_old", "detection_v4_valland_20261029_new"),
}
METRICS = {"precision": "metrics/precision(B)", "recall": "metrics/recall(B)", "map50": "metrics/mAP50(B)", "map50_95": "metrics/mAP50-95(B)"}


def load(name):
    path = RUNS / name / "analysis.json"
    return json.loads(path.read_text()) if path.exists() else None


def summarise(a):
    animal = a["results"]["conf_0.25"]["animal"]
    if "per_class_ap" not in a:  # the 28 Oct 1280 review predates per-class AP output; use the matched final file it reproduces
        matched = json.loads((RUNS / "detection_pilot_recovery_20261025_01" / "matched_final_1280.json").read_text())
        assert matched["checkpoint_sha256"] == a["checkpoint_sha256"] and matched["metrics"] == a["framework_metrics"]
        a = dict(a, per_class_ap=matched["per_class"])
    return {
        "checkpoint_sha256": a["checkpoint_sha256"],
        "images": a["images"],
        "targets": a["targets"],
        "imgsz": a["imgsz"],
        "nms_iou": a.get("nms_iou") or 0.7,
        "overall": {k: a["framework_metrics"][v] for k, v in METRICS.items()},
        "per_class_ap50": {c: v["ap50"] for c, v in a["per_class_ap"].items()},
        "per_class_ap50_95": {c: v["ap50_95"] for c, v in a["per_class_ap"].items()},
        "animal_conf025": {
            "targets": animal["targets"], "recall": animal["recall"], "predictions": animal["predictions"], "precision": animal["precision"],
            "by_size": {s: {k: b[k] for k in ("targets", "found", "recall", "predictions", "precision")} for s, b in animal["by_size"].items()},
        },
    }


def main():
    OUT.mkdir(exist_ok=True)
    report = {"old_checkpoint_sha256": OLD_SHA, "pairs": {}, "missing": []}
    for key, (old_name, new_name) in PAIRS.items():
        old, new = load(old_name), load(new_name)
        if old is None or new is None:
            report["missing"].append(key)
            continue
        o, n = summarise(old), summarise(new)
        assert o["checkpoint_sha256"] == OLD_SHA, key
        assert (o["images"], o["targets"], o["imgsz"], o["nms_iou"]) == (n["images"], n["targets"], n["imgsz"], n["nms_iou"]), key
        delta = {c: n["per_class_ap50"][c] - o["per_class_ap50"][c] for c in o["per_class_ap50"]}
        report["pairs"][key] = {
            "old_run": old_name, "new_run": new_name, "old": o, "new": n,
            "overall_delta": {k: n["overall"][k] - o["overall"][k] for k in METRICS},
            "ap50_delta": delta,
            "dropped_more_than_0_5_points": {c: d for c, d in delta.items() if d < -0.005},
        }
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2))
    plot(report)
    print(json.dumps({k: {"overall_delta": v["overall_delta"], "drops": v["dropped_more_than_0_5_points"]}
                      for k, v in report["pairs"].items()}, indent=1), "missing:", report["missing"])


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key, title in (("validation_1536_nms060", "Original validation, 1536 input / NMS 0.6"),
                       ("validation_1280_nms070", "Original validation, 1280 input / NMS 0.7")):
        pair = report["pairs"].get(key)
        if not pair:
            continue
        classes = list(pair["old"]["per_class_ap50"])
        old = [100 * pair["old"]["per_class_ap50"][c] for c in classes]
        new = [100 * pair["new"]["per_class_ap50"][c] for c in classes]
        fig, ax = plt.subplots(figsize=(11, 4.8))
        x = range(len(classes))
        ax.bar([i - 0.2 for i in x], old, 0.4, label="old (f7bb7e4c, 25 Oct)", color="#9aa5b1")
        ax.bar([i + 0.2 for i in x], new, 0.4, label="new (806a7163, 29 Oct v4)", color="#2a6fdb")
        for i, (a, b) in enumerate(zip(old, new)):
            d = b - a
            ax.text(i + 0.2, b + 1, f"{d:+.1f}", ha="center", fontsize=8, color="#b42318" if d < -0.5 else "#1f2933")
        ax.set_xticks(list(x), classes, rotation=30, ha="right")
        ax.set_ylabel("AP50 (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Per-class AP50 before/after overnight training — {title}\n(number = change in points; red = dropped more than 0.5)")
        ax.legend(loc="upper right", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f"ap50_{key}.png", dpi=130)
        plt.close(fig)

    sizes = ("small", "medium", "large")
    rows = [(k, lbl) for k, lbl in (("validation_1536_nms060", "original validation"), ("val_land_1536_nms060", "land-only validation"))
            if k in report["pairs"]]
    fig, axes = plt.subplots(1, len(rows), figsize=(5.5 * len(rows), 4), squeeze=False)
    for ax, (key, label) in zip(axes[0], rows):
        pair = report["pairs"][key]
        old = [100 * (pair["old"]["animal_conf025"]["by_size"][s]["recall"] or 0) for s in sizes]
        new = [100 * (pair["new"]["animal_conf025"]["by_size"][s]["recall"] or 0) for s in sizes]
        n = [pair["old"]["animal_conf025"]["by_size"][s]["targets"] for s in sizes]
        ax.bar([i - 0.2 for i in range(3)], old, 0.4, label="old", color="#9aa5b1")
        ax.bar([i + 0.2 for i in range(3)], new, 0.4, label="new", color="#2a6fdb")
        ax.set_xticks(range(3), [f"{s}\n({c} boxes)" for s, c in zip(sizes, n)])
        ax.set_ylabel("animal recall at confidence 0.25 (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Animal recall by size — {label}\n1536 / NMS 0.6")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "animal_by_size.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
