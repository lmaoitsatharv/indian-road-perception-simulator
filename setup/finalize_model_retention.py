"""Publish the measured retention decision and all fresh simulator comparisons."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/model_retention_20261021_01"
TRIAL = ROOT / "runs/road_rehearsal_20261021_01"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local(value):
    if len(value) > 2 and value[1] == ":" and ROOT.as_posix().startswith("/mnt/"):
        return Path("/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/"))
    return Path(value)


def percent(value):
    return f"{100 * value:.4f}%"


def main():
    before = load(OUT / "protection.json")
    after = {name: sha(ROOT / name) for name in before["protected_files"]}
    assert after == before["protected_files"] and load(OUT / "scope_verification.json")["passed"]
    state = load(TRIAL / "run_status.json")
    assert state["phase"] == "completed" and state["protected_files_unchanged"]
    assert load(TRIAL / "independent_recount.json")["passed"]
    comparison, chosen = load(TRIAL / "comparison.json"), load(OUT / "simulation_selection.json")
    reports = {key: load(OUT / ("fresh_" + key) / "report.json") for key in ("original", "previous03", "selected")}
    frozen = max(datetime.fromisoformat(chosen["frozen_utc"]),
                 datetime.fromisoformat(comparison["selection"]["frozen_utc"]))
    expected_models = {"original": "93a8faf3d43b77f0ce74d5cdbaf94124728d371c3b68b2a3386f9f72fc106725",
                       "previous03": "2eb4c06641bcb8226176500f9a4f2c1ef66a4472db3006787ecaeb57bf59165e",
                       "selected": chosen["chosen"]["checkpoint_sha256"]}
    for key, report in reports.items():
        assert report["status"] == "completed" and report["protected_files_unchanged"] and report["model_weights_unchanged"]
        assert datetime.fromisoformat(report["created_utc"]) > frozen
        assert report["runtime"]["checkpoint_sha256"] == expected_models[key]
        assert report["runtime"]["device"] == "/GPU:0"
        assert load(OUT / ("fresh_" + key) / "independent_verification.json")["passed"]
    baseline = comparison["baseline"]
    results = {"completed_utc": datetime.now(timezone.utc).isoformat(), "real_inference_unchanged": True,
               "protected_files": after, "reserved_test_images_read": False,
               "new_rehearsal": {"baseline": {"real_validation": baseline["real_validation"]["metrics"],
                                             "development": baseline["development"]["metrics"],
                                             "real_video_guards": {k: v["metrics"] for k, v in baseline["real_video_guards"].items()}},
                                 "epochs": [{"epoch": row["epoch"], "strict_gates": row["strict_gates"],
                                             "real_validation": row["evaluations"]["real_validation"]["metrics"],
                                             "development": row["evaluations"]["development"]["metrics"],
                                             "real_video_guards": {k: v["metrics"] for k, v in row["evaluations"]["real_video_guards"].items()}}
                                            for row in comparison["epochs"]], "shared_selection": comparison["selection"]},
               "simulation_only_selection": chosen,
               "fresh_synthetic": {key: {"metrics": report["metrics"], "frames": sum(e["frames"] for e in report["episodes"]),
                                         "checkpoint_sha256": report["runtime"]["checkpoint_sha256"]} for key, report in reports.items()},
               "real_path": "Unchanged LockedRoadSegmenter for real images, videos and unknown sources",
               "simulation_path": "Opt-in recorded CARLA evaluation only; no camera appearance classifier and no actor control",
               "limits": ["New seed in the same synthetic town, not an independent real-road test",
                          "Reference uses depth and OpenDRIVE lane geometry with ignored uncertain edges",
                          "The18human-labelled frames are repeated development guards, not whole-video accuracy",
                          "Real outputs are preserved by leaving that deployed path unchanged; a new model is not guaranteed on unseen roads",
                          "No segmentation model or RL policy controls the actors"]}
    results["fresh_synthetic"]["delta_selected_vs_original_pp"] = 100 * (reports["selected"]["metrics"]["road_iou"] - reports["original"]["metrics"]["road_iou"])
    results["fresh_synthetic"]["delta_selected_vs_previous03_pp"] = 100 * (reports["selected"]["metrics"]["road_iou"] - reports["previous03"]["metrics"]["road_iou"])
    results["simulation_confirmation_improved_vs_original"] = results["fresh_synthetic"]["delta_selected_vs_original_pp"] > 0
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = ["# Simulation adaptation with protected real inference", "",
             "## Decision", "",
             "Real images, videos and unknown sources continue to use the unchanged locked morning-best model. "
             "Its checkpoint, inference implementation, split and original three-model catalog are byte-identical to this task's protection snapshot.", "",
             "The new rehearsal trial uses 1,024 real training images, 110 synthetic training images and teacher-logit preservation. "
             "All 687 real validation images and the separate 12 scoring / 6 diagnostic human-reviewed frames are evaluation-only. "
             "Road IoU and 5px boundary IoU must each be no lower than the matched baseline on every guard; exact integer ratios determine the decision.", "",
             "| New trial | Real VAL road IoU | Boundary IoU5 | Synthetic DEV road IoU | All real guards pass |",
             "|---|---:|---:|---:|---|"]
    b = baseline["real_validation"]["metrics"]
    lines.append(f"| Locked morning-best baseline | {percent(b['road_iou'])} | {percent(b['boundary_iou_5px'])} | {percent(baseline['development']['metrics']['road_iou'])} | Reference |")
    for row in comparison["epochs"]:
        r, d = row["evaluations"]["real_validation"]["metrics"], row["evaluations"]["development"]["metrics"]
        lines.append(f"| Rehearsal epoch {row['epoch']} | {percent(r['road_iou'])} | {percent(r['boundary_iou_5px'])} | {percent(d['road_iou'])} | {row['real_retention_passed']} |")
    lines += ["", f"Shared-model selection: **{comparison['selection']['selected_epoch']}** (None means rejected). Real deployment remains unchanged.", "",
              "## Separate simulator choice", "",
              f"Chosen by the highest score on the same 60 development frames: `{chosen['chosen']['checkpoint']}`. "
              f"Development road IoU **{percent(chosen['chosen']['dev_road_iou'])}**, baseline **{percent(chosen['baseline_development_iou'])}**.", "",
              "Historical trial 02 was rejected for real-image retention. It is eligible only for the explicitly separate simulator path. "
              "Reusing an older checkpoint is a routing/selection improvement, not a newly trained accuracy gain. "
              "The frozen candidate pool, checkpoint hashes and selection time are in simulation_pool_plan.json and simulation_selection.json.", "",
              "## Fresh confirmation after selection", "",
              "Clear mixed-traffic seed 137 was recorded at 1Hz with 110 paired frames and no drops. "
              "Its pixels were scored only after both model selections were frozen. No model was reselected using these scores.", "",
              "| Model | Fresh synthetic road IoU | Dice | Evaluated pixels |", "|---|---:|---:|---:|"]
    for key, label in (("original", "Locked morning best"), ("previous03", "Previous experimental 03"), ("selected", "Frozen simulator-only choice")):
        m = reports[key]["metrics"]
        lines.append(f"| {label} | {percent(m['road_iou'])} | {percent(m['dice'])} | {percent(m['evaluated_fraction'])} |")
    lines += ["", "## Evidence", "",
              "- [All 110 fresh five-column comparisons](index.html)",
              "- [Full new rehearsal report](../road_rehearsal_20261021_01/MODEL_REPORT.md)",
              "- [Independent rehearsal confusion/boundary recount](../road_rehearsal_20261021_01/independent_recount.json)",
              "- [Original fresh gallery](fresh_original/index.html)",
              "- [Previous 03 fresh gallery](fresh_previous03/index.html)",
              "- [Selected fresh gallery](fresh_selected/index.html)",
              "- [Real deployment protection checks](scope_verification.json)", "",
              "## Limits", "", *["- " + limit for limit in results["limits"]], "",
              "No reserved 689-image real test image was opened or evaluated. The completed locked-model test result remains historical and does not transfer to simulator weights."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pictures = OUT / "comparisons"; pictures.mkdir(exist_ok=False)
    episode = reports["original"]["episodes"][0]
    source = local(episode["source_capture"])
    rows = {key: {row["frame"]: row for row in report["episodes"][0]["rows"]} for key, report in reports.items()}
    entries = []
    for row in episode["rows"]:
        frame = row["frame"]
        rgb = np.array(Image.open(source / "road_camera" / row["source"]["rgb_file"]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR))
        target = np.array(Image.open(OUT / "fresh_original" / row["reference"]))
        reference = np.zeros((512, 512, 3), np.uint8); reference[target == 1] = 255; reference[target == 255] = (130, 65, 175)
        cells = [("Camera", rgb), ("Reference: purple ignored", reference)]
        for key, title in (("original", "Original"), ("previous03", "Previous03"), ("selected", "Simulator choice")):
            record = rows[key][frame]
            assert record["source"]["rgb_sha256"] == row["source"]["rgb_sha256"]
            other_reference = np.array(Image.open(OUT / ("fresh_" + key) / record["reference"]))
            np.testing.assert_array_equal(target, other_reference)
            mask = np.array(Image.open(OUT / ("fresh_" + key) / record["mask"]))
            overlay = rgb.copy(); road = mask == 1
            overlay[road] = (.55 * overlay[road] + .45 * np.array([25, 215, 140])).astype(np.uint8)
            cells.append((f"{title}: IoU{100 * record['metrics']['road_iou']:.2f}%", overlay))
        canvas = Image.new("RGB", (1600, 213), "#121d2b"); draw = ImageDraw.Draw(canvas)
        for index, (title, pixels) in enumerate(cells):
            draw.text((index * 320 + 5, 8), title, fill="white")
            canvas.paste(Image.fromarray(pixels).resize((320, 180), Image.Resampling.NEAREST if index == 1 else Image.Resampling.BILINEAR), (index * 320, 30))
        relative = f"comparisons/frame_{frame:07d}.jpg"
        canvas.save(OUT / relative, quality=88)
        entries.append({"frame": frame, "sim_s": row["sim_s"], "comparison": relative,
                        "scores": {key: rows[key][frame]["metrics"]["road_iou"] for key in rows}})
    (OUT / "gallery_manifest.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    cards = "\n".join(f'<section><h2>{entry["sim_s"]:.2f}s / frame{entry["frame"]}</h2><a href="{entry["comparison"]}"><img loading="lazy" src="{entry["comparison"]}" alt="Camera, fixed reference and three model predictions"></a></section>' for entry in entries)
    table = "".join(f'<tr><td>{html.escape(key)}</td><td>{percent(report["metrics"]["road_iou"])}</td></tr>' for key, report in reports.items())
    page = f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Road-model retention comparison</title><style>body{{background:#101722;color:#e6edf5;font:16px system-ui;margin:24px auto;max-width:1600px;padding:0 15px}}a{{color:#73d6f5}}section{{margin:20px 0;background:#172333;padding:12px;border-radius:10px}}img{{width:100%;height:auto}}h2{{font-size:17px}}td{{padding:8px 24px 8px 0}}p{{max-width:1000px;line-height:1.6}}</style><h1>Simulation improvement, protected real predictions</h1><p>Real photos and videos keep the original locked model. The separate simulator choice was frozen using development scores before this110-frame check. Green shows predicted road; purple reference pixels are ignored. These are synthetic geometry-assisted scores, not real-road accuracy.</p><table><tr><th>Fresh simulator comparison</th><th>Road IoU</th></tr>{table}</table><p><a href="REPORT.md">Full report and training decisions</a> · <a href="results.json">Exact results</a> · <a href="../road_rehearsal_20261021_01/index.html">Real-image guards and training comparisons</a></p>{cards}</html>'
    (OUT / "index.html").write_text(page, encoding="utf-8")
    assert len(entries) == 110 and all((OUT / entry["comparison"]).is_file() for entry in entries)
    (OUT / "gallery_check.json").write_text(json.dumps({"passed": True, "comparisons": len(entries), "all_references_identical": True,
                                                        "method": "Static files and image shapes checked; browser interaction not automated"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"real_inference_unchanged": True, "fresh_road_iou": {k: v["metrics"]["road_iou"] for k, v in reports.items()},
                      "gallery": str(OUT / "index.html")}))


if __name__ == "__main__":
    main()
