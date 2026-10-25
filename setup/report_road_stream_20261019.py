"""Independently recount stream records and make the next-step report/hub."""
import argparse
import csv
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from idd_inference import ROOT, file_sha256, write_json_atomic


def link_source_gallery():
    path = ROOT / "media/real_driving_20261017/index.html"
    page = path.read_text(encoding="utf-8")
    if 'id="model-results"' in page:
        return
    navigation = '''</p><article id="model-results"><h2>Road-model results are ready</h2><p><a href="../../runs/real_video_20261019_02/index.html">Watch original vs predicted-road overlays</a> · <a href="../../runs/real_stream_20261019_01/index.html">Streaming results and human label editor</a></p></article><article>'''
    if "</p><article>" not in page:
        raise ValueError("The source gallery layout changed; add its result links explicitly.")
    path.write_text(page.replace("</p><article>", navigation, 1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--label-pack", type=Path, required=True)
    args = parser.parse_args()
    run, pack = args.run.resolve(), args.label_pack.resolve()
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    baseline = ROOT / config["reference_run"]
    old = json.loads((baseline / "run_status.json").read_text(encoding="utf-8"))
    source_clips = {c["clip_id"]: c for c in old["clips"]}
    assert status["state"] == "completed"
    assert status["preservation"]["all_protected_files_unchanged"]
    assert status["preservation"]["in_memory_weights_and_bn_unchanged"]
    for relative, sha in status["preservation"]["protected_files_after"].items():
        assert file_sha256(ROOT / relative) == sha
    all_rows = {"lossless": [], "source_paced": []}
    verified = []
    for result in status["results"]:
        clip = source_clips[result["clip_id"]]
        n, rate = clip["processed_frames"], Fraction(clip["source_video"]["fps_fraction"])
        with (run / result["rows_file"]).open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == result["processed_frames"]
        processed = [int(r["frame_index"]) for r in rows]
        dropped = result["dropped_queue_frame_indices"] + result["dropped_stale_frame_indices"]
        assert sorted(processed+dropped) == list(range(n))
        assert result["maximum_observed_queue_size"] <= 2 and result["decoder_thread_stopped"]
        for row in rows:
            index = int(row["frame_index"])
            assert abs(float(row["timestamp_s"]) - float(Fraction(index) / rate)) < 1e-9
            assert Fraction(int(row["source_pts"])) * Fraction(row["source_time_base"]) == Fraction(index) / rate
            with Image.open(baseline / clip["folder"] / "masks_01" / f"frame_{index:06d}.png") as image:
                mask = np.array(image)
            assert mask.dtype == np.uint8 and mask.shape == (512, 512) and np.isin(mask, [0, 1]).all()
            assert hashlib.sha256(mask.tobytes()).hexdigest() == row["mask_sha256"]
            assert float(row["road_fraction"]) == float(mask.mean())
            assert np.isfinite(float(row["prediction_call_ms"])) and float(row["prediction_call_ms"]) >= 0
            assert row["stale_at_completion"] in ("True", "False")
        assert sum(r["stale_at_completion"] == "True" for r in rows) == result["results_stale_at_completion"]
        all_rows[result["mode"]].extend(rows)
        verified.append({"clip": result["clip_id"], "mode": result["mode"], "mask_hashes_verified": len(rows),
                         "csv_sha256": file_sha256(run / result["rows_file"]), "frame_accounting_complete": True})
    modes = {}
    for mode, rows in all_rows.items():
        results = [r for r in status["results"] if r["mode"] == mode]
        modes[mode] = {"processed_frames": len(rows), "combined_processing_fps": len(rows)/sum(r["wall_s"] for r in results),
                       "dropped_frames": sum(len(r["dropped_queue_frame_indices"])+len(r["dropped_stale_frame_indices"]) for r in results),
                       "inference_median_ms": float(np.median([float(r["inference_ms"]) for r in rows])),
                       "inference_p95_ms": float(np.percentile([float(r["inference_ms"]) for r in rows], 95)),
                       "stale_results": sum(r["stale_at_completion"] == "True" for r in rows)}
        if mode == "source_paced":
            ages = [float(r["source_scheduled_to_result_ms"]) for r in rows]
            modes[mode].update({"scheduled_release_to_result_median_ms": float(np.median(ages)),
                                "scheduled_release_to_result_p95_ms": float(np.percentile(ages, 95)),
                                "scheduled_release_to_result_max_ms": float(max(ages))})
    labels = json.loads((pack / "label_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert labels["accuracy"] is None and labels["state"] == "awaiting_human_labels"
    assert len(manifest["frames"]) == 18
    assert sum(f["group"] == "time_selected_scoring" for f in manifest["frames"]) == 12
    for frame in manifest["frames"]:
        assert file_sha256(pack / frame["image"]) == frame["image_sha256"]
        assert file_sha256(pack / frame["prediction"]) == frame["prediction_sha256"]
        with Image.open(pack / frame["image"]) as image:
            assert image.size == (512, 512) and image.mode == "RGB"
    evidence = {"state": "passed", "modes": modes, "records": verified,
                "compared_predictions": sum(len(v) for v in all_rows.values()),
                "all_predictions_bitwise_identical_to_reference": True,
                "all_frame_indices_and_source_timestamps_verified": True,
                "label_pack_frames": 18, "real_human_labels_created": False, "real_video_accuracy": None,
                "actual_label_editor_browser_interactions_verified": False,
                "browser_limit": "Local file URLs were rejected by the browser policy during the prior step. No alternate protocol/surface was used to bypass that restriction. Syntax and mask serialization were checked separately.",
                "checkpoint_and_reference_preservation_verified": True, "model_loaded_by_audit": False,
                "reserved_dataset_images_read": False}
    write_json_atomic(run / "independent_verification.json", evidence)
    for path in [Path(__file__), ROOT / "road_video_labels.py", ROOT / "setup/road_label_editor_template.html",
                 ROOT / "setup/verify_road_stream_20261019.py", ROOT / "setup/verify_road_labels_20261019.py"]:
        shutil.copy2(path, run / "source_snapshot" / path.name)
    (pack / "source_snapshot").mkdir(exist_ok=True)
    for path in [ROOT / "road_video_labels.py", ROOT / "setup/road_label_editor_template.html"]:
        shutil.copy2(path, pack / "source_snapshot" / path.name)
    # Scientific artifact: historical export vs current lossless/paced capacity.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(3)
    old_rates = [c["processing_fps_including_decode_resize_mask_write_render_encode"] for c in old["clips"]]
    fast_rates = [r["processed_fps"] for r in status["results"] if r["mode"] == "lossless"]
    axes[0].bar(x-.17, old_rates, .34, label="Earlier diagnostic export (includes saving)", color="#9b7189")
    axes[0].bar(x+.17, fast_rates, .34, label="Current lean file stream (no display/encoding)", color="#268972")
    axes[0].axhline(30, color="#3e4c5e", linestyle="--", linewidth=1)
    axes[0].set(xticks=x, xticklabels=[c["clip_id"] for c in old["clips"]], ylabel="Measured processing fps", title="Same predictions, different pipeline work")
    axes[0].legend(fontsize=8, loc="upper left")
    for result in status["results"]:
        if result["mode"] != "source_paced":
            continue
        with (run / result["rows_file"]).open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        ages = np.sort([float(r["source_scheduled_to_result_ms"]) for r in rows])
        axes[1].plot(ages, np.linspace(0, 1, len(ages)), label=result["clip_id"])
    axes[1].axvline(33.3333, color="#777", linestyle="--", linewidth=1, label="33.3 ms source interval")
    axes[1].set(xlabel="Scheduled file-frame release to result (ms)", ylabel="Fraction of results", title="30 fps paced replay latency (no physical camera)")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Locked model unchanged. Recorded-file benchmark; display, encoding and sensor capture excluded.", fontsize=11)
    figure.savefig(run / "stream_measurements.png", dpi=160)
    plt.close(figure)
    fast, paced = modes["lossless"], modes["source_paced"]
    rows = []
    for clip in old["clips"]:
        a = next(r for r in status["results"] if r["mode"] == "lossless" and r["clip_id"] == clip["clip_id"])
        b = next(r for r in status["results"] if r["mode"] == "source_paced" and r["clip_id"] == clip["clip_id"])
        rows.append(f'| {clip["clip_id"]} | {clip["processed_frames"]:,} | {a["processed_fps"]:.2f} | {b["processed_fps"]:.2f} | {b["source_scheduled_to_result_ms"]["p95"]:.2f} ms | {len(b["dropped_queue_frame_indices"])+len(b["dropped_stale_frame_indices"])} |')
    report = f'''# Lean road-video pipeline and human-label preparation — 19 October 2026

## What is complete

- Reused the unchanged locked morning-best model in a two-frame bounded stream.
- Processed all 3,690 frames once in lossless capacity mode and once at their
  original 30 fps release schedule. All **7,380 masks** exactly match the earlier
  saved binary predictions. This is output equivalence, not accuracy.
- Lossless capacity: **{fast["combined_processing_fps"]:.2f} fps** across the three clips.
- Paced replay: **{paced["combined_processing_fps"]:.2f} processed fps**, **{paced["dropped_frames"]} dropped frames**,
  **{paced["stale_results"]} results beyond the predeclared 100 ms stale limit**.
- Paced scheduled-release-to-result median **{paced["scheduled_release_to_result_median_ms"]:.2f} ms**,
  p95 **{paced["scheduled_release_to_result_p95_ms"]:.2f} ms**, max **{paced["scheduled_release_to_result_max_ms"]:.2f} ms**.
- [Prepared the blank human-labelling page](../{pack.name}/index.html):
  **12 fixed time-selected scoring frames + 6 separately reported diagnostics**.
  No human reference masks or real-video accuracy score have been fabricated.

## Measured comparison

| Clip | Frames in each mode | Lean lossless fps | Source-paced processed fps | Paced release-to-result p95 | Drops |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The earlier diagnostic export averaged 16.23 fps and includes per-frame PNG
writes, drawing and H.264 encoding. The lean stream excludes those operations
and overlaps CPU decoding/resize with GPU prediction. It still computes a mask
hash and lightweight in-memory metrics for every result. CSV export and reference
mask reads happen after timing. Its model normalization, threshold, weights,
precision policy and outputs have not changed.

The old exporter was measured in an earlier run, not in a randomized matched
timing experiment. Treat the comparison as throughput for different workloads,
not an isolated causal speedup or evidence of improved segmentation accuracy.

![Processing and latency measurements](stream_measurements.png)

## Meaning of the real-time replay

The file decoder released frames at 30 fps from local storage. These are not
physical camera exposure/driver timestamps. Latency includes file decoding,
RGB conversion, direct 512x512 resize, queueing, single normalization and
synchronized prediction/result copy. It excludes camera transport, display,
video encoding, calibration, actor perception, planning and control.

The measured ~29.9 fps includes end-of-stream cleanup. All source frames were
processed. Some individual latencies exceed the 33.3 ms source interval; the
measurements do not promise a hard 33.3 ms deadline. The maximum observed delay
was {paced["scheduled_release_to_result_max_ms"]:.2f} ms during these 123 seconds.
This supports the next live-camera experiment; it is not a complete real-time
driving system or a general performance guarantee under all machine loads.

## Overload and stopping behavior

The queue capacity is two frames. Lossless mode blocks the producer instead of
dropping. Paced mode removes the oldest waiting frame when full. Inputs already
more than 100 ms old are rejected before inference; late completions are counted
and marked, not hidden. Every decoded frame is accounted for as processed,
queue-dropped or stale-dropped. A synthetic slow consumer verified overload,
latest-frame retention and stale handling; real clips had no drops in this run.

Own-run STOP.json, producer failures and consumer failures propagate and clean
up the owned decoder thread. The process has a 1,200-second wall-time guard and
3,072 MiB TensorFlow GPU pool cap. No old controller or automation was restarted.

## Labels and accuracy: what still requires a person

The editor starts every pixel as unlabelled/ignored (255); it does not prefill
the model mask. Road/non-road labels are 1/0 in this new external review pack;
the existing dataset and prediction masks remain unchanged 0/1 files.

Open [the label page](../{pack.name}/index.html), draw the visible road, remove
vehicle/person bodies and mark uncertain edges ignored. Review each full frame,
then export the JSON. The scoring tool checks pack/image/prediction hashes,
label values, review flags and at least 80% coverage. It publishes a primary
aggregate only after all 12 scoring frames have reviewed labels. The 6 known
diagnostic frames are reported separately and do not inflate the primary score.

These frames come from already inspected development clips. Their future
human-reviewed score is a labelled-frame development result, **not whole-video
accuracy or a new independent test score**. Camera viewpoint and overlap with
prior source data remain limitations. A human-reviewed mask may also need a
second annotator to resolve kerb/shoulder definitions.

The browser editor's JavaScript syntax and JS/Python mask serialization passed
checks. Its actual drawing/click controls were not automatically browser-tested:
the browser tool rejects local file URLs, and no alternate surface/protocol was
used to bypass that restriction. The page is saved for manual local use.

## Verification and preservation

[Independent audit](independent_verification.json) rechecked every frame index,
source timestamp, saved-mask hash and road fraction in both modes. The fixed
label pack's 18 RGB images and copied prediction hashes were checked too.

All three checkpoint files, in-memory weights/BatchNorm, source clips, lock,
split manifest, selected final-test evidence and previous diagnostic status/audit
remained unchanged. No training or IDD image-content reads occurred. Model load
still validates split CSV metadata. The completed 689-image test result was not
rerun or used to tune this pipeline.

Synthetic stream checks: `../real_stream_contract_20261019_01/verification.json`.
Synthetic label checks: `../real_label_contract_20261019_01/verification.json`.
The synthetic scores in the latter folder are test fixtures, not real-video
human accuracy. Real-label status remains `awaiting_human_labels` with null accuracy.

Exact runtime/configuration, CSV records, model/file preservation hashes and
source snapshots are retained in this run. [Run status](run_status.json) ·
[Configuration](config.json) · [Label instructions](../{pack.name}/README.md).

## Next bounded development step

Obtain reviewed labels and inspect the scorer's road-edge/vehicle errors. Then
test a real camera/video acquisition path with display enabled and measured
capture-to-result age. Camera calibration and separate obstacle/actor
observations are still required before camera masks guide the simulated car.
STOP/SLOW/PROCEED, independent braking and eventual RL remain separate stages.

Source footage: MOTOR, Varun A. Paturkar, Shankar Gangisetty and C. V. Jawahar,
ICRA 2026; CC BY-NC 4.0, non-commercial research with attribution.
The bounded producer/consumer implementation uses Python's
[standard synchronized queue](https://docs.python.org/3.12/library/queue.html)
and [thread events](https://docs.python.org/3.12/library/threading.html).
'''
    (run / "STREAM_REPORT.md").write_text(report, encoding="utf-8")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Road model · next development step</title>
<style>body{{margin:0;background:#101720;color:#e8eff7;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:30px}}h1{{font-size:38px;line-height:1.2}}a{{color:#89ddbd}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}section,.card{{background:#1c2838;border:1px solid #3a4b62;border-radius:12px;padding:22px;margin:20px 0}}.big{{font-size:36px;color:#94e6bf}}img{{width:100%;height:auto}}.button{{display:inline-block;padding:12px 18px;background:#284e45;color:#fff;border-radius:7px;text-decoration:none;margin:5px}}small{{color:#b8c8db}}</style>
<main><small>TESLA PROJECT / STREAMING & HUMAN REVIEW / 19 OCTOBER 2026</small><h1>Faster video predictions. Same road model.</h1>
<div class="cards"><div class="card"><div class="big">{fast["combined_processing_fps"]:.1f} fps</div>Lean file-stream capacity<br><small>No display or video recording</small></div><div class="card"><div class="big">0 drops</div>All 3,690 frames at 30 fps input<br><small>Recorded-file paced replay</small></div><div class="card"><div class="big">7,380 matches</div>Every mask matches the earlier result<br><small>Two modes, unchanged predictions</small></div></div>
<section><h2>Next: give accuracy a human reference</h2><p>We still cannot calculate these videos' accuracy without labels. A blank offline editor is ready with 12 time-selected scoring frames and 6 separate diagnostic examples. It shows original frames and starts with no model-generated labels.</p><a class="button" href="../{pack.name}/index.html">Open the human label editor</a><a class="button" href="../{pack.name}/README.md">How to label and score</a><p><small>Future scores apply to these reviewed development frames, not the entire videos or an independent road-safety test.</small></p></section>
<section><h2>See the existing predictions</h2><a class="button" href="../real_video_20261019_02/index.html">Play original vs road-overlay videos</a><p><a href="STREAM_REPORT.md">Full report and limits</a> · <a href="independent_verification.json">Verification</a> · <a href="config.json">Exact setup</a></p></section>
<section><h2>Measured pipeline work</h2><img src="stream_measurements.png" alt="Measured throughput and paced-file result latency"><p>The old diagnostic exporter saved every mask and encoded video. The new benchmark measures a lean decoding/prediction path. Physical camera capture, display, planning and car control remain untested by this benchmark.</p></section>
<p><small>All model files and the previous completed test result remain unchanged. Source: MOTOR, Paturkar, Gangisetty and Jawahar, ICRA 2026. CC BY-NC 4.0, non-commercial research with attribution.</small></p></main></html>'''
    (run / "index.html").write_text(page, encoding="utf-8")
    link_source_gallery()
    print(json.dumps({"audit": "passed", "lossless_fps": fast["combined_processing_fps"],
                      "paced_fps": paced["combined_processing_fps"], "predictions_matched": evidence["compared_predictions"],
                      "paced_latency_p95_ms": paced["scheduled_release_to_result_p95_ms"],
                      "human_labels": "pending", "real_video_accuracy": None}), flush=True)


if __name__ == "__main__":
    main()
