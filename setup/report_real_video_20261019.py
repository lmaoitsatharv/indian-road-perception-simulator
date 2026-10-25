"""Make the measured report/plots from a completed, independently audited run."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    audit = json.loads((run / "independent_verification.json").read_text(encoding="utf-8"))
    review = json.loads((run / "visual_review.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed" and audit["state"] == "passed"
    figure, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    tables, temporal = [], []
    for row, clip in enumerate(status["clips"]):
        with (run / clip["folder"] / "per_frame.csv").open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        times = np.array([float(f["timestamp_s"]) for f in frames])
        latency = np.array([float(f["inference_ms"]) for f in frames])
        changes = np.array([float(f["raw_mask_change_fraction"])*100 if f["raw_mask_change_fraction"] else np.nan for f in frames])
        axes[row, 0].plot(times, latency, linewidth=.6, color="#236cb0")
        axes[row, 0].axhline(1000/30, color="#b44837", linestyle="--", linewidth=1, label="33.3 ms (inference only)")
        axes[row, 0].set(title=f'{clip["clip_id"]}: synchronized inference', ylabel="Milliseconds / frame")
        axes[row, 1].plot(times, changes, linewidth=.7, color="#21885e")
        axes[row, 1].set(title=f'{clip["clip_id"]}: raw consecutive mask changes', ylabel="Changed pixels (%)")
        for axis in axes[row]:
            axis.set_xlabel("Source clip time (seconds)")
            axis.grid(alpha=.2)
        tables.append(f'| {clip["clip_id"]} | {clip["processed_frames"]:,} | {clip["source_video"]["duration_s"]:.1f} s | {clip["inference_ms"]["median"]:.2f} ms | {clip["inference_ms"]["p95"]:.2f} ms | {clip["processing_fps_including_decode_resize_mask_write_render_encode"]:.2f} |')
        temporal.append(f'| {clip["clip_id"]} | {100*clip["raw_mask_change_fraction"]["mean"]:.2f}% | {100*clip["raw_mask_change_fraction"]["p95"]:.2f}% | {100*clip["consecutive_mask_iou"]["mean"]:.2f}% |')
    figure.suptitle("External video diagnostics: timings and unaligned mask changes\nMask changes include real camera/scene motion; no video ground-truth accuracy is available", fontsize=13)
    figure.savefig(run / "temporal_diagnostics.png", dpi=150)
    plt.close(figure)
    findings = []
    for finding in review["findings"]:
        findings.append(f'| [{finding["clip_id"]}, {finding["timestamp_s"]:.2f} s]({finding["comparison"]}) | {finding["observation"]} |')
    total_duration = sum(c["source_video"]["duration_s"] for c in status["clips"])
    total_examples = sum(len(c["examples"]) for c in status["clips"])
    source = config["source_manifest"]
    text = f'''# Real Indian video road-model report — 19 October 2026

## Completed result

The locked **morning-best** checkpoint processed **{audit["total_frames"]:,} frames**
from three original MOTOR recordings, totalling **{total_duration:.0f} seconds** at
30 fps. [Open the playable original/overlay demo](index.html).
The result is an offline perception diagnostic; it does not control the MATLAB car.

Every decoded source frame has a matching, lossless **uint8 0/1** road mask.
All three display MP4s were decoded again, and every presentation timestamp and
frame count was checked. {total_examples} review frames also have raw RGB,
float32 probabilities, masks, overlays and metadata. An independent audit
recounted every saved mask, its road fraction and consecutive-mask statistics.

No road ground-truth masks accompany these clips. **Video accuracy, precision,
recall and ground-truth IoU are unavailable.** The previously completed 689-image
test result is separate; that test was not rerun or used for changes here.

## Actual processing measurements

| Clip | Frames | Source duration | Inference median | Inference p95 | Complete processing fps |
|---|---:|---:|---:|---:|---:|
{chr(10).join(tables)}

Combined inference median: **{audit["combined_inference_ms_median"]:.2f} ms**;
p95: **{audit["combined_inference_ms_p95"]:.2f} ms**. Inference-only throughput
from the mean latency is **{audit["combined_inference_only_fps_from_mean_latency"]:.2f} fps**.
Combined complete export throughput is **{audit["combined_processing_fps"]:.2f} fps**.

Inference is the synchronized TensorFlow call and output copy. The separate
`prediction_call_ms` CSV column also includes normalization. Complete export
includes sequential decoding, resize, prediction, all mask PNG writes, drawing
and MP4 encoding/flush. It excludes model startup, source preflight, selected
example export and verification. Therefore the complete diagnostic exporter is
not a demonstrated live 30 fps system, even when network inference alone is faster.
Playback at 30 fps is the original footage's timeline, not processing speed.

Model loading/fingerprinting took {status["model_load_s"]:.2f} s and synthetic
warmup took {status["warmup_s"]:.2f} s. Total inference-run wall time including
preflight, startup, example export and output verification was
{status["total_wall_s_including_model_preflight_export_verification"]:.2f} s.
The subsequent independent recount/report time is excluded from that total.

Actual output device: `{status["output_device"]}`.
GPU: {status["runtime"]["gpu_details"]["device_name"]}; TensorFlow
{status["runtime"]["tensorflow"]}, Keras {status["runtime"]["keras"]},
PyAV {config["pyav"]}, GPU pool limit {config["gpu_memory_mib"]:,} MiB.

## Temporal diagnostics, with limits

| Clip | Mean raw changed pixels | p95 raw changed pixels | Mean adjacent-prediction mask overlap |
|---|---:|---:|---:|
{chr(10).join(temporal)}

These compare predictions at fixed camera pixels in adjacent frames, excluding
clip boundaries. They have **no motion compensation**: camera movement, vehicles,
occlusion, viewpoint changes and actual road changes contribute. The adjacent
overlap column is prediction-versus-prediction, **not ground-truth IoU or an
accuracy score**. These numbers cannot by themselves quantify pure model flicker.

![Timing and temporal diagnostic plots](temporal_diagnostics.png)

## Visual review and saved issue examples

This is qualitative inspection of selected frames, not a pixel-labelled
evaluation or a claim that every other frame is correct.

| Example | Visible observation |
|---|---|
{chr(10).join(findings)}

Selection used five fixed times per clip, plus minimum/maximum road area and
largest raw mask change. Automatic selections are review candidates, not known
errors. The visual review covered {review["distinct_frames_inspected"]} distinct
saved frames. Ground-truth definitions at kerbs, shoulders and under vehicles
still need explicit labels; no source annotations were edited.

## Exact inference and display protocol

- Checkpoint: `{status["runtime"]["checkpoint_sha256"]}`; same morning-best model.
- Original 1920x1080 RGB -> explicit direct bilinear 512x512 resize. This changes
  aspect ratio and has no calibrated camera projection. The exact centre-pixel
  transform is saved in selected `frame.json` files.
- Single raw-RGB normalization `/127.5 -1`, batch one, `training=False`, fixed
  `>=0.5` threshold. No optimizer, weight/BatchNorm updates, morphology, temporal
  smoothing, augmentation or retraining. No model was selected using these clips.
- Display-only overlay: thresholded mask resized with nearest interpolation;
  40% green blend on predicted road. H.264/yuv420p, 1280x408, CRF20, veryfast,
  two encoder CPU threads, MP4 faststart. No source has an audio stream.
- All frames' binary data remains 0/1. Only display previews use white 255.
  Full float32 probabilities are retained for selected examples, not every frame.

## Detected source fault and recovery

The two longer publisher MP4s declare 1,146 and 1,656 frames in their headers,
but decode to **1,140 and 1,650**. Their full decoded PTS sequences are continuous
at 30 fps, and decoded durations exactly match their 38/55-second stream durations.
The first 900-frame clip has consistent metadata.

The initial exporter intentionally stopped at the frame-count mismatch before
loading the model. That attempt and its unchanged-file checks remain in
`../real_video_20261019_01/`. The corrected exporter records both counts and
uses the verified decoded timeline. No invented, repeated or dropped frames
were inserted to force the incorrect header counts. The synthetic video check
was rerun after this correction and passed. Variable-rate or missing timestamps
remain a rejected input, rather than being silently retimed.

## Preservation and verification

- All three retained checkpoint files, final lock/catalog, split manifest,
  source-video bytes, source manifest and selected final-test evidence files
  have identical before/after SHA256s.
- In-memory model weights **and BatchNorm state** have identical before/after
  fingerprints. Input normalization, threshold and architecture are unchanged.
- No reserved IDD image contents were read. Only the split CSV metadata was
  checked by the existing lock validation; the clips live outside that dataset.
- No old overnight automation/controller was started. The inference process
  was bounded to 1,800 seconds, with an own-run stop marker and 3,072 MiB GPU cap.
- RGB/channel-order, explicit resize, 0/1 masks, all-frame export, MP4 PTS/
  duration roundtrip, unsupported variable timing, no overwrite and no invented
  accuracy checks passed in `../real_video_contract_20261019_02/verification.json`.
- [Independent verification](independent_verification.json), [exact configuration](config.json),
  [run status](run_status.json) and the exact source snapshot are saved here.

Automated in-app browser playback was **not** verified: its URL policy rejected
local `file:` pages. No protocol workaround was used. The MP4s passed complete
decode/timestamp checks and the gallery is saved for manual local playback.
See [UI verification limits](ui_verification.json). Play/Pause/seek are standard
HTML video controls; actual interaction with them is not claimed as tested here.

## Next development step

Define and label a small separate real-video review set, especially shadowed
road edges, kerbs and occluded obstacles. Add motion-aware temporal evaluation
and benchmark a lean live stream separately from this export-heavy diagnostic.
Before camera predictions steer a simulated car, establish camera geometry and
separate obstacle/actor observations, then test STOP/SLOW/PROCEED and braking
against held-out scenarios. The current renderer/real camera appearance gap
and the motorcycle viewpoint remain open issues; no RL policy is trained here.

## Source and rights

{source["attribution"]}.
[MOTOR dataset]({source["source_repository"]}),
[author project]({source["source_project"]}).
Pinned revision `{source["revision"]}`.
License: **{source["license"]}**, non-commercial research with attribution.
The videos in this result are derived analysis displays; original MP4 bytes
are preserved in `media/real_driving_20261017/`. Publisher behavior labels are
metadata, not pixel-level segmentation truth. Independence/overlap with all
prior training sources has not been established.
'''
    (run / "VIDEO_REPORT.md").write_text(text, encoding="utf-8")
    shutil.copy2(Path(__file__), run / "source_snapshot" / Path(__file__).name)
    print(json.dumps({"report": str(run / "VIDEO_REPORT.md"), "frames": audit["total_frames"],
                      "inference_median_ms": audit["combined_inference_ms_median"],
                      "processing_fps": audit["combined_processing_fps"]}))


if __name__ == "__main__":
    main()
