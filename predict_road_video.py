"""Offline real-video diagnostics for the unchanged, locked road segmenter.

PyAV preserves decoded presentation timestamps. Every frame is predicted once;
binary data stays 0/1, while the MP4 is only a display. No accuracy is computed
because these external clips have no road masks. No planning/control is run.
"""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from idd_inference import (ROOT, LockedRoadSegmenter, RoadPrediction, file_sha256,
                           load_final_lock, write_json_atomic, write_prediction)

PANEL = (640, 360)
VIDEO_SIZE = (1280, 408)
FONT = ImageFont.load_default(size=16)
SMALL_FONT = ImageFont.load_default(size=13)
FIELDS = ["frame_index", "source_pts", "source_time_base", "timestamp_s",
          "inference_ms", "prediction_call_ms", "road_fraction",
          "raw_mask_change_fraction", "consecutive_mask_iou"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def inspect_video(path):
    """Decode metadata to count frames and verify the CFR contract explicitly."""
    stamps = []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Select a source with exactly one video stream.")
        stream = container.streams.video[0]
        rate = stream.average_rate
        if not rate or rate <= 0 or rate > 120:
            raise ValueError("A finite source frame rate in (0,120] is required.")
        if int(stream.metadata.get("rotate", "0")) % 360:
            raise ValueError("Rotated video requires a separately declared transform.")
        width, height = stream.width, stream.height
        if not width or not height or width * height > 3840 * 2160:
            raise ValueError("Only video up to 3840x2160 is supported by this bounded demo.")
        advertised_frames = stream.frames
        container_duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                raise ValueError("Missing timestamp or corrupt frame; refusing an ambiguous replay.")
            if (frame.width, frame.height) != (width, height):
                raise ValueError("Resolution changed inside a source video.")
            rotation = getattr(frame, "rotation", 0)
            if rotation and int(rotation) % 360:
                raise ValueError("Rotated frame requires a separately declared transform.")
            stamp = Fraction(frame.pts) * frame.time_base
            if stamps and stamp - stamps[-1] != 1 / rate:
                raise ValueError("This exporter requires CFR input; variable/missing frames are not silently retimed.")
            stamps.append(stamp)
            if len(stamps) > 18000 or stamp - stamps[0] > 180:
                raise ValueError("Source exceeds the bounded three-minute demo limit.")
        if not stamps:
            raise ValueError("Source has no decodable video frames.")
        duration = float(stamps[-1] - stamps[0] + 1 / rate)
        if container_duration is not None and abs(container_duration - duration) > float(1 / rate) + 1e-6:
            raise ValueError("Decoded duration does not match the source timeline; possible missing frames.")
        # Two publisher MP4s overstate nb_frames by six, but have continuous
        # timestamps and exactly matching durations. Trust the verified decode,
        # retain the discrepancy, and never insert invented frames to match it.
        return {"width": width, "height": height, "frames": len(stamps),
                "fps": float(rate), "fps_fraction": str(rate),
                "time_base": str(stream.time_base), "first_pts_time_s": float(stamps[0]),
                "first_pts_time_fraction": str(stamps[0]),
                "last_pts_time_s": float(stamps[-1]),
                "duration_s": duration, "container_duration_s": container_duration,
                "container_declared_frames": advertised_frames,
                "declared_frame_count_matches_decoded": not advertised_frames or advertised_frames == len(stamps),
                "audio_streams": len(container.streams.audio),
                "source_codec": stream.codec_context.name,
                "constant_frame_rate_verified": True}


def resize_input(rgb, *, allow_resize):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Decoder must supply raw uint8 RGB, not BGR/normalized input.")
    if rgb.shape[:2] != (512, 512) and not allow_resize:
        raise ValueError("Pass --resize to opt into direct bilinear 512x512 resizing.")
    return np.array(Image.fromarray(rgb).resize((512, 512), Image.Resampling.BILINEAR))


def mask_change(previous, current):
    if previous is None:
        return None, None
    union = np.count_nonzero(previous | current)
    intersection = np.count_nonzero(previous & current)
    return float(np.mean(previous != current)), float(intersection / union) if union else 1.0


def display_pair(rgb_panel, mask):
    """Display-only enlargement of an already thresholded mask; no filtering."""
    road = np.array(Image.fromarray(mask).resize(PANEL, Image.Resampling.NEAREST)) == 1
    overlay = rgb_panel.copy()
    overlay[road] = (.6 * rgb_panel[road] + .4 * np.array([30, 220, 110])).astype(np.uint8)
    return overlay


def make_display(rgb_panel, mask, timestamp_s, frame_index):
    image = Image.new("RGB", VIDEO_SIZE, (18, 25, 35))
    image.paste(Image.fromarray(rgb_panel), (0, 28))
    image.paste(Image.fromarray(display_pair(rgb_panel, mask)), (640, 28))
    draw = ImageDraw.Draw(image)
    draw.text((12, 5), f"Original | {timestamp_s:.2f}s | frame {frame_index}", fill="white", font=FONT)
    draw.text((652, 5), "Locked morning-best | green = predicted road", fill="white", font=FONT)
    draw.text((12, 390), "Offline playback at source speed. Green is a prediction, not a safe-driving area. No labelled video accuracy.", fill=(208, 215, 225), font=SMALL_FONT)
    return np.array(image)


class DisplayWriter:
    def __init__(self, path, rate):
        self.container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        try:
            self.stream = self.container.add_stream("libx264", rate=rate)
            self.stream.width, self.stream.height = VIDEO_SIZE
            self.stream.pix_fmt = "yuv420p"
            self.stream.codec_context.time_base = 1 / rate
            self.stream.codec_context.thread_count = 2
            self.stream.options = {"crf": "20", "preset": "veryfast"}
        except BaseException:
            self.container.close()
            raise

    def append(self, rgb, index):
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = index
        frame.time_base = self.stream.codec_context.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self, *, finish=True):
        try:
            if finish:
                for packet in self.stream.encode():
                    self.container.mux(packet)
        finally:
            self.container.close()


def weight_fingerprint(model):
    digest = hashlib.sha256()
    for weight in model.weights:
        array = weight.numpy()
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    a = np.array(values, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def protected_files(manifest_path):
    retained = json.loads((ROOT / "models/retained_models.json").read_text(encoding="utf-8"))
    paths = [ROOT / m["checkpoint"] for m in retained["models"]]
    paths += [ROOT / p for p in ["models/segmentation_final_lock.json", "models/retained_models.json",
              "dataset/splits/pair_splits.csv", "runs/test_evaluation_20261003_morning_best/test_metrics.json",
              "runs/test_evaluation_20261003_morning_best/TEST_REPORT.md",
              "runs/test_evaluation_20261003_morning_best/EVALUATION_LOCK.json"]]
    paths.append(manifest_path)
    return paths


def capture_candidate(index, timestamp, model_rgb, panel, prediction):
    # Only eight selected frames are retained in RAM; full videos are streamed.
    return {"index": index, "timestamp": timestamp, "rgb": model_rgb.copy(),
            "panel": panel.copy(), "prediction": RoadPrediction(prediction.probability.copy(),
            prediction.mask.copy(), prediction.inference_ms, prediction.output_device)}


def export_examples(folder, cached, meta, source, lock):
    selected = {}
    for reason, value in cached.items():
        item = selected.setdefault(value["index"], {"value": value, "reasons": []})
        item["reasons"].append(reason)
    results = []
    sx, sy = 512 / meta["width"], 512 / meta["height"]
    for index, item in sorted(selected.items()):
        value = item["value"]
        stem = f"frame_{index:06d}"
        source_meta = {"input_file": source["file"], "input_sha256": source["sha256"],
                       "clip_id": source["clip_id"], "source_frame_index": index,
                       "timestamp_source": "decoded PTS relative to first video frame",
                       "oriented_source_size_wh": [meta["width"], meta["height"]],
                       "resize_policy": "explicit direct bilinear resize to 512x512; aspect ratio changes",
                       "source_to_model_pixel_centers": [[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]],
                       "calibration_note": "No camera intrinsics or metric projection are supplied."}
        example = write_prediction(folder / "examples", stem, value["rgb"], value["prediction"],
                                   source_meta, timestamp_s=value["timestamp"], model_lock=lock)
        binary_display = Image.fromarray(value["prediction"].mask * 255).resize(PANEL, Image.Resampling.NEAREST)
        preview = Image.new("RGB", (1920, 408), (18, 25, 35))
        preview.paste(Image.fromarray(value["panel"]), (0, 28))
        preview.paste(binary_display.convert("RGB"), (640, 28))
        preview.paste(Image.fromarray(display_pair(value["panel"], value["prediction"].mask)), (1280, 28))
        d = ImageDraw.Draw(preview)
        d.text((12, 5), f"Original | {value['timestamp']:.2f}s | frame {index}", fill="white", font=FONT)
        d.text((652, 5), "Predicted mask | white is display-only 255", fill="white", font=FONT)
        d.text((1292, 5), "Locked morning-best | predicted road overlay", fill="white", font=FONT)
        d.text((12, 390), "Review example, not labelled accuracy. Camera-mask predictions do not identify every obstacle.", fill="white", font=SMALL_FONT)
        preview.save(example / "comparison.jpg", quality=93)
        results.append({"frame_index": index, "timestamp_s": value["timestamp"], "selection_reasons": item["reasons"],
                        "comparison": f"examples/{stem}/comparison.jpg", "frame_metadata": f"examples/{stem}/frame.json"})
    return results


def process_clip(path, source, meta, folder, segmenter, args, deadline, progress):
    folder.mkdir()
    (folder / "masks_01").mkdir()
    (folder / "examples").mkdir()
    rate = Fraction(meta["fps_fraction"])
    origin = Fraction(meta["first_pts_time_fraction"])
    periodic = {int(round((meta["frames"] - 1) * q)): f"fixed_{q:.2f}" for q in (0, .25, .5, .75, 1)}
    cached, rows, previous = {}, [], None
    extrema = {"minimum_road_fraction": float("inf"), "maximum_road_fraction": -1., "largest_raw_mask_change": -1.}
    started = time.perf_counter()
    writer = DisplayWriter(folder / "comparison.mp4", rate)
    complete = False
    try:
        with av.open(str(path)) as container, (folder / "per_frame.csv").open("w", newline="", encoding="utf-8") as log:
            csv_writer = csv.DictWriter(log, fieldnames=FIELDS)
            csv_writer.writeheader()
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Bounded inference wall-time limit reached; partial evidence retained.")
                if args.stop_file and args.stop_file.exists():
                    raise InterruptedError("This video run was stopped by its own STOP marker.")
                stamp = Fraction(frame.pts) * frame.time_base - origin
                if stamp != Fraction(index, 1) / rate or index >= meta["frames"]:
                    raise ValueError("Source timestamps changed between preflight and prediction.")
                if frame.is_corrupt:
                    raise ValueError("Corrupt video frame during prediction.")
                rgb = frame.to_ndarray(format="rgb24")
                model_rgb = resize_input(rgb, allow_resize=args.resize)
                call_started = time.perf_counter()
                prediction = segmenter.predict_rgb(model_rgb)
                prediction_call_ms = (time.perf_counter() - call_started) * 1000
                mask = prediction.mask
                if mask.dtype != np.uint8 or mask.shape != (512, 512) or not np.isin(mask, [0, 1]).all():
                    raise ValueError("The locked predictor did not return a binary 0/1 mask.")
                Image.fromarray(mask).save(folder / "masks_01" / f"frame_{index:06d}.png", compress_level=1)
                change, consecutive_iou = mask_change(previous, mask)
                row = {"frame_index": index, "source_pts": frame.pts, "source_time_base": str(frame.time_base),
                       "timestamp_s": float(stamp), "inference_ms": prediction.inference_ms,
                       "prediction_call_ms": prediction_call_ms, "road_fraction": float(mask.mean()),
                       "raw_mask_change_fraction": change, "consecutive_mask_iou": consecutive_iou}
                rows.append(row)
                csv_writer.writerow(row)
                panel = np.array(Image.fromarray(rgb).resize(PANEL, Image.Resampling.BILINEAR))
                writer.append(make_display(panel, mask, float(stamp), index), index)
                reasons = [periodic[index]] if index in periodic else []
                for key, score, minimize in [("minimum_road_fraction", row["road_fraction"], True),
                                              ("maximum_road_fraction", row["road_fraction"], False),
                                              ("largest_raw_mask_change", change, False)]:
                    if score is not None and ((score < extrema[key]) if minimize else (score > extrema[key])):
                        extrema[key] = score
                        reasons.append(key)
                if reasons:
                    value = capture_candidate(index, float(stamp), model_rgb, panel, prediction)
                    for reason in reasons:
                        cached[reason] = value
                previous = mask
                if (index + 1) % 150 == 0 or index + 1 == meta["frames"]:
                    log.flush()
                    progress(index + 1, meta["frames"], float(stamp))
            if len(rows) != meta["frames"]:
                raise ValueError("Source frame count changed during prediction.")
        complete = True
    finally:
        writer.close(finish=complete)
    processing_s = time.perf_counter() - started
    examples = export_examples(folder, cached, meta, source, segmenter.lock)
    output_info = inspect_video(folder / "comparison.mp4")
    if output_info["frames"] != len(rows) or abs(output_info["duration_s"] - meta["duration_s"]) > 1e-6:
        raise ValueError("Encoded display did not preserve frame count and duration.")
    clip = {"clip_id": source["clip_id"], "description": source["description"],
            "source_file": source["file"], "folder": folder.name, "source_video": meta,
            "processed_frames": len(rows), "processing_wall_s": processing_s,
            "processing_fps_including_decode_resize_mask_write_render_encode": len(rows) / processing_s,
            "inference_ms": distribution([r["inference_ms"] for r in rows]),
            "prediction_call_ms": distribution([r["prediction_call_ms"] for r in rows]),
            "road_fraction": distribution([r["road_fraction"] for r in rows]),
            "raw_mask_change_fraction": distribution([r["raw_mask_change_fraction"] for r in rows]),
            "consecutive_mask_iou": distribution([r["consecutive_mask_iou"] for r in rows]),
            "output_video": output_info, "examples": examples,
            "video_accuracy_available": False, "mask_values": [0, 1],
            "temporal_diagnostics_note": "Unaligned consecutive camera masks: real motion and viewpoint changes contribute. Not accuracy or a pure flicker measure.",
            "processing_timing_note": "Warm processing, excludes model load/warmup, preflight, example export and output verification; includes sequential decoding, resizing, prediction, mask PNG writes, drawing and video encoding/flush."}
    write_json_atomic(folder / "clip_result.json", clip)
    return clip


def build_gallery(folder, status, manifest):
    sections = []
    for clip in status["clips"]:
        name = html.escape(clip["folder"])
        examples = "".join(f'<a href="{name}/{e["comparison"]}" target="_blank"><img loading="lazy" src="{name}/{e["comparison"]}" alt="Frame {e["frame_index"]} at {e["timestamp_s"]:.2f} seconds"><span>{e["timestamp_s"]:.2f}s · frame {e["frame_index"]} · {html.escape(", ".join(e["selection_reasons"]))}</span></a>' for e in clip["examples"])
        sections.append(f'''<section><h2>{html.escape(clip["description"])}</h2>
<p>{clip["processed_frames"]:,} frames · {clip["source_video"]["duration_s"]:.1f} seconds · 30 fps source playback · GPU median {clip["inference_ms"]["median"]:.1f} ms/frame · complete processing {clip["processing_fps_including_decode_resize_mask_write_render_encode"]:.1f} fps</p>
<video controls preload="metadata" playsinline poster="{name}/{clip["examples"][0]["comparison"]}"><source src="{name}/comparison.mp4" type="video/mp4"></video>
<p><a href="{name}/comparison.mp4" download>Save comparison video</a> · <a href="{name}/per_frame.csv">Every frame's timestamps and timings</a> · <a href="{name}/clip_result.json">Measurements</a></p>
<details><summary>Selected frames: original, predicted mask and overlay</summary><div class="examples">{examples}</div></details></section>''')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Road model · real Indian video</title>
<style>body{margin:0;background:#101720;color:#e8eff7;font:16px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:clamp(25px,4vw,42px);line-height:1.15}h2{font-size:22px}a{color:#83d8bd}section,.note{background:#1a2431;border:1px solid #344154;border-radius:12px;padding:20px;margin:24px 0}p{max-width:1000px}video{width:100%;background:#000;border-radius:8px}.examples{display:grid;gap:16px;margin-top:16px}.examples img{width:100%;height:auto;display:block}.examples span{font-size:14px}summary{cursor:pointer}small{color:#b6c5d6}</style>
<main><small>TESLA PROJECT / EXTERNAL VIDEO DIAGNOSTICS / 19 OCTOBER 2026</small><h1>Your locked road model on real Indian driving video</h1>
<p>The left view is the original footage; the right view shows the same frame with the predicted road in green. Playback keeps the source timing. All frames were processed once, with no smoothing or retraining.</p>
<div class="note"><strong>What this demonstrates</strong><p>Real-camera inference and a reviewable video export. It does not drive the simulated car. Green does not mean safe to drive: this model has no separate obstacle detector or calibrated metre-based projection. These motorcycle-camera clips have no labelled road masks, so no video accuracy or IoU score is claimed. Raw consecutive-mask changes include normal camera motion.</p></div>
''' + "\n".join(sections) + f'''<section><h2>Evidence and source</h2><p><a href="VIDEO_REPORT.md">Full report and visual review</a> · <a href="run_status.json">Run status and preservation checks</a> · <a href="config.json">Exact configuration</a></p>
<p>{html.escape(manifest["attribution"])}. <a href="{html.escape(manifest["source_repository"], quote=True)}">MOTOR source dataset</a>. {html.escape(manifest["license"])}; non-commercial research with attribution. Videos here are derived analysis displays; original source bytes are preserved.</p></section></main></html>'''
    (folder / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "media/real_driving_20261017/clips.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--gpu-memory-mib", type=int, default=3072)
    parser.add_argument("--max-wall-seconds", type=int, default=1800)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if not 60 <= args.max_wall_seconds <= 3600:
        parser.error("Use a wall-time limit between 60 and 3600 seconds.")
    output = args.output_dir.expanduser().resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        parser.error("Store this project's diagnostics in a new directory under runs/.")
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_binary_road_masks_provided") is not False:
        parser.error("This protocol expects explicitly unlabelled external videos.")
    sources = manifest["clips"]
    if not 1 <= len(sources) <= 3 or len({s["file"] for s in sources}) != len(sources):
        parser.error("Supply one to three unique external clips.")
    source_paths = []
    for source in sources:
        path = (manifest_path.parent / source["file"]).resolve(strict=True)
        if not path.is_relative_to(manifest_path.parent):
            parser.error("A source path escapes the manifest directory.")
        if file_sha256(path) != source["sha256"]:
            parser.error(f"Source hash changed: {path.name}")
        source_paths.append(path)
    load_final_lock()
    output.mkdir(parents=True, exist_ok=False)
    args.stop_file = (args.stop_file or output / "STOP.json").resolve()
    paths = protected_files(manifest_path) + source_paths
    before = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in ["predict_road_video.py", "idd_inference.py", "idd_model.py", "idd_boundary.py",
                 "run_road_video_prediction.ps1", "models/segmentation_final_lock.json"]:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    config = {"created_utc": utc_now(), "argv": sys.argv, "manifest": str(manifest_path.relative_to(ROOT)),
              "source_manifest": manifest, "device": args.device, "gpu_memory_mib": args.gpu_memory_mib,
              "pyav": av.__version__, "ffmpeg_libraries": {k: list(v) for k, v in av.library_versions.items()},
              "resize_opt_in": args.resize, "input_size": [512, 512],
              "resize_policy": "direct bilinear; aspect ratio changes; same existing image-inference convention",
              "normalization": "existing predict_rgb: raw uint8 RGB /127.5 -1 exactly once",
              "threshold": .5, "comparator": ">=", "postprocessing": "none; threshold only",
              "predict_every_frame": True, "batch_size": 1, "training": False,
              "audio": "no audio streams in these source clips; analysis display has no audio",
              "display_codec": "libx264/yuv420p, CRF20, veryfast, 2 CPU threads, MP4 faststart",
              "display_size_wh": list(VIDEO_SIZE), "mask_storage": "one lossless uint8 0/1 PNG per decoded frame",
              "probability_storage": "float32 arrays for selected review examples only",
              "no_ground_truth": True, "accuracy_metrics": None,
              "temporal_diagnostics": "raw consecutive masks without motion compensation; confounded by real motion",
              "max_wall_seconds": args.max_wall_seconds, "stop_file": str(args.stop_file),
              "old_automations_or_controllers_started": False,
              "protected_files_before": before}
    write_json_atomic(output / "config.json", config)
    status = {"state": "preflight", "started_utc": utc_now(), "clips": [], "training_performed": False,
              "reserved_test_images_read": False, "segmentation_accuracy_recomputed": False}
    write_json_atomic(output / "run_status.json", status)
    run_started = time.perf_counter()
    deadline = time.monotonic() + args.max_wall_seconds
    segmenter, weights_before = None, None
    try:
        metadata = []
        for path in source_paths:
            meta = inspect_video(path)
            metadata.append(meta)
            print(json.dumps({"preflight": path.name, **meta}), flush=True)
        write_json_atomic(output / "source_video_metadata.json", metadata)
        status["state"] = "loading_model"
        write_json_atomic(output / "run_status.json", status)
        load_start = time.perf_counter()
        segmenter = LockedRoadSegmenter(device=args.device, gpu_memory_mib=args.gpu_memory_mib)
        weights_before = weight_fingerprint(segmenter.model)
        status["model_load_s"] = time.perf_counter() - load_start
        warm_start = time.perf_counter()
        warmup = segmenter.warmup()
        status["warmup_s"] = time.perf_counter() - warm_start
        status["runtime"] = segmenter.runtime
        status["output_device"] = warmup.output_device
        for index, (source, path, meta) in enumerate(zip(sources, source_paths, metadata), 1):
            status["state"] = "predicting"
            status["active_clip"] = source["clip_id"]
            def progress(done, total, timestamp):
                status["progress"] = {"clip": source["clip_id"], "frames": done, "total": total, "timestamp_s": timestamp}
                status["checked_utc"] = utc_now()
                write_json_atomic(output / "run_status.json", status)
                print(json.dumps(status["progress"]), flush=True)
            result = process_clip(path, source, meta, output / f"{index:02d}_{source['clip_id']}",
                                  segmenter, args, deadline, progress)
            status["clips"].append(result)
            write_json_atomic(output / "run_status.json", status)
        status["state"] = "completed"
    except BaseException as error:
        status["state"] = "stopped" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        after = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
        status["preservation"] = {"file_hashes_before": before, "file_hashes_after": after,
                                  "all_protected_files_unchanged": before == after}
        if segmenter is not None and weights_before:
            weights_after = weight_fingerprint(segmenter.model)
            status["preservation"].update({"weights_and_bn_sha256_before": weights_before,
                                           "weights_and_bn_sha256_after": weights_after,
                                           "in_memory_weights_and_bn_unchanged": weights_before == weights_after})
        status["finished_utc"] = utc_now()
        status["total_wall_s_including_model_preflight_export_verification"] = time.perf_counter() - run_started
        if before != after or status["preservation"].get("in_memory_weights_and_bn_unchanged") is False:
            status["state"] = "failed"
            status["preservation_error"] = "Protected file or model state changed. Do not treat this run as verified."
        status.pop("active_clip", None)
        write_json_atomic(output / "run_status.json", status)
    if status["state"] != "completed":
        raise RuntimeError("Video run did not meet the preservation gate.")
    build_gallery(output, status, manifest)
    print(json.dumps({"state": status["state"], "output": str(output), "clips": len(status["clips"]),
                      "frames": sum(c["processed_frames"] for c in status["clips"])}), flush=True)


if __name__ == "__main__":
    main()
