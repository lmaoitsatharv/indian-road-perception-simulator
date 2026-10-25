"""Bounded file-stream benchmark, not a physical-camera or driving controller.

Reuses the locked batch-one predictor and the exact existing RGB resize. A
two-frame producer/consumer queue overlaps decoding and inference. Lossless
mode measures capacity; source-paced mode tests latest-frame behavior at 30 fps.
No mask PNG/video encoding is in the measured path. Hashes are compared with
every corresponding saved diagnostic mask after timing has stopped.
"""
import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from queue import Empty, Full, Queue
import shutil
from threading import Event, Thread
import time

import av
import numpy as np
from PIL import Image

from idd_inference import ROOT, LockedRoadSegmenter, file_sha256, write_json_atomic
from predict_road_video import distribution, protected_files, resize_input, utc_now, weight_fingerprint


@dataclass(frozen=True)
class StreamFrame:
    index: int
    timestamp_s: float
    source_pts: int
    time_base: str
    rgb: np.ndarray
    scheduled_monotonic: float | None
    ready_monotonic: float
    decode_rgb_ms: float
    resize_ms: float


def run_stream(path, metadata, segmenter, *, mode, capacity, stale_ms, deadline,
               stop_file=None, progress=lambda *_: None):
    if mode not in ("lossless", "source_paced") or capacity not in (1, 2):
        raise ValueError("Use lossless/source_paced mode and one or two queued frames.")
    if not np.isfinite(stale_ms) or stale_ms <= 0:
        raise ValueError("The stale-frame budget must be positive and finite.")
    queue = Queue(maxsize=capacity)
    cancel, finished = Event(), Event()
    dropped, stale, produced, errors, rows = [], [], [], [], []
    origin = Fraction(metadata["first_pts_time_fraction"])
    rate = Fraction(metadata["fps_fraction"])
    started = time.monotonic()
    max_queue_size = 0

    def guard():
        if time.monotonic() >= deadline:
            raise TimeoutError("The bounded stream benchmark reached its wall-time limit.")
        if stop_file and Path(stop_file).exists():
            raise InterruptedError("This run's STOP marker was written.")

    def produce():
        nonlocal max_queue_size
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                decoder = iter(container.decode(stream))
                for index in range(metadata["frames"]):
                    guard()
                    if cancel.is_set():
                        return
                    scheduled = started + float(Fraction(index) / rate) if mode == "source_paced" else None
                    if scheduled is not None and cancel.wait(max(0., scheduled - time.monotonic())):
                        return
                    t0 = time.monotonic()
                    frame = next(decoder)
                    if frame.is_corrupt or frame.pts is None or frame.time_base is None:
                        raise ValueError("Corrupt frame or missing source presentation timestamp.")
                    stamp = Fraction(frame.pts) * frame.time_base - origin
                    if stamp != Fraction(index) / rate:
                        raise ValueError("Input source timeline changed from its verified baseline.")
                    rgb = frame.to_ndarray(format="rgb24")
                    t1 = time.monotonic()
                    model_rgb = resize_input(rgb, allow_resize=True)
                    ready = time.monotonic()
                    packet = StreamFrame(index, float(stamp), frame.pts, str(frame.time_base),
                                         model_rgb, scheduled, ready, (t1-t0)*1000, (ready-t1)*1000)
                    produced.append(index)
                    if mode == "lossless":
                        while not cancel.is_set():
                            guard()
                            try:
                                queue.put(packet, timeout=.1)
                                break
                            except Full:
                                continue
                    else:
                        # The producer alone inserts frames. Remove the oldest
                        # queued one on overflow; never increase buffer capacity.
                        while not cancel.is_set():
                            try:
                                queue.put_nowait(packet)
                                break
                            except Full:
                                try:
                                    dropped.append(queue.get_nowait().index)
                                except Empty:
                                    continue
                    max_queue_size = max(max_queue_size, queue.qsize())
                try:
                    next(decoder)
                except StopIteration:
                    pass
                else:
                    raise ValueError("More source frames appeared than the verified count.")
        except BaseException as error:
            errors.append(ValueError("Fewer decoded frames than the verified source count.") if isinstance(error, StopIteration) else error)
            cancel.set()
        finally:
            finished.set()

    producer = Thread(target=produce, name="road-file-decoder", daemon=True)
    producer.start()
    try:
        while True:
            guard()
            if errors:
                raise errors[0]
            try:
                packet = queue.get(timeout=.1)
            except Empty:
                if finished.is_set():
                    break
                continue
            receive = time.monotonic()
            age_before = (receive-packet.scheduled_monotonic)*1000 if packet.scheduled_monotonic is not None else None
            if age_before is not None and age_before > stale_ms:
                stale.append(packet.index)
                continue
            prediction = segmenter.predict_rgb(packet.rgb)
            complete = time.monotonic()
            digest = hashlib.sha256(prediction.mask.tobytes()).hexdigest()
            age_complete = (complete-packet.scheduled_monotonic)*1000 if packet.scheduled_monotonic is not None else None
            rows.append({"frame_index": packet.index, "timestamp_s": packet.timestamp_s,
                         "source_pts": packet.source_pts, "source_time_base": packet.time_base,
                         "decode_rgb_ms": packet.decode_rgb_ms, "resize_ms": packet.resize_ms,
                         "queue_wait_ms": (receive-packet.ready_monotonic)*1000,
                         "prediction_call_ms": (complete-receive)*1000,
                         "inference_ms": prediction.inference_ms,
                         "source_scheduled_to_result_ms": age_complete,
                         "stale_at_completion": age_complete is not None and age_complete > stale_ms,
                         "road_fraction": float(prediction.mask.mean()), "mask_sha256": digest})
            if len(rows) % 150 == 0:
                progress(len(rows), len(produced), packet.timestamp_s)
    finally:
        cancel.set()
        producer.join(timeout=5)
    elapsed = time.monotonic() - started
    if producer.is_alive():
        raise RuntimeError("Owned decoder thread did not stop within five seconds.")
    if errors:
        raise errors[0]
    processed = [r["frame_index"] for r in rows]
    partitions = processed + dropped + stale
    if sorted(produced) != list(range(metadata["frames"])) or sorted(partitions) != sorted(produced):
        raise RuntimeError("Processed/dropped/stale frame accounting is incomplete or duplicated.")
    if len(set(partitions)) != len(partitions) or max_queue_size > capacity:
        raise RuntimeError("Frame queue accounting violated its bounds.")
    if mode == "lossless" and (dropped or stale):
        raise RuntimeError("Lossless mode silently lost frames.")
    if not rows:
        raise RuntimeError("No fresh predictions were produced by this stream.")
    result = {"mode": mode, "decoded_frames": len(produced), "processed_frames": len(rows),
              "wall_s": elapsed, "processed_fps": len(rows) / elapsed,
              "source_duration_s": metadata["duration_s"], "source_fps": metadata["fps"],
              "queue_capacity_frames": capacity, "maximum_observed_queue_size": max_queue_size,
              "dropped_queue_frame_indices": dropped, "dropped_stale_frame_indices": stale,
              "stale_budget_ms": stale_ms, "results_stale_at_completion": sum(r["stale_at_completion"] for r in rows),
              "decoder_thread_stopped": not producer.is_alive(),
              "inference_ms": distribution([r["inference_ms"] for r in rows]),
              "decode_rgb_ms": distribution([r["decode_rgb_ms"] for r in rows]),
              "resize_ms": distribution([r["resize_ms"] for r in rows]),
              "queue_wait_ms": distribution([r["queue_wait_ms"] for r in rows]),
              "source_scheduled_to_result_ms": distribution([r["source_scheduled_to_result_ms"] for r in rows]),
              "physical_camera_tested": False, "display_or_video_encoding_in_timed_path": False,
              "timing_note": "Includes file open/decode/RGB resize, bounded queue, prediction, mask hashing and in-memory metrics; excludes model startup, source preflight, saved-mask equivalence reads and final CSV export. Source-paced ages start at scheduled local-file frame release, not physical camera exposure."}
    return result, rows


def compare_saved_masks(rows, baseline_folder):
    mismatches = []
    for row in rows:
        with Image.open(baseline_folder / "masks_01" / f"frame_{row['frame_index']:06d}.png") as image:
            mask = np.array(image)
        if mask.dtype != np.uint8 or mask.shape != (512, 512) or not np.isin(mask, [0, 1]).all():
            raise ValueError("The saved reference mask no longer matches its binary contract.")
        if hashlib.sha256(mask.tobytes()).hexdigest() != row["mask_sha256"]:
            mismatches.append(row["frame_index"])
    return {"masks_compared": len(rows), "mismatch_frame_indices": mismatches,
            "all_compared_masks_bitwise_identical": not mismatches,
            "not_an_accuracy_measure": True}


def save_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs/real_video_20261019_02")
    parser.add_argument("--max-wall-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 60 <= args.max_wall_seconds <= 1800:
        parser.error("Choose a bounded duration from 60 to 1800 seconds.")
    out = args.output_dir.resolve()
    if not out.is_relative_to((ROOT / "runs").resolve()):
        parser.error("Choose a fresh output directory inside runs/.")
    baseline = args.baseline.resolve(strict=True)
    previous = json.loads((baseline / "run_status.json").read_text(encoding="utf-8"))
    audit = json.loads((baseline / "independent_verification.json").read_text(encoding="utf-8"))
    if previous["state"] != "completed" or audit["state"] != "passed":
        parser.error("The reference video run must have a completed independent audit.")
    manifest_path = ROOT / "media/real_driving_20261017/clips.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = previous["clips"]
    sources = {s["clip_id"]: s for s in manifest["clips"]}
    if len(clips) != 3 or set(sources) != {c["clip_id"] for c in clips}:
        parser.error("Reference clips no longer match the established three-video selection.")
    verified_audit = {c["clip_id"]: c for c in audit["clips"]}
    source_paths = []
    for clip in clips:
        if Path(clip["folder"]).name != clip["folder"] or clip["folder"] in (".", ".."):
            parser.error("Invalid reference folder name.")
        source = sources[clip["clip_id"]]
        path = (manifest_path.parent / source["file"]).resolve(strict=True)
        if not path.is_relative_to(manifest_path.parent) or file_sha256(path) != source["sha256"]:
            parser.error("Source path/hash differs from the saved original.")
        if file_sha256(baseline / clip["folder"] / "per_frame.csv") != verified_audit[clip["clip_id"]]["per_frame_csv_sha256"]:
            parser.error("The verified reference frame index was changed.")
        source_paths.append(path)
    out.mkdir(parents=True, exist_ok=False)
    stop_file = out / "STOP.json"
    paths = protected_files(manifest_path) + source_paths + [baseline / "run_status.json", baseline / "independent_verification.json"]
    before = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for name in ["benchmark_road_stream.py", "idd_inference.py", "predict_road_video.py", "run_road_stream.ps1"]:
        shutil.copy2(ROOT / name, snapshot / name)
    config = {"created_utc": utc_now(), "reference_run": str(baseline.relative_to(ROOT)),
              "source_manifest": manifest, "modes": ["lossless", "source_paced"],
              "queue_capacity_frames": 2, "stale_budget_ms": 100,
              "resize": "identical direct Pillow bilinear RGB 512x512",
              "normalization": "existing predict_rgb: /127.5 -1 once",
              "batch_size": 1, "threshold": .5, "comparator": ">=", "training": False,
              "device": "gpu", "gpu_memory_limit_mib": 3072, "max_wall_seconds": args.max_wall_seconds,
              "pyav": av.__version__, "decoder_threads": 2,
              "timed_output": "mask SHA256 and small in-memory metrics only; no PNG/display/MP4 encoding",
              "source_pacing": "local file decoded on scheduled 30 fps arrivals, not a physical camera",
              "accuracy_available": False, "old_automations_started": False,
              "protected_files_before": before}
    write_json_atomic(out / "config.json", config)
    status = {"state": "loading_model", "started_utc": utc_now(), "results": [],
              "training_performed": False, "reserved_dataset_images_read": False,
              "reference_unchanged": True, "physical_camera_tested": False}
    write_json_atomic(out / "run_status.json", status)
    segmenter = None
    fingerprint = None
    began = time.monotonic()
    deadline = began + args.max_wall_seconds
    try:
        segmenter = LockedRoadSegmenter(device="gpu", gpu_memory_mib=3072)
        fingerprint = weight_fingerprint(segmenter.model)
        segmenter.warmup()
        status["runtime"] = segmenter.runtime
        for mode in config["modes"]:
            for clip, path in zip(clips, source_paths):
                status["state"] = "benchmarking"
                status["active"] = {"mode": mode, "clip": clip["clip_id"]}
                write_json_atomic(out / "run_status.json", status)
                def progress(processed, decoded, timestamp):
                    status["progress"] = {"mode": mode, "clip": clip["clip_id"], "processed": processed,
                                          "decoded": decoded, "timestamp_s": timestamp}
                    status["checked_utc"] = utc_now()
                    write_json_atomic(out / "run_status.json", status)
                    print(json.dumps(status["progress"]), flush=True)
                result, rows = run_stream(path, clip["source_video"], segmenter, mode=mode, capacity=2,
                                          stale_ms=100, deadline=deadline, stop_file=stop_file, progress=progress)
                result["clip_id"] = clip["clip_id"]
                result["reference_export_fps_historical"] = clip["processing_fps_including_decode_resize_mask_write_render_encode"]
                result["saved_mask_comparison"] = compare_saved_masks(rows, baseline / clip["folder"])
                result["rows_file"] = f"{mode}_{clip['clip_id']}.csv"
                save_rows(out / result["rows_file"], rows)
                status["results"].append(result)
                write_json_atomic(out / "run_status.json", status)
        status["state"] = "completed"
        if not all(r["saved_mask_comparison"]["all_compared_masks_bitwise_identical"] for r in status["results"]):
            status["state"] = "completed_with_mask_differences"
    except BaseException as error:
        status["state"] = "stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed"
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        after = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
        preservation = {"protected_files_before": before, "protected_files_after": after,
                        "all_protected_files_unchanged": before == after}
        if segmenter is not None and fingerprint:
            after_fingerprint = weight_fingerprint(segmenter.model)
            preservation.update({"weights_and_bn_before": fingerprint, "weights_and_bn_after": after_fingerprint,
                                 "in_memory_weights_and_bn_unchanged": fingerprint == after_fingerprint})
        status["preservation"] = preservation
        status["total_wall_s"] = time.monotonic()-began
        status["finished_utc"] = utc_now()
        status.pop("active", None)
        if before != after or preservation.get("in_memory_weights_and_bn_unchanged") is False:
            status["state"] = "failed_preservation"
        write_json_atomic(out / "run_status.json", status)
    print(json.dumps({"state": status["state"], "output": str(out)}), flush=True)
    if status["state"] not in ("completed", "completed_with_mask_differences"):
        raise RuntimeError("Stream benchmark failed its completion gate.")


if __name__ == "__main__":
    main()
