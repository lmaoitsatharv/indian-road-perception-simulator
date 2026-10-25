"""Predict road regions with the locked model; never train or overwrite a run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import math

from idd_inference import LockedRoadSegmenter, read_rgb, write_prediction, file_sha256, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", required=True, help="Raw image path; repeat for several images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory, refused if it already exists.")
    parser.add_argument("--resize", action="store_true", help="Explicitly resize a new non-512x512 image to the trained square input.")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-memory-mib", type=int, default=3072)
    parser.add_argument("--timestamp-s", type=float, default=None, help="Caller-provided capture timestamp; omit for unknown offline time.")
    args = parser.parse_args()
    if len({p.expanduser().resolve() for p in args.image}) != len(args.image):
        parser.error("Input image paths must be unique.")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error("Output directory already exists. Choose a new name to preserve prior results.")
    if args.timestamp_s is not None and (not math.isfinite(args.timestamp_s) or args.timestamp_s < 0):
        parser.error("timestamp-s must be finite and nonnegative.")
    if args.timestamp_s is not None and len(args.image) != 1:
        parser.error("A capture timestamp belongs to one image; use separate calls for separately timed frames.")
    # Preflight input decoding before creating an output or loading TensorFlow.
    # Keep only one RGB frame in RAM, even if a caller supplies a long list.
    input_hashes = [read_rgb(path, resize=args.resize)[1]["input_sha256"] for path in args.image]
    segmenter = LockedRoadSegmenter(device=args.device, gpu_memory_mib=args.gpu_memory_mib)
    before = file_sha256(segmenter.checkpoint)
    if before != segmenter.lock["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during model loading; prediction was not started.")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "images": [], "runtime": segmenter.runtime,
              "threshold": .5, "training_started": False, "test_evaluation_rerun": False,
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    status_path = output / "run_status.json"
    started = time.perf_counter()
    try:
        write_json_atomic(status_path, record)
        segmenter.warmup()
        for index, path in enumerate(args.image):
            rgb, metadata = read_rgb(path, resize=args.resize)
            if metadata["input_sha256"] != input_hashes[index]:
                raise RuntimeError(f"Input image changed after preflight: {metadata['input_file']}")
            prediction = segmenter.predict_rgb(rgb)
            stem = f"{index+1:03d}_{args.image[index].stem}"
            folder = write_prediction(output, stem, rgb, prediction, metadata,
                                      timestamp_s=args.timestamp_s, model_lock=segmenter.lock)
            record["images"].append({"input": metadata["input_file"], "output": str(folder),
                                      "road_fraction": float(prediction.mask.mean()),
                                      "inference_ms": prediction.inference_ms, "device": prediction.output_device})
            record["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(status_path, record)
        if file_sha256(segmenter.checkpoint) != before:
            raise RuntimeError("Checkpoint changed during inference; these outputs require investigation.")
        record.update(status="completed", checkpoint_unchanged=True, elapsed_seconds=time.perf_counter()-started)
        print(json.dumps(record, indent=2))
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=str(error), error_type=type(error).__name__)
        raise
    finally:
        try:
            record["checkpoint_unchanged"] = file_sha256(segmenter.checkpoint) == before
        except OSError as error:
            record.update(checkpoint_unchanged=False, checkpoint_integrity_error=str(error))
        record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter()-started)
        write_json_atomic(status_path, record)


if __name__ == "__main__":
    main()
