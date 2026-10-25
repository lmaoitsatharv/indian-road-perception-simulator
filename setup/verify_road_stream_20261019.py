"""Exercise queue overload, stop/exception cleanup and exact reference masks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from threading import enumerate as live_threads
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from idd_inference import RoadPrediction, write_json_atomic
from predict_road_video import inspect_video
from benchmark_road_stream import compare_saved_masks, run_stream


class FakePredictor:
    def __init__(self, delay=0, fail_at=None):
        self.delay, self.fail_at, self.calls = delay, fail_at, 0

    def predict_rgb(self, rgb):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected consumer failure")
        if self.delay:
            time.sleep(self.delay)
        mask = (rgb[..., 0] > rgb[..., 2]).astype(np.uint8)
        return RoadPrediction(mask.astype(np.float32), mask, .1, "synthetic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference = Path("runs/real_video_contract_20261019_02").resolve()
    path = reference / "fixture.mp4"
    meta = inspect_video(path)
    result, rows = run_stream(path, meta, FakePredictor(), mode="lossless", capacity=2,
                              stale_ms=100, deadline=time.monotonic()+20)
    assert result["processed_frames"] == 12 and result["decoder_thread_stopped"]
    assert compare_saved_masks(rows, reference / "export")["all_compared_masks_bitwise_identical"]
    overload, sampled = run_stream(path, meta, FakePredictor(.4), mode="source_paced", capacity=2,
                                  stale_ms=1000, deadline=time.monotonic()+15)
    assert overload["dropped_queue_frame_indices"] and overload["processed_frames"] < 12
    assert sampled[-1]["frame_index"] == 11
    assert overload["maximum_observed_queue_size"] <= 2 and overload["decoder_thread_stopped"]
    assert len(sampled) + len(overload["dropped_queue_frame_indices"]) + len(overload["dropped_stale_frame_indices"]) == 12
    stale, _ = run_stream(path, meta, FakePredictor(.4), mode="source_paced", capacity=2,
                         stale_ms=60, deadline=time.monotonic()+15)
    assert stale["dropped_stale_frame_indices"]
    assert stale["results_stale_at_completion"] > 0
    try:
        run_stream(path, meta, FakePredictor(fail_at=3), mode="lossless", capacity=2,
                   stale_ms=100, deadline=time.monotonic()+10)
    except RuntimeError as error:
        assert "injected" in str(error)
    else:
        raise AssertionError("Predictor failure was swallowed")
    stop = out / "STOP.json"
    stop.write_text('{"reason":"synthetic stop"}', encoding="utf-8")
    try:
        run_stream(path, meta, FakePredictor(), mode="source_paced", capacity=2,
                   stale_ms=100, deadline=time.monotonic()+10, stop_file=stop)
    except InterruptedError:
        pass
    else:
        raise AssertionError("Own-run stop ignored")
    bad_meta = dict(meta, frames=13)
    try:
        run_stream(path, bad_meta, FakePredictor(), mode="lossless", capacity=2,
                   stale_ms=100, deadline=time.monotonic()+10)
    except ValueError as error:
        assert "Fewer" in str(error)
    else:
        raise AssertionError("Decoder EOF mismatch was swallowed")
    assert not [t for t in live_threads() if t.name == "road-file-decoder"]
    evidence = {"state": "passed", "synthetic_only": True, "tensorflow_or_idd_images_read": False,
                "lossless_all_masks_match": True, "overload": overload,
                "stale_rejection": stale,
                "checks": ["bounded queue", "all-frame accounting", "newest frame survives overload",
                           "stale input rejected", "stale completion reported", "RGB masks identical",
                           "consumer exception cleanup", "producer error propagation", "own STOP", "no decoder threads left"]}
    write_json_atomic(out / "verification.json", evidence)
    print(json.dumps({"state": "passed", "checks": evidence["checks"]}), flush=True)


if __name__ == "__main__":
    main()
