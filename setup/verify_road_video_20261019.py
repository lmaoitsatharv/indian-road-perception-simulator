"""Small end-to-end video contract check. Uses a fake predictor, no dataset/TF."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import numpy as np
from PIL import Image

from idd_inference import RoadPrediction, file_sha256, write_json_atomic
from predict_road_video import inspect_video, mask_change, process_clip, resize_input


class FakeSegmenter:
    lock = {"checkpoint_sha256": "synthetic-only", "threshold": .5,
            "threshold_comparator": ">=", "mask_labels": [0, 1]}

    def predict_rgb(self, rgb):
        assert rgb.dtype == np.uint8 and rgb.shape == (512, 512, 3)
        # A red-vs-blue fixture exposes accidental BGR decoding and channel swaps.
        mask = (rgb[..., 0] > rgb[..., 2]).astype(np.uint8)
        return RoadPrediction(mask.astype(np.float32), mask, .1, "synthetic-fixture")


def write_fixture(path, *, variable=False):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=Fraction(6))
        stream.width, stream.height = 640, 360
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 6)
        stream.options = {"crf": "15"}
        for i in range(12):
            rgb = np.zeros((360, 640, 3), np.uint8)
            rgb[:, :320] = [230, 20, 10]
            rgb[:, 320:] = [10, 20, 230]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i + (int(i >= 6) if variable else 0)
            frame.time_base = Fraction(1, 6)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "fixture.mp4"
    write_fixture(fixture)
    meta = inspect_video(fixture)
    assert meta["frames"] == 12 and meta["duration_s"] == 2.
    assert meta["fps_fraction"] == "6"
    with av.open(str(fixture)) as container:
        rgb = next(container.decode(video=0)).to_ndarray(format="rgb24")
    assert rgb[100, 100, 0] > 200 and rgb[100, 500, 2] > 200
    try:
        resize_input(rgb, allow_resize=False)
    except ValueError:
        pass
    else:
        raise AssertionError("Resize requires explicit opt-in")
    raw = resize_input(rgb, allow_resize=True)
    assert raw.dtype == np.uint8 and raw[100, 100, 0] > 200
    a, b = np.zeros((3, 3), np.uint8), np.ones((3, 3), np.uint8)
    assert mask_change(None, a) == (None, None)
    assert mask_change(a, a) == (0., 1.)
    assert mask_change(a, b) == (1., 0.)
    source = {"file": fixture.name, "sha256": file_sha256(fixture),
              "clip_id": "synthetic", "description": "Synthetic RGB/timestamp fixture"}
    result = process_clip(fixture, source, meta, out / "export", FakeSegmenter(),
                          SimpleNamespace(resize=True, stop_file=None), time.monotonic()+60,
                          lambda done, total, stamp: None)
    masks = sorted((out / "export/masks_01").glob("*.png"))
    assert len(masks) == 12
    for path in masks:
        mask = np.array(Image.open(path))
        assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 1}
        assert mask[100, 100] == 1 and mask[100, 400] == 0
    assert result["output_video"]["frames"] == 12
    assert result["output_video"]["duration_s"] == 2
    assert not result["video_accuracy_available"]
    variable = out / "variable.mp4"
    write_fixture(variable, variable=True)
    try:
        inspect_video(variable)
    except ValueError as error:
        assert "CFR" in str(error)
    else:
        raise AssertionError("Variable-frame timestamps must not be silently retimed")
    # The same export directory must not be reused and overwrite its evidence.
    try:
        process_clip(fixture, source, meta, out / "export", FakeSegmenter(),
                     SimpleNamespace(resize=True, stop_file=None), time.monotonic()+60,
                     lambda *x: None)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing export was overwritten")
    evidence = {"state": "passed", "frames": 12, "duration_s": 2,
                "checks": ["RGB channel order", "explicit resize", "lossless 0/1 masks",
                           "decoded PTS and duration roundtrip", "all frames exported",
                           "variable-frame timing rejected", "no overwrite", "no accuracy invented"],
                "tensorflow_or_dataset_read": False, "pyav": av.__version__}
    write_json_atomic(out / "verification.json", evidence)
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
