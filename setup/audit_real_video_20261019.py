"""Recount saved video masks/timestamps without loading TensorFlow or IDD data."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import numpy as np
from PIL import Image, ImageDraw

from idd_inference import file_sha256, write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    if status["state"] != "completed":
        raise RuntimeError("Audit requires a completed inference run.")
    assert status["preservation"]["all_protected_files_unchanged"]
    assert status["preservation"]["in_memory_weights_and_bn_unchanged"]
    assert status["reserved_test_images_read"] is False
    clips, all_latency, all_change = [], [], []
    for clip in status["clips"]:
        folder = run / clip["folder"]
        with (folder / "per_frame.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        n = clip["source_video"]["frames"]
        assert len(rows) == n == clip["processed_frames"]
        assert len(list((folder / "masks_01").glob("*.png"))) == n
        rate = Fraction(clip["source_video"]["fps_fraction"])
        previous = None
        origin = Fraction(clip["source_video"]["first_pts_time_fraction"])
        for index, row in enumerate(rows):
            assert int(row["frame_index"]) == index
            stamp = Fraction(int(row["source_pts"])) * Fraction(row["source_time_base"]) - origin
            assert stamp == Fraction(index) / rate
            assert abs(float(stamp) - float(row["timestamp_s"])) < 1e-9
            with Image.open(folder / "masks_01" / f"frame_{index:06d}.png") as image:
                mask = np.array(image)
            assert mask.dtype == np.uint8 and mask.shape == (512, 512)
            assert np.isin(mask, [0, 1]).all()
            assert abs(float(mask.mean()) - float(row["road_fraction"])) < 1e-12
            if previous is None:
                assert row["raw_mask_change_fraction"] == "" and row["consecutive_mask_iou"] == ""
            else:
                change = float(np.mean(previous != mask))
                inter = np.count_nonzero((previous == 1) & (mask == 1))
                union = np.count_nonzero((previous == 1) | (mask == 1))
                iou = inter / union if union else 1.
                assert abs(change - float(row["raw_mask_change_fraction"])) < 1e-12
                assert abs(iou - float(row["consecutive_mask_iou"])) < 1e-12
                all_change.append(change)
            previous = mask
            latency = float(row["inference_ms"])
            assert np.isfinite(latency) and latency >= 0
            all_latency.append(latency)
        with av.open(str(folder / "comparison.mp4")) as output:
            stream = output.streams.video[0]
            assert stream.codec_context.name == "h264"
            assert stream.codec_context.pix_fmt == "yuv420p"
            assert (stream.width, stream.height) == (1280, 408)
            assert not output.streams.audio
            count = 0
            for index, frame in enumerate(output.decode(video=0)):
                assert Fraction(frame.pts) * frame.time_base == Fraction(index) / rate
                count += 1
            assert count == n
        for example in clip["examples"]:
            f = folder / Path(example["frame_metadata"]).parent
            probability = np.load(f / "road_probability.npy", allow_pickle=False)
            with Image.open(f / "road_mask_01.png") as image:
                mask = np.array(image)
            assert probability.dtype == np.float32
            assert np.isfinite(probability).all()
            assert np.array_equal(mask, (probability >= .5).astype(np.uint8))
            with Image.open(folder / "masks_01" / f"frame_{example['frame_index']:06d}.png") as image:
                assert np.array_equal(mask, np.array(image))
        # Full-width review contact sheet, with no modifications to source data.
        sheet = Image.new("RGB", (1920, len(clip["examples"]) * 436), (18, 25, 35))
        draw = ImageDraw.Draw(sheet)
        for i, example in enumerate(clip["examples"]):
            with Image.open(folder / example["comparison"]) as image:
                sheet.paste(image, (0, i * 436))
            draw.text((12, i * 436 + 413), ", ".join(example["selection_reasons"]), fill="white")
        sheet.save(folder / "review_contact_sheet.jpg", quality=90)
        clips.append({"clip_id": clip["clip_id"], "all_binary_masks_recounted": n,
                      "source_pts_rows_verified": n, "display_frames_decoded": count,
                      "sample_probability_exports_verified": len(clip["examples"]),
                      "display_sha256": file_sha256(folder / "comparison.mp4"),
                      "per_frame_csv_sha256": file_sha256(folder / "per_frame.csv")})
        print(json.dumps(clips[-1]), flush=True)
    total_frames = sum(c["all_binary_masks_recounted"] for c in clips)
    processing_s = sum(c["processing_wall_s"] for c in status["clips"])
    evidence = {"state": "passed", "clips": clips, "total_frames": total_frames,
                "combined_inference_ms_median": float(np.median(all_latency)),
                "combined_inference_ms_p95": float(np.percentile(all_latency, 95)),
                "combined_inference_only_fps_from_mean_latency": 1000 / float(np.mean(all_latency)),
                "combined_processing_fps": total_frames / processing_s,
                "raw_consecutive_mask_change_mean": float(np.mean(all_change)),
                "raw_consecutive_mask_change_p95": float(np.percentile(all_change, 95)),
                "all_saved_mask_values_binary": True, "all_masks_match_road_fraction_csv": True,
                "all_temporal_statistics_recounted": True, "all_display_pts_verified": True,
                "video_ground_truth_available": False, "video_accuracy_claimed": False,
                "tensorflow_loaded_by_audit": False, "reserved_dataset_images_read": False}
    write_json_atomic(run / "independent_verification.json", evidence)
    shutil.copy2(Path(__file__), run / "source_snapshot" / Path(__file__).name)
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
