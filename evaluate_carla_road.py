"""Score a road model on verified paired CARLA frames; never controls actors."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from idd_inference import ROOT, LockedRoadSegmenter, file_sha256, write_json_atomic
from predict_road_video import distribution, protected_files, weight_fingerprint
from carla_road_reference import build_reference, checked_appearance

ROAD_IDS, IGNORE_IDS = (1, 24), (0,)
CLASS_NAMES = {0: "unlabeled", 1: "road", 2: "sidewalk", 3: "building", 4: "wall", 5: "fence",
    6: "pole", 7: "traffic_light", 8: "traffic_sign", 9: "vegetation", 10: "terrain", 11: "sky",
    12: "pedestrian", 13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train",
    18: "motorcycle", 19: "bicycle", 20: "static", 21: "dynamic", 22: "other", 23: "water",
    24: "road_line", 25: "ground", 26: "bridge", 27: "rail_track", 28: "guard_rail",
    29: "owned_asphalt_appearance_instance"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference(labels):
    labels = np.asarray(labels)
    if labels.dtype != np.uint8 or labels.ndim != 2 or not np.isin(labels, [0,1,255]).all():
        raise ValueError("Expected reference labels 0/1/255, not raw native tags or a color palette")
    return labels == 1, labels != 255


def confusion(prediction, tags):
    prediction = np.asarray(prediction)
    road, valid = reference(tags)
    if prediction.shape != road.shape or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Prediction must be a matching 0/1 mask")
    pred = prediction.astype(bool)
    return dict(tp=int(np.sum(pred & road & valid)), fp=int(np.sum(pred & ~road & valid)),
                fn=int(np.sum(~pred & road & valid)), tn=int(np.sum(~pred & ~road & valid)),
                ignored=int(np.sum(~valid)), total=int(valid.size))


def metrics(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    def ratio(a, b):
        return a / b if b else None
    return {**counts, "road_iou": ratio(tp, tp + fp + fn), "dice": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "pixel_accuracy": ratio(tp + tn, tp + fp + fn + tn),
            "evaluated_fraction": ratio(tp + fp + fn + tn, counts["total"])}


def safe_source(folder, name, expected_hash):
    path = (folder / name).resolve(strict=True)
    if not path.is_relative_to(folder.resolve()) or file_sha256(path) != expected_hash:
        raise ValueError("Capture path or fingerprint mismatch")
    return path


def inspect_capture(episode):
    config = load(episode / "road_camera/capture_config.json")
    result = load(episode / "result.json")
    launcher = load(episode.parent / "launcher_result.json")
    if not (launcher["status"] == "completed" and launcher["owned_job_handles_closed"]
            and result["cleanup_complete"] and result["runtime_passed"]):
        raise ValueError("Use a completed capture with clean owned-job shutdown")
    if (config["schema_version"] not in (2, 3) or config["carla_version"] != "0.9.16" or config["size_wh"] != [960, 540]
            or config["reference"]["road_ids"] != list(ROAD_IDS)
            or config["reference"]["ignore_ids"] != list(IGNORE_IDS)):
        raise ValueError("Capture protocol/mapping differs from this evaluation")
    if config.get("surface_style") == "asphalt" and config["schema_version"] != 3:
        raise ValueError("Asphalt pilot lacks instance evidence; use a schema-3 capture")
    if config["schema_version"] == 3:
        ids = config["reference"].get("appearance_actor_ids", [])
        if (config["reference"].get("appearance_native_class") != 3 or not ids
                or len(ids) != len(set(ids)) or any(type(i) is not int or not 0 < i <= 65535 for i in ids)):
            raise ValueError("Invalid owned-asphalt instance protocol")
    summary = load(episode / "road_camera/capture_summary.json")
    rows = [json.loads(line) for line in (episode / "road_camera/frames.jsonl").read_text().splitlines() if line.strip()]
    if not rows or len(rows) != summary["paired_frames"] or any(summary["dropped"].values()):
        raise ValueError("Empty/incomplete capture or dropped queue entries; inspect before evaluation")
    for i, row in enumerate(rows):
        if (abs(row["rgb_timestamp_s"] - row["labels_timestamp_s"]) > 1e-6
                or abs(row["rgb_timestamp_s"] - row["depth_timestamp_s"]) > 1e-6
                or abs(row["rgb_timestamp_s"] - row["timestamp_s"]) > 1e-6
                or row["decision"]["world_frame"] != row["frame"]
                or i and (row["frame"] <= rows[i-1]["frame"] or row["sim_s"] <= rows[i-1]["sim_s"])):
            raise ValueError("Camera pair/frame/timestamp alignment failed")
        if config["schema_version"] == 3 and (not row.get("instance_file") or
                abs(row["rgb_timestamp_s"]-row["instance_timestamp_s"]) > 1e-6 or row["appearance_pixels"] <= 0):
            raise ValueError("Missing/alignment-failed asphalt-instance evidence")
    return config, rows, result, summary


def panel(rgb, tags, mask, row, score):
    road, valid = reference(tags)
    source = Image.fromarray(rgb).resize((480, 270), Image.Resampling.BILINEAR)
    truth = np.zeros((512, 512, 3), np.uint8)
    truth[road] = 255
    truth[~valid] = [110, 72, 130]
    overlay = rgb.copy()
    overlay[mask == 1] = (.60 * overlay[mask == 1] + .40 * np.array([30, 220, 110])).astype(np.uint8)
    errors = rgb.copy()
    errors[road & (mask == 0) & valid] = [245, 65, 65]
    errors[~road & (mask == 1) & valid] = [40, 135, 250]
    image = Image.new("RGB", (960, 626), (17, 25, 36))
    draw = ImageDraw.Draw(image)
    font, small = ImageFont.load_default(size=18), ImageFont.load_default(size=14)
    iou = "n/a" if score["road_iou"] is None else f"{100*score['road_iou']:.2f}%"
    draw.text((12, 8), f"Front camera | {row['sim_s']:.2f}s | Frame {row['frame']} | Reference road IoU {iou}", font=font)
    for title, picture, x, y in (("Camera image", source, 0, 55),
            ("Simulator road label (purple = ignored)", Image.fromarray(truth), 480, 55),
            ("Locked model prediction", Image.fromarray(overlay), 0, 353),
            ("Red: missed road | Blue: extra road", Image.fromarray(errors), 480, 353)):
        draw.text((x + 10, y - 20), title, font=small)
        image.paste(picture.resize((480, 270), Image.Resampling.NEAREST), (x, y))
    return image


def gallery(out, result):
    data = json.dumps(result, allow_nan=False).replace("<", "\\u003c")
    template = (ROOT / "setup/road_observer_template.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(template.replace("__REPORT_DATA__", data), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--simulation-model", type=Path,
                        help="Opt-in simulator-only model manifest; default remains the locked real-road model.")
    parser.add_argument("--max-wall-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 30 <= args.max_wall_seconds <= 3600 or len(args.episode) > 8:
        parser.error("Use 1..8 episodes and a 30..3600-second bound")
    sources = [p.resolve(strict=True) for p in args.episode]
    captures = [inspect_capture(p) for p in sources]
    simulation_files = []
    if args.simulation_model:
        from idd_simulation_inference import load_simulation_model_spec
        _, model_files = load_simulation_model_spec(args.simulation_model)
        simulation_files = [args.simulation_model.resolve(), *model_files.values()]
    if sum(len(c[1]) for c in captures) > 6000:
        raise ValueError("Capture exceeds the 6000-frame observation bound")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").mkdir()
    for name in ("evaluate_carla_road.py", "carla_road_reference.py", "idd_inference.py", "predict_road_video.py", "setup/road_observer_template.html"):
        shutil.copy2(ROOT / name, out / "source" / Path(name).name)
    if args.simulation_model:
        shutil.copy2(ROOT / "idd_simulation_inference.py", out / "source/idd_simulation_inference.py")
    deadline = time.monotonic() + args.max_wall_seconds
    protected = protected_files(sources[0] / "road_camera/frames.jsonl") + simulation_files
    before = {str(p): file_sha256(p) for p in protected}
    status = {"status": "loading_model", "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "paired_simulator_camera_observation", "model_control": False, "episodes": [],
        "reference_mapping": captures[0][0]["reference"],
        "resize": "RGB direct bilinear 960x540 to 512x512; class IDs nearest-neighbour; no crop",
        "source_to_model_pixel_centers": [[512/960, 0, (512/960-1)/2], [0, 512/540, (512/540-1)/2], [0, 0, 1]],
        "simulation_model_opt_in": bool(args.simulation_model),
        "limits": "Recorded camera replay; actual sampling is saved per episode. Agreement with a depth/geometry-assisted simulator reference on selected synthetic scenes; not real-road accuracy or live perception latency. Native road tags are absent on this generated map. Unknown pixels outside the geometry range and uncertain map edges are excluded with coverage reported. The 0.1 m map raster and 0.2 m boundary exclusion limit reference precision. No tuning or control uses these predictions. Simulator-only model opt-in does not replace real-image or real-video inference."}
    write_json_atomic(out / "report.json", status)
    try:
        if args.simulation_model:
            from idd_simulation_inference import SimulationRoadSegmenter
            segmenter = SimulationRoadSegmenter(args.simulation_model, device=args.device)
        else:
            segmenter = LockedRoadSegmenter(device=args.device)
        weights_before = weight_fingerprint(segmenter.model)
        warmup = segmenter.warmup()
        status.update(status="predicting", runtime=segmenter.runtime, warmup_ms=warmup.inference_ms)
        write_json_atomic(out / "report.json", status)
        combined = {k: 0 for k in ("tp", "fp", "fn", "tn", "ignored", "total")}
        for source, (config, records, native, capture_summary) in zip(sources, captures):
            name = f"{len(status['episodes'])+1:02d}_{native['case']}_seed{native['seed']}"
            folder = out / name
            for part in ("masks_01", "reference_01_ignore255", "comparisons", "examples"):
                (folder / part).mkdir(parents=True, exist_ok=False)
            counts, class_total, class_extra = {k: 0 for k in combined}, np.zeros(30, np.int64), np.zeros(30, np.int64)
            rows, worst = [], []
            grid_config = config["reference"]["grid"]
            grid_path = safe_source(source / "road_camera", grid_config["file"], grid_config["sha256"])
            with np.load(grid_path, allow_pickle=False) as archive:
                footprint = archive["driving"]
            started = time.perf_counter()
            for index, row in enumerate(records):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Observation processing reached its wall limit")
                camera = source / "road_camera"
                rgb_path = safe_source(camera, row["rgb_file"], row["rgb_sha256"])
                labels_path = safe_source(camera, row["labels_file"], row["labels_sha256"])
                depth_path = safe_source(camera, row["depth_file"], row["depth_sha256"])
                with Image.open(rgb_path) as original:
                    rgb = np.array(original.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR))
                with Image.open(labels_path) as original:
                    native_tags = np.array(original.convert("L"))
                with np.load(depth_path, allow_pickle=False) as archive:
                    depth = archive["meters"]
                appearance = None
                disagreement = np.zeros(native_tags.shape, bool)
                if config["schema_version"] == 3:
                    instance_path = safe_source(camera, row["instance_file"], row["instance_sha256"])
                    with Image.open(instance_path) as original:
                        instance = np.array(original.convert("RGB"))
                    appearance, disagreement = checked_appearance(native_tags, instance, config, row)
                native_reference, geometry_check = build_reference(native_tags, depth, row["camera_to_world"], config, footprint, appearance)
                native_reference[disagreement] = 255
                geometry_check["semantic_instance_disagreement_pixels_ignored"] = int(disagreement.sum())
                if (geometry_check["ground_height_abs_median_m"] is None
                        or geometry_check["ground_height_abs_median_m"] > .04
                        or geometry_check["near_map_road_flat_fraction"] < .85):
                    raise ValueError("Depth projection failed the independent near-road flat-surface check")
                tags = np.array(Image.fromarray(native_reference).resize((512, 512), Image.Resampling.NEAREST))
                analysis_classes = native_tags.copy()
                if appearance is not None:
                    analysis_classes[appearance] = 29  # Report separately; native source pixels remain unchanged.
                native_ids = np.array(Image.fromarray(analysis_classes).resize((512, 512), Image.Resampling.NEAREST))
                prediction = segmenter.predict_rgb(rgb)
                values = confusion(prediction.mask, tags)
                score = metrics(values)
                for key in counts:
                    counts[key] += values[key]
                class_total += np.bincount(native_ids.ravel(), minlength=30)
                class_extra += np.bincount(native_ids[prediction.mask == 1], minlength=30)
                reference_road, valid = reference(tags)
                target = reference_road.astype(np.uint8)
                target[~valid] = 255
                stem = f"frame_{row['frame']:07d}"
                Image.fromarray(prediction.mask).save(folder / "masks_01" / f"{stem}.png")
                Image.fromarray(target).save(folder / "reference_01_ignore255" / f"{stem}.png")
                comparison = f"{name}/comparisons/{stem}.jpg"
                panel(rgb, tags, prediction.mask, row, score).save(out / comparison, quality=85)
                saved = {"frame": row["frame"], "sim_s": row["sim_s"], "source": row,
                    "inference_ms": prediction.inference_ms, "mask_sha256": hashlib.sha256(prediction.mask.tobytes()).hexdigest(),
                    "mask": f"{name}/masks_01/{stem}.png", "reference": f"{name}/reference_01_ignore255/{stem}.png",
                    "comparison": comparison, "metrics": score, "geometry_check": geometry_check}
                rows.append(saved)
                if score["road_iou"] is not None:
                    worst.append((score["road_iou"], index, prediction.probability.copy()))
                    worst.sort(key=lambda value: (value[0], value[1]))
                    del worst[8:]
                if index % 50 == 0:
                    write_json_atomic(out / "progress.json", {"episode": name, "processed": index+1, "total": len(records)})
                    print(f"{name}: {index+1}/{len(records)}", flush=True)
            for _, index, probability in worst:
                np.save(folder / "examples" / f"frame_{records[index]['frame']:07d}_probability.npy", probability, allow_pickle=False)
            for key in combined:
                combined[key] += counts[key]
            classes = [{"id": i, "name": CLASS_NAMES[i], "pixels": int(class_total[i]),
                "predicted_road_pixels": int(class_extra[i]),
                "predicted_road_fraction": float(class_extra[i] / class_total[i]) if class_total[i] else None}
                for i in range(30) if i < 29 or class_total[i]]
            episode = {"name": name, "case": native["case"], "seed": native["seed"], "frames": len(rows),
                "source_capture": str(source), "camera_config": config, "capture_summary": capture_summary,
                "source_manifest_sha256": file_sha256(source / "road_camera/frames.jsonl"),
                "render_quality": next(value.split('=',1)[1] for value in load(source.parent/"launcher_result.json")["server_argv"] if value.startswith('-quality-level=')),
                "metrics": metrics(counts), "class_breakdown": classes,
                "inference_ms": distribution([r["inference_ms"] for r in rows]),
                "processing_wall_s_including_export": time.perf_counter()-started,
                "rows": rows, "worst_frames": [rows[i]["comparison"] for _, i, _ in worst],
                "native_run": {**{k: native[k] for k in ("ego_goal_reached", "completed_overtakes", "exit_reason")},
                    "collision_events": len(native["collision_events"]), "sampled_body_overlaps": len(native["all_actor_overlaps"])}}
            write_json_atomic(folder / "frames.json", rows)
            status["episodes"].append(episode)
            write_json_atomic(out / "report.json", status)
        after = {str(p): file_sha256(p) for p in protected}
        status.update(metrics=metrics(combined), protected_files_unchanged=before == after,
            model_weights_unchanged=weights_before == weight_fingerprint(segmenter.model),
            protected_hashes=after, status="completed")
        if not status["protected_files_unchanged"] or not status["model_weights_unchanged"]:
            raise RuntimeError("Protected model/protocol state changed")
        write_json_atomic(out / "report.json", status)
        gallery(out, status)
        print(json.dumps({"status": "completed", "frames": sum(e["frames"] for e in status["episodes"]),
                          "metrics": status["metrics"], "output": str(out)}), flush=True)
    except BaseException as error:
        status.update(status="failed", error=str(error))
        write_json_atomic(out / "report.json", status)
        raise


if __name__ == "__main__":
    main()
