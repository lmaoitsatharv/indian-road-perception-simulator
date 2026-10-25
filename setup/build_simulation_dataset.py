"""Build/version/audit simulation pairs from synchronized native evidence; no ML inference."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carla_road_reference import build_reference
from prepare_simulation_dataset import sha, save, protected, DEFAULT


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def png(path, mode=None):
    with Image.open(path) as image:
        return np.array(image.convert(mode) if mode else image)


def source_file(camera, row, kind):
    path = camera / row[f"{kind}_file"]
    if path.parent != camera or sha(path) != row[f"{kind}_sha256"]:
        raise ValueError(f"Source hash or path mismatch: {path}")
    return path


def candidates(episode, expected):
    camera = episode / "road_camera"
    rows = [json.loads(s) for s in (camera / "frames.jsonl").read_text().splitlines() if s]
    summary, native = load(camera / "capture_summary.json"), load(episode / "result.json")
    if (not native["runtime_passed"] or native["cleanup_warnings"] or not native["cleanup_complete"]
            or native["exit_reason"] != "bounded_test_complete" or any(summary["dropped"].values())
            or summary["paired_frames"] != len(rows)):
        raise ValueError(f"Incomplete or unhealthy episode: {episode}")
    if len(rows) < expected-2 or len({r["frame"] for r in rows}) != len(rows):
        raise ValueError("Expected a complete set of distinct paired frames for the declared duration/rate")
    return rows


def diverse_selection(rows, camera, count, existing_pixels):
    """Remove exact duplicates, prefer changing views, then cover the full available episode."""
    unique, duplicates, low_change = [], 0, 0
    previous = None
    local_hashes = set()
    for row in rows:
        rgb = png(source_file(camera, row, "rgb"), "RGB")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest in local_hashes or digest in existing_pixels:
            duplicates += 1
            continue
        local_hashes.add(digest)
        thumbnail = np.array(Image.fromarray(rgb).resize((48,27), Image.Resampling.BILINEAR), dtype=np.float32)
        movement = float(np.abs(thumbnail-previous).mean()) if previous is not None else 255.
        # A stopped camera with only sub-pixel render noise is not a useful extra example.
        if movement < .75:
            low_change += 1
            continue
        previous = thumbnail
        unique.append((row, digest, movement))
    if len(unique) < count:
        raise ValueError(f"Only {len(unique)} changing unique views for requested {count}; capture more episodes instead of padding")
    indices = np.linspace(0, len(unique)-1, count).round().astype(int)
    chosen = [unique[i] for i in indices]
    existing_pixels.update(digest for _,digest,_ in chosen)
    return chosen, {"raw_pairs": len(rows), "exact_duplicates_rejected": duplicates,
                    "low_change_rejected": low_change, "changing_candidates": len(unique), "selected": count,
                    "minimum_change_mae_rgb_0_255": .75}


def panel(image, reference, title, path):
    road, ignore = reference == 1, reference == 255
    overlay = image.copy()
    overlay[road] = (image[road]*.55 + np.array([40,230,100])*.45).astype(np.uint8)
    overlay[ignore] = (image[ignore]*.45 + np.array([220,70,220])*.55).astype(np.uint8)
    mask = np.zeros_like(image)
    mask[road], mask[ignore] = [255,255,255], [220,70,220]
    canvas = Image.new("RGB", (768, 290), "#101824")
    draw = ImageDraw.Draw(canvas)
    draw.text((8,6), title, fill="white")
    for i, value in enumerate((image,mask,overlay)):
        canvas.paste(Image.fromarray(value).resize((256,256), Image.Resampling.NEAREST if i==1 else Image.Resampling.BILINEAR), (256*i,30))
    canvas.save(path, quality=88)


def process_episode(output, episode, spec, all_rows, pixel_hashes, previews, start_id=None):
    camera = episode / "road_camera"
    native_rows = candidates(episode, round(spec["sim_seconds"]*spec["capture_hz"]))
    selected, diversity = diverse_selection(native_rows, camera, spec["pairs_to_select"], pixel_hashes)
    config = load(camera / "capture_config.json")
    assert not config.get("weather_probe_only"),"Diagnostic probes must never enter the training dataset"
    assert config["schema_version"] == 3 and config["camera_profile"] == spec["camera_profile"]
    grid = camera / config["reference"]["grid"]["file"]
    assert sha(grid) == config["reference"]["grid"]["sha256"]
    with np.load(grid, allow_pickle=False) as archive:
        footprint = archive["driving"]
    totals = Counter()
    split = spec["split"]
    for i, (source, pixel_sha, movement) in enumerate(selected):
        pair_id = start_id+i if start_id is not None else len(all_rows)
        tags = png(source_file(camera, source, "labels"), "RGB")[:,:,0]
        instance = png(source_file(camera, source, "instance"), "RGB")
        disagreement = instance[:,:,0] != tags
        assert int(disagreement.sum()) == source.get("semantic_instance_disagreement_pixels",0)
        assert disagreement.sum() <= tags.size*.001
        ids = instance[:,:,1].astype(np.uint16) + 256*instance[:,:,2].astype(np.uint16)
        appearance = np.isin(ids, config["reference"]["appearance_actor_ids"])
        assert int(appearance.sum()) == source["appearance_pixels"]
        appearance &= ~disagreement
        with np.load(source_file(camera, source, "depth"), allow_pickle=False) as archive:
            depth = archive["meters"]
        reference, geometry = build_reference(tags, depth, source["camera_to_world"], config, footprint, appearance)
        reference[disagreement] = 255
        if (geometry["ground_height_abs_median_m"] is None or geometry["ground_height_abs_median_m"] > .04
                or geometry["near_map_road_flat_fraction"] < .85):
            raise ValueError("Source geometry failed independent near-ground checks")
        yy = np.minimum(539, ((np.arange(512)+.5)*540/512).astype(int))
        xx = np.minimum(959, ((np.arange(512)+.5)*960/512).astype(int))
        labels = reference[np.ix_(yy,xx)]
        resized_tags, resized_appearance = tags[np.ix_(yy,xx)], appearance[np.ix_(yy,xx)]
        paving = np.isin(ids[np.ix_(yy,xx)], config["streetscape"]["nonroad_actor_ids"])
        known_nonroad = ~np.isin(resized_tags,[0,1,24]) & ~resized_appearance & ~disagreement[np.ix_(yy,xx)]
        assert not (labels[paving] == 1).any() and (labels[known_nonroad] == 0).all()
        road, valid = (labels == 1).astype(np.uint8), (labels != 255).astype(np.uint8)
        if not road.any() or valid.mean() < .85:
            raise ValueError("Empty road mask or insufficient trusted-pixel coverage")
        image = np.array(Image.open(camera / source["rgb_file"]).convert("RGB").resize((512,512), Image.Resampling.BILINEAR))
        files = {"image": ("images",f"Image_{pair_id}.png",image),
                 "mask": ("masks",f"Mask_{pair_id}.png",road),
                 "valid": ("valid",f"Valid_{pair_id}.png",valid),
                 "reference": ("reference",f"Reference_{pair_id}.png",labels)}
        paths, hashes = {}, {}
        for key, (folder,name,pixels) in files.items():
            path = output / "splits" / split / folder / name
            Image.fromarray(pixels).save(path, compress_level=1)
            paths[key] = str(path.relative_to(output)).replace("\\","/")
            hashes[key] = sha(path)
        row = {"id": pair_id, "split": split, **spec, "source_episode": str(episode),
            "source_frame": source["frame"], "sim_s": source["sim_s"], "source": source,
            "source_rgb_pixel_sha256": pixel_sha, "change_from_previous_candidate_mae": movement,
            "files": paths, "sha256": hashes, "source_map_sha256": sha(episode.parent / "map.xodr"),
            "road_pixels": int(road.sum()), "valid_pixels": int(valid.sum()),
            "ignored_pixels": int((valid==0).sum()), "known_nonroad_pixels_checked": int(known_nonroad.sum()),
            "footpath_pixels_checked": int(paving.sum()), "geometry": geometry,
            "native_semantic_counts": source["semantic_counts"]}
        all_rows.append(row)
        for k in ("road_pixels","valid_pixels","ignored_pixels","known_nonroad_pixels_checked","footpath_pixels_checked"):
            totals[k] += row[k]
        if split != "test" and i in (len(selected)//3, 2*len(selected)//3):
            name = f"{spec['episode_id']}_{i:03}.jpg"
            panel(image, labels, f"{split} | {spec['layout']} | {spec['weather']} | {spec['case']} | {source['sim_s']:.1f}s", output / "previews" / name)
            previews.append({"path": name, "episode_id": spec["episode_id"], "split": split})
    return {"episode_id": spec["episode_id"], "layout": spec["layout"], "split": split,
            "source_episode": str(episode), "diversity": diversity, "pixel_totals": dict(totals),
            "native_result": load(episode / "result.json"), "source_capture_config_sha256": sha(camera/"capture_config.json")}


def episode_worker(output,episode,spec,start_id):
    rows,previews=[],[]
    summary=process_episode(output,episode,spec,rows,set(),previews,start_id)
    return rows,summary,previews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--watch", action="store_true", help="Process completed maps while later maps capture")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = load(output / "plan.json")
    if (output / "splits").exists():
        raise FileExistsError("Preserve earlier preparation; use a new dataset version")
    for split in ("train","val","test"):
        for folder in ("images","masks","valid","reference"):
            (output / "splits" / split / folder).mkdir(parents=True,exist_ok=False)
    (output / "previews").mkdir(exist_ok=False)
    rows, summaries, previews, pixel_hashes, done = [], [], [], set(), set()
    started = time.monotonic()
    offsets,offset={},0
    for spec in protocol["episodes"]:
        offsets[spec["episode_id"]]=offset
        offset+=spec["pairs_to_select"]
    pending,submitted={},set()
    with ThreadPoolExecutor(max_workers=3,thread_name_prefix="simulation-preprocess") as executor:
        while len(done) < len(protocol["episodes"]):
            capture = load(output / "capture.json")
            for episode_name in capture.get("episodes",[]):
                episode = Path(episode_name)
                spec = load(episode / "dataset_episode.json")
                if spec["episode_id"] in submitted:
                    continue
                expected = next(e for e in protocol["episodes"] if e["episode_id"] == spec["episode_id"])
                assert expected == spec
                future=executor.submit(episode_worker,output,episode,spec,offsets[spec["episode_id"]])
                pending[future]=spec["episode_id"]
                submitted.add(spec["episode_id"])
            for future in list(pending):
                if not future.done():
                    continue
                episode_id=pending.pop(future)
                added,summary,pictures=future.result()
                hashes={r["source_rgb_pixel_sha256"] for r in added}
                assert not hashes.intersection(pixel_hashes),"Exact RGB duplicate crossed independently processed episodes"
                pixel_hashes.update(hashes)
                rows.extend(added); summaries.append(summary); previews.extend(pictures); done.add(episode_id)
                save(output / "preprocessing_progress.json", {"completed_episodes": len(done), "pairs": len(rows),
                     "latest_episode": episode_id, "elapsed_s": time.monotonic()-started})
                print(json.dumps({"episode": episode_id, "total_pairs": len(rows), "diversity": summary["diversity"]}),flush=True)
            if len(done)==len(protocol["episodes"]):
                break
            if capture["status"]=="failed_preserved" or (not pending and (not args.watch or capture["status"]=="captured")):
                raise RuntimeError("Not all planned healthy captures are available; partial preprocessing remains preserved")
            time.sleep(1)
    rows.sort(key=lambda r:r["id"])
    summaries.sort(key=lambda s:offsets[s["episode_id"]])
    previews.sort(key=lambda p:(offsets[p["episode_id"]],p["path"]))
    counts = dict(Counter(r["split"] for r in rows))
    assert counts == protocol["split_counts"] and len(pixel_hashes)==5000
    manifest = output / "splits/pair_splits.csv"
    with manifest.open("w",newline="",encoding="utf-8") as stream:
        fields = ["id","split","episode_id","layout","case","weather","seed","source_frame"]
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    with (output / "provenance.jsonl").open("w",encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
    save(output / "episode_audit.json", summaries)
    html = '<!doctype html><meta charset="utf-8"><title>Simulation dataset previews</title><style>body{background:#101824;color:white;font:16px sans-serif;max-width:1100px;margin:30px auto}img{width:100%}figure{margin:24px 0}</style><h1>Simulation dataset: train/validation previews</h1><p>RGB | road reference | overlay. White/green = road; black = non-road; magenta = ignore. No model predictions.</p>'
    html += "".join(f'<figure><img src="{p["path"]}" loading="lazy"><figcaption>{p["episode_id"]} / {p["split"]}</figcaption></figure>' for p in previews)
    (output / "previews/index.html").write_text(html,encoding="utf-8")
    totals = Counter()
    for s in summaries:
        totals.update(s["pixel_totals"])
    report = {"status": "prepared_pending_independent_verification", "training_started": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "counts": counts, "pairs": len(rows),
        "episodes": len(done), "layouts": protocol["layouts"], "manifest_sha256": sha(manifest),
        "plan_sha256": sha(output / "plan.json"), "provenance_sha256": sha(output/"provenance.jsonl"),
        "unique_source_rgb_pixel_hashes": len(pixel_hashes), "cross_split_episode_overlap": 0,
        "pixel_totals": dict(totals), "valid_pixel_coverage": totals["valid_pixels"] / (len(rows)*512*512),
        "protected_unchanged": protected() == protocol["protected_sha256"], "test_model_evaluation_performed": False,
        "input_preprocessing": "512x512 bilinear RGB; float32 once to [-1,1] in memory; nearest labels; train-only paired flip + optional existing shadow profile",
        "mask_contract": "masks contain 0/1; valid contains 0/1; reference contains 0/1/255. A mask value at valid=0 is a placeholder and MUST NOT enter any loss or metric.",
        "limits": protocol["limits"], "previews": "previews/index.html", "elapsed_s": time.monotonic()-started}
    assert report["protected_unchanged"]
    save(output / "dataset_report.json", report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
