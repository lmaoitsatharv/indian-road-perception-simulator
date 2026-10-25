"""Independent continuity, identity, geometry, camera and asset-label audit."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import carla
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Reuse the workspace's existing Pillow read-only. The CARLA environment keeps
# its own earlier NumPy/CARLA imports and no package installation is performed.
try:
    from PIL import Image
except ModuleNotFoundError:
    sys.path.append(str(ROOT/".venv/Lib/site-packages"))
    from PIL import Image
sys.path.insert(0,str(ROOT/"integration"))
from india_combined_map import build_combined,validate_combined

ASSET_LABELS = {"asphalt":1,"indian_sign":8,"shop_board":3,"ground":10,"median":2}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rectangle(row):
    c,s = math.cos(row["yaw"]),math.sin(row["yaw"])
    x,y,a,b = row["x"],row["y"],row["ex"],row["ey"]
    return [(x+u*c-v*s,y+u*s+v*c) for u,v in ((-a,-b),(a,-b),(a,b),(-a,b))]


def overlap(a,b):
    # Separate implementation using projections on unit-length normals.
    for polygon in (a,b):
        for p,q in zip(polygon,polygon[1:]+polygon[:1]):
            axis = (q[1]-p[1],p[0]-q[0])
            length = math.hypot(*axis)
            if length < 1e-9:
                continue
            axis = axis[0]/length,axis[1]/length
            pa = [x*axis[0]+y*axis[1] for x,y in a]
            pb = [x*axis[0]+y*axis[1] for x,y in b]
            if max(pa) <= min(pb)+1e-7 or max(pb) <= min(pa)+1e-7:
                return False
    return True


def audit(session,output,protection,require_tour):
    output.mkdir(parents=True,exist_ok=False)
    config = json.loads((session/"config.json").read_text())
    result = json.loads((session/"result.json").read_text())
    assert result["passed"] and result["cleanup_complete"],"Native run failed"
    assert result["world_generate_calls"] == 1 and len(result["world_ids"]) == 1
    assert result["model_control"] is False and result["vehicle_pose_teleport"] is False
    xml,network = build_combined()
    assert (session/"map.xodr").read_text() == xml
    maps = validate_combined(xml,network)
    actors = json.loads((session/"actors.json").read_text())["actors"]
    expected = {a["actor_id"] for a in actors if a["blueprint"].startswith("vehicle.")}
    assert len(expected) == result["native_vehicles"] == 25
    assert len({a["actor_id"] for a in actors}) == len(actors)
    bus_ids = {a["actor_id"] for a in actors if a["project_class"] == "bus"}
    assert len(bus_ids) == 1
    map_native = carla.Map("independent_town_trace_audit",xml)
    visited = defaultdict(set)
    seen_worlds,seen_frames = set(),set()
    previous,trace_rows,max_step,geometry_pairs = {},0,0.,0
    route_distance_first,route_distance_last = {},{}
    for line in (session/"actor_trace.jsonl").open():
        row = json.loads(line)
        seen_worlds.add(row["world_id"])
        assert row["frame"] not in seen_frames
        seen_frames.add(row["frame"])
        states = row["actors"]
        assert {a["id"] for a in states} == expected,"Vehicle replaced/disappeared"
        for a in states:
            assert map_native.get_waypoint(carla.Location(x=a["x"],y=a["y"],z=0),False,carla.LaneType.Driving) is not None
            if a["id"] in bus_ids:
                street = next(s for s in network["streets"] if s["id"] == a["road_id"])
                assert street["divided"] and a["lane"] == 3,"Oversized bus entered a narrow road"
            visited[a["name"]].add(a["district"])
            route_distance_first.setdefault(a["name"],a["distance_m"])
            route_distance_last[a["name"]] = a["distance_m"]
            if a["id"] in previous:
                old = previous[a["id"]]
                step = math.hypot(a["x"]-old["x"],a["y"]-old["y"])
                assert step < 1.3,(a["name"],"Position discontinuity",step)
                max_step = max(max_step,step)
                assert a["distance_m"] >= old["distance_m"]-1e-8
            previous[a["id"]] = a
            if a["speed"] > .3:
                assert abs(math.remainder(a["yaw"]-a["expected_yaw"],2*math.pi)) < math.radians(70)
        polygons = [rectangle(a) for a in states]
        for i,a in enumerate(states):
            for j in range(i+1,len(states)):
                b = states[j]
                if math.hypot(a["x"]-b["x"],a["y"]-b["y"]) < 17:
                    assert not overlap(polygons[i],polygons[j]),(row["frame"],a["name"],b["name"],"Independent overlap")
                    geometry_pairs += 1
        trace_rows += 1
    assert seen_worlds == set(result["world_ids"])
    assert trace_rows == result["ticks"]
    assert {k:sorted(v) for k,v in visited.items()} == result["districts_visited"]
    assert not result["collisions"] and not result["vehicle_centre_departures"]
    assert len(visited["ego"]) == 5 if require_tour else len(visited["ego"]) >= 1
    scene = json.loads((session/"scenery.json").read_text())
    assert len(scene["districts_present"]) == 5
    lookup = np.full(65536,255,dtype=np.uint8)
    kinds = defaultdict(list)
    for item in scene["objects"]:
        kind = item.get("kind")
        if kind in ASSET_LABELS and "actor_id" in item:
            actor_id = item["actor_id"]
            assert 0 < actor_id < 65536
            assert lookup[actor_id] == 255
            lookup[actor_id] = ASSET_LABELS[kind]
            kinds[kind].append(actor_id)
    masks = output/"asset_masks"
    masks.mkdir()
    image_count,sets,disagreement_pixels,total_pixels = 0,0,0,0
    native_tags,overrides = Counter(),Counter()
    last_time,last_frame = None,None
    pair_manifest = (output/"label_pairs.jsonl").open("w",encoding="utf-8")
    for line in (session/"camera_frames.jsonl").open():
        row = json.loads(line)
        assert row["world_id"] in seen_worlds
        streams = row["streams"]
        assert len(streams) == 5
        assert {x["frame"] for x in streams.values()} == {row["source_frame"]}
        times = {x["timestamp"] for x in streams.values()}
        assert len(times) == 1
        timestamp = next(iter(times))
        if last_time is not None:
            assert abs(timestamp-last_time-1.) < .002
            assert row["source_frame"]-last_frame == 20
        last_time,last_frame = timestamp,row["source_frame"]
        arrays = {}
        for name,capture in streams.items():
            path = (session/capture["file"]).resolve()
            assert path.is_relative_to(session) and sha(path) == capture["sha256"]
            with Image.open(path) as image:
                assert image.size == (960,540)
                if name in ("semantic_segmentation","instance_segmentation","depth"):
                    arrays[name] = np.asarray(image).copy()
            image_count += 1
        tags = arrays["semantic_segmentation"][...,0]
        instance = arrays["instance_segmentation"]
        assert tags.max() <= 28
        disagreement = tags != instance[...,0]
        count = int(disagreement.sum())
        assert count <= tags.size*.001
        disagreement_pixels += count
        total_pixels += tags.size
        ids = instance[...,1].astype(np.uint16)+256*instance[...,2].astype(np.uint16)
        depth = arrays["depth"]
        metres = (depth[...,0].astype(float)+256*depth[...,1].astype(float)+65536*depth[...,2].astype(float))/16777215*1000
        assert np.isfinite(metres).all() and metres.min() >= 0 and metres.max() <= 1000
        mask = tags.copy()
        mask[mask == 0] = 255
        mapped = lookup[ids]
        owned = mapped != 255
        mask[owned] = mapped[owned]
        mask[disagreement] = 255
        for kind,actor_ids in kinds.items():
            pixels = np.isin(ids,actor_ids)&~disagreement
            assert np.all(mask[pixels] == ASSET_LABELS[kind])
            overrides[kind] += int(pixels.sum())
        retained = ~owned & ~disagreement & (tags != 0)
        assert np.array_equal(mask[retained],tags[retained])
        path = masks/f"{row['source_frame']:08d}.png"
        Image.fromarray(mask).save(path)
        assert np.array_equal(np.asarray(Image.open(path)),mask)
        unique,counts = np.unique(tags,return_counts=True)
        native_tags.update({int(k):int(v) for k,v in zip(unique,counts)})
        pair_manifest.write(json.dumps({"frame":row["source_frame"],"mask":str(path.relative_to(output)),"sha256":sha(path),
            "rgb":str((session/streams["rgb"]["file"]).relative_to(ROOT)),"world_id":row["world_id"],"district":row["district"]})+"\n")
        sets += 1
    pair_manifest.close()
    assert sets == result["saved_camera_sets"] and overrides["indian_sign"] > 0 and overrides["asphalt"] > 0
    protected = json.loads(protection.read_text())
    protection_result = {name:sha(ROOT/name) == digest for name,digest in protected.items()}
    assert all(protection_result.values())
    proof = {"passed":True,"session":str(session.relative_to(ROOT)),"single_native_world":True,
        "same_vehicle_ids_every_tick":True,"world_ids":sorted(seen_worlds),"map":maps,
        "physics_trace_rows":trace_rows,"independent_vehicle_body_pairs":geometry_pairs,"max_step_distance_m":max_step,
        "ego_visited_all_five":len(visited["ego"]) == 5,"ego_districts":sorted(visited["ego"]),
        "ego_distance_m":route_distance_last["ego"]-route_distance_first["ego"],
        "vehicle_count":25,"walker_count":result["native_walkers"],"camera_sets":sets,"hashed_reopened_images":image_count,
        "wide_bus_stayed_on_divided_corridor":True,
        "native_semantic_pixels":total_pixels,"semantic_instance_edge_disagreements":disagreement_pixels,
        "asset_label_pixel_counts":dict(overrides),"native_tag_counts":dict(native_tags),
        "native_sensor_evidence_unchanged":True,"asset_label_masks_training_allowed":False,
        "protected_files_unchanged":protection_result,
        "limits":["Illustrative traffic and generic assets; no live autorickshaw",
            "Vehicle-body checks are sampled, not a continuous safety proof; sidewalk people not independently body-audited",
            "Asset masks are inspected development previews; no full depth/map road-edge label audit or dataset split",
            "Completed download verification is independent of this town and does not mean the detection archive was extracted"]}
    (output/"verification.json").write_text(json.dumps(proof,indent=2)+"\n")
    print(json.dumps({"passed":True,"single_world":True,"ego_all_five":proof["ego_visited_all_five"],
        "camera_sets":sets,"images":image_count,"trace_rows":trace_rows,"protected_files":len(protected)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--protection",type=Path,required=True)
    p.add_argument("--require-tour",action="store_true")
    args = p.parse_args()
    session,output = args.session.resolve(),args.output.resolve()
    if not session.is_relative_to(ROOT/"runs") or not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Keep native evidence under runs/")
    audit(session,output,args.protection,args.require_tour)
