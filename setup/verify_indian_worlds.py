"""Independent map, captured-stream, actor-motion and protection checks."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET

import carla
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"integration"))
from india_world_maps import PROFILES,build_world,validate_world,road_specs
from india_world_traffic import routes_for


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offline(folder):
    folder.mkdir(parents=True,exist_ok=False)
    results = []
    for key,profile in PROFILES.items():
        xml = build_world(key)
        native = carla.Map("independent_"+key,xml)
        result = validate_world(key,xml)
        routes = routes_for(key,xml,native)
        max_gap = 0.
        for route in routes.values():
            gaps = np.linalg.norm(np.diff(route.points[:,:2],axis=0),axis=1)
            assert gaps.max() < 1.,key
            max_gap = max(max_gap,float(gaps.max()))
            for index in range(2,len(route.points)-2,11):
                p = route.points[index]
                wp = native.get_waypoint(carla.Location(x=p[0],y=p[1],z=0),False,carla.LaneType.Driving)
                assert wp is not None,(key,index,"route outside native road")
                delta = route.points[index+1,:2]-route.points[index-1,:2]
                error = abs(math.remainder(math.atan2(delta[1],delta[0])-p[2],2*math.pi))
                assert error < .15,(key,index,"heading error",error)
        # Widths remain positive, medians/shoulders are distinct native lane types.
        tree = ET.fromstring(xml)
        for road in tree.findall("road"):
            for lane in road.findall("lanes/laneSection/*/lane"):
                if lane.get("type") == "driving":
                    width = lane.find("width")
                    for s in np.linspace(0,float(road.get("length")),101):
                        w = sum(float(width.get(c))*s**i for i,c in enumerate("abcd"))
                        assert 2.4 < w < 4.,(key,"invalid lane width",w)
        if profile.median:
            assert len(tree.findall("road/lanes/laneSection/*/lane[@type='median']")) == 2
        if profile.shoulder:
            assert len(tree.findall("road/lanes/laneSection/*/lane[@type='shoulder']")) == 2
            assert not tree.findall("road/lanes/laneSection/*/lane[@type='sidewalk']")
        result.update(route_count=len(routes),max_route_point_gap_m=max_gap,independent_heading_check=True)
        (folder/(key+".xodr")).write_text(xml)
        results.append(result)
    assert len({r["sha256"] for r in results}) == 5
    write(folder/"verification.json",{"passed":True,"worlds":results,"all_lht":True})
    return results


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def native(session):
    checks = []
    result = json.loads((session/"result.json").read_text())
    for world_result in result["worlds"]:
        key = world_result["profile"]
        folder = session/key
        rows = [json.loads(line) for line in (folder/"camera_frames.jsonl").read_text().splitlines()]
        assert rows and len(rows) == world_result["paired_frames"]
        frames,timestamps,visible_ids = [],[],set()
        mismatch_count,total_pixels,checked_images = 0,0,0
        for row in rows:
            frames.append(row["source_frame"])
            captures = row["streams"]
            assert len(captures) == 5
            assert {c["frame"] for c in captures.values()} == {row["source_frame"]}
            assert len({c["timestamp"] for c in captures.values()}) == 1
            timestamps.append(captures["rgb"]["timestamp"])
            arrays = {}
            for kind,capture in captures.items():
                path = (folder/capture["file"]).resolve()
                assert path.is_relative_to(folder.resolve()) and fingerprint(path) == capture["sha256"]
                image = pygame.image.load(str(path))
                assert image.get_size() == ((640,480) if kind == "overview" else (960,540))
                checked_images += 1
                if kind in ("semantic_segmentation","instance_segmentation","depth"):
                    arrays[kind] = pygame.surfarray.array3d(image).swapaxes(0,1)
            tags = arrays["semantic_segmentation"][...,0]
            instance = arrays["instance_segmentation"]
            assert tags.max() <= 28
            difference = int(np.count_nonzero(tags != instance[...,0]))
            mismatch_count += difference
            total_pixels += tags.size
            assert difference < tags.size*.001,(key,"semantic/instance misalignment")
            ids = instance[...,1].astype(np.uint16)+256*instance[...,2].astype(np.uint16)
            visible_ids.update(map(int,np.unique(ids)))
            depth = arrays["depth"]
            decoded = (depth[...,0].astype(float)+depth[...,1].astype(float)*256+depth[...,2].astype(float)*65536)/16777215*1000
            assert np.isfinite(decoded).all() and decoded.min() >= 0 and decoded.max() <= 1000
        assert frames == sorted(set(frames))
        expected_step = 1./world_result.get("saved_capture_hz",5)
        assert all(abs(b-a-expected_step) < .001 for a,b in zip(timestamps,timestamps[1:])),(key,"capture gap")
        traces = [json.loads(line) for line in (folder/"actor_trace.jsonl").read_text().splitlines()]
        assert len(traces) == world_result["ticks"]
        first,last = traces[0]["actors"],traces[-1]["actors"]
        displacements = {a["name"]:math.hypot(b["x"]-a["x"],b["y"]-a["y"]) for a,b in zip(first,last)}
        assert displacements["ego"] > 4.
        assert any(abs(a["steer"]) > .01 and a["throttle"] > 0 for row in traces for a in row["actors"])
        assert not world_result["collisions"] and not world_result["body_overlaps"]
        assert not world_result["road_departures"] and not world_result["wrong_way"]
        assert world_result["cleanup_complete"] and world_result["passed"]
        actors = json.loads((folder/"actors.json").read_text())["actors"]
        assert len([a for a in actors if a["blueprint"].startswith("vehicle.")]) == 7
        assert {"car","motorcycle","bus","other_vehicle","bicycle"} <= {a["project_class"] for a in actors}
        scenery = json.loads((folder/"scenery.json").read_text())
        assert not scenery["raw_semantics_are_training_ready"]
        asphalt = set(scenery["surface"]["asphalt_actor_ids"])
        solids = {a["actor_id"] for a in scenery["objects"] if a.get("collisions_enabled")}
        assert not asphalt & solids
        signs = [a for a in scenery["objects"] if a.get("kind") == "indian_sign"]
        assert signs and all(a["project_class"] == "traffic_sign" and a["project_label_requires_instance_override"] for a in signs)
        checks.append({"profile":key,"passed":True,"frames":len(rows),"images_reopened_hashed":checked_images,
            "semantic_instance_disagreement_pixels":mismatch_count,"compared_semantic_pixels":total_pixels,
            "visible_native_ids":len(visible_ids),"native_signs":len(signs),"ego_displacement_m":displacements["ego"],
            "all_displacements_m":displacements,"collision_events":0,"road_departures":0,"wrong_way_events":0,
            "model_control":False,"raw_streams_training_ready":False})
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--session",type=Path)
    p.add_argument("--protection",type=Path)
    args = p.parse_args()
    folder = args.output.resolve()
    if not folder.is_relative_to(ROOT/"runs"):
        raise ValueError("New run evidence directory required")
    maps = offline(folder/"maps")
    checks = native(args.session.resolve()) if args.session else []
    protection = {}
    if args.protection:
        before = json.loads(args.protection.read_text())
        protection = {name:fingerprint(ROOT/name) == digest for name,digest in before.items()}
        assert all(protection.values())
    report = {"passed":True,"maps":maps,"native":checks,"protected_files_unchanged":protection,
        "native_session":str(args.session) if args.session else None}
    write(folder/"verification.json",report)
    print(json.dumps({"passed":True,"maps":len(maps),"native_worlds":len(checks),"protected_files":len(protection)}))


if __name__ == "__main__":
    main()
