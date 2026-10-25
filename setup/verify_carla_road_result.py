"""Recount saved predictions and check depth-projected road points against OpenDRIVE."""
import argparse
import hashlib
import json
from pathlib import Path

import carla
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def channel(path):
    image = pygame.image.load(str(path))
    return pygame.surfarray.array3d(image)[:,:,0].T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    args = parser.parse_args()
    report = load(args.output / "report.json")
    assert report["status"] == "completed" and report["model_control"] is False
    assert report["model_weights_unchanged"] and report["protected_files_unchanged"]
    sums, checked, point_checks, mismatches, known_occlusions, appearance_points = [0,0,0,0,0,0],0,0,0,0,0
    footpath_pixels = 0
    for episode, capture in zip(report["episodes"], args.capture, strict=True):
        road_map = carla.Map("saved_observation_map", (capture.parent / "map.xodr").read_text())
        rows = episode["rows"]
        assert rows == load(args.output / episode["name"] / "frames.json")
        for i,row in enumerate(rows):
            prediction, target = channel(args.output/row["mask"]), channel(args.output/row["reference"])
            assert prediction.shape == target.shape == (512,512)
            assert set(np.unique(prediction)) <= {0,1} and set(np.unique(target)) <= {0,1,255}
            assert hashlib.sha256(prediction.tobytes()).hexdigest() == row["mask_sha256"]
            good, truth, guessed = target != 255, target == 1, prediction == 1
            counts = [int(np.sum(good & truth & guessed)),int(np.sum(good & ~truth & guessed)),
                int(np.sum(good & truth & ~guessed)),int(np.sum(good & ~truth & ~guessed)),int(np.sum(~good)),target.size]
            assert counts == [row["metrics"][k] for k in ("tp","fp","fn","tn","ignored","total")]
            sums = [a+b for a,b in zip(sums,counts)]
            checked += 1
            if i % 25:
                continue
            source = row["source"]
            tags = channel(capture / "road_camera" / source["labels_file"])
            appearance = np.zeros(tags.shape, bool)
            if episode["camera_config"]["schema_version"] == 3:
                path = capture / "road_camera" / source["instance_file"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == source["instance_sha256"]
                rgb = pygame.surfarray.array3d(pygame.image.load(str(path))).swapaxes(0,1)
                disagreement = rgb[:,:,0] != tags
                different = int(disagreement.sum())
                assert different == source.get("semantic_instance_disagreement_pixels",0)
                if different:
                    assert episode["camera_config"].get("dataset_edge_disagreement_policy") == (
                        "At most 0.1% semantic/instance disagreement; preserve both native streams and ignore every disagreeing pixel")
                    assert different <= tags.size*.001
                ids = rgb[:,:,1].astype(np.uint16) + 256*rgb[:,:,2].astype(np.uint16)
                appearance = np.isin(ids, episode["camera_config"]["reference"]["appearance_actor_ids"])
                assert int(appearance.sum()) == source["appearance_pixels"]
                appearance &= ~disagreement
                yy = np.minimum(539, ((np.arange(512)+.5)*540/512).astype(int))
                xx = np.minimum(959, ((np.arange(512)+.5)*960/512).astype(int))
                assert (target[disagreement[np.ix_(yy,xx)]] == 255).all()
                assert (tags[appearance] == 3).all()
                paving_ids = episode["camera_config"].get("streetscape", {}).get("nonroad_actor_ids", [])
                if paving_ids:
                    assert not set(paving_ids).intersection(episode["camera_config"]["reference"]["appearance_actor_ids"])
                    # Match the exact nearest-neighbour label resize. Visible
                    # footpath instances must never become positive driving road.
                    yy = np.minimum(539, ((np.arange(512)+.5)*540/512).astype(int))
                    xx = np.minimum(959, ((np.arange(512)+.5)*960/512).astype(int))
                    paving = np.isin(ids[np.ix_(yy,xx)], paving_ids)
                    assert not (target[paving] == 1).any(), "Footpath was labeled as driving road"
                    footpath_pixels += int(paving.sum())
            with np.load(capture / "road_camera" / source["depth_file"],allow_pickle=False) as value:
                depth=value["meters"]
            k=np.array(episode["camera_config"]["intrinsics_source"])
            transform=np.array(source["camera_to_world"])
            ys,xs=np.where(target==1)
            for n in np.linspace(0,len(xs)-1,min(40,len(xs)),dtype=int):
                y,x=int((ys[n]+.5)*540/512),int((xs[n]+.5)*960/512)
                z=depth[y,x]
                ray=np.array([z,(x-k[0,2])*z/k[0,0],-(y-k[1,2])*z/k[1,1],1])
                point=transform@ray
                assert tags[y,x] in (0,1,24) or appearance[y,x], "An unrelated known actor was labeled road"
                appearance_points += int(appearance[y,x])
                assert abs(point[2]) < .09, "A road reference point is not on the flat road surface"
                waypoint=road_map.get_waypoint(carla.Location(x=float(point[0]),y=float(point[1]),z=float(point[2])),False,carla.LaneType.Driving)
                point_checks+=1
                if waypoint is None:
                    mismatches+=1
            for tag in (3,12,13,14,15,16,18,19):
                yy,xx=np.where((tags==tag) & ~appearance)
                if len(xx):
                    # The nearest-label resize must preserve non-road for known actors.
                    sy=np.minimum(511,((yy+.5)*512/540).astype(int))
                    sx=np.minimum(511,((xx+.5)*512/960).astype(int))
                    exact_y=np.minimum(539,((sy+.5)*540/512).astype(int))
                    exact_x=np.minimum(959,((sx+.5)*960/512).astype(int))
                    exact=(tags[exact_y,exact_x]==tag) & ~appearance[exact_y,exact_x]
                    assert not (target[sy[exact],sx[exact]]==1).any()
                    known_occlusions+=int(exact.sum())
        for example in (args.output/episode["name"]/"examples").glob("*_probability.npy"):
            mask=channel(args.output/episode["name"]/"masks_01"/(example.stem.replace('_probability','')+'.png'))
            probability=np.load(example,allow_pickle=False)
            assert probability.dtype==np.float32 and np.isfinite(probability).all()
            assert np.array_equal(mask,(probability>=.5).astype(np.uint8))
    assert sums == [report["metrics"][k] for k in ("tp","fp","fn","tn","ignored","total")]
    assert checked and point_checks and mismatches / point_checks < .01
    result={"passed":True,"frames_recounted":checked,"projected_road_points":point_checks,
        "points_outside_opendrive_driving_lanes":mismatches,"known_actor_pixels_checked":known_occlusions,
        "owned_appearance_road_points_checked":appearance_points,
        "raised_footpath_pixels_checked_nonroad":footpath_pixels,
        "mask_values_0_1":True,"selected_probability_thresholds_verified":True,
        "method":"Independent saved-PNG confusion recount, sampled depth-to-map query and known-actor occlusion checks"}
    (args.output/"independent_verification.json").write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
