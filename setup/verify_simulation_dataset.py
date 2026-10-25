"""Independent saved-file/geometry audit; never predicts, trains or selects a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter,defaultdict

import carla
import numpy as np
import pygame

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"setup"))
from prepare_simulation_dataset import DEFAULT,sha,save,protected


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pixels(path):
    return pygame.surfarray.array3d(pygame.image.load(str(path))).swapaxes(0,1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT)
    args=parser.parse_args()
    output=args.output.resolve()
    protocol,report=load(output/"plan.json"),load(output/"dataset_report.json")
    rows=[json.loads(s) for s in (output/"provenance.jsonl").read_text().splitlines()]
    with (output/"splits/pair_splits.csv").open(newline="",encoding="utf-8") as stream:
        manifest=list(csv.DictReader(stream))
    assert len(rows)==len(manifest)==5000
    assert report["manifest_sha256"]==sha(output/"splits/pair_splits.csv")
    assert report["provenance_sha256"]==sha(output/"provenance.jsonl")
    counts=Counter(); groups=defaultdict(set); seeds=defaultdict(set); maps={}; totals=Counter()
    rgb_hashes=set(); raw_hashes=set(); geometry_points=0; map_failures=0; native_checks=0
    episode_config={}
    features=[]; splits=[]; previous={}; low_change_selected=0
    minimum_contrast=255.; brightness=[]
    minimum_day_sky=255.
    for i,row in enumerate(rows):
        assert row["id"]==int(manifest[i]["id"])==i and row["split"]==manifest[i]["split"]
        split=row["split"]
        sky_median=row["source"].get("sky_rgb_median")
        if row["weather"]!="dusk" and row["source"].get("sky_pixels",0)>100:
            assert sky_median>=10,"A black daytime sky entered the selected dataset"
            minimum_day_sky=min(minimum_day_sky,sky_median)
        counts[split]+=1; groups[row["episode_id"]].add(split); seeds[row["seed"]].add(split)
        arrays={}
        for kind,relative in row["files"].items():
            path=output/relative
            assert path.resolve().is_relative_to(output) and sha(path)==row["sha256"][kind]
            arrays[kind]=pixels(path)
        image=arrays["image"]
        mask,valid,target=[arrays[k][:,:,0] for k in ("mask","valid","reference")]
        assert image.shape==(512,512,3) and mask.shape==valid.shape==target.shape==(512,512)
        contrast=float(image.std())
        assert contrast>=3. and int(image.max())-int(image.min())>=20,"Blank or nearly constant RGB capture"
        minimum_contrast=min(minimum_contrast,contrast)
        brightness.append(float(image.mean()))
        assert set(np.unique(target)) <= {0,1,255}
        assert set(np.unique(mask)) <= {0,1} and set(np.unique(valid)) <= {0,1}
        assert np.array_equal(mask,(target==1).astype(np.uint8))
        assert np.array_equal(valid,(target!=255).astype(np.uint8))
        assert int(mask.sum())==row["road_pixels"] and int(valid.sum())==row["valid_pixels"]
        for k in ("road_pixels","valid_pixels","ignored_pixels","known_nonroad_pixels_checked","footpath_pixels_checked"):
            totals[k]+=row[k]
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        assert digest not in rgb_hashes, "Duplicate resized image, including across splits"
        rgb_hashes.add(digest)
        assert row["source_rgb_pixel_sha256"] not in raw_hashes
        raw_hashes.add(row["source_rgb_pixel_sha256"])
        # Tiny RGB descriptor: adjacent-frame and cross-split similarity diagnostics,
        # never a substitute for episode grouping or a claim of independent cities.
        thumb=image.reshape(16,32,16,32,3).mean(axis=(1,3)).astype(np.float32)
        descriptor=thumb.reshape(-1)
        features.append(descriptor); splits.append(split)
        key=row["episode_id"]
        if key in previous:
            low_change_selected+=float(np.abs(descriptor-previous[key]).mean())<.75
        previous[key]=descriptor
        episode=Path(row["source_episode"])
        if key not in episode_config:
            config=load(episode/"road_camera/capture_config.json")
            assert np.isclose(config["actual_weather"]["scattering_intensity"],1.)
            assert np.isclose(config["actual_weather"]["mie_scattering_scale"],.03)
            assert load(episode/"dataset_episode.json")==next(e for e in protocol["episodes"] if e["episode_id"]==key)
            assert config["camera_profile"]==row["camera_profile"]
            map_path=episode.parent/"map.xodr"
            assert sha(map_path)==row["source_map_sha256"]
            maps[row["layout"]]=carla.Map(row["layout"],map_path.read_text())
            episode_config[key]=config
        # A separate native-depth projection and CARLA lane query checks road labels
        # without calling the reference-builder implementation.
        if i%25==0:
            config=episode_config[key]; camera=episode/"road_camera"; source=row["source"]
            tags=pixels(camera/source["labels_file"])[:,:,0]
            instance=pixels(camera/source["instance_file"])
            ids=instance[:,:,1].astype(np.uint16)+256*instance[:,:,2].astype(np.uint16)
            yy=np.minimum(539,((np.arange(512)+.5)*540/512).astype(int))
            xx=np.minimum(959,((np.arange(512)+.5)*960/512).astype(int))
            source_tags=tags[np.ix_(yy,xx)]
            source_ids=ids[np.ix_(yy,xx)]
            asphalt=np.isin(source_ids,config["reference"]["appearance_actor_ids"])
            known=~np.isin(source_tags,[0,1,24]) & ~asphalt
            disagreement=(instance[:,:,0]!=tags)[np.ix_(yy,xx)]
            assert (target[known & ~disagreement]==0).all()
            assert (target[disagreement]==255).all()
            paving=np.isin(source_ids,config["streetscape"]["nonroad_actor_ids"])
            assert not (target[paving]==1).any()
            native_checks+=int((known & ~disagreement).sum())
            with np.load(camera/source["depth_file"],allow_pickle=False) as archive:
                depth=archive["meters"]
            k=np.array(config["intrinsics_source"]); transform=np.array(source["camera_to_world"])
            ys,xs=np.where(target==1)
            for n in np.linspace(0,len(xs)-1,12,dtype=int):
                y,x=yy[ys[n]],xx[xs[n]]; d=depth[y,x]
                point=transform@np.array([d,(x-k[0,2])*d/k[0,0],-(y-k[1,2])*d/k[1,1],1.])
                assert abs(point[2])<.09
                wp=maps[row["layout"]].get_waypoint(carla.Location(x=float(point[0]),y=float(point[1]),z=float(point[2])),False,carla.LaneType.Driving)
                geometry_points+=1; map_failures+=wp is None
        if (i+1)%500==0:
            print(json.dumps({"independently_checked_pairs":i+1}),flush=True)
    assert dict(counts)==protocol["split_counts"]
    assert len(groups)==len(seeds)==len(protocol["episodes"]) and all(len(s)==1 for s in groups.values()) and all(len(s)==1 for s in seeds.values())
    assert len(maps)==4 and map_failures/geometry_points < .01
    assert dict(totals)==report["pixel_totals"]
    assert protected()==protocol["protected_sha256"]
    features=np.array(features,dtype=np.float32)
    # Closest training view for held-out frames, measured by thumbnail RMS pixel
    # difference. Report, do not silently reshuffle held-out episodes after capture.
    train=features[np.array(splits)=="train"]
    nearest={}
    for split in ("val","test"):
        held=features[np.array(splits)==split]
        distances=[]
        for block in np.array_split(held,10):
            d=(np.sum(block*block,axis=1)[:,None]+np.sum(train*train,axis=1)[None,:]-2*block@train.T)/features.shape[1]
            distances.extend(np.sqrt(np.maximum(0,d.min(axis=1))).tolist())
        nearest[split]={"minimum_rmse_0_255":float(np.min(distances)),"median_rmse_0_255":float(np.median(distances)),
                        "pairs_below_rmse_2":int(np.sum(np.array(distances)<2.))}
    verification={"passed":True,"pairs_recounted":5000,"unique_resized_rgb_images":5000,"unique_source_rgb_images":5000,
        "episode_overlap_across_splits":0,"seed_overlap_across_splits":0,"distinct_map_topologies":4,
        "projected_road_points":geometry_points,"points_outside_driving_lanes":map_failures,
        "sampled_native_nonroad_pixels_checked":native_checks,"selected_consecutive_low_change_pairs":int(low_change_selected),
        "cross_split_thumbnail_similarity":nearest,"protected_unchanged":True,"training_started":False,
        "blank_rgb_frames":0,"minimum_rgb_standard_deviation":minimum_contrast,
        "invalid_black_daytime_sky_frames":0,"minimum_daytime_sky_median":minimum_day_sky,
        "image_brightness_mean_range":[min(brightness),max(brightness)],
        "test_read_for_file_and_label_qa_only":True,"model_predictions_or_test_model_scores":False}
    save(output/"independent_verification.json",verification)
    print(json.dumps(verification,indent=2))


if __name__=="__main__":
    main()
