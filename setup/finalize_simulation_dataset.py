"""Publish preparation evidence only after both independent and TensorFlow checks pass."""
from collections import Counter
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from prepare_simulation_dataset import DEFAULT,ROOT,save,sha,protected


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT)
    output=parser.parse_args().output.resolve()
    if not output.is_relative_to(ROOT/"dataset"):
        raise ValueError("Expected a dataset version inside this project")
    dataset_relative=output.relative_to(ROOT).as_posix()
    report=load(output/"dataset_report.json")
    verification=load(output/"independent_verification.json")
    preprocessing=load(output/"preprocessing_check.json")
    plan=load(output/"plan.json")
    capture=load(output/"capture.json")
    map_preflight=load(output/"map_preflight.json")
    assert map_preflight["passed"] and map_preflight["distinct_map_hashes"]==4
    assert capture["status"]=="captured" and verification["passed"] and preprocessing["passed"]
    assert protected()==plan["protected_sha256"]
    rows=[json.loads(s) for s in (output/"provenance.jsonl").read_text().splitlines()]
    breakdown={key:dict(Counter(r[key] for r in rows)) for key in ("split","layout","weather","case")}
    cases_by_split={split:dict(Counter(r["case"] for r in rows if r["split"]==split)) for split in ("train","val","test")}
    visible={name:sum(sum(r["native_semantic_counts"][i] for i in ids)>10 for r in rows)
             for name,ids in {"pedestrian_or_rider":[12,13],"bus":[16],"motorcycle":[18],"bicycle":[19]}.items()}
    episode_audit=load(output/"episode_audit.json")
    assert len(episode_audit)==32 and all(e["native_result"]["runtime_passed"] for e in episode_audit)
    actor_counts=[]
    for episode in episode_audit:
        actors=load(Path(episode["source_episode"])/"actors.json")
        actor_counts.append({"episode_id":episode["episode_id"],"layout":episode["layout"],"split":episode["split"],
            "other_vehicles":len(actors["vehicles"])-1,"pedestrians":len(actors["pedestrians"])})
    source_folder=output/"source"
    source_folder.mkdir(exist_ok=True)
    source_names=["simulation_data.py","carla_road_reference.py","idd_data.py","idd_experiments.py",
                  "setup/prepare_simulation_dataset.py","setup/build_simulation_dataset.py",
                  "setup/verify_simulation_dataset.py","setup/check_simulation_preprocessing.py",
                  "setup/finalize_simulation_dataset.py"]
    source_hashes={}
    for name in source_names:
        target=source_folder/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
        source_hashes[name]=sha(target)
    # Illustrate actual saved map footprints and separate native camera examples.
    # These overview pictures are documentation, never training examples.
    from build_simulation_dataset import panel
    overview=Image.new("RGB",(1000,1060),"#101824")
    cameras=Image.new("RGB",(1536,580),"#101824")
    draw=ImageDraw.Draw(overview)
    try:
        font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20)
    except OSError:
        font=ImageFont.load_default()
    names={"crossroad":"Crossroads","t_north":"T-junction A","t_south":"T-junction B","straight":"Straight street"}
    for i,layout in enumerate(plan["layouts"]):
        chosen=min((r for r in rows if r["split"]=="train" and r["layout"]==layout and r["case"]=="mixed"),key=lambda r:abs(r["sim_s"]-25))
        camera=Path(chosen["source_episode"])/"road_camera"
        config=load(camera/"capture_config.json")
        grid=config["reference"]["grid"]
        with np.load(camera/grid["file"],allow_pickle=False) as archive:
            footprint=archive["driving"]
        y0,x0=[round((-65-origin)/grid["cell_m"]) for origin in grid["origin_xy_m"][::-1]]
        crop=footprint[y0:y0+1300,x0:x0+1300]
        colors=np.where(crop[...,None].astype(bool),np.array([125,145,165],np.uint8),np.array([27,43,52],np.uint8))
        x,y=(i%2)*500,(i//2)*510+40
        overview.paste(Image.fromarray(colors).resize((460,460),Image.Resampling.NEAREST),(x+20,y+30))
        draw.text((x+20,y),names[layout]+" | left-hand traffic",fill="white",font=font)
        rgb=np.array(Image.open(output/chosen["files"]["image"]).convert("RGB"))
        reference=np.array(Image.open(output/chosen["files"]["reference"]))
        example=output/f"{layout}_overview.jpg"
        panel(rgb,reference,f"{names[layout]} | {chosen['weather']} | {chosen['case']} | {chosen['sim_s']:.1f}s",example)
        cameras.paste(Image.open(example),((i%2)*768,(i//2)*290))
    draw.text((20,12),"Four actual saved road footprints (central 130 m); shared neighbourhood assets",fill="white",font=font)
    overview.save(output/"map_overview.png")
    cameras.save(output/"camera_overview.png")
    report.update(status="ready_for_separate_simulation_training",breakdown=breakdown,
        cases_by_split=cases_by_split,actor_counts_per_episode=actor_counts,
        map_preflight=map_preflight,
        visible_native_class_frames=visible,independent_verification=verification,
        preprocessing_verified=True,source_sha256=source_hashes,
        discarded_exact_duplicate_candidates=sum(e["diversity"]["exact_duplicates_rejected"] for e in episode_audit),
        discarded_low_change_candidates=sum(e["diversity"]["low_change_rejected"] for e in episode_audit))
    save(output/"dataset_report.json",report)
    portable={k:report[k] for k in ("status","training_started","counts","pairs","episodes","layouts",
        "manifest_sha256","unique_source_rgb_pixel_hashes","valid_pixel_coverage","protected_unchanged",
        "breakdown","visible_native_class_frames","discarded_exact_duplicate_candidates","discarded_low_change_candidates")}
    portable.update(independent_verification=verification,map_preflight=map_preflight,preprocessing_verified=True,
        cases_by_split=cases_by_split,actor_counts_per_episode=actor_counts,
        dataset=dataset_relative,guide="docs/SIMULATION_DATASET.md",limits=plan["limits"])
    save(ROOT/"docs/results/simulation_dataset.json",portable)
    coverage=report["valid_pixel_coverage"]*100
    text=f'''# Simulation-only dataset preparation result

Completed preparation: **5,000 unique image/mask pairs**, four custom left-hand
road layouts and 32 fresh episodes. Split: **4,000 train / 500 validation /
500 test**. Training has not started.

All six preparation stages in `docs/SIMULATION_DATASET.md` are complete. The real
model, its inference path, original checkpoints and real split retain their saved
SHA-256 fingerprints. No real test image was opened by this work.

| Map | Pairs |
|---|---:|
'''
    text+=''.join(f'| {key} | {value} |\n' for key,value in breakdown["layout"].items())
    text+=f'''
Cases include mixed traffic, overtaking/slow cycles, oncoming vehicles, crossing
pedestrians, return-gap interactions and junction traffic. Four weather conditions,
sunlight directions, camera profiles, positions, speeds and colours vary.

All 5,000 resized RGB images and source RGB images are distinct. Selection removed
{report['discarded_low_change_candidates']:,} low-change candidates and
{report['discarded_exact_duplicate_candidates']:,} exact duplicate candidates.
Every episode and seed belongs to exactly one split. Related roads and shared
assets remain correlated; this is not unseen-city evaluation.

Visible road is 1, known non-road is 0 and uncertain reference pixels are 255.
Binary 0/1 targets include separate 0/1 validity masks, so uncertain pixels never
become false background labels. Trusted-pixel coverage is **{coverage:.3f}%**.

Independent verification recounted every saved pair, checked hashes and target /
validity consistency, queried {verification['projected_road_points']:,} depth-projected
road points against native map lanes ({verification['points_outside_driving_lanes']}
outside), and checked sampled actor/footpath occlusions. TensorFlow consumed all
4,000 training and 500 validation pairs with exact once-only normalization and
paired mask/validity augmentation. The new simulation test split received file /
label QA only; no model prediction, accuracy measurement or selection used it.

The future training recipe is saved in `training_recipe.json`. The separate masked
training adapter and actual training/evaluation are the next stage. Do not pass
this data to the unchanged fixed-count real-data trainer.

Previews: `previews/index.html` (train/validation only).
Evidence: `dataset_report.json`, `independent_verification.json`,
`preprocessing_check.json`, `provenance.jsonl`, `episode_audit.json`.
Manifest SHA-256: `{report['manifest_sha256']}`.

Limits: generic installed assets, no authentic auto-rickshaw, no rural / highway /
unstructured-road collection, no calibrated real-traffic claim, no learned driving
or RL. Automatic geometry labels retain conservative ignored edges and range limits.
'''
    (output/"DATASET_REPORT.md").write_text(text,encoding="utf-8")
    progress=ROOT/"runs/simulation_preparation_summary.md"
    current=progress.read_text(encoding="utf-8") if progress.exists() else ""
    heading="# Separate simulation dataset prepared — 21 October 2026"
    if heading not in current:
        intro=f'''{heading}

A new simulation-only dataset is ready: **5,000 unique RGB/mask pairs**, split
4,000 train / 500 validation / 500 test by whole episodes. Thirty-two fresh
Indian-style left-hand episodes cover crossroads, both T-junction orientations
and straight streets, six traffic case families, four weather conditions and
varied camera / lighting / actor settings. Repeated and low-change views were
screened out; adjacent frames never cross splits.

Masks come from native semantic / instance / depth evidence plus actual map
geometry, without model predictions. Binary masks have separate validity masks;
uncertain reference pixels are ignored. Trusted-pixel coverage is {coverage:.3f}%.
All 5,000 pairs passed independent saved-file / geometry checks; all 4,500 train /
validation pairs passed actual TensorFlow preprocessing checks. No model was
trained, and the new test split has no model score. The real model, inference
files, original checkpoints and real split remain unchanged.

Dataset: `{dataset_relative}/`. Read
[the staged guide](docs/SIMULATION_DATASET.md),
[portable results](docs/results/simulation_dataset.json), and
the local `{dataset_relative}/DATASET_REPORT.md`.
These related generated maps share generic assets; no unseen-city or real-road
generalisation is established. Failed pilot captures/preparation are preserved.

'''
        progress.write_text(intro+current,encoding="utf-8")
    print(json.dumps(portable,indent=2))


if __name__=="__main__":
    main()
