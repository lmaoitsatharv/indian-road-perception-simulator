"""Publish verified fresh-scene scores, comparisons and limits; never selects/trains."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "setup"))
from prepare_simulation_dataset import save,sha,protected

OUT=ROOT / "runs/simulation_confirmation_20261023_01"


def load(path):
    return json.loads(Path(path).read_text())


def local(value):
    return Path("E:/"+value[7:]) if value.startswith("/mnt/e/") else Path(value)


def percent(value):
    return f"{value*100:.4f}%"


def overlap(counts):
    tp,fp,fn=(counts[k] for k in ("tp","fp","fn"))
    return {"road_iou":tp/(tp+fp+fn),"dice":2*tp/(2*tp+fp+fn)}


def overlay(image,prediction,reference):
    pixels=image.copy()
    p,v,t=prediction==1,reference!=255,reference==1
    pixels[p & v]=(pixels[p & v]*.55+np.array([30,210,140])*.45).astype(np.uint8)
    pixels[~v]=(pixels[~v]*.5+np.array([200,40,200])*.5).astype(np.uint8)
    pixels[p & ~t & v]=[245,45,45]
    pixels[~p & t & v]=[30,120,255]
    return pixels


def main():
    plan,capture,selection=(load(OUT / n) for n in ("plan.json","capture_verification.json","selection.json"))
    assert load(OUT / "evaluation_status.json")["status"]=="completed"
    assert capture["passed"] and load(OUT / "protocol_verification.json")["passed"]
    assert protected()==plan["protected_sha256"]
    assert sha(ROOT / selection["checkpoint"])==selection["checkpoint_sha256"]
    reports={key:load(OUT / key / "report.json") for key in ("fresh_model","locked_baseline")}
    verifications={key:load(OUT / key / "independent_verification.json") for key in reports}
    for key,report in reports.items():
        assert report["status"]=="completed" and report["protected_files_unchanged"] and report["model_weights_unchanged"]
        assert verifications[key]["passed"] and verifications[key]["frames_recounted"]==480
        assert len(report["episodes"])==8 and sum(e["frames"] for e in report["episodes"])==480
    assert reports["fresh_model"]["runtime"]["checkpoint_sha256"]==selection["checkpoint_sha256"]
    assert reports["locked_baseline"]["runtime"]["checkpoint_sha256"]==load(ROOT / "models/segmentation_final_lock.json")["checkpoint_sha256"]
    folder=OUT / "comparisons"
    folder.mkdir(exist_ok=False)
    entries=[]
    episode_summaries=[]
    layout_counts={}
    changes={"improved":0,"worsened":0,"equal":0}
    for native,new,base in zip(capture["episodes"],reports["fresh_model"]["episodes"],reports["locked_baseline"]["episodes"],strict=True):
        assert new["name"]==base["name"] and new["source_manifest_sha256"]==base["source_manifest_sha256"]
        summary={k:native[k] for k in ("episode_id","layout","weather","case","seed","frames")}
        summary.update(fresh_model=new["metrics"],locked_baseline=base["metrics"],
                       delta_road_iou_pp=100*(new["metrics"]["road_iou"]-base["metrics"]["road_iou"]))
        episode_summaries.append(summary)
        for key,ep in (("fresh_model",new),("locked_baseline",base)):
            group=layout_counts.setdefault(native["layout"],{}).setdefault(key,dict.fromkeys(("tp","fp","fn","tn","ignored","total"),0))
            for field in group:
                group[field]+=ep["metrics"][field]
        for nr,br in zip(new["rows"],base["rows"],strict=True):
            assert nr["source"]==br["source"] and nr["frame"]==br["frame"]
            reference=np.array(Image.open(OUT / "fresh_model" / nr["reference"]))
            assert np.array_equal(reference,np.array(Image.open(OUT / "locked_baseline" / br["reference"])))
            image=np.array(Image.open(local(new["source_capture"]) / "road_camera" / nr["source"]["rgb_file"]).convert("RGB").resize((512,512),Image.Resampling.BILINEAR))
            new_mask=np.array(Image.open(OUT / "fresh_model" / nr["mask"]))
            base_mask=np.array(Image.open(OUT / "locked_baseline" / br["mask"]))
            target=np.zeros_like(image)
            target[reference==1]=255
            target[reference==255]=[200,40,200]
            ni,bi=nr["metrics"]["road_iou"],br["metrics"]["road_iou"]
            n,b=nr["metrics"],br["metrics"]
            difference=n["tp"]*(b["tp"]+b["fp"]+b["fn"])-b["tp"]*(n["tp"]+n["fp"]+n["fn"])
            category="improved" if difference>0 else ("worsened" if difference<0 else "equal")
            changes[category]+=1
            canvas=Image.new("RGB",(1024,292),"#101923")
            draw=ImageDraw.Draw(canvas)
            labels=("Camera RGB","Reference (purple ignored)",f"Locked real model: {bi*100:.2f}% IoU",f"New simulation model: {ni*100:.2f}% IoU")
            panels=(image,target,overlay(image,base_mask,reference),overlay(image,new_mask,reference))
            for column,(label,panel) in enumerate(zip(labels,panels)):
                draw.text((column*256+6,7),label,fill="white")
                canvas.paste(Image.fromarray(panel).resize((256,256),Image.Resampling.NEAREST if column==1 else Image.Resampling.BILINEAR),(column*256,30))
            name=f"{new['name']}_frame{nr['frame']:07d}.jpg"
            canvas.save(folder / name,quality=88)
            entries.append({"episode_id":native["episode_id"],"layout":native["layout"],"case":native["case"],
                "weather":native["weather"],"seed":native["seed"],"frame":nr["frame"],"sim_s":nr["sim_s"],
                "comparison":"comparisons/"+name,"fresh_model_road_iou":ni,"locked_baseline_road_iou":bi,
                "delta_pp":100*(ni-bi),"change":category})
    for values in layout_counts.values():
        for group in values.values():
            group.update(overlap(group))
    warnings=[{"episode_id":e["episode_id"],"unexercised_behaviour_criteria":e["unexercised_behaviour_criteria"]}
              for e in capture["episodes"] if e["unexercised_behaviour_criteria"]]
    results={"status":"completed","completed_utc":datetime.now(timezone.utc).isoformat(),
        "checkpoint":selection["checkpoint"],"checkpoint_sha256":selection["checkpoint_sha256"],"selected_epoch":10,
        "threshold":.5,"episodes":8,"frames":480,"trained_or_tuned_on_confirmation":False,
        "fresh_model":reports["fresh_model"]["metrics"],"locked_baseline":reports["locked_baseline"]["metrics"],
        "delta_road_iou_pp":100*(reports["fresh_model"]["metrics"]["road_iou"]-reports["locked_baseline"]["metrics"]["road_iou"]),
        "per_episode":episode_summaries,"per_layout":layout_counts,"per_frame_changes":changes,
        "worst_fresh_model_frame":min(entries,key=lambda e:e["fresh_model_road_iou"]),
        "independent_verification":{k:{f:v for f,v in verification.items() if f!="method"} for k,verification in verifications.items()},
        "reference_arrays_identical_between_models":True,"geometry_and_assets_verified":True,
        "all_native_health_checks_passed":True,"all_scripted_behaviour_criteria_passed":not warnings,
        "behaviour_warnings":warnings,"native_collision_events":sum(e["collision_events"] for e in capture["episodes"]),
        "close_clearance_alerts":sum(e["close_clearance_alerts"] for e in capture["episodes"]),
        "minimum_body_gap_m":min(e["minimum_body_gap_m"] for e in capture["episodes"] if e["minimum_body_gap_m"] is not None),
        "real_inference_unchanged":True,"real_test_images_read":False,"model_control":False,
        "scope":"recorded_carla_only","simulation_model_manifest":(OUT / "simulation_model.json").relative_to(ROOT).as_posix(),
        "gallery":(OUT / "index.html").relative_to(ROOT).as_posix(),"report":"docs/SIMULATION_CONFIRMATION.md",
        "limits":plan["limits"]}
    assert protected()==plan["protected_sha256"]
    save(OUT / "results.json",results)
    save(OUT / "comparison_frames.json",entries)
    save(ROOT / "docs/results/simulation_confirmation.json",results)
    worst=min(entries,key=lambda e:e["fresh_model_road_iou"])
    selected=[]
    for episode in episode_summaries:
        selected.append(min((e for e in entries if e["episode_id"]==episode["episode_id"]),key=lambda e:e["fresh_model_road_iou"]))
    sheet=Image.new("RGB",(1024,len(selected)*316),"#101923")
    draw=ImageDraw.Draw(sheet)
    for index,entry in enumerate(selected):
        draw.text((8,index*316+5),f"{entry['layout']} | {entry['case']} | {entry['weather']} | seed {entry['seed']} | {entry['sim_s']:.1f}s",fill="white")
        sheet.paste(Image.open(OUT / entry["comparison"]),(0,index*316+22))
    sheet.save(OUT / "worst_per_episode.jpg",quality=90)
    document='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Fresh Indian-style simulation model checks</title><style>body{font:16px system-ui;background:#111923;color:#eef4fc;max-width:1300px;margin:auto;padding:24px}p{line-height:1.6}img{max-width:100%;height:auto;display:block}summary{cursor:pointer;padding:16px;background:#203144;margin-top:16px}article{padding:12px 0;border-bottom:1px solid #364353}a{color:#8de7e9}table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #364353;text-align:left}</style><h1>Fresh Indian-style simulation checks</h1>'
    document+=f'<p>Eight fresh runs, four changed left-hand map layouts, all 480 frames. The epoch-10 model and 0.5 threshold were fixed before capture. New model IoU: <strong>{percent(results["fresh_model"]["road_iou"])}</strong>; locked real-road baseline: <strong>{percent(results["locked_baseline"]["road_iou"])}</strong>.</p><p>Green: predicted road. Red: extra road. Blue: missed road. Purple: ignored reference. Recorded comparison only; the road model does not control vehicles. Generic assets and related road families limit generalization claims.</p>'
    document+='<p><a href="results.json">Exact results</a> · <a href="fresh_model/index.html">New-model frame viewer</a> · <a href="locked_baseline/index.html">Baseline frame viewer</a> · <a href="worst_per_episode.jpg">Worst frame in each episode</a></p>'
    document+='<p>Controller diagnostics: three close-clearance alerts; two pedestrian episodes did not exercise the historical vehicle-yield criterion. These records remain included.</p>'
    for episode in episode_summaries:
        rows=[e for e in entries if e["episode_id"]==episode["episode_id"]]
        document+=f'<details><summary>{html.escape(episode["layout"])} · {html.escape(episode["case"])} · {episode["weather"]} · seed {episode["seed"]} · 60 frames · model IoU {percent(episode["fresh_model"]["road_iou"])}</summary>'
        for entry in rows:
            document+=f'<article><p>{entry["sim_s"]:.1f}s · new model {entry["fresh_model_road_iou"]*100:.3f}% IoU · baseline {entry["locked_baseline_road_iou"]*100:.3f}% · difference {entry["delta_pp"]:+.3f} pp</p><a href="{entry["comparison"]}"><img loading="lazy" src="{entry["comparison"]}" alt="RGB, reference and both model overlays"></a></article>'
        document+='</details>'
    (OUT / "index.html").write_text(document+'\n</html>',encoding="utf-8")
    lines=["# Fresh simulation confirmation — 23 October 2026","",
        "The fixed epoch-10 simulation model was checked on **eight new 60-second Indian-style left-hand episodes**, "
        "covering four changed map layouts and all six traffic-case families. All 480 frames at 1 Hz were scored; "
        "none were removed based on accuracy. No checkpoint, threshold or model weights changed.","",
        "| Model | Road IoU | Dice | Valid-pixel coverage |","|---|---:|---:|---:|"]
    for name,key in (("New simulation-only model","fresh_model"),("Locked real-road baseline on the same frames","locked_baseline")):
        m=results[key]
        lines.append(f"| {name} | {percent(m['road_iou'])} | {percent(m['dice'])} | {percent(m['evaluated_fraction'])} |")
    lines += ["",f"New-model minus baseline road IoU: **{results['delta_road_iou_pp']:+.4f} percentage points**.","",
        "These measurements cover new seeds, installed building assets, asphalt appearances and junction spacing "
        "within the same illustrative road families. They do not establish real-road or unseen-city performance.","",
        "## Scene changes fixed before capture","",
        "Junction half-spacing changed from 12 m to 10 or 11 m; the four canonical OpenDRIVE map hashes "
        "differ from training. Approach lengths, 4.5 m lanes and LHT lane rules remain fixed. New low-rise/apartment "
        "meshes (House12, Apartment01 and Apartment02) replace the original Apartment03/04 facades. Road texture "
        "colour/grain, sun direction, camera heights (1.30/1.45 m), pitch/yaw and FOV (92/98 degrees) also vary.","",
        "Preflight checked 30,600 footpath samples with no driving-lane intrusion and verified original default-map "
        "hashes remain unchanged. Native capture metadata confirms the new meshes actually spawned. All cameras "
        "were synchronized, no evidence frames dropped, and the 360 daytime frames with substantial visible sky "
        "passed the brightness check.","",
        "## Per-episode comparison","","| Layout | Case | Weather | Seed | Baseline IoU | New model IoU |","|---|---|---|---:|---:|---:|"]
    for row in episode_summaries:
        lines.append(f"| {row['layout']} | {row['case']} | {row['weather']} | {row['seed']} | {percent(row['locked_baseline']['road_iou'])} | {percent(row['fresh_model']['road_iou'])} |")
    lines += ["",f"Per-frame comparison: {changes['improved']} improved, {changes['worsened']} worsened, {changes['equal']} equal. "
        "Counts use exact integer-ratio comparisons. Weather, appearance and geometry changes are bundled; "
        "this check does not isolate their individual effects.","",
        "## Independent checks and remaining limitations","",
        "Both sets of 480 saved masks were independently recounted against their 0/1/255 references. "
        "Depth-projected sampled road points were checked against each native OpenDRIVE map; known actors "
        "and raised footpaths remained non-road in the reference. The two models used identical capture hashes "
        "and identical reference arrays. Uncertain pixels are excluded, with coverage above. Small declared "
        "semantic/instance edge mismatches (at most 0.1%) are excluded exactly, matching the dataset policy; "
        "historical captures without that declaration still reject any mismatch.","",
        "All eight native health checks passed with zero collisions, body overlaps, logged road departures or "
        "wrong-way events. Three close-clearance alerts remain, with a minimum sampled body gap of "
        f"{results['minimum_body_gap_m']:.3f} m. Two pedestrian runs (seeds 4201/4409) did not exercise the "
        "historical vehicle-yield criterion: pedestrians waited for a gap. Those behaviour criteria remain marked "
        "unexercised, not passed. Some mixed runs did not finish the full route within 60 seconds. "
        "This is a perception comparison with independent rule-based traffic, not a driving-safety evaluation.","",
        f"Worst new-model frame: {worst['layout']}, seed {worst['seed']}, {worst['sim_s']:.1f}s, "
        f"road IoU **{percent(worst['fresh_model_road_iou'])}**. Inspect its saved comparison and the worst frame "
        "from every episode; pooled overlap can hide local errors. Stationary intervals remain included and "
        "adjacent frames are correlated.","",
        "## Evidence and next step","",
        "- Gallery of all 480 matched comparisons: `runs/simulation_confirmation_20261023_01/index.html`.",
        "- Counts, per-frame changes and per-layout breakdown: `results.json` and `comparison_frames.json` in that folder.",
        "- Fixed simulator-only route: `simulation_model.json`; selection and capture protocol: `selection.json` / `plan.json`.",
        "- Native and protocol verification: `capture_verification.json`, `preflight.json`, `protocol_verification.json`.",
        "- Independent pixel/depth checks: `fresh_model/independent_verification.json` and `locked_baseline/independent_verification.json`.",
        "- Portable results: [results/simulation_confirmation.json](results/simulation_confirmation.json).","",
        "Real-image/video inference, real checkpoints, model catalog/lock, original splits and the trained simulation "
        "checkpoint remain unchanged. No reserved real-test images were opened; no training or RL occurred. "
        "The opt-in manifest is for recorded CARLA evaluation only, not shared real deployment.","",
        "A following implementation step is bounded live observation: measure capture-to-result age and "
        "stale-frame behaviour before integrating segmentation into an interactive preview. Vehicle/pedestrian "
        "detection/tracking and any controller use remain separate future work. This fresh set is now inspected "
        "and cannot serve as untouched confirmation after tuning."]
    (ROOT / "docs/SIMULATION_CONFIRMATION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Preserve the code actually used, including post-capture audit/report tools.
    source=OUT / "source"
    source.mkdir(exist_ok=False)
    names=set(load(OUT / "scene_sources.json")) | set(load(OUT / "evaluation_sources.json"))
    names |= {"setup/verify_confirmation_capture.py","setup/finalize_simulation_confirmation.py"}
    for name in sorted(names):
        target=source / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT / name,target)
    save(OUT / "final_source_hashes.json",{name:sha(ROOT / name) for name in sorted(names)})
    print(json.dumps({"status":"completed","frames":480,"fresh_model":results["fresh_model"],
                      "locked_baseline":results["locked_baseline"],"delta_pp":results["delta_road_iou_pp"],
                      "per_frame_changes":changes,"worst_frame":worst},indent=2))


if __name__=="__main__":
    main()
