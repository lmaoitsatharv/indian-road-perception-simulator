"""Independent source, timing, stale-display, mask and matched-control audit."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from carla_road_reference import build_reference,checked_appearance
from setup.prepare_simulation_dataset import protected,save,sha


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def distribution(values):
    return {"count":len(values),"median_ms":float(np.median(values)),"p95_ms":float(np.percentile(values,95)),
            "max_ms":float(max(values))} if values else None


def episode(record):
    owner=load(record)["launcher"]
    assert owner["status"]=="completed" and owner["owned_job_handles_closed"]
    paths=sorted(Path(owner["output"]).glob("episode_*"))
    assert len(paths)==1
    return paths[0]


def compare_controls(baseline,live):
    def trace(folder):
        with (folder / "actor_trace.csv").open(newline="") as stream:
            return {(round(float(row["sim_s"])*20),row["name"]):row for row in csv.DictReader(stream)}
    base,observed=trace(baseline),trace(live)
    assert set(base)==set(observed)
    fields=("x","y","yaw_rad","speed_mps","route_s","target_mps","throttle","brake","steer")
    maxima={field:0. for field in fields}
    for key,row in observed.items():
        for field in fields:
            if row[field] and base[key][field]:
                maxima[field]=max(maxima[field],abs(float(row[field])-float(base[key][field])))
    reasons=sum(r["reason"]!=base[k]["reason"] or r["pass_state"]!=base[k]["pass_state"] for k,r in observed.items())
    return {"actor_steps":len(base),"maximum_absolute_differences":maxima,
            "decision_reason_or_pass_state_mismatches":reasons,
            "strict_numeric_replay_match":all(maxima[f]<=1e-5 for f in ("target_mps","throttle","brake","steer"))
                 and all(maxima[f]<=1e-4 for f in ("x","y","yaw_rad","speed_mps","route_s"))}


def verify(output):
    state=load(output / "status.json")
    config=load(output / "session.json")
    worker=load(output / "worker_status.json")
    assert state["status"]=="completed" and state["protected_files_unchanged"]
    assert worker["status"]=="completed" and worker["model_weights_unchanged"] and state["worker_host_exit_code"]==0
    assert protected()==state["protected_sha256"]
    assert sha(ROOT / config["checkpoint"])==state["checkpoint_sha256"]
    live=episode(output / "live_capture.json")
    native=load(live / "result.json")
    assert native["runtime_passed"] and native["cleanup_complete"] and not native["cleanup_warnings"]
    assert not native["collision_events"]
    summary=load(live / "live_road/summary.json")
    assert summary["thread_stopped"] and summary.get("presentation_deadline_violations",0)==0
    assert summary["pending_capacity"]==summary["inflight_capacity"]==1 and summary["model_control"] is False
    ticks={r["frame"]:r for r in rows(live / "live_road/source_ticks.jsonl")}
    receipts=rows(live / "live_road/receipts.jsonl")
    displayed=rows(live / "live_road/display.jsonl")
    processed={r["frame"]:r for r in rows(output / "worker_frames.jsonl")}
    assert len(receipts)==len({r["frame"] for r in receipts})==summary["responses"] and len(receipts)>50
    assert len(processed)==worker["requests"]
    for row in receipts:
        tick=ticks[row["frame"]]
        assert row["session"]==config["session"] and row["tick_start_ns"]==tick["tick_start_ns"]
        assert row["tick_return_ns"]==tick["tick_return_ns"]
        assert row["received_ns"]>=row["sent_ns"]>=row["sensor_callback_ns"]>=row["tick_start_ns"]
        assert abs(row["source_step_to_result_upper_ms"]-(row["received_ns"]-tick["tick_start_ns"])/1e6)<1e-9
        assert abs(row["callback_to_result_ms"]-(row["received_ns"]-row["sensor_callback_ns"])/1e6)<1e-9
        assert processed[row["frame"]]["input_rgb_sha256"]==row["input_rgb_sha256"]
    presented=[]
    first_presented=[]
    shown_ids=set()
    for row in displayed:
        if row["source_frame"] is not None:
            expected=(row["presented_ns"]-row["source_tick_start_ns"])/1e6
            assert abs(expected-row["age_upper_ms"])<1e-9
        if row["overlay_shown"]:
            assert row["state"]=="fresh" and row["state_at_draw"]=="fresh"
            assert row["age_upper_ms"]<=config["max_age_ms"] and row["simulation_age_ms"]<=config["max_sim_age_ms"]
            assert row["source_frame"] in processed
            if row["source_frame"] not in shown_ids:
                first_presented.append(row["age_upper_ms"])
            shown_ids.add(row["source_frame"])
            presented.append(row["age_upper_ms"])
        assert row["model_control"] is False
    assert presented
    held_late=[(b["presented_ns"]-a["source_tick_start_ns"])/1e6-config["max_age_ms"]
        for a,b in zip(displayed,displayed[1:]) if a["overlay_shown"] and
        (b["presented_ns"]-a["source_tick_start_ns"])/1e6>config["max_age_ms"]]
    # A completed redraw can be gated; the OS may delay the next redraw while
    # synchronous simulator/render work runs. Report that separate limitation
    # instead of treating a timing reserve as a hard scheduling guarantee.
    if state["check"]:
        assert summary["pause_probe_completed"]
        assert {f["kind"] for f in worker["faults"]}=={"delay","disconnect"}
        disconnect=next(f["frame"] for f in worker["faults"] if f["kind"]=="disconnect")
        offline=[r for r in displayed if r["current_frame"]>=disconnect and r["state_at_draw"]=="offline"]
        assert offline and all(not r["overlay_shown"] for r in offline)
        assert any(r["overlay_shown"] and r["presented_ns"]>offline[-1]["presented_ns"] for r in displayed)
        stale=[r for r in displayed if r["state_at_draw"]=="stale"]
        assert stale and all(not r["overlay_shown"] for r in stale)
        spans={}
        for row in displayed:
            spans.setdefault(row["current_frame"],[]).append(row)
        paused=[group for group in spans.values() if group[-1]["presented_ns"]-group[0]["presented_ns"]>=1_000_000_000]
        assert paused and any(r["state_at_draw"]=="stale" for group in paused for r in group)
    # Re-open the actual captured sensor images: source bytes, normalization input
    # and masks are checked independently from the UI/transport implementation.
    camera=live / "road_camera"
    captures={r["frame"]:r for r in rows(camera / "frames.jsonl")}
    camera_config=load(camera / "capture_config.json")
    capture_summary=load(camera / "capture_summary.json")
    assert not any(capture_summary["dropped"].values())
    with np.load(camera / camera_config["reference"]["grid"]["file"],allow_pickle=False) as archive:
        footprint=archive["driving"]
    counts=dict.fromkeys(("tp","fp","fn","tn","ignored","total"),0)
    probabilities_checked=0
    for receipt in receipts:
        frame=receipt["frame"]
        capture=captures[frame]
        image=np.array(Image.open(camera / capture["rgb_file"]).convert("RGB"))
        assert hashlib.sha256(image.tobytes()).hexdigest()==receipt["source_rgb_sha256"]
        resized=np.array(Image.fromarray(image).resize((512,512),Image.Resampling.BILINEAR))
        assert hashlib.sha256(resized.tobytes()).hexdigest()==receipt["input_rgb_sha256"]
        mask=np.array(Image.open(output / "predictions" / f"frame_{frame:07d}.png"))
        assert mask.shape==(512,512) and set(np.unique(mask)).issubset({0,1})
        assert hashlib.sha256(mask.tobytes()).hexdigest()==receipt["mask_sha256"]
        probability_path=output / "probabilities" / f"frame_{frame:07d}.npy"
        if probability_path.exists():
            probability=np.load(probability_path,allow_pickle=False)
            assert probability.dtype==np.float32 and np.isfinite(probability).all()
            assert np.array_equal(mask,(probability>=.5).astype(np.uint8))
            probabilities_checked+=1
        tags=np.array(Image.open(camera / capture["labels_file"]).convert("L"))
        instance=np.array(Image.open(camera / capture["instance_file"]).convert("RGB"))
        appearance,mismatch=checked_appearance(tags,instance,camera_config,capture)
        with np.load(camera / capture["depth_file"],allow_pickle=False) as archive:
            depth=archive["meters"]
        reference,_=build_reference(tags,depth,capture["camera_to_world"],camera_config,footprint,appearance)
        reference[mismatch]=255
        reference=np.array(Image.fromarray(reference).resize((512,512),Image.Resampling.NEAREST))
        t,v,p=reference==1,reference!=255,mask==1
        current={"tp":int(np.count_nonzero(t & p & v)),"fp":int(np.count_nonzero(~t & p & v)),
                 "fn":int(np.count_nonzero(t & ~p & v)),"tn":int(np.count_nonzero(~t & ~p & v)),
                 "ignored":int(np.count_nonzero(~v)),"total":v.size}
        for key,value in current.items():
            counts[key]+=value
    assert probabilities_checked>0
    # Compare same-seed actions/poses by simulation tick; absolute world frames
    # and wall clocks may differ between separately owned CARLA processes.
    matched=None
    if state["check"]:
        baseline=episode(output / "baseline_capture.json")
        matched=compare_controls(baseline,live)
        assert matched["decision_reason_or_pass_state_mismatches"]==0
        save(output / "control_comparison.json",matched)
        assert sha(baseline.parent / "source/india_behavior.py")==sha(live.parent / "source/india_behavior.py")==state["rule_controller_sha256"]
    # Verify the actual native call boundary and the observer's actuator surface.
    native_tree=ast.parse((live.parent / "source/carla_india_scene.py").read_text())
    controller_calls=[n for n in ast.walk(native_tree) if isinstance(n,ast.Call) and
        isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and
        n.func.value.id=="traffic" and n.func.attr=="step"]
    assert len(controller_calls)==1 and [ast.unparse(a) for a in controller_calls[0].args]==["snapshot","sim_time"]
    preview_tree=ast.parse((live.parent / "source/india_live_preview.py").read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="apply_control" for n in ast.walk(preview_tree))
    tp,fp,fn=(counts[k] for k in ("tp","fp","fn"))
    result={"passed":True,"verification_scope":"Frame identity, stale suppression at completed redraws, model outputs, cleanup and controller input isolation; held-display deadline compliance and numeric replay equality are reported separately",
        "session":config["session"],"live_episode":str(live),
        "camera_frames":len(captures),"received_results":len(receipts),"shown_unique_frames":len(shown_ids),
        "source_rgb_resize_and_mask_hashes_verified":True,"probabilities_checked":probabilities_checked,
        "fresh_presentations":len(presented),"stale_overlay_presentations":0,
        "held_overlay_intervals_crossing_deadline":len(held_late),
        "maximum_held_overlay_deadline_overrun_ms":max(held_late,default=0),
        "held_overlay_deadline_met":not held_late,
        "redraw_freshness_gate_passed":True,
        "presentation_reserve_ms":config.get("presentation_reserve_ms",50),
        "source_step_to_result_upper_ms":distribution([r["source_step_to_result_upper_ms"] for r in receipts]),
        "normal_source_step_to_result_upper_ms":distribution([r["source_step_to_result_upper_ms"] for r in receipts if r["injected_delay_ms"]==0]),
        "callback_to_result_ms":distribution([r["callback_to_result_ms"] for r in receipts]),
        "gpu_inference_ms":distribution([r["inference_ms"] for r in receipts]),
        "presented_source_age_upper_ms":distribution(presented),"fault_probe":state["check"],
        "first_presentation_source_age_upper_ms":distribution(first_presented),
        "faults":worker["faults"],"queue_replacements":summary["queue_replacements"],
        "connection_errors":summary["connection_errors"],"matched_rule_control":matched,
        "controller_source_unchanged":True,"model_outputs_excluded_from_controller_inputs":True,
        "received_frame_reference_counts":counts,
        "received_frame_quality":{"road_iou":tp/(tp+fp+fn),"dice":2*tp/(2*tp+fp+fn),
                                  "coverage":1-counts["ignored"]/counts["total"]},
        "worker_and_native_cleanup_passed":True,"real_and_simulation_weights_unchanged":True,
        "model_control":False,"age_is_conservative_upper_bound":True,
        "actual_exposure_wall_timestamp_available":False}
    assert protected()==state["protected_sha256"]
    save(output / "independent_verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    verify(args.output.resolve())
