"""Publish the verified live-observer implementation, timings and known limits."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from setup.prepare_simulation_dataset import save,sha,protected


def load(path):
    return json.loads(Path(path).read_text())


def main():
    normal=ROOT / "runs/simulation_live_20261023_07"
    faults=ROOT / "runs/simulation_live_20261023_05"
    n=load(normal / "independent_verification.json")
    f=load(faults / "independent_verification.json")
    assert n["passed"] and f["passed"] and n["connection_errors"]==0
    assert n["received_results"]==n["camera_frames"] and n["redraw_freshness_gate_passed"]
    assert f["matched_rule_control"]["strict_numeric_replay_match"]
    state=load(normal / "status.json")
    assert state["status"]=="completed" and protected()==state["protected_sha256"]
    native=load(Path(n["live_episode"]) / "result.json")
    snapshots=normal / "screenshots"
    snapshots.mkdir(exist_ok=False)
    sources={"fresh":Path(n["live_episode"]) / "live_road/fresh.png",
             "stale":Path(f["live_episode"]) / "live_road/stale.png",
             "offline":Path(f["live_episode"]) / "live_road/offline.png"}
    for name,path in sources.items():
        shutil.copy2(path,snapshots / (name+".png"))
    result={"status":"completed","completed_utc":datetime.now(timezone.utc).isoformat(),
        "launcher":"Start_Live_Road_Preview.cmd","powershell_launcher":"Start_Live_Road_Preview.ps1",
        "report":"docs/LIVE_SIMULATION_OBSERVATION.md","camera_hz":5,"pending_capacity":1,"inflight_capacity":1,
        "wall_freshness_limit_ms":500,"simulation_freshness_limit_ms":500,"presentation_reserve_ms":100,
        "age_is_conservative_source_tick_start_upper_bound":True,"actual_exposure_wall_timestamp_available":False,
        "held_overlay_deadline_met":n["held_overlay_deadline_met"],
        "display_timing_limitation":{
            "intervals_crossing_deadline":n["held_overlay_intervals_crossing_deadline"],
            "maximum_overrun_ms":n["maximum_held_overlay_deadline_overrun_ms"],
            "reason":"The synchronous preview can hold its previous image while a simulator/UI operation blocks"},
        "normal_run":normal.relative_to(ROOT).as_posix(),"normal":n,
        "fault_check_run":faults.relative_to(ROOT).as_posix(),"fault_check":f,
        "normal_run_speed_ratio":native["simulated_time_per_wall_second"],
        "normal_native_wall_s":native["loop_wall_s"],"normal_sim_s":native["sim_s"],
        "live_model_manifest":(normal / "model_manifest.json").relative_to(ROOT).as_posix(),
        "checkpoint_sha256":state["checkpoint_sha256"],"real_inference_and_weights_unchanged":True,
        "controller_source_unchanged":True,"model_control":False,"rl_started":False,
        "prototype_history":[
            {"run":"runs/simulation_live_20261023_01","result":"Failed before inference: seed4601 required pedestrian spawn failed; preserved"},
            {"run":"runs/simulation_live_20261023_02","result":"Early 2 GiB fault prototype; strict numeric replay mismatch retained"},
            {"run":"runs/simulation_live_20261023_03","result":"Normal prototype exposed unexpected idle disconnects; preserved"},
            {"run":"runs/simulation_live_20261023_04","result":"Transport logging identified timeouts and long initial GPU inference stalls"},
            {"run":"runs/simulation_live_20261023_06","result":"Fixed transport processed150/150 with no reconnects; screen-hold audit motivated larger presentation reserve"}],
        "limits":["Known seed4101 scene used to validate engineering behavior, not a new held-out accuracy test",
                  "Measured on this laptop at1x/Low graphics; no hard real-time scheduling or exact exposure-time guarantee",
                  "Freshness gates are checked at redraw; prolonged OS/window/GPU stalls can delay redraw",
                  "Road segmentation only; object detection/tracking and control use remain future work"]}
    save(normal / "live_results.json",result)
    save(ROOT / "docs/results/live_simulation_observation.json",result)
    def timing_row(label,key):
        values=n[key]
        return f"| {label} | {values['median_ms']:.2f} ms | {values['p95_ms']:.2f} ms | {values['max_ms']:.2f} ms |"
    lines=["# Live simulation road observation — 23 October 2026","",
        "The simulation model now runs on the native CARLA front-camera stream while the simulator is active. "
        "A separate GPU service supplies frame-aligned masks to the preview. The rule driver, real-image/video "
        "inference and all checkpoint weights remain unchanged. No training or RL occurred.","",
        "## Start the preview","",
        "Double-click **Start_Live_Road_Preview.cmd** in the project folder. It loads the existing WSL GPU "
        "runtime and starts an owned Indian-style left-hand scene. Allow model/simulator startup to finish. "
        "The default scene is mixed traffic, seed 4101, clear weather, 1x playback and Low graphics, bounded to 110 "
        "simulation seconds. The scene uses the already verified new low-rise assets and 10 m junction half-spacing.","",
        "The top-left panel shows the current front camera. The top-right panel shows the latest admissible "
        "road mask on **its own matching source image**, with source-frame number and age. Green means "
        "predicted road. Waiting/stale/offline states replace that overlay with a blank labelled panel. "
        "The small lower-left image retains the existing follow/driver/junction view and the lower HUD retains "
        "the existing driving explanation. SPACE pauses/resumes; S saves a screenshot; ESC closes the owned "
        "simulator and GPU service.","",
        "PowerShell options, when wanted:","","```powershell",
        ".\\Start_Live_Road_Preview.ps1 -Seconds 60 -Scenario mixed -Weather clear",
        ".\\Start_Live_Road_Preview.ps1 -Seconds 40 -Check","```","",
        "`-Check` enables explicit delay/disconnection/pause tests and a matched no-model control comparison. "
        "Normal launch does not inject faults. Existing `Start_Indian_Traffic` behavior remains available separately.","",
        "## Measured normal operation","",
        f"The final normal run received **{n['received_results']}/{n['camera_frames']} results**, with **zero reconnects**, "
        f"in {native['sim_s']:.1f} simulated seconds. Camera sampling was 5 Hz. No over-age mask was admitted "
        "at a completed redraw. All native and GPU worker processes "
        "and the transport thread closed cleanly.","",
        f"**Display timing limitation:** {n['held_overlay_intervals_crossing_deadline']} intervals held the "
        f"previous overlay beyond the 500 ms age target, by up to {n['maximum_held_overlay_deadline_overrun_ms']:.2f} ms. "
        "The next redraw replaced it with a fresh result or a blank stale panel. A synchronous simulator/UI "
        "operation can postpone that redraw. The per-redraw freshness gate passed; continuous on-screen "
        "deadline compliance did not. This preview is suitable for observation, with that measured limitation.","",
        "| Timing | Median | 95th percentile | Maximum |","|---|---:|---:|---:|",
        timing_row("Source-step start → result received (age upper bound)","source_step_to_result_upper_ms"),
        timing_row("Camera callback → result received","callback_to_result_ms"),
        timing_row("GPU inference stage","gpu_inference_ms"),
        timing_row("Source-step start → first preview presentation (upper bound)","first_presentation_source_age_upper_ms"),
        timing_row("Source age across all fresh preview redraws (upper bound)","presented_source_age_upper_ms"),"",
        "Later redraws may reuse a fresh result until a newer one arrives, so their source age differs from "
        "the latency to the first display of each unique result. All absolute timestamps used in these age "
        "differences originate from the same Windows monotonic clock. Linux timestamps are never subtracted "
        "from Windows timestamps; the worker uses its own clock only for local stage durations.","",
        "CARLA does not provide an exact camera-exposure wall timestamp in this pipeline. The timestamp "
        "immediately before the source physics step is a conservative earlier bound, including sensor delivery, "
        "transport, resize and inference. These are measured age bounds, not exact exposure latency or a hard "
        "real-time guarantee. Prolonged OS/GPU/UI stalls can delay the next redraw.","",
        "## Freshness, bounded buffering and recovery","",
        "There is at most one in-flight request and one pending frame. A newer pending frame replaces the "
        "older one instead of growing a queue. Received frame/session identities and pixel hashes must match. "
        "Raw RGB is resized bilinearly to 512x512 and normalized once; probability threshold remains 0.5. "
        "Every mask is rendered on its matching resized source RGB, never a newer camera frame.","",
        "Freshness gates apply a 500 ms Windows wall-age bound and 500 ms of simulated age. The final display "
        "uses a 100 ms presentation reserve, so it stops admitting an older mask at 400 ms rather than waiting "
        "until its last permissible instant. The independent audit checks both completed presentations and "
        "intervals for which the previous overlay remained on screen. This is a redraw-based policy, not a "
        "hard-scheduling guarantee. Pausing the simulated world does not freeze the wall-age clock.","",
        f"The 40 s controlled run received {f['received_results']}/{f['camera_frames']} results. Its 0.9 s delay "
        "produced an over-age result that was hidden; its deliberate TCP disconnection produced an offline "
        "panel and recovered automatically; the 1.2 s pause also aged the mask out while the world frame stayed "
        f"fixed. {f['queue_replacements']} pending frames were replaced and the only connection error was the "
        "deliberate disconnection. The fault run used the earlier 50 ms presentation reserve; the final normal "
        "run verified the stricter 100 ms reserve and retained its measured display overruns. The transport's "
        "3 s timeout never extends the 500 ms overlay admission gate.","",
        "The model runtime reserves 1 GiB of GPU memory to leave room for CARLA. Earlier 2 GiB/short-idle-timeout "
        "prototypes showed unnecessary reconnects and GPU stalls; they remain preserved under runs01–04. "
        "Run 06 had no reconnects but a held-screen audit found a 15.67 ms tail beyond the nominal deadline on "
        "three intervals. Run 07 used the larger reserve; its separate held-screen result is reported above. "
        "Increasing a timing reserve alone cannot guarantee redraw scheduling.","",
        "## Controller and model isolation","",
        f"The final fault test matched **{f['matched_rule_control']['actor_steps']:,} actor/control records** "
        "from the same-seed no-model baseline exactly: positions, heading, speed, route position, throttle, "
        "brake, steering, targets, decision reasons and pass states all matched. A repeated no-model baseline "
        "also matched exactly. This is evidence for this 40 s scene, not universal simulator determinism.","",
        "The controller source is byte-identical to the protection snapshot. Its call still receives only "
        "`snapshot` and `sim_time`. The preview has no actuator calls, and no image/mask is supplied to the "
        "driver. Model weights, real deployment/catalog/split metadata and the old recorded-only simulator "
        "loader remain unchanged. The new live-observer manifest has its own native-CARLA observation-only "
        "scope; neither recorded nor live scope grants real deployment or vehicle control.","",
        "An independent script reopened each actual captured camera image and verified the source bytes, "
        "exact resize, prediction hashes, saved probability thresholds, native reference masks, all timing "
        "arithmetic and source-frame identities. It confirmed stale/offline suppression and recovery. The "
        "source scene/seed was already inspected; any saved road-IoU sanity score is not a new independent "
        "held-out test result. The completed 500-image test and 480-frame confirmation remain their own reports.","",
        "## Evidence","",
        f"- Final normal run: `{normal.relative_to(ROOT).as_posix()}/`.",
        f"- Controlled fault/control comparison: `{faults.relative_to(ROOT).as_posix()}/`.",
        "- Each run has `independent_verification.json`, `worker_frames.jsonl`, worker/capture status and model manifest.",
        "- Native episode `live_road/` contains receipts, source ticks, every display event and transport-error logs.",
        "- Model masks and selected probability arrays remain beside the worker logs.",
        "- Portable result: [results/live_simulation_observation.json](results/live_simulation_observation.json).","",
        "Normal fresh view:","",f"![Live road preview](<{(snapshots/'fresh.png').as_posix()}>)","",
        "Controlled stale-result suppression:","",f"![Stale road mask hidden](<{(snapshots/'stale.png').as_posix()}>)","",
        "Controlled service-disconnection state:","",f"![Disconnected observer](<{(snapshots/'offline.png').as_posix()}>)","",
        "The first new-seed attempt (run01, seed4601) failed during pedestrian spawning before inference. "
        "That failed capture is retained; the latency/controller checks use the previously verified seed4101. "
        "Earlier prototype timing/control differences are also retained, rather than overwritten.","",
        "## Following work","",
        "If strict continuous display expiry is required, decouple preview refresh from blocking simulator "
        "calls and evaluate OS/GPU stalls explicitly. The current per-redraw expiry policy is measured above.","",
        "Add independently evaluated vehicle/pedestrian detection and tracking in observation-only mode, "
        "including missed actors, timing and stale-track behavior. Road masks alone do not provide object "
        "identity or motion. Any later planning/controller/RL use remains a separate step."]
    document="\n".join(lines)+"\n"
    (ROOT / "docs/LIVE_SIMULATION_OBSERVATION.md").write_text(document,encoding="utf-8")
    (normal / "LIVE_REPORT.md").write_text(document,encoding="utf-8")
    names=["Start_Live_Road_Preview.cmd","Start_Live_Road_Preview.ps1","simulation_live_protocol.py",
           "idd_live_simulation_inference.py","integration/india_live_preview.py","integration/india_road_observer.py",
           "integration/carla_india_scene.py","setup/carla/run_session.py","setup/live_simulation_worker.py",
           "setup/run_live_simulation.py","setup/verify_live_protocol.py","setup/verify_live_simulation.py",
           "setup/check_live_repeatability.py","setup/finalize_live_simulation.py"]
    source=normal / "source"
    source.mkdir(exist_ok=False)
    for name in names:
        target=source / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT / name,target)
    save(normal / "final_source_hashes.json",{name:sha(ROOT / name) for name in names})
    print(json.dumps({"status":"completed","frames":n["received_results"],
        "result_age_upper_median_ms":n["source_step_to_result_upper_ms"]["median_ms"],
        "gpu_inference_median_ms":n["gpu_inference_ms"]["median_ms"],
        "camera_errors":n["connection_errors"],"stale_presentations":n["stale_overlay_presentations"]}))


if __name__=="__main__":
    main()
