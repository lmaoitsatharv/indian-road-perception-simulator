"""Summarize saved scene, reference and recording evidence without rerunning a model."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    before = load("runs/carla_asphalt_observation_20261020_02/report.json")
    after = load("runs/carla_paving_observation_20261020_01/report.json")
    audit = load("runs/carla_paving_observation_20261020_01/independent_verification.json")
    assert before["status"] == after["status"] == "completed" and audit["passed"]
    capture_names = ["carla_india_smoke_20261020T172949_077967Z_seed7",
                     "carla_india_smoke_20261020T173813_970813Z_seed42",
                     "carla_india_smoke_20261020T174627_026693Z_seed11"]
    captures = []
    for name in capture_names:
        folder = ROOT / "runs" / name
        episode = next(folder.glob("episode_*"))
        result = json.loads((episode/"result.json").read_text())
        capture = json.loads((episode/"road_camera/capture_summary.json").read_text())
        native = json.loads((episode/"independent_audit.json").read_text())
        assert native["passed"] and not any(capture["dropped"].values())
        captures.append({"source": str(episode.relative_to(ROOT)).replace("\\","/"),
            "seed": result["seed"], "road_camera_hz": 5,
            "simulated_seconds_per_wall_second": result["simulated_time_per_wall_second"],
            "paired_frames": capture["paired_frames"], "dropped": capture["dropped"],
            "write_pair_ms_median": capture.get("write_pair_ms_median"),
            "producer_backpressure_s": capture.get("producer_backpressure_s"),
            "overtakes": result["completed_overtakes"], "goal": result["ego_goal_reached"],
            "native_audit": native["checks"], "minimum_body_gap_m": native["min_gap_m"]})
    screenshot = load("setup/road_boundary_fix_20261020/screenshot_comparison.json")
    trials, candidate, model_folder = [], None, None
    for folder in sorted((ROOT/"runs").glob("road_boundary_adaptation_20261020_0[23]")):
        if not (folder/"comparison.json").exists():
            continue
        value = json.loads((folder/"comparison.json").read_text(encoding="utf-8"))
        config = json.loads((folder/"config.json").read_text(encoding="utf-8"))
        trials.append({"source": str(folder.relative_to(ROOT)).replace("\\","/"),
            "learning_rate": config["learning_rate"], "selected_epoch": value["selection"]["selected_epoch"],
            "baseline_real": value["baseline_real"], "baseline_development": value["baseline_development"],
            "epochs": value["epochs"], "protected_files_unchanged": value["protected_files_unchanged"]})
        candidate, model_folder = value, folder
    fixed_path = model_folder/"fixed_case_comparison.json" if model_folder else None
    fixed_cases = json.loads(fixed_path.read_text(encoding="utf-8")) if fixed_path and fixed_path.exists() else None
    final_paving = load("runs/carla_india_live_20261020T180848_160979Z_seed7/streetscape.json")
    report = {"date": "2026-10-20", "scene_version": "lht-neighbourhood-v4.2",
        "before_scene_metrics": before["metrics"], "after_scene_metrics": after["metrics"],
        "scene_iou_gain_pp": 100*(after["metrics"]["road_iou"]-before["metrics"]["road_iou"]),
        "scene_frames": sum(e["frames"] for e in after["episodes"]),
        "scene_change": "224 raised, textured footpath slabs; driving-lane footprint and vehicle policy retained",
        "reference_audit": audit, "screenshot_40_5s": screenshot, "recording_runs": captures,
        "model_candidate": candidate, "model_trials": trials, "candidate_known_cases": fixed_cases,
        "final_native_footpath_check": {"source": "runs/carla_india_live_20261020T180848_160979Z_seed7",
            "native_meshes": len(final_paving["native_bounds"]),
            "native_height_checked_m": final_paving["top_height_m"],
            "native_bounds_driving_lane_overlap_samples": final_paving["native_bounds_driving_lane_overlap_samples"],
            "read_after_static_setup_tick": True},
        "preserved_failed_startup_check": "runs/carla_india_live_20261020T180748_297093Z_seed7",
        "protected_files_unchanged": after["protected_files_unchanged"],
        "model_control": False, "rl_trained": False,
        "limits": ["Geometry-assisted simulator references are not human annotations or independent real-road accuracy",
            "Paving, visible occlusions and eligible-pixel coverage change between scenes; fresh runs are not a perfect single-factor benchmark",
            "Recorded speed comparisons use different mixed seeds and interactive controls, rather than a controlled same-seed microbenchmark",
            "One rejected training recording had an unrealistic black daytime overcast sky; it was preserved and replaced before any optimization",
            "Original three checkpoints, split manifest and completed real-data test remain untouched",
            "A roughly 0.48 m bus/parked-car clearance remains reported; zero observed collisions is not a safety proof"]}
    target = ROOT/"runs/road_boundary_fix_20261020_01"
    target.mkdir(exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False)+"\n"
    (target/"results.json").write_text(text,encoding="utf-8")
    # Portable scene/recording evidence; candidate's detailed local manifest stays local.
    portable = dict(report); portable.pop("model_candidate")
    (ROOT/"docs/results/road_boundary_fix.json").write_text(json.dumps(portable,indent=2)+"\n",encoding="utf-8")
    lines = ["# Road-boundary and recording fixes — 20 October 2026", "",
        "## Scene and reference", "",
        f"The same locked model scored **{100*before['metrics']['road_iou']:.2f}% → {100*after['metrics']['road_iou']:.2f}% road IoU** across 1,100 simulator frames.",
        "Raised paving with block joints now distinguishes footpaths from asphalt. The underlying driving-lane definition was retained.",
        "At the user's 40.5-second view, full-frame IoU changed from 77.05% to 83.03%.",
        "On commonly evaluated pixels where both references agree, IoU changed from 77.32% to 83.11%.",
        "The references disagree on 825 of 256,794 common valid pixels, reflecting visible-geometry/occlusion differences.", "",
        "The independent audit recounted all 1,100 masks and checked 1,760 projected road points: none fell outside driving lanes.",
        f"It also checked {audit['raised_footpath_pixels_checked_nonroad']:,} footpath pixels remain non-road, plus {audit['known_actor_pixels_checked']:,} known-actor pixels.", "",
        "## Recording performance", "",
        "Four analysis streams used to encode/save on the simulation tick thread. Their lossless writing now uses two bounded background encoders, with ordered commits and failure propagation.",
        "RGB/semantic/instance pixels and float32 depth remain exact. No frame dropping or depth quantization was used for the speed gain.",
        "A lower 1 Hz analysis-capture option is also available; it does not change the physics or viewer camera.", "",
        "| Saved run | Paired frames | Simulated time / wall time |", "| --- | ---: | ---: |"]
    for run in captures:
        lines.append(f"| Mixed seed {run['seed']} | {run['paired_frames']} | {run['simulated_seconds_per_wall_second']:.2f}× |")
    lines += ["", "The final parallel-encoder run averaged 2.72× with steady windows near 2.95× at selected 3×. No paired frames were dropped.",
        "It measured about 4.2 ms per world tick and 1.1 ms for traffic controls; image encoding was the substantial bottleneck.",
        "The before/after runs use different seeds and interactive control histories, so the ratio is an observed throughput comparison, not a strict causal benchmark.", "",
        "## Separate model candidate", "",
        "Read `runs/road_boundary_adaptation_20261020_03/MODEL_REPORT.md` for the candidate's measured real-validation, synthetic-development and withheld comparisons. Trial _02 failed retention and remains preserved.",
        "The training recording with the invalid overcast sky was rejected before optimization; `road_boundary_adaptation_20261020_01` preserves that preparation and its reason.",
        "The three original model files and deployment lock remain protected. The new candidate uses a separate checkpoint and never opens the reserved real-data test images.", "",
        "## Evidence and limits", "",
        "- Full locked-model replay: `runs/carla_paving_observation_20261020_01/index.html`.",
        "- Machine-readable report: `runs/road_boundary_fix_20261020_01/results.json`.",
        "- Current user-facing simulation remains hand coded using simulator state; road predictions are not connected to control and no RL was trained.",
        "- Reference, weather, same-town adaptation and close-clearance limits remain explicit in the saved reports.", ""]
    if candidate:
        model_lines = ["", "### Measured model trials", "",
            "| Trial / epoch | Decoder LR | Real validation IoU | Real boundary IoU (5px) | Simulator development IoU | Retention passed |",
            "| --- | ---: | ---: | ---: | ---: | --- |"]
        for trial in trials:
            for epoch in trial["epochs"]:
                model_lines.append(f"| {Path(trial['source']).name} / {epoch['epoch']} | {trial['learning_rate']:g} | {100*epoch['real_validation']['road_iou']:.3f}% | {100*epoch['real_validation']['boundary_iou_5px']:.3f}% | {100*epoch['synthetic_development']['road_iou']:.3f}% | {epoch['real_retention_guard_passed']} |")
        selected_epoch = candidate["selection"]["selected_epoch"]
        model_lines += ["", f"Latest completed trial: `{model_folder.relative_to(ROOT).as_posix()}`. Selected epoch: **{selected_epoch if selected_epoch else 'none'}**. The deployment lock is unchanged."]
        if selected_epoch:
            model_lines += [f"The frozen candidate's withheld simulator IoU was **{100*candidate['baseline_heldout']['road_iou']:.3f}% → {100*candidate['candidate_heldout']['road_iou']:.3f}%** on {candidate['withheld_images']} frames, decoded after selection."]
        else:
            model_lines += ["No candidate qualified in this trial. Its withheld images were left unopened."]
        if fixed_cases:
            example = next(row for row in fixed_cases["cases"] if row["name"] == "after_paving")
            model_lines += [f"On the known 40.5-second development example, the frozen candidate improved the new scene from {100*example['baseline']['road_iou']:.3f}% to {100*example['candidate']['road_iou']:.3f}% IoU. This example was not used to change selection.",
                "Model gallery: `runs/road_boundary_adaptation_20261020_03/index.html`."]
        lines += model_lines+[""]
    lines += ["## Final native startup check", "",
        "All 224 native footpath mesh bounds were checked after a static setup tick: top height 0.16 m and zero sampled driving-lane intrusions.",
        "An initial assertion read stale actor bounds before world advancement and stopped startup. That failure was preserved; reading after the setup tick fixed the check, and the viewer launched successfully.", ""]
    (ROOT/"docs/ROAD_BOUNDARY_FIX.md").write_text("\n".join(lines),encoding="utf-8")
    (target/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"scene_iou_gain_pp":report["scene_iou_gain_pp"],"frames":report["scene_frames"]}))


if __name__ == "__main__":
    main()
