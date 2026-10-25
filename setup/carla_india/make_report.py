"""Assemble a report only from explicitly selected, independently checked episodes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, nargs="+")
    args = parser.parse_args()
    selected, rows, comfort = [], [], []
    for episode in args.episodes:
        episode = episode.resolve()
        audit, result = load(episode / "independent_audit.json"), load(episode / "result.json")
        if not audit["passed"] or not audit["completed"]:
            raise RuntimeError(f"Unverified or incomplete scenario: {episode}")
        ego = audit["actors"]["ego"]
        selected.append({"case": audit["case"], "seed": audit["seed"], "episode": str(episode),
                         "audit": str(episode / "independent_audit.json"), "checks": audit["checks"],
                         "distance_m": ego["distance_m"], "goal_reached": audit["goal_reached"],
                         "goal_time_s": result["ego_goal_time_s"], "overtakes": result["completed_overtakes"],
                         "min_traffic_body_gap_m": audit["min_gap_m"], "camera_images": result["camera_images"],
                         "sim_s": result["sim_s"], "camera_delay_frames": result["camera_max_delivery_delay_frames"]})
        gap = "Not within the 4 m audit search band" if audit["min_gap_m"] is None else f"{audit['min_gap_m']:.3f} m"
        goal = f"{result['ego_goal_time_s']:.2f} s" if audit["goal_reached"] else "Not reached within this test"
        rows.append(f"| {audit['case']} | {audit['seed']} | {result['sim_s']:.0f} s | {ego['distance_m']:.2f} m | {result['completed_overtakes']} | {goal} | {gap} | Pass |")
        comfort.append(f"| {audit['case']} / {audit['seed']} | {ego['max_speed_mps']:.2f} | {ego['braking_500ms_max_mps2']:.2f} | {ego['abs_jerk_500ms_p95_mps3']:.2f} | {ego['abs_jerk_500ms_max_mps3']:.2f} |")
    expected = {(name, 7) for name in ("clear_overtake", "oncoming", "pedestrian", "return_gap", "junction", "mixed")} | {("mixed", 42)}
    if not expected <= {(s["case"], s["seed"]) for s in selected}:
        raise RuntimeError("Missing a required controlled scenario or mixed seed")
    sources = [ROOT / "integration" / f for f in ("india_behavior.py", "india_road.py", "india_scenery.py", "carla_india_scene.py")]
    sources += [ROOT / "setup/carla" / f for f in ("run_session.py", "windows_job.py", "viewer_entry.py")]
    sources += [Path(__file__), ROOT / "setup/carla_india/verify_run.py"]
    manifest = {"verified_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "episodes": selected,
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "renderer": "Vulkan, Epic, windowed", "final_runtime_cap_hz": 20,
                "claim": "Finite scenario verification; not calibrated traffic or real-world safety"}
    camera_review_file = ROOT / "setup/carla_india/camera_review.json"
    camera_review = load(camera_review_file) if camera_review_file.exists() else None
    if camera_review:
        manifest["final_visual_review"] = camera_review
    (ROOT / "setup/carla_india/verified_scene.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    mixed7 = next(Path(s["episode"]) for s in selected if (s["case"], s["seed"]) == ("mixed", 7))
    follow = sorted(mixed7.glob("follow_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    junction = sorted(mixed7.glob("junction_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    pictures = "\n\n".join(f"![Actual CARLA camera capture]({p.as_posix()})" for p in ([follow[-1]] if follow else []) + ([junction[len(junction)//2]] if junction else []))
    if camera_review:
        pictures += f"\n\n![Corrected driver camera]({Path(camera_review['driver_capture']).as_posix()})"
    evidence = "\n".join(f"- [{s['case']} / seed {s['seed']}]({Path(s['episode']).as_posix()}/independent_audit.json)" for s in selected)
    report = f"""# Indian traffic scenario — checked build, 19 October 2026

## What is ready

Native 3D CARLA neighbourhood with a true left-hand road network, twelve junction routes, follow/driver/junction cameras, mixed vehicle sizes, walking and crossing people, bus stopping, roadside activity and a separate parking bay. Vehicles use throttle, brake and steering in CARLA physics. There are no driving pose teleports or disabled collisions.

Play **Start_Indian_Traffic.cmd**. Use **Tab** for cameras, **Space** to pause, **R** to start a fresh episode, **S** to save an image and **Esc** to close. The default session lasts at most ten minutes and pauses at the ego goal. Use the window titled **Indian traffic neighbourhood**; a separate CARLA engine window may also be present.

The scene has 11 vehicles including the ego and parked car, and 8 people in the mixed scenario. There is a scooter, motorcycle, two cycles, bus, van and several cars. These are available CARLA assets, not authentic models for every Indian vehicle class.

## Verified results

Seven completed scenario checks are selected below. All have zero native vehicle collision events, zero sampled traffic-body overlaps, zero road departures and zero wrong-way events. Their saved images match the corresponding world-frame timestamps. Body checks include moving people and the parked car; they are two-dimensional checks at 20 Hz on this flat map, with native collision sensors on every vehicle.

| Scenario | Seed | Simulated duration | Ego travel | Completed passes | Ego goal | Closest sampled traffic-body gap | Independent audit |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
{chr(10).join(rows)}

The oncoming case intentionally keeps the ego behind the cyclist while the pass is blocked. Route completion is not claimed for a case where the goal was not reached. Both 110-second mixed cases must reach their declared goal to pass the final audit.

The case requirements are checked against actual events: a completed clear-road pass, initial oncoming blockage, a direct ego yield and completed pedestrian crossing, a merge delayed by the return-gap scooter, and a junction yield/release. Runtime success alone cannot satisfy these requirements.

## Ride quality

Actual motion is measured from positions. The following braking/jerk values use half-second differences, consistently across builds, to reduce amplification of frame-level position noise. The raw 20 Hz peak values remain in each result and can be much larger. A controller command limit is not a measured comfort guarantee.

| Scenario / seed | Max ego speed (m/s) | Max braking over 0.5 s (m/s²) | Absolute jerk p95 over 0.5 s (m/s³) | Max absolute jerk over 0.5 s (m/s³) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(comfort)}

The earlier clear-road build recorded 12.37 m/s² braking and 9.10 m/s³ jerk p95. The checked controller reduces those observed values in that case. Stopping positions and headway were also corrected, so this is a build comparison, not an isolated causal experiment. High transient jerk and emergency braking still need further work and calibration.

## Problems found and addressed

1. **Incorrect early pass prediction:** a stopped oncoming car could accelerate after a pass began. The gate now accounts for its target approach speed, and the controlled oncoming case must show a blocked pass.
2. **A masked pedestrian test:** a slow cycle was doing the yielding in front of the ego. The direct-yield test now isolates the ego response.
3. **Overwide bus placement:** the installed bus's conservative body rectangle did not fit its roadside offset. It is now centred within its carriageway.
4. **Overlapping people:** crossing endpoints were moved to the outer footpath, separated from walking tracks. People already crossing continue walking while vehicles yield. Mixed-scene pedestrians first accept a vehicle gap.
5. **Unnecessary junction queues:** geometric conflict reservations permit compatible opposing flows and following traffic, while conflicting paths retain priority rules and headway checks.
6. **Destination blockage:** NPCs formerly stopped at the ego's destination. The ego now has its own finish point near x=112 m; NPC stopping areas are farther toward the map edge. Parking is in a separate bay.
7. **Abrupt stopping and overspeed:** earlier braking, longer headway, anti-windup, acceleration feedback and lower normal brake commands reduce the measured harshness. All raw metrics remain saved.
8. **GPU renderer failures:** the experimental off-screen launch failed with VK_ERROR_SURFACE_LOST_KHR and was removed. An accelerated windowed suite later failed with VK_ERROR_DEVICE_LOST. The accepted long mixed checks use fresh, windowed sessions at the 20 Hz cap. These GPU failures remain in `native_failures/crashes.json`; broad long-session renderer reliability is not proven.
9. **Slow failure reporting:** actor bounds are cached at spawning, runtime RPC waits are bounded to ten seconds, and the launcher watches new native crash records belonging to process IDs from its own Windows Job Object. The ownership filter was checked against saved real crash evidence and a foreign PID.

The replay key was exercised: it removed the previous actors, created a new ego actor, restored the same starting position and preserved the earlier episode. Pause/resume checks verified an unchanged world frame; camera changes and screenshots exercised the actual key handlers.

## Actual captures

{pictures}

## Evidence

Each linked episode contains the independent audit, per-frame trace, actor settings, incidents, collision records, camera alignment records and actual PNG captures. Its parent folder retains the exact source snapshot and generated OpenDRIVE map. Failed and interrupted development runs were preserved.

The first four accepted controlled cases preceded the actor-bound caching change. The saved `controlled_to_final_policy.diff` confirms that only when bounds are fetched changed; the controller rules and road/scenery files are unchanged. Junction and mixed checks then passed with cached bounds and the real-time cap. The final camera-eye adjustment was visually checked in a separate live preview and is recorded in `camera_review.json`.

{evidence}

Current source hashes and the selected evidence are in **verified_scene.json**. **replay_control_check.json** and **native_failure_check.json** record the control and crash-recognition checks. Recovery stops acted only through the owned controller/Job Objects.

## Limits and the next development step

- This is an illustrative Indian-style neighbourhood, not a calibrated Indian city. Native building/vehicle assets remain generic. Authentic autorickshaws, local storefronts, denser irregular junctions and footage-based traffic calibration are still needed.
- The junction reservation and passing rules are hand-coded. Desired speeds, gap thresholds and pedestrian choices are scenario settings, not parameters established from an Indian traffic study.
- The camera is for observation. The locked segmentation model does not control this car, and no RL policy was trained. The existing three road models, original data, splits and completed reserved test evaluation are preserved.
- Short seeded checks and zero sampled overlaps do not establish continuous safety or real-road readiness. Static scenery is covered by native vehicle collision sensors, not by the traffic-only 2D pair audit.
- The next ML stage should begin with the bounded STOP / SLOW / PROCEED target-speed scope, retaining independent emergency and geometric safety checks. Compare it against this measured rule baseline before making performance claims.

For controls and scenario selection, read **INDIAN_SIMULATION.md** in the project root. No overnight automation was resumed.
"""
    (ROOT / "setup/carla_india/SCENARIO_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"verified_episodes": len(selected), "report": str(ROOT / "setup/carla_india/SCENARIO_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
