"""Native CARLA LHT neighbourhood: physics-controlled traffic and auditable episodes."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import shutil
import time
import traceback
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import carla
import numpy as np
import pygame
from india_road import junction_xml, validate_offline, LAYOUT_ARMS
from india_scenery import add_scenery
from india_behavior import Traffic, CASES, DT, polygon_gap
from india_playback import Playback, SPEEDS
from india_decisions import describe_ego
from india_conditions import PRESETS, weather_parameters
from india_signals import SignalLights
from india_camera import configure_rgb
from carla_starter import write_json

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("follow", "driver", "junction")
CAMERAS = {
    "follow": carla.Transform(carla.Location(x=-7.5, z=3.3), carla.Rotation(pitch=-14)),
    "driver": carla.Transform(carla.Location(x=.78, y=.30, z=1.10)),
    "junction": carla.Transform(carla.Location(x=-28, y=-22, z=16), carla.Rotation(yaw=37, pitch=-27)),
}


def road_markings(world, surface_style, active_arms=None):
    """Visual debug paint only; it does not change physics or road-model labels."""
    white = carla.Color(205, 200, 185) if surface_style == "asphalt" else carla.Color(70, 70, 65)
    for axis in (0, 1):
        for side in (-1, 1):
            if active_arms is not None and axis == 1 and ("south" if side > 0 else "north") not in active_arms:
                continue
            for start in range(16, 149, 7):
                xy = [(side * a, 0) if axis == 0 else (0, side * a) for a in (start, start + 3)]
                world.debug.draw_line(carla.Location(x=xy[0][0], y=xy[0][1], z=.04),
                                      carla.Location(x=xy[1][0], y=xy[1][1], z=.04),
                                      thickness=.08 if surface_style == "asphalt" else .025, color=white, life_time=1800)
        for edge in (-4.43, 4.43):
            for start, end in ((-151, -13), (13, 151)):
                if active_arms is not None and axis == 1 and ("south" if start > 0 else "north") not in active_arms:
                    continue
                a, b = ((start, edge), (end, edge)) if axis == 0 else ((edge, start), (edge, end))
                world.debug.draw_line(carla.Location(x=a[0], y=a[1], z=.04),
                                      carla.Location(x=b[0], y=b[1], z=.04),
                                      thickness=.10 if surface_style == "asphalt" else .035, color=white, life_time=1800)


def destroy_episode(client, world, actors, sensors):
    warnings = []
    for sensor in sensors:
        try:
            sensor.stop()
        except RuntimeError as error:
            warnings.append(str(error))
    replies = client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in reversed(actors)], False)
    warnings.extend(r.error for r in replies if r.has_error())
    world.tick(10)
    remaining = {a.id for a in world.get_actors()}
    if remaining.intersection(a.id for a in actors):
        warnings.append("Owned episode actor remained after batch destroy")
    return warnings


def episode(client, world, xml, args, out, case, seed, display, font, deadline):
    out.mkdir(exist_ok=False)
    owned, sensors, collisions, incidents = [], [], [], []
    frames = queue.Queue(maxsize=8)
    tick_count, image_count, sim_time = 0, 0, 0.
    stage_times = {}
    def measured(stage, started):
        now = time.perf_counter()
        stage_times.setdefault(stage, []).append(now-started)
        return now
    errors, warnings, geometry, offroad, wrongway = [], [], [], [], []
    state_cache, camera_records, stats, last_states = {}, [], {}, {}
    view, paused, running = args.view, False, True
    shown_view = args.view
    pause_frame, pause_results = None, []
    control_tests = {"pause": False, "resume": False, "tab": False, "screenshot": False, "screenshot_saved": False}
    pending_shot, shot_count, last_surface = False, 0, None
    mode_changed_at = -20
    camera = traffic = observer = live_preview = None
    lights = SignalLights(args.weather)
    close_pairs = set()
    playback = Playback(args.speed, dt=DT)
    speed_changes, speed_check_posted = [], set()
    saved_speed_hud = False
    last_decision, last_decision_signature = None, None
    last_image_frame = None
    pending_hud_shot = False
    saved_decision_huds = set()
    goal_time, exit_reason = None, "wall_limit"
    loop_start = None
    overlap_keys, lane_keys = set(), set()
    min_gap = float("inf")
    trace_file = (out / "actor_trace.csv").open("w", newline="", encoding="utf-8")
    trace = csv.writer(trace_file)
    trace.writerow(["frame", "sim_s", "id", "name", "type", "x", "y", "yaw_rad", "ex", "ey", "speed_mps",
                    "target_mps", "throttle", "brake", "steer", "reason", "route_s", "pass_state"])
    decision_file = (out / "ego_decisions.jsonl").open("w", encoding="utf-8")

    def emit(t, kind, name, **detail):
        incidents.append(dict(sim_s=round(t, 3), kind=kind, actor=name, **detail))

    def progress(stage):
        write_json(out / "progress.json", {"stage": stage, "case": case, "seed": seed,
                                          "sim_s": sim_time, "ticks": tick_count, "images": image_count,
                                          "playback_speed": playback.speed, "measured_speed": playback.actual})

    def result(cleanup):
        return {"case": case, "seed": seed, "exit_reason": exit_reason,
                "sim_s": sim_time, "ticks": tick_count, "camera_images": image_count,
                "loop_wall_s": None if loop_start is None else time.monotonic() - loop_start,
                "simulated_time_per_wall_second": None if loop_start is None else sim_time / max(.001, time.monotonic() - loop_start),
                "initial_playback_speed": args.speed, "final_playback_speed": playback.speed,
                "playback_speed_changes": speed_changes, "playback_speed_samples": playback.samples,
                "playback_physics_dt": DT, "playback_measurement_excludes_pauses": True,
                "decision_trace": "ego_decisions.jsonl", "decision_labels_camera_frame_linked": True,
                "decision_overlay_huds": sorted(saved_decision_huds),
                "camera_frames_aligned": bool(camera_records) and all(r["aligned"] for r in camera_records),
                "camera_max_delivery_delay_frames": max((r["delay"] for r in camera_records), default=None),
                "ego_goal_reached": goal_time is not None, "ego_goal_time_s": goal_time,
                "completed_overtakes": traffic.overtakes if traffic else 0,
                "overtake_block_reasons": sorted(traffic.blocked) if traffic else [],
                "collision_events": collisions, "all_actor_overlaps": geometry,
                "offroad_events": offroad, "wrongway_events": wrongway,
                "sampled_min_body_gap_m": min_gap if math.isfinite(min_gap) else None,
                "close_clearance_alerts": sum(e["kind"] == "close_body_clearance" for e in incidents),
                "light_state_changes": len(lights.records),
                "native_light_flags_verified": bool(lights.records) and all(r["verified"] for r in lights.records),
                "actor_stats": list(stats.values()), "errors": errors, "cleanup_warnings": warnings,
                "control_tests": control_tests, "pause_checks": pause_results, "cleanup_complete": cleanup,
                "runtime_passed": bool(tick_count >= 100 and image_count >= 50 and not errors and not collisions
                                       and not geometry and not offroad and not wrongway),
                "limits": ["Seeded illustrative Indian-style behaviour; not calibrated against traffic counts or trajectories",
                           "Rule-based controller uses simulator ground truth; no camera-model control and no RL",
                           "Geometric conflict reservations at the unsignalised junction are conservative and uncalibrated",
                           "Available scooters/bikes/bus/van/cars; no authentic autorickshaw asset installed",
                           "2D body checks are sampled at 20 Hz on a flat road; not a continuous safety proof"]}

    try:
        progress("spawning")
        traffic = Traffic(world, xml, owned, case, seed, emit, dataset_variation=bool(args.dataset_plan))
        write_json(out / "actors.json", traffic.metadata())
        for _ in range(35):
            world.tick(10)
        for d in traffic.drivers:
            sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=d.actor)
            owned.append(sensor)
            sensors.append(sensor)
            def collision(event, observed=d.name):
                impulse = event.normal_impulse
                collisions.append({"frame": event.frame, "time": event.timestamp, "actor": observed,
                                   "other_id": event.other_actor.id, "other_type": event.other_actor.type_id,
                                   "impulse": math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)})
            sensor.listen(collision)
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for key, value in {"image_size_x": "960", "image_size_y": "540", "fov": "95"}.items():
            bp.set_attribute(key, value)
        rgb_postprocessing = configure_rgb(bp)
        camera = world.spawn_actor(bp, carla.Transform())
        owned.append(camera)
        sensors.append(camera)
        def receive(image):
            try:
                if frames.full():
                    frames.get_nowait()
                frames.put_nowait(image)
            except queue.Empty:
                pass
        camera.listen(receive)
        if args.observe_road:
            from india_road_observer import RoadObserver
            if args.live_road_config:
                from india_live_preview import LiveRoadPreview
                live_preview = LiveRoadPreview(args.live_road_config, out)
            observer = RoadObserver(world, traffic.ego.actor, owned, sensors, out / "road_camera", args.surface_style,
                                    args.road_camera_hz, getattr(args, "dataset_camera", None),live_sink=live_preview)
        first_time = world.get_snapshot().timestamp.elapsed_seconds
        loop_start = time.monotonic()
        # Actor settling and sensor creation are not camera-delivery waiting time.
        no_image = loop_start
        clock = pygame.time.Clock()
        progress("running")
        while running and time.monotonic() < deadline:
            if live_preview is not None:
                live_preview.probe_events(tick_count,paused)
            if (ROOT / "setup/carla/STOP_DEMO.json").exists():
                exit_reason = "stop_marker"
                break
            if args.smoke and tick_count >= 15 and not control_tests["pause"]:
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                control_tests["pause"] = True
                pause_started = time.monotonic()
            elif args.smoke and control_tests["pause"] and not control_tests["resume"] and time.monotonic() - pause_started > .25:
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                control_tests["resume"] = True
            if args.smoke and tick_count >= 40 and not control_tests["tab"]:
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
                control_tests["tab"] = True
            if args.smoke and tick_count >= 65 and not control_tests["screenshot"]:
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
                control_tests["screenshot"] = True
            if args.replay_check and tick_count >= 260 and not args.replay_posted:
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
                args.replay_posted = True
            if args.speed_check:
                for step, key in ((100, pygame.K_EQUALS), (160, pygame.K_KP_PLUS),
                                  (220, pygame.K_PLUS), (280, pygame.K_3), (400, pygame.K_1),
                                  (440, pygame.K_MINUS), (480, pygame.K_2),
                                  (540, pygame.K_KP_MINUS), (600, pygame.K_3)):
                    if tick_count >= step and step not in speed_check_posted:
                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
                        speed_check_posted.add(step)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    exit_reason, running = "user_closed", False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        frame = world.get_snapshot().frame
                        if not paused:
                            pause_frame = frame
                        elif pause_frame is not None:
                            pause_results.append({"paused_frame": pause_frame, "resumed_frame": frame, "passed": pause_frame == frame})
                        paused = not paused
                    elif event.key == pygame.K_TAB:
                        view, mode_changed_at = VIEWS[(VIEWS.index(view) + 1) % len(VIEWS)], tick_count
                    elif event.key == pygame.K_s:
                        pending_hud_shot = True
                        if paused and last_surface is not None:
                            pygame.image.save(last_surface, str(out / f"{shown_view}_paused_{tick_count}.png"))
                            control_tests["screenshot_saved"] = True
                            shot_count += 1
                        else:
                            pending_shot = True
                    elif event.key == pygame.K_r:
                        exit_reason, running = "replay_requested", False
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_EQUALS,
                                       pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_1, pygame.K_2, pygame.K_3):
                        old_speed = playback.speed
                        if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                            playback.select({pygame.K_1: 1., pygame.K_2: 2., pygame.K_3: 3.}[event.key])
                        else:
                            playback.adjust(-1 if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) else 1)
                        if old_speed != playback.speed:
                            change = {"sim_s": sim_time, "old_speed": old_speed, "speed": playback.speed,
                                      "key": pygame.key.name(event.key), "paused": paused}
                            speed_changes.append(change)
                            emit(sim_time, "playback_speed_changed", "viewer", **{k: v for k, v in change.items() if k != "sim_s"})
                            progress("paused" if paused else "running")
            if not running:
                break
            if not paused:
                stage_start = time.perf_counter()
                snapshot = world.get_snapshot()
                sim_time = snapshot.timestamp.elapsed_seconds - first_time
                states = traffic.step(snapshot, sim_time)
                stage_start = measured("traffic_control", stage_start)
                lights.verify(traffic.drivers, snapshot.frame)
                decision = describe_ego(traffic, states, sim_time, snapshot.frame)
                signature = (decision["action"], decision["controller_reason"], decision["passing"]["reason"],
                             decision["passing"].get("leader"), tuple(decision["passing"].get("blocking_actors", [])))
                if signature != last_decision_signature:
                    emit(sim_time, "ego_decision_changed", "ego", source_frame=snapshot.frame,
                         action=decision["action"], controller_reason=decision["controller_reason"],
                         pass_reason=decision["passing"]["reason"], hud_lines=decision["hud_lines"])
                    last_decision_signature = signature
                meta = {d.actor.id: d for d in traffic.drivers}
                names = {d.actor.id: d.name for d in traffic.drivers}
                names.update({p.actor.id: p.name for p in traffic.walkers})
                for actor_id, state in states.items():
                    name = names[actor_id]
                    previous = last_states.get(actor_id)
                    moved = 0 if previous is None else math.hypot(state["x"] - previous["x"], state["y"] - previous["y"])
                    speed = moved / DT if previous is not None else 0
                    stat = stats.setdefault(actor_id, {"id": actor_id, "name": name, "distance_m": 0., "max_speed_mps": 0.,
                                                       "max_braking_mps2": 0., "max_abs_jerk_mps3": 0., "stopped_s": 0.})
                    stat["distance_m"] += moved
                    stat["max_speed_mps"] = max(stat["max_speed_mps"], speed)
                    if speed < .15:
                        stat["stopped_s"] += DT
                    if previous is not None and tick_count > 5:
                        acceleration = (speed - previous["measured_speed"]) / DT
                        stat["max_braking_mps2"] = max(stat["max_braking_mps2"], -acceleration)
                        if "measured_accel" in previous:
                            stat["max_abs_jerk_mps3"] = max(stat["max_abs_jerk_mps3"], abs(acceleration - previous["measured_accel"]) / DT)
                        state["measured_accel"] = acceleration
                    state["measured_speed"] = speed
                    d = meta.get(actor_id)
                    control = d.actor.get_control() if d else None
                    if d:
                        lights.update(d, control, sim_time, snapshot.frame)
                    trace.writerow([snapshot.frame, sim_time, actor_id, name, "vehicle" if d else "pedestrian",
                                    state["x"], state["y"], state["yaw"], state["ex"], state["ey"], speed,
                                    d.target if d else "", control.throttle if d else "", control.brake if d else "",
                                    control.steer if d else "", d.reason if d else "walk", d.s if d else "", d.pass_state if d else ""])
                    if d:
                        outside = any(not (abs(x) <= 12.2 and abs(y) <= 12.2 or abs(x) <= 152 and abs(y) <= 4.62 or abs(y) <= 152 and abs(x) <= 4.62
                                           or d.parked and 12 < x < 152 and -6.9 <= y <= -4.5)
                                      for x, y in state["polygon"])
                        wrong = ((abs(state["x"]) > 18 and abs(math.cos(state["yaw"])) > .8 and state["y"] * math.cos(state["yaw"]) > .1)
                                 or (abs(state["y"]) > 18 and abs(math.sin(state["yaw"])) > .8 and state["x"] * math.sin(state["yaw"]) < -.1))
                        for bad, label, destination in ((outside, "offroad", offroad), (wrong, "wrongway", wrongway)):
                            if bad and (actor_id, label) not in lane_keys:
                                destination.append({"frame": snapshot.frame, "sim_s": sim_time, "actor": name, "x": state["x"], "y": state["y"]})
                                emit(sim_time, label, name)
                                lane_keys.add((actor_id, label))
                last_states = states
                ids = list(states)
                for i, a in enumerate(ids):
                    for b in ids[i + 1:]:
                        sa, sb = states[a], states[b]
                        if math.hypot(sa["x"] - sb["x"], sa["y"] - sb["y"]) > math.hypot(sa["ex"], sa["ey"]) + math.hypot(sb["ex"], sb["ey"]) + 4:
                            continue
                        overlap, gap = polygon_gap(sa["polygon"], sb["polygon"])
                        min_gap = min(min_gap, gap)
                        pair = (a, b)
                        if 0 < gap < .5 and pair not in close_pairs:
                            emit(sim_time, "close_body_clearance", names[a], other=names[b], gap_m=gap,
                                 diagnostic_threshold_m=.5)
                            close_pairs.add(pair)
                        elif gap > .7:
                            close_pairs.discard(pair)
                        if overlap and (a, b) not in overlap_keys:
                            geometry.append({"frame": snapshot.frame, "sim_s": sim_time, "a": names[a], "b": names[b]})
                            emit(sim_time, "body_overlap", names[a], other=names[b])
                            overlap_keys.add((a, b))
                if traffic.ego.finished and states[traffic.ego.actor.id]["speed"] < .15 and goal_time is None:
                    goal_time = sim_time
                    emit(sim_time, "ego_goal_stopped", "ego")
                    if not args.smoke and not args.suite:
                        paused = True
                stage_start = measured("actor_audit_and_logging", stage_start)
                if args.smoke or args.suite:
                    auto_view = VIEWS[min(2, int(sim_time // 12))]
                    if view != auto_view and tick_count > 80:
                        view, mode_changed_at = auto_view, tick_count
                transform = CAMERAS[view]
                if view != "junction":
                    ego_transform = traffic.ego.actor.get_transform()
                    location = ego_transform.transform(carla.Location(transform.location))
                    transform = carla.Transform(location, carla.Rotation(pitch=CAMERAS[view].rotation.pitch, yaw=ego_transform.rotation.yaw))
                camera.set_transform(transform)
                stage_start = measured("viewer_camera_transform", stage_start)
                tick_started_ns = time.perf_counter_ns() if live_preview is not None else None
                frame = world.tick(10)
                tick_returned_ns = time.perf_counter_ns() if live_preview is not None else None
                fresh = world.get_snapshot()
                if live_preview is not None:
                    live_preview.register_tick(frame,fresh.timestamp.elapsed_seconds,tick_started_ns,tick_returned_ns)
                stage_start = measured("world_tick", stage_start)
                decision.update(world_frame=frame, world_timestamp=fresh.timestamp.elapsed_seconds)
                if observer is not None:
                    observer.tick(fresh, first_time, decision)
                stage_start = measured("road_capture_enqueue", stage_start)
                decision_file.write(json.dumps(decision, allow_nan=False) + "\n")
                state_cache[frame] = {"time": fresh.timestamp.elapsed_seconds, "view": view, "tick": tick_count,
                                      "decision": decision}
                for key in list(state_cache):
                    if key < frame - 20:
                        del state_cache[key]
                image = None
                while not frames.empty():
                    candidate = frames.get_nowait()
                    if candidate.frame in state_cache:
                        image = candidate
                if image is not None:
                    capture = state_cache[image.frame]
                    shown_view = capture["view"]
                    last_decision, last_image_frame = capture["decision"], image.frame
                    aligned = abs(capture["time"] - image.timestamp) < 1e-5
                    if not aligned:
                        raise RuntimeError("RGB image timestamp differs from its world snapshot")
                    camera_records.append({"frame": image.frame, "view": capture["view"], "time": image.timestamp,
                                           "delay": frame - image.frame, "aligned": aligned,
                                           "decision_source_frame": last_decision["source_frame"],
                                           "decision_world_frame": last_decision["world_frame"],
                                           "decision_hud_lines": last_decision["hud_lines"]})
                    rgb = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)[:, :, :3][:, :, ::-1]
                    last_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                    image_count += 1
                    no_image = time.monotonic()
                    if image_count in (30, 180, 300, 420, 600, 800, 1000) or pending_shot or tick_count - mode_changed_at == 8:
                        pygame.image.save(last_surface, str(out / f"{capture['view']}_{image.frame}.png"))
                        if pending_shot:
                            control_tests["screenshot_saved"] = True
                        pending_shot, shot_count = False, shot_count + 1
                elif time.monotonic() - no_image > 10:
                    raise TimeoutError("No aligned RGB image for ten seconds")
                tick_count += 1
                if tick_count % 20 == 0:
                    trace_file.flush()
                    decision_file.flush()
                    progress("running")
                measured("viewer_frame_processing", stage_start)
                case_seconds = min(args.sim_seconds, 60) if args.suite and case != "mixed" else args.sim_seconds
                if (args.smoke or args.suite) and sim_time >= case_seconds:
                    exit_reason = "bounded_test_complete"
                    break
            render_started = time.perf_counter()
            display.fill((16, 23, 31))
            live_draw = None
            if live_preview is not None:
                live_draw = live_preview.draw(display,font,last_surface)
            elif last_surface:
                display.blit(last_surface, (0, 0))
            top = f"{shown_view.upper()} | {'PAUSED' if paused else 'LIVE'} | {case} | Seed {seed} | {args.weather} | {sim_time:.1f}s | Passes {traffic.overtakes}"
            display.blit(font.render(top, True, (235, 235, 235)), (12, 550))
            playback.observe(time.monotonic(), sim_time, paused)
            actual = "paused" if paused else (f"{playback.actual:.2f}x" if playback.actual is not None else "measuring")
            speed_text = f"Speed {playback.speed:g}x | Actual {actual} | -/+ adjust | 1 / 2 / 3 direct"
            if args.observe_road:
                speed_text += f" | Recording {args.road_camera_hz} Hz"
            display.blit(font.render(speed_text, True, (180, 235, 170)), (12, 577))
            decision_lines = last_decision["hud_lines"] if last_decision else ["Waiting for the first time-aligned camera frame", ""]
            display.blit(font.render(decision_lines[0], True, (235, 235, 235)), (12, 604))
            display.blit(font.render(decision_lines[1], True, (255, 215, 130)), (12, 631))
            display.blit(font.render("TAB camera | SPACE pause | R replay | S screenshot | ESC close", True, (140, 210, 240)), (12, 658))
            footer = ("Live road preview | Driver: unchanged rules | Model / RL control: OFF" if live_preview is not None
                      else "Synthetic LHT traffic | Rules + ground truth | Road model / RL not connected")
            display.blit(font.render(footer, True, (240, 199, 112)), (12, 685))
            pygame.display.flip()
            if live_preview is not None:
                live_preview.presented(display,live_draw)
            if last_decision is not None:
                category = last_decision["passing"]["reason"]
                capture_decision_hud = (args.smoke and category in ("oncoming_gap", "pedestrian_conflict", "junction_too_close",
                                        "junction_zone", "return_gap", "passing", "returning", "overtake_completed")
                                        and category not in saved_decision_huds)
                if pending_hud_shot or capture_decision_hud:
                    label = category if capture_decision_hud else f"manual_{last_image_frame}_{shot_count}"
                    hud_path = out / f"decision_{label}.png"
                    pygame.image.save(display, str(hud_path))
                    write_json(hud_path.with_suffix(".json"), {"image_frame": last_image_frame,
                        "view": shown_view, "paused": paused, "decision": last_decision})
                    control_tests["decision_screenshot_saved"] = True
                    if capture_decision_hud:
                        saved_decision_huds.add(category)
                    pending_hud_shot = False
            if args.speed_check and playback.speed == 3 and playback.actual is not None and not saved_speed_hud:
                pygame.image.save(display, str(out / "speed_controls_3x.png"))
                saved_speed_hud = True
            measured("display", render_started)
            pacing_started = time.perf_counter()
            clock.tick(playback.fps_limit(paused))
            measured("requested_speed_pacing", pacing_started)
    except Exception:
        errors.append(traceback.format_exc())
        print(errors[-1], flush=True)
        exit_reason = "error"
    finally:
        if live_preview is not None:
            try:
                live_preview.finish()
            except Exception:
                errors.append(traceback.format_exc())
        if traffic is not None:
            try:
                lights.verify(traffic.drivers, world.get_snapshot().frame)
            except Exception:
                errors.append(traceback.format_exc())
        if observer is not None:
            try:
                observer.finish()
            except Exception:
                errors.append(traceback.format_exc())
        trace_file.close()
        decision_file.close()
        write_json(out / "performance.json", {"stages": {name: {
            "calls": len(values), "total_s": sum(values),
            "median_ms": float(np.median(values))*1000,
            "p95_ms": float(np.percentile(values, 95))*1000}
            for name, values in stage_times.items()}, "physics_dt_s": DT,
            "note": "Measured elapsed stages. Camera encoding runs concurrently; timing totals are not CPU utilization."})
        write_json(out / "incidents.json", {"events": incidents})
        write_json(out / "light_states.json", {"changes": lights.records, "source": "native vehicle-light flags"})
        write_json(out / "camera_frames.json", {"frames": camera_records})
        write_json(out / "result.json", result(False))
        try:
            warnings.extend(destroy_episode(client, world, owned, sensors))
        except Exception:
            warnings.append(traceback.format_exc())
        record = result(True)
        record["screenshots"] = shot_count
        write_json(out / "result.json", record)
        print(json.dumps({k: record[k] for k in ("case", "seed", "sim_s", "runtime_passed", "completed_overtakes", "errors", "cleanup_warnings")}), flush=True)
    args.speed = playback.speed
    return record


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=2050)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--sim-seconds", type=float, default=60)
    parser.add_argument("--speed", type=float, choices=SPEEDS, default=1.)
    parser.add_argument("--speed-check", action="store_true")
    parser.add_argument("--case", choices=CASES, default="mixed")
    parser.add_argument("--view", choices=VIEWS, default="follow")
    parser.add_argument("--india-probe", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--replay-check", action="store_true")
    parser.add_argument("--observe-road", action="store_true", help="Capture matching forward RGB and semantic frames")
    parser.add_argument("--dataset-plan", type=Path, help="Predeclared episode-separated dataset capture plan")
    parser.add_argument("--live-road-config",type=Path,help="Explicit bounded native-CARLA live-observer session")
    parser.add_argument("--road-camera-hz", type=int, choices=(1,5), default=5,
                        help="Paired evidence rate; independent of the physics and viewer camera")
    parser.add_argument("--surface-style", choices=("legacy", "asphalt"), default="asphalt")
    parser.add_argument("--weather", choices=tuple(PRESETS), default="clear")
    args, _ = parser.parse_known_args()
    if args.live_road_config and (not args.observe_road or args.road_camera_hz!=5):
        parser.error("Live road preview requires the explicit 5 Hz observation camera")
    dataset_entries = None
    dataset_layout = "crossroad"
    confirmation_scene = {}
    if args.dataset_plan:
        if not args.smoke or not args.observe_road or args.suite or args.replay_check or args.speed_check:
            parser.error("Dataset plan requires bounded road capture without interactive test suites")
        dataset_protocol = json.loads(args.dataset_plan.read_text(encoding="utf-8"))
        dataset_entries = dataset_protocol["episodes"]
        dataset_layout = dataset_protocol.get("layout", "crossroad")
        confirmation_scene = dataset_protocol.get("confirmation_scene", {})
        if confirmation_scene:
            from india_scenery import CONFIRMATION_FACADES
            from india_surface import CONFIRMATION_PALETTES
            if (dataset_protocol.get("role") != "fresh_confirmation_only"
                    or confirmation_scene.get("scenery") not in CONFIRMATION_FACADES
                    or confirmation_scene.get("asphalt_palette") not in CONFIRMATION_PALETTES
                    or not 10 <= confirmation_scene.get("junction_radius_m", 0) <= 12):
                parser.error("Confirmation scenes require declared supported geometry/appearance")
        if dataset_layout not in LAYOUT_ARMS:
            parser.error("Unknown left-hand dataset layout")
        if not dataset_entries or len({e["seed"] for e in dataset_entries}) != len(dataset_entries):
            parser.error("Dataset episodes need distinct seeds")
        for entry in dataset_entries:
            if (entry["case"] not in CASES or entry["weather"] not in PRESETS
                    or entry["split"] not in ("train", "val", "test")
                    or not 5 <= entry["sim_seconds"] <= 150):
                parser.error("Invalid dataset episode")
    args.replay_posted = False
    if args.speed_check and (not args.smoke or args.speed != 1 or args.sim_seconds < 36):
        parser.error("Speed control check needs --smoke, --speed 1 and at least 36 simulated seconds")
    if not 0 < args.seconds <= 1800 or not 5 <= args.sim_seconds <= 150:
        parser.error("Wall time 1..1800 seconds; test simulation time 5..150 seconds")
    args.output.mkdir(exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for name in ("carla_india_scene.py", "india_behavior.py", "india_road.py", "india_dataset_maps.py", "india_scenery.py", "india_playback.py", "india_decisions.py", "india_road_observer.py", "india_surface.py", "india_streetscape.py", "india_evidence_io.py", "india_conditions.py", "india_signals.py", "india_pedestrian_spacing.py", "india_camera.py"):
        shutil.copy2(ROOT / "integration" / name, source / name)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["dataset_plan"] = str(args.dataset_plan.resolve()) if args.dataset_plan else None
    config["live_road_config"] = str(args.live_road_config.resolve()) if args.live_road_config else None
    if args.live_road_config:
        shutil.copy2(ROOT / "integration/india_live_preview.py",source / "india_live_preview.py")
        shutil.copy2(ROOT / "simulation_live_protocol.py",source / "simulation_live_protocol.py")
    if dataset_entries:
        config["dataset_episodes"] = dataset_entries
    config.update(architecture="Hand-coded route following + gap checks + emergency braking; not RL", dt=DT,
                  scenario_version="lht-neighbourhood-v4.2", model_control=False, collision_disable=False,
                  appearance_collisions=False, weather_parameters=PRESETS[args.weather],
                  close_clearance_diagnostic_m=.5, pedestrian_entry_person_clearance_m=1.2,
                  live_pose_teleport=False, target_ego_mps=6.8, junction_mps=3.8,
                  requested_normal_accel_mps2=[-3.5, 2.1], normal_brake_cap=.25,
                  emergency_brake=1., rendered_camera=[960, 540],
                  playback_speeds=list(SPEEDS), playback_changes_physics=False,
                  decision_overlay="Observed controller reasons and thresholds; not consumed by the policy",
                  note="Actuator limits are commands; actual acceleration and jerk are measured separately")
    write_json(args.output / "config.json", config)
    owned, outcomes, warnings, errors = [], [], [], []
    world = original = None
    deadline = time.monotonic() + args.seconds
    pygame.init()
    display = pygame.display.set_mode((960, 714))
    pygame.display.set_caption("Indian traffic | Live road preview | SPACE pause | ESC close" if args.live_road_config else
                               "Indian traffic neighbourhood | -/+ speed to 3x | TAB camera | SPACE pause | R replay")
    font = pygame.font.Font(None, 24)
    display.fill((16, 23, 31))
    display.blit(font.render("Building the left-hand neighbourhood and traffic...", True, (235, 235, 235)), (80, 280))
    pygame.display.flip()
    try:
        owner = json.loads((ROOT / "setup/carla/server_owner.json").read_text(encoding="utf-8"))
        if owner.get("port") != args.port or owner.get("output") != str(args.output):
            raise RuntimeError("Refusing a world change without matching dedicated-server ownership")
        xml = junction_xml(dataset_layout, junction_radius=confirmation_scene.get("junction_radius_m", 12.))
        (args.output / "map.xodr").write_text(xml, encoding="utf-8")
        if dataset_layout == "crossroad":
            map_check = validate_offline(xml)
        else:
            from india_dataset_maps import validate_layout
            map_check = validate_layout(xml, dataset_layout)
        write_json(args.output / "map_check.json", map_check)
        client = carla.Client("127.0.0.1", args.port)
        client.set_timeout(120)
        write_json(args.output / "progress.json", {"stage": "generating_left_hand_world"})
        params = carla.OpendriveGenerationParameters(vertex_distance=1., max_road_length=30., wall_height=0.,
                    additional_width=0., smooth_junctions=True, enable_mesh_visibility=True, enable_pedestrian_navigation=True)
        world = client.generate_opendrive_world(xml, params)
        client.set_timeout(10)
        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, DT
        settings.substepping, settings.max_substep_delta_time, settings.max_substeps = True, .01, 10
        world.apply_settings(settings)
        world.set_weather(weather_parameters(args.weather, physical_atmosphere=bool(args.dataset_plan)))
        world.get_spectator().set_transform(CAMERAS["junction"])
        scenery = add_scenery(world, owned, confirmation_scene.get("scenery"))
        if args.surface_style != "legacy":
            from india_surface import apply_surface
            from india_streetscape import apply_streetscape
            active_arms = LAYOUT_ARMS[dataset_layout] if args.dataset_plan else None
            apply_surface(world, owned, args.output, args.surface_style, active_arms,
                          palette=confirmation_scene.get("asphalt_palette"))
            apply_streetscape(world, owned, args.output, active_arms)
        road_markings(world, args.surface_style, LAYOUT_ARMS[dataset_layout] if args.dataset_plan else None)
        write_json(args.output / "scenery.json", {"objects": scenery, "paint": "Visual debug lines only",
                                                  "confirmation_scene": confirmation_scene})
        cases = list(CASES) if args.suite else [args.case]
        plan = [(case, args.seed) for case in cases] + ([("mixed", 42)] if args.suite and args.seed != 42 else [])
        if dataset_entries:
            plan = [(entry["case"], entry["seed"]) for entry in dataset_entries]
        for episode_index, (case, seed) in enumerate(plan):
            if dataset_entries:
                entry = dataset_entries[episode_index]
                args.weather, args.sim_seconds = entry["weather"], entry["sim_seconds"]
                args.dataset_camera = entry["camera_profile"]
                condition = weather_parameters(args.weather, physical_atmosphere=True)
                condition.sun_azimuth_angle = entry["sun_azimuth_deg"]
                for key,value in entry.get("weather_probe_overrides",{}).items():
                    if not entry["camera_profile"].get("weather_probe_only") or key not in ("cloudiness","sun_altitude_angle","sun_azimuth_angle"):
                        raise ValueError("Weather overrides are restricted to declared diagnostic probes")
                    setattr(condition,key,value)
                world.set_weather(condition)
            while time.monotonic() < deadline:
                output = args.output / f"episode_{len(outcomes) + 1:02}_{case}_seed{seed}"
                write_json(args.output / "progress.json", {"stage": "episode", "output": str(output), "completed": len(outcomes)})
                record = episode(client, world, xml, args, output, case, seed, display, font, deadline)
                if dataset_entries:
                    write_json(output / "dataset_episode.json", entry)
                outcomes.append(record)
                if record["exit_reason"] == "replay_requested":
                    continue
                break
            if not outcomes or outcomes[-1]["exit_reason"] in ("user_closed", "stop_marker", "error", "wall_limit"):
                break
    except Exception:
        errors.append(traceback.format_exc())
        print(errors[-1], flush=True)
    finally:
        write_json(args.output / "result.json", {"episodes": outcomes, "errors": errors, "cleanup_complete": False})
        if world is not None and original is not None:
            try:
                world.apply_settings(original)
                replies = client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in reversed(owned)], False)
                warnings.extend(r.error for r in replies if r.has_error())
            except Exception:
                warnings.append(traceback.format_exc())
        pygame.quit()
        summary = {"kind": "indian_lht_neighbourhood", "episodes": outcomes, "errors": errors,
                   "cleanup_warnings": warnings, "cleanup_complete": not warnings,
                   "passed": bool(outcomes) and not errors and not warnings and all(r["runtime_passed"] and not r["cleanup_warnings"] for r in outcomes)}
        write_json(args.output / "result.json", summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
