"""Live CARLA installation/demo check. Stock traffic rules, not the Indian scenario or learned driving."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import queue
import random
import shutil
import time
import traceback
import xml.etree.ElementTree as ET

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import carla
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
VIEWS = {
    "follow": carla.Transform(carla.Location(x=-6.5, z=3.0), carla.Rotation(pitch=-13)),
    "driver": carla.Transform(carla.Location(x=0.25, y=0.32, z=1.35)),
}


def write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2050)
    parser.add_argument("--tm-port", type=int, default=8050)
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=600, help="Wall time limit (maximum 1800)")
    parser.add_argument("--smoke", action="store_true", help="Bounded camera/control verification")
    parser.add_argument("--smoke-frames", type=int, default=400)
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--walkers", type=int, default=10)
    parser.add_argument("--renderer", default="vulkan")
    parser.add_argument("--quality", default="Epic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1800:
        parser.error("--seconds must be >0 and <=1800")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or ROOT / "runs" / f"carla_setup_{stamp}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "carla_starter_source.py")
    shutil.copy2(ROOT / "setup/carla/run_session.py", output / "run_session_source.py")
    lock = ROOT / "setup/carla/viewer.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Another CARLA viewer may be running. Inspect setup/carla/viewer.lock before retrying.")
    lock_identity = {"pid": os.getpid(), "output": str(output)}
    os.write(lock_fd, json.dumps(lock_identity).encode("utf-8"))
    os.close(lock_fd)
    rng = random.Random(args.seed)
    world = tm = original_settings = camera = collision_sensor = ego = None
    owned, controllers, collisions = [], [], []
    collision_sensors, actor_stats, actor_last_positions, actor_last_times = [], {}, {}, {}
    errors, warnings = [], []
    frames, distance, speed_max, image_frames = 0, 0.0, 0.0, 0
    state_cache = {}
    capture_delays = []
    shown_view = "follow"
    no_image_since = time.monotonic()
    first_timestamp = last_timestamp = None
    saved_views, camera_frames = set(), []
    paused = False
    pause_observed = False
    pause_checks, pause_frame = [], None
    smoke_actions = {"pause": False, "resume": False, "tab": False, "screenshot": False, "escape": False}
    key_screenshot_saved = False
    smoke_pause_at = None
    view = "follow"
    sim_dt = 0.05
    started = time.monotonic()
    deadline = started + args.seconds
    last_position = None
    exit_reason = "time_limit"
    images = queue.Queue(maxsize=8)
    config = vars(args).copy()
    config["output"] = str(output)
    config.update({"kind": "stock_map_installation_check", "camera_drives_vehicle": False,
                   "controller": "CARLA Traffic Manager autopilot", "rl": False,
                   "fixed_delta_seconds": sim_dt, "view_modes": list(VIEWS),
                   "seed_note": "Seeded setup, not independently verified deterministic replay."})
    write_json(output / "config.json", config)
    trace_file = (output / "ego_trace.csv").open("w", newline="", encoding="utf-8")
    trace = csv.writer(trace_file)
    trace.writerow(["frame", "sim_seconds", "x", "y", "z", "speed_mps", "throttle", "brake", "steer", "camera"])
    actor_file = (output / "actor_trace.csv").open("w", newline="", encoding="utf-8")
    actor_trace = csv.writer(actor_file)
    actor_trace.writerow(["frame", "sim_seconds", "actor_id", "type", "x", "y", "speed_mps"])

    def progress(stage):
        write_json(output / "progress.json", {"stage": stage, "wall_seconds": time.monotonic() - started,
                                              "utc": datetime.now(timezone.utc).isoformat()})

    def capture_result(cleanup_complete):
        moving_walkers = sum(item["type"].startswith("walker.pedestrian.") and item["distance_m"] > 0.3 for item in actor_stats.values())
        moving_npcs = sum(item["type"].startswith("vehicle.") and item["id"] != config.get("ego_actor_id")
                          and item["distance_m"] > 0.3 for item in actor_stats.values())
        runtime_passed = bool(args.smoke and frames >= args.smoke_frames and saved_views == set(VIEWS)
                              and all(smoke_actions.values()) and pause_observed
                              and pause_checks and all(check["world_stayed_still"] for check in pause_checks)
                              and key_screenshot_saved and distance > 1.0 and image_frames == frames
                              and config.get("spawned_npcs", 0) > 0 and config.get("spawned_walkers", 0) > 0
                              and moving_walkers > 0 and moving_npcs > 0
                              and not collisions and not errors)
        return {"exit_reason": exit_reason, "frames": frames, "camera_images": image_frames,
                "wall_seconds": time.monotonic() - started,
                "sim_seconds": None if last_timestamp is None else last_timestamp - first_timestamp + sim_dt,
                "ego_distance_m": distance, "ego_max_speed_mps": speed_max,
                "saved_views": sorted(saved_views), "collisions": collisions,
                "ego_collision_sensor_only": False, "errors": errors, "cleanup_warnings": warnings,
                "actor_stats": list(actor_stats.values()), "moving_walkers": moving_walkers, "moving_npcs": moving_npcs,
                "actor_speed_source": "Finite differences of time-aligned XY positions; native pedestrian velocity may report zero",
                "collision_coverage": "Vehicle-mounted sensors on ego and all spawned NPC vehicles",
                "cleanup_complete": cleanup_complete,
                "camera_frame_alignment": bool(camera_frames) and all(item["world_frame"] == item["image_frame"] for item in camera_frames),
                "camera_delivery_delay_frames_max": max(capture_delays, default=None),
                "smoke_key_events": smoke_actions, "pause_observed": pause_observed,
                "pause_checks": pause_checks, "key_screenshot_saved": key_screenshot_saved,
                "runtime_check_passed": runtime_passed,
                "smoke_passed": runtime_passed and cleanup_complete and not warnings,
                "limitations": ["Stock map road rules; not a calibrated Indian junction",
                                "Traffic Manager controls ego; camera is observation only",
                                "No road-model or RL control; no general overtaking validation",
                                "Vehicle collision sensors do not establish continuous all-actor safety"]}

    try:
        progress("opening_window")
        pygame.init()
        display = pygame.display.set_mode((960, 600))
        pygame.display.set_caption("Tesla project - CARLA live starter | TAB camera | SPACE pause | ESC exit")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()
        client = carla.Client("127.0.0.1", args.port)
        client.set_timeout(90)
        progress("connecting")
        server_version = client.get_server_version()
        if server_version != client.get_client_version():
            raise RuntimeError(f"Client/server mismatch: {client.get_client_version()} / {server_version}")
        world = client.get_world()
        # Map changes are allowed only in the dedicated server started by our launcher.
        owner = json.loads((ROOT / "setup/carla/server_owner.json").read_text(encoding="utf-8"))
        if owner.get("port") != args.port:
            raise RuntimeError("No matching owned-server record; refusing to change the world.")
        maps = client.get_available_maps()
        matches = [name for name in maps if name.rsplit("/", 1)[-1] == args.map]
        if not matches:
            raise RuntimeError(f"Map {args.map} unavailable; installed maps: {maps}")
        progress("loading_map")
        if world.get_map().name.rsplit("/", 1)[-1] != args.map:
            raise RuntimeError("Startup map differs from the requested map; refusing an unsafe implicit map reload")
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = sim_dt
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(args.seed)
        tm.set_global_distance_to_leading_vehicle(3.0)
        tm.global_percentage_speed_difference(30)
        world.set_pedestrians_seed(args.seed)
        world.set_pedestrians_cross_factor(0.15)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        progress("reading_map")
        carla_map = world.get_map()
        roads = ET.fromstring(carla_map.to_opendrive()).findall("road")
        config.update({"server_version": server_version, "map_loaded": carla_map.name,
                       "map_road_rules": sorted({road.attrib.get("rule", "RHT(default)") for road in roads}),
                       "available_maps": maps, "quality": args.quality, "renderer": args.renderer})
        library = world.get_blueprint_library()
        spawn_points = list(carla_map.get_spawn_points())
        rng.shuffle(spawn_points)
        # Prefer an initial straight segment so the installation check can observe motion.
        ego_spawns = []
        for spawn in spawn_points:
            waypoint = carla_map.get_waypoint(spawn.location)
            ahead = waypoint.next(20) if waypoint is not None else []
            if waypoint is not None and not waypoint.is_junction and ahead and not ahead[0].is_junction:
                ego_spawns.append(spawn)
        ego_bp = library.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "tesla_setup_ego")
        for spawn in ego_spawns + spawn_points:
            ego = world.try_spawn_actor(ego_bp, spawn)
            if ego is not None:
                owned.append(ego)
                break
        if ego is None:
            raise RuntimeError("No free spawn location for ego vehicle")
        progress("spawning_traffic")
        ego.set_autopilot(True, args.tm_port)
        tm.auto_lane_change(ego, False)
        tm.vehicle_percentage_speed_difference(ego, 35)
        vehicle_bps = sorted(library.filter("vehicle.*"), key=lambda bp: bp.id)
        npc_count = 0
        ego_location = ego.get_location()
        nearby_spawns = sorted(spawn_points, key=lambda spawn: spawn.location.distance(ego_location))
        for spawn in nearby_spawns:
            if npc_count >= args.vehicles:
                break
            if spawn.location.distance(ego_location) < 12:
                continue
            bp = rng.choice(vehicle_bps)
            bp.set_attribute("role_name", "tesla_setup_npc")
            actor = world.try_spawn_actor(bp, spawn)
            if actor is not None:
                owned.append(actor)
                actor.set_autopilot(True, args.tm_port)
                npc_count += 1
        walker_count = 0
        walker_bps = sorted(library.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
        for _ in range(args.walkers * 20):
            if walker_count >= args.walkers or not walker_bps:
                break
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            if loc.distance(ego_location) > 100:
                continue
            bp = rng.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            actor = world.try_spawn_actor(bp, carla.Transform(loc))
            if actor is None:
                continue
            owned.append(actor)
            controller = world.try_spawn_actor(library.find("controller.ai.walker"), carla.Transform(), actor)
            if controller is not None:
                controllers.append(controller)
                walker_count += 1
            else:
                actor.destroy()
                owned.remove(actor)
        world.tick()
        progress("starting_walkers")
        for controller in controllers:
            controller.start()
            target = world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)
            controller.set_max_speed(rng.uniform(0.9, 1.5))
        camera_bp = library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "960")
        camera_bp.set_attribute("image_size_y", "540")
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", "0")
        camera = world.spawn_actor(camera_bp, VIEWS[view], attach_to=ego)
        owned.append(camera)

        def receive(image):
            try:
                images.put_nowait(image)
            except queue.Full:
                # Display backlog is bounded. Never label a stale image as the current tick.
                try:
                    images.get_nowait()
                except queue.Empty:
                    pass
                images.put_nowait(image)

        camera.listen(receive)
        for vehicle in [actor for actor in owned if actor.type_id.startswith("vehicle.")]:
            sensor = world.spawn_actor(library.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)
            owned.append(sensor)
            collision_sensors.append(sensor)

            def record_collision(event, observed_id=vehicle.id):
                impulse = event.normal_impulse
                collisions.append({"frame": event.frame, "sim_seconds": event.timestamp,
                                   "observed_vehicle_id": observed_id,
                                   "other_id": event.other_actor.id, "other_type": event.other_actor.type_id,
                                   "impulse_norm": math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)})

            sensor.listen(record_collision)
        for actor in owned:
            if actor.type_id.startswith(("vehicle.", "walker.pedestrian.")):
                actor_stats[actor.id] = {"id": actor.id, "type": actor.type_id, "distance_m": 0.0, "max_speed_mps": 0.0}
        config.update({"spawned_npcs": npc_count, "spawned_walkers": walker_count,
                       "ego_actor_id": ego.id, "vehicle_blueprints_available": [bp.id for bp in vehicle_bps]})
        write_json(output / "config.json", config)
        progress("live_camera")
        no_image_since = time.monotonic()
        print(f"CARLA {server_version}: {config['map_loaded']}, {npc_count} vehicles, {walker_count} walkers", flush=True)
        print("TAB: driver/follow | SPACE: pause | S: screenshot | ESC: close and clean up", flush=True)
        running, last_surface = True, None
        while running and time.monotonic() < deadline:
            if (ROOT / "setup/carla/STOP_DEMO.json").exists():
                exit_reason = "stop_marker"
                break
            screenshot = False
            # Exercise the real pygame key handlers during the bounded smoke check.
            if args.smoke:
                if frames >= 15 and not smoke_actions["pause"]:
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                    smoke_actions["pause"] = True
                    smoke_pause_at = time.monotonic()
                elif smoke_actions["pause"] and not smoke_actions["resume"] and time.monotonic() - smoke_pause_at > 0.3:
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                    smoke_actions["resume"] = True
                if frames >= 35 and not smoke_actions["tab"]:
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
                    smoke_actions["tab"] = True
                if frames >= 60 and not smoke_actions["screenshot"]:
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
                    smoke_actions["screenshot"] = True
                if frames >= args.smoke_frames and not smoke_actions["escape"]:
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
                    smoke_actions["escape"] = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                    exit_reason = "smoke_complete" if args.smoke and smoke_actions["escape"] else "user_closed"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        current_frame = world.get_snapshot().frame
                        if not paused:
                            pause_frame = current_frame
                        else:
                            pause_checks.append({"paused_at_frame": pause_frame, "resumed_at_frame": current_frame,
                                                 "world_stayed_still": pause_frame == current_frame})
                        paused = not paused
                    elif event.key == pygame.K_TAB:
                        view = "driver" if view == "follow" else "follow"
                        camera.set_transform(VIEWS[view])
                    elif event.key == pygame.K_s:
                        screenshot = True
            if not running:
                break
            if paused:
                pause_observed = True
                if screenshot and last_surface is not None:
                    pygame.image.save(last_surface, str(output / f"{shown_view}_paused_{frames}.png"))
                    key_screenshot_saved = True
            if not paused:
                frame = world.tick(10)
                snapshot = world.get_snapshot()
                for actor_id, stats in actor_stats.items():
                    observed = snapshot.find(actor_id)
                    if observed is None:
                        continue
                    actor_location = observed.get_transform().location
                    actor_velocity = observed.get_velocity()
                    actor_speed = math.hypot(actor_velocity.x, actor_velocity.y)
                    previous = actor_last_positions.get(actor_id)
                    if previous is not None:
                        actor_distance = math.hypot(actor_location.x - previous.x, actor_location.y - previous.y)
                        stats["distance_m"] += actor_distance
                        elapsed = snapshot.timestamp.elapsed_seconds - actor_last_times[actor_id]
                        actor_speed = actor_distance / elapsed if elapsed > 0 else 0.0
                    actor_last_positions[actor_id] = actor_location
                    actor_last_times[actor_id] = snapshot.timestamp.elapsed_seconds
                    stats["max_speed_mps"] = max(stats["max_speed_mps"], actor_speed)
                    actor_trace.writerow([frame, snapshot.timestamp.elapsed_seconds, actor_id, stats["type"],
                                          actor_location.x, actor_location.y, actor_speed])
                actor_snapshot = snapshot.find(ego.id)
                if actor_snapshot is None or snapshot.frame != frame:
                    raise RuntimeError("Missing ego state or unexpected world snapshot frame")
                transform = actor_snapshot.get_transform()
                velocity = actor_snapshot.get_velocity()
                control = ego.get_control()
                state_cache[frame] = {"time": snapshot.timestamp.elapsed_seconds, "location": transform.location,
                                      "velocity": velocity, "control": control, "view": view}
                for old_frame in list(state_cache):
                    if old_frame < frame - 20:
                        del state_cache[old_frame]
                image = None
                # GPU readback arrives a frame later on this machine. Keep advancing,
                # then pair each image with its actual capture snapshot, never a newer pose.
                while True:
                    try:
                        candidate = images.get_nowait()
                    except queue.Empty:
                        break
                    if candidate.frame > frame:
                        raise RuntimeError(f"Camera advanced beyond world tick: {candidate.frame} > {frame}")
                    if candidate.frame in state_cache:
                        image = candidate
                if image is not None:
                    no_image_since = time.monotonic()
                    capture = state_cache[image.frame]
                    sim_time = capture["time"]
                    if abs(image.timestamp - sim_time) > 1e-5:
                        raise RuntimeError("Camera timestamp does not match its capture snapshot")
                    if first_timestamp is None:
                        first_timestamp = sim_time
                    last_timestamp = sim_time
                    shown_view = capture["view"]
                    rgb = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3][:, :, ::-1]
                    last_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                    image_frames += 1
                    capture_delays.append(frame - image.frame)
                    camera_frames.append({"world_frame": image.frame, "image_frame": image.frame,
                                          "latest_world_frame": frame, "view": shown_view,
                                          "image_timestamp": image.timestamp, "snapshot_timestamp": sim_time})
                    if (shown_view == "follow" and frames == 25) or (shown_view == "driver" and frames == 50) or screenshot:
                        pygame.image.save(last_surface, str(output / f"{shown_view}_{image.frame}.png"))
                        saved_views.add(shown_view)
                        if screenshot:
                            key_screenshot_saved = True
                    velocity = capture["velocity"]
                    speed = math.hypot(velocity.x, velocity.y)
                    speed_max = max(speed_max, speed)
                    location = capture["location"]
                    if last_position is not None:
                        distance += math.hypot(location.x - last_position.x, location.y - last_position.y)
                    last_position = location
                    control = capture["control"]
                    trace.writerow([image.frame, sim_time, location.x, location.y, location.z, speed,
                                    control.throttle, control.brake, control.steer, shown_view])
                    frames += 1
                    if frames % 20 == 0:
                        trace_file.flush()
                        actor_file.flush()
                        progress("live_camera")
                elif time.monotonic() - no_image_since > 10:
                    raise TimeoutError("RGB stream produced no usable image for ten seconds while ticking")
            display.fill((18, 24, 33))
            if last_surface is not None:
                display.blit(last_surface, (0, 0))
            label = f"{shown_view.upper()} | {'PAUSED' if paused else 'LIVE'} | TAB camera | SPACE pause | ESC exit"
            display.blit(font.render(label, True, (240, 240, 240)), (12, 550))
            display.blit(font.render("Stock-map setup check | CARLA autopilot | Road model not connected", True, (245, 200, 110)), (12, 576))
            pygame.display.flip()
            clock.tick(20)
    except Exception:
        errors.append(traceback.format_exc())
        exit_reason = "error"
        print(errors[-1], flush=True)
    finally:
        trace_file.flush()
        actor_file.flush()
        write_json(output / "camera_frames.json", {"frames": camera_frames})
        write_json(output / "result.json", capture_result(False))
        progress("cleanup_camera_stop")
        if camera is not None:
            try:
                camera.stop()
            except Exception as error:
                warnings.append(f"camera stop: {error}")
        for sensor in collision_sensors:
            try:
                sensor.stop()
            except Exception as error:
                warnings.append(f"collision stop: {error}")
        progress("cleanup_restore_sync")
        if tm is not None:
            try:
                tm.set_synchronous_mode(False)
            except Exception as error:
                warnings.append(f"TM restore: {error}")
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception as error:
                warnings.append(f"settings restore: {error}")
        progress("cleanup_actors")
        for controller in controllers:
            try:
                controller.stop()
            except Exception as error:
                warnings.append(f"walker stop: {error}")
        destroy_ids = [actor.id for actor in list(reversed(controllers)) + list(reversed(owned))]
        if destroy_ids:
            try:
                responses = client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in destroy_ids], False)
                warnings.extend(response.error for response in responses if response.has_error())
                time.sleep(0.2)
                remaining = set(destroy_ids).intersection(actor.id for actor in world.get_actors())
                if remaining:
                    warnings.append(f"Owned actors still present after cleanup: {sorted(remaining)}")
            except Exception as error:
                warnings.append(f"batch actor cleanup: {error}")
        progress("cleanup_window")
        trace_file.close()
        actor_file.close()
        pygame.quit()
        if lock.exists():
            try:
                if json.loads(lock.read_text(encoding="utf-8")) == lock_identity:
                    lock.unlink()
            except (OSError, ValueError):
                warnings.append("Viewer lock changed; left it in place for inspection")
        result = capture_result(True)
        write_json(output / "result.json", result)
        progress("finished")
        write_json(output / "camera_frames.json", {"frames": camera_frames})
        print(json.dumps(result, indent=2), flush=True)
    return 1 if errors or (args.smoke and not result["smoke_passed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
