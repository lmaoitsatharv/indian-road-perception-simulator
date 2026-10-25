"""Native preview and capture for five separate Indian road worlds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import time
import traceback

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
import carla
import numpy as np
import pygame

from carla_starter import write_json
from india_camera import configure_rgb
from india_conditions import weather_parameters
from india_world_maps import PROFILES,build_world,validate_world
from india_world_scenery import WorldScenery
from india_world_traffic import WorldTraffic
from india_behavior import polygon_gap

ROOT = Path(__file__).resolve().parents[1]
DT = .05


def save_native(image, path):
    values = np.frombuffer(image.raw_data,dtype=np.uint8).reshape(image.height,image.width,4)
    pygame.image.save(pygame.surfarray.make_surface(values[...,[2,1,0]].swapaxes(0,1)),str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_surface(image):
    array = np.frombuffer(image.raw_data,dtype=np.uint8).reshape(image.height,image.width,4)
    return pygame.surfarray.make_surface(array[...,[2,1,0]].swapaxes(0,1))


def wait_frame(pending, frame):
    deadline = time.monotonic()+12
    while time.monotonic() < deadline:
        try:
            image = pending.get(timeout=min(1,max(.01,deadline-time.monotonic())))
        except queue.Empty:
            continue
        if image.frame == frame:
            return image
        if image.frame > frame:
            raise RuntimeError("Camera advanced beyond expected world frame")
    raise TimeoutError("Native camera did not deliver its exact world frame")


def cleanup(client,world,owned,sensors):
    failures = []
    for sensor in sensors:
        try:
            sensor.stop()
        except RuntimeError as error:
            failures.append(str(error))
    replies = client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in reversed(owned)],False)
    failures.extend(r.error for r in replies if r.has_error())
    world.tick(15)
    remaining = set(a.id for a in world.get_actors()) & {a.id for a in owned}
    if remaining:
        failures.append("Owned actors remain: "+str(sorted(remaining)))
    return failures


def run_world(client,key,args,folder,display):
    folder.mkdir(exist_ok=False)
    owned,sensors,streams = [],[],{}
    result = {"profile":key,"passed":False,"errors":[],"cleanup_complete":False,
        "model_control":False,"rl_trained":False,"training_started":False,
        "wrong_way_definition":"Vehicle heading differs over 70 degrees from its connected native route while moving; connector/approach overlaps do not select a nearest opposite lane"}
    trace_file = frame_file = None
    world = None
    try:
        profile = PROFILES[key]
        xml = build_world(key)
        (folder/"map.xodr").write_text(xml,encoding="utf-8")
        write_json(folder/"map_check.json",validate_world(key,xml))
        display.fill((16,23,29))
        font = pygame.font.Font(None,32)
        display.blit(font.render("Building "+profile.title+"...",True,(241,231,210)),(70,240))
        pygame.display.flip()
        params = carla.OpendriveGenerationParameters(vertex_distance=.6,max_road_length=35.,wall_height=0.,
            additional_width=0.,smooth_junctions=True,enable_mesh_visibility=True,enable_pedestrian_navigation=True)
        client.set_timeout(120)
        world = client.generate_opendrive_world(xml,params)
        client.set_timeout(20)
        settings = world.get_settings()
        settings.synchronous_mode,settings.fixed_delta_seconds = True,DT
        settings.substepping,settings.max_substep_delta_time,settings.max_substeps = True,.01,10
        world.apply_settings(settings)
        condition = weather_parameters(args.weather,physical_atmosphere=True)
        world.set_weather(condition)
        library = world.get_blueprint_library()
        write_json(folder/"native_asset_inventory.json",{"server_version":client.get_server_version(),
            "vehicles":[{"id":bp.id,"wheels":bp.get_attribute("number_of_wheels").as_int() if bp.has_attribute("number_of_wheels") else None,
                "base_type":str(bp.get_attribute("base_type")) if bp.has_attribute("base_type") else None} for bp in library.filter("vehicle.*")],
            "autorickshaw_blueprints":[bp.id for bp in library.filter("vehicle.*") if any(w in bp.id.lower() for w in ("rickshaw","bajaj","tuktuk","tuk_tuk"))]})
        scenery = WorldScenery(world,owned,key,xml)
        write_json(folder/"progress.json",{"stage":"building_native_scenery"})
        record = scenery.build()
        write_json(folder/"scenery.json",record)
        write_json(folder/"progress.json",{"stage":"spawning_native_traffic"})
        traffic = WorldTraffic(world,client,owned,key,xml,args.seed)
        sensors.extend(traffic.sensor_actors)
        write_json(folder/"actors.json",{"actors":traffic.records,"autorickshaw":"Unavailable in packaged runtime; source model staged separately"})
        for kind in ("rgb","semantic_segmentation","instance_segmentation","depth"):
            bp = library.find("sensor.camera."+kind)
            for name,value in {"image_size_x":"960","image_size_y":"540","fov":"95","sensor_tick":"0.20"}.items():
                bp.set_attribute(name,value)
            if kind == "rgb":
                configure_rgb(bp)
            sensor = world.spawn_actor(bp,carla.Transform(carla.Location(x=1.30,y=.23,z=1.45)),attach_to=traffic.ego.actor)
            owned.append(sensor)
            sensors.append(sensor)
            pending = queue.Queue()
            sensor.listen(pending.put)
            streams[kind] = pending
        bp = library.find("sensor.camera.rgb")
        for name,value in {"image_size_x":"640","image_size_y":"480","fov":"85","sensor_tick":"0.20"}.items():
            bp.set_attribute(name,value)
        configure_rgb(bp)
        midpoint = world.get_map().get_waypoint_xodr(1,1,profile.length/2).transform.location
        cx,cy = (0.,0.) if profile.arms else (midpoint.x,midpoint.y)
        overview = world.spawn_actor(bp,carla.Transform(carla.Location(x=cx-45,y=cy-60,z=105),
            carla.Rotation(pitch=-54.5,yaw=53.13)))
        owned.append(overview)
        sensors.append(overview)
        streams["overview"] = queue.Queue()
        overview.listen(streams["overview"].put)
        for name in streams:
            (folder/name).mkdir()
        trace_file = (folder/"actor_trace.jsonl").open("w",encoding="utf-8")
        frame_file = (folder/"camera_frames.jsonl").open("w",encoding="utf-8")
        start_time = world.get_snapshot().timestamp.elapsed_seconds
        deadline = time.monotonic()+args.seconds
        paused,exit_requested,frame_count,ticks = False,False,0,0
        saved_frames = 0
        last_rgb,last_overview = None,None
        overlaps,departures,wrong_way = [],[],[]
        min_gap = float("inf")
        controls = {"pause":False,"resume":False,"screenshot":False}
        wall_start = time.monotonic()
        while time.monotonic() < deadline:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    exit_requested = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    pygame.image.save(display,str(folder/f"manual_{ticks:06d}.png"))
                    controls["screenshot"] = True
            if exit_requested or (ROOT/"setup/carla/STOP_DEMO.json").exists():
                break
            if paused:
                time.sleep(.025)
                continue
            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds-start_time
            if sim_time >= args.sim_seconds:
                break
            states,trace = traffic.update(snapshot,sim_time)
            trace_file.write(json.dumps({"frame":snapshot.frame,"sim_s":sim_time,"actors":trace})+"\n")
            # Native map and independently sampled body geometry, not model masks.
            bodies = list(states.items())
            for i,(aid,a) in enumerate(bodies):
                for bid,b in bodies[i+1:]:
                    if math.hypot(a["x"]-b["x"],a["y"]-b["y"]) > 20:
                        continue
                    overlap,gap = polygon_gap(a["polygon"],b["polygon"])
                    min_gap = min(min_gap,gap)
                    if overlap:
                        overlaps.append({"frame":snapshot.frame,"a":aid,"b":bid,"gap_m":gap})
            trace_by_id = {row["id"]:row for row in trace}
            for actor_id,state in states.items():
                wp = world.get_map().get_waypoint(carla.Location(x=state["x"],y=state["y"],z=0),False,carla.LaneType.Driving)
                if wp is None:
                    departures.append({"frame":snapshot.frame,"id":actor_id,"xy":[state["x"],state["y"]]})
                # Junction approaches overlap connector surfaces. Nearest-lane
                # lookup can select the opposite approach even on a legal turn;
                # compare direction with the explicitly connected native route.
                elif state["speed"] > .2 and abs(math.remainder(state["yaw"]-trace_by_id[actor_id]["expected_yaw"],2*math.pi)) > math.radians(70):
                    wrong_way.append({"frame":snapshot.frame,"id":actor_id})
            native_frame = world.tick(15)
            ticks += 1
            # sensor_tick may be phase-aligned to map creation, not our loop.
            # Wait for RGB at its own exact frame, then require all paired sensors.
            time.sleep(.001)
            if not streams["rgb"].empty():
                rgb = streams["rgb"].get(timeout=10)
                bundle = {"rgb":rgb}
                for name in ("semantic_segmentation","instance_segmentation","depth","overview"):
                    bundle[name] = wait_frame(streams[name],rgb.frame)
                if len({image.frame for image in bundle.values()}) != 1:
                    raise RuntimeError("Camera streams are not aligned")
                frame_count += 1
                if frame_count%5 == 0:
                    captures = {}
                    for name,image in bundle.items():
                        path = folder/name/f"{image.frame:08d}.png"
                        captures[name] = {"file":str(path.relative_to(folder)),"sha256":save_native(image,path),"frame":image.frame,
                            "timestamp":image.timestamp,"camera_to_world":image.transform.get_matrix()}
                    frame_file.write(json.dumps({"source_frame":rgb.frame,"sim_s":rgb.timestamp-start_time,"streams":captures})+"\n")
                    saved_frames += 1
                last_rgb,last_overview = image_surface(rgb),image_surface(bundle["overview"])
            display.fill((16,23,29))
            if last_rgb is not None:
                display.blit(last_rgb,(0,55))
            if last_overview is not None:
                display.blit(pygame.transform.smoothscale(last_overview,(320,240)),(960,55))
            display.blit(font.render(profile.title+"  |  Indian left-hand traffic",True,(242,223,174)),(20,14))
            small = pygame.font.Font(None,23)
            lines = ["Native physics traffic", "SPACE pause / resume", "S save picture | ESC close", "",
                f"Lane width: {profile.width:.2f} m",f"Lanes each way: {profile.lanes_each_way}",
                f"Median: {profile.median*2:.1f} m",f"Shoulder: {profile.shoulder:.2f} m", "",
                "Rule driver; no model control", "Generic installed vehicles", "Autorickshaw: source only"]
            for i,line in enumerate(lines):
                display.blit(small.render(line,True,(204,212,211)),(973,315+i*23))
            status = f"{sim_time:.1f}s  |  {len(traffic.drivers)} vehicles  |  {len(traffic.walkers)} people  |  Paired captures: {frame_count}"
            display.blit(small.render(status,True,(210,223,219)),(20,629))
            display.blit(small.render("Hindi / English shops and signs  |  Indian-style synthetic roads",True,(184,201,198)),(20,665))
            pygame.display.flip()
            if ticks in (100,240) and last_rgb is not None:
                pygame.image.save(display,str(folder/f"preview_{ticks:04d}.png"))
                pygame.image.save(last_rgb,str(folder/f"front_{ticks:04d}.png"))
                pygame.image.save(last_overview,str(folder/f"overview_{ticks:04d}.png"))
            if args.smoke and ticks == 50:
                before = world.get_snapshot().frame
                time.sleep(.18)
                after = world.get_snapshot().frame
                controls["pause"] = controls["resume"] = before == after
            if ticks%40 == 0:
                write_json(folder/"progress.json",{"stage":"running","ticks":ticks,"sim_s":sim_time,
                    "paired_captures":frame_count,"collisions":len(traffic.collisions),"departures":len(departures)})
            if not args.smoke:
                time.sleep(max(0.,DT-(time.monotonic()-wall_start-ticks*DT)))
        result.update(ticks=ticks,paired_frames=saved_frames,processed_camera_frames=frame_count,
            preview_hz=5,saved_capture_hz=1,sim_seconds=world.get_snapshot().timestamp.elapsed_seconds-start_time,
            wall_seconds=time.monotonic()-wall_start,collisions=traffic.collisions,body_overlaps=overlaps,
            road_departures=departures,wrong_way=wrong_way,sampled_min_vehicle_gap_m=min_gap if math.isfinite(min_gap) else None,
            max_displacement_m=traffic.max_travel,pause_checks=controls,
            native_assets_used=len(owned),raw_reference_training_ready=False)
        result["passed"] = bool(ticks >= 100 and frame_count >= 15 and not traffic.collisions and not overlaps and
            not departures and not wrong_way and traffic.max_travel["ego"] > 4)
    except Exception:
        result["errors"].append(traceback.format_exc())
        print(result["errors"][-1],flush=True)
    finally:
        if trace_file:
            trace_file.close()
        if frame_file:
            frame_file.close()
        if world is not None:
            try:
                result["cleanup_warnings"] = cleanup(client,world,owned,sensors)
                result["cleanup_complete"] = not result["cleanup_warnings"]
            except Exception:
                result["cleanup_warnings"] = [traceback.format_exc()]
        result["passed"] = result["passed"] and result["cleanup_complete"]
        write_json(folder/"result.json",result)
    return result,exit_requested if "exit_requested" in locals() else False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--india-world",choices=(*PROFILES,"all","asset_probe"),required=True)
    p.add_argument("--port",type=int,default=2050)
    p.add_argument("--seconds",type=int,default=600)
    p.add_argument("--sim-seconds",type=float,default=60.)
    p.add_argument("--seed",type=int,default=7101)
    p.add_argument("--weather",choices=("clear","overcast","wet","dusk"),default="clear")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args,unknown = p.parse_known_args()
    if not 5 <= args.sim_seconds <= 150 or not 1 <= args.seconds <= 1800:
        p.error("Bounded simulation 5..150s and wall limit 1..1800s required")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Keep evidence inside runs/")
    owner = json.loads((ROOT/"setup/carla/server_owner.json").read_text())
    if owner.get("port") != args.port or owner.get("output") != str(output):
        raise RuntimeError("Dedicated server ownership does not match")
    output.mkdir(exist_ok=False)
    (output/"source").mkdir()
    for file in ("india_world_maps.py","india_world_scene.py","india_world_scenery.py","india_world_traffic.py"):
        shutil.copy2(ROOT/"integration"/file,output/"source"/file)
    write_json(output/"config.json",{"schema":"indian_worlds_v1","profile":args.india_world,"seed":args.seed,
        "weather":args.weather,"native_physics":True,"old_controller_modified":False,
        "model_control":False,"training_started":False,"created_utc":datetime.now(timezone.utc).isoformat()})
    pygame.init()
    display = pygame.display.set_mode((1280,714))
    pygame.display.set_caption("Indian worlds | SPACE pause | S picture | ESC close")
    client = carla.Client("127.0.0.1",args.port)
    results = []
    try:
        if args.india_world == "asset_probe":
            from india_world_asset_probe import probe
            return probe(client,output)
        for key in PROFILES if args.india_world == "all" else (args.india_world,):
            measured,quit_requested = run_world(client,key,args,output/key,display)
            results.append(measured)
            write_json(output/"result.json",{"worlds":results,"passed":all(r["passed"] for r in results)})
            if quit_requested or measured["errors"]:
                break
    finally:
        pygame.quit()
    return 0 if results and all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
