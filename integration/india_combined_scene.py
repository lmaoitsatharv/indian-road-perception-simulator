"""One live Indian town: every district and all traffic share one native world."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
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
from india_behavior import polygon_gap
from india_camera import configure_rgb
from india_conditions import weather_parameters
from india_combined_map import build_combined,validate_combined,BY_ID,NODES,NODE_NAMES
from india_combined_scenery import CombinedScenery
from india_combined_traffic import CombinedTraffic
from india_world_scene import save_native,image_surface,wait_frame,cleanup

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"market_street":(225,166,79),"divided_main_road":(98,172,219),"rural_road":(116,177,112),
    "residential_colony":(191,150,204),"irregular_junction":(227,115,104)}
NAMES = {"market_street":"Bazaar market","divided_main_road":"Divided main road","rural_road":"Village roads",
    "residential_colony":"Residential colony","irregular_junction":"Irregular junction"}


def minimap(native,network):
    surface = pygame.Surface((300,275))
    surface.fill((23,38,40))
    low = np.array([-360.,-330.])
    high = np.array([540.,230.])
    scale = min(275/(high[0]-low[0]),240/(high[1]-low[1]))
    centre = (low+high)/2
    def xy(x,y):
        return int(150+(x-centre[0])*scale),int(138+(y-centre[1])*scale)
    for spec in network["streets"]:
        street = BY_ID[spec["id"]]
        points = []
        for s in np.arange(.1,spec["length"]-.1,2.):
            t = native.get_waypoint_xodr(street.id,street.lanes[0],float(s)).transform
            points.append(xy(t.location.x,t.location.y))
        pygame.draw.lines(surface,COLORS[street.district],False,points,4 if street.divided else 3)
    for node,(x,y) in NODES.items():
        pygame.draw.circle(surface,(226,222,202),xy(x,-y),4)
    return surface,xy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port",type=int,default=2050)
    parser.add_argument("--seed",type=int,default=8101)
    parser.add_argument("--sim-seconds",type=float,default=900)
    parser.add_argument("--seconds",type=int,default=1800)
    parser.add_argument("--weather",choices=("clear","overcast","wet","dusk"),default="clear")
    parser.add_argument("--smoke",action="store_true")
    args,unknown = parser.parse_known_args()
    if not 10 <= args.sim_seconds <= 1500 or not 1 <= args.seconds <= 1800:
        parser.error("Combined-world simulation 10..1500s, wall limit 1..1800s")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Keep evidence inside runs/")
    owner = json.loads((ROOT/"setup/carla/server_owner.json").read_text())
    if owner.get("port") != args.port or owner.get("output") != str(output):
        raise RuntimeError("Dedicated native server ownership mismatch")
    output.mkdir(exist_ok=False)
    (output/"source").mkdir()
    for name in ("india_combined_map.py","india_combined_scenery.py","india_combined_traffic.py","india_combined_scene.py"):
        shutil.copy2(ROOT/"integration"/name,output/"source"/name)
    xml,network = build_combined()
    (output/"map.xodr").write_text(xml)
    write_json(output/"network.json",network)
    write_json(output/"map_check.json",validate_combined(xml,network))
    write_json(output/"config.json",{"schema":"one_connected_indian_town_v1","seed":args.seed,"weather":args.weather,
        "single_world":True,"sim_seconds":args.sim_seconds,"model_control":False,"rl_trained":False,
        "created_utc":datetime.now(timezone.utc).isoformat(),"preview_hz":5,"saved_camera_hz":1,
        "vehicle_pose_teleport":False,"original_models_unchanged":True})
    pygame.init()
    display = pygame.display.set_mode((1280,800))
    pygame.display.set_caption("One Indian town | TAB camera | SPACE pause | S photo | ESC close")
    font,small = pygame.font.Font(None,29),pygame.font.Font(None,22)
    def progress(stage,**detail):
        write_json(output/"progress.json",{"stage":stage,**detail})
        if stage != "running":
            display.fill((17,28,31))
            display.blit(font.render("Building one connected Indian town...",True,(238,218,177)),(65,260))
            display.blit(small.render(stage.replace("_"," "),True,(201,215,211)),(65,310))
            pygame.display.flip()
            pygame.event.pump()
    world,client,trace_file,frame_file = None,None,None,None
    owned,sensors,streams = [],[],{}
    result = {"passed":False,"errors":[],"single_world":True,"world_generate_calls":0,"cleanup_complete":False,
        "model_control":False,"rl_trained":False,"vehicle_pose_teleport":False}
    try:
        client = carla.Client("127.0.0.1",args.port)
        client.set_timeout(120)
        progress("generating_connected_opendrive_world")
        world = client.generate_opendrive_world(xml,carla.OpendriveGenerationParameters(vertex_distance=.9,max_road_length=35.,
            wall_height=0.,additional_width=0.,smooth_junctions=True,enable_mesh_visibility=True,enable_pedestrian_navigation=True))
        result["world_generate_calls"] += 1
        client.set_timeout(25)
        settings = world.get_settings()
        settings.synchronous_mode,settings.fixed_delta_seconds = True,.05
        settings.substepping,settings.max_substep_delta_time,settings.max_substeps = True,.01,10
        world.apply_settings(settings)
        world.set_weather(weather_parameters(args.weather,physical_atmosphere=True))
        native_map = world.get_map()
        start_world_id = world.id
        scenery = CombinedScenery(world,owned,xml,network)
        write_json(output/"scenery.json",scenery.build(progress))
        progress("spawning_traffic_across_all_five_districts")
        traffic = CombinedTraffic(world,client,owned,network,args.seed)
        sensors.extend(traffic.sensors)
        write_json(output/"actors.json",{"actors":traffic.records,"routes":{
            key:{"nodes":route.nodes,"length_m":route.length,"max_join_gap_m":route.max_gap,"parts":route.parts}
            for key,route in traffic.routes.items()},"all_exist_in_one_world":True,"autorickshaw_runtime":False})
        library = world.get_blueprint_library()
        mounts = {"driver":carla.Transform(carla.Location(x=1.35,y=.25,z=1.45)),
            "follow":carla.Transform(carla.Location(x=-7.5,z=3.4),carla.Rotation(pitch=-14))}
        for kind in ("rgb","semantic_segmentation","instance_segmentation","depth"):
            bp = library.find("sensor.camera."+kind)
            for name,value in {"image_size_x":"960","image_size_y":"540","fov":"95","sensor_tick":"0.20"}.items():
                bp.set_attribute(name,value)
            if kind == "rgb":
                configure_rgb(bp)
            sensor = world.spawn_actor(bp,mounts["driver"],attach_to=traffic.ego.actor)
            owned.append(sensor)
            sensors.append(sensor)
            streams[kind] = queue.Queue()
            sensor.listen(streams[kind].put)
        bp = library.find("sensor.camera.rgb")
        for name,value in {"image_size_x":"960","image_size_y":"540","fov":"95","sensor_tick":"0.20"}.items():
            bp.set_attribute(name,value)
        configure_rgb(bp)
        follow = world.spawn_actor(bp,mounts["follow"],attach_to=traffic.ego.actor)
        owned.append(follow)
        sensors.append(follow)
        streams["follow"] = queue.Queue()
        follow.listen(streams["follow"].put)
        for name in ("rgb","semantic_segmentation","instance_segmentation","depth","follow"):
            (output/name).mkdir()
        (output/"screenshots").mkdir()
        map_base,map_xy = minimap(native_map,network)
        pygame.image.save(map_base,str(output/"town_plan.png"))
        trace_file = (output/"actor_trace.jsonl").open("w",encoding="utf-8")
        frame_file = (output/"camera_frames.jsonl").open("w",encoding="utf-8")
        start_sim = world.get_snapshot().timestamp.elapsed_seconds
        wall_start = time.monotonic()
        deadline = wall_start+args.seconds
        ticks,received,saved = 0,0,0
        paused,view = False,"follow"
        latest,last_trace = {},[]
        overlaps,departures,direction_errors = [],[],[]
        min_gap = float("inf")
        # Progress gate: contacts and departures alone cannot see a gridlock.
        stopped_since,longest_stopped = None,0.
        ego_mark,ego_mark_s,longest_ego_stall = 0.,0.,0.
        saved_regions = set()
        pause_check = None
        exit_reason = "duration"
        world_ids = {start_world_id}
        while time.monotonic() < deadline:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    exit_reason = "user_closed"
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    view = "driver" if view == "follow" else "follow"
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    pygame.image.save(display,str(output/"screenshots"/f"manual_{ticks:07d}.png"))
            if exit_reason == "user_closed" or (ROOT/"setup/carla/STOP_DEMO.json").exists():
                break
            if paused:
                time.sleep(.025)
                continue
            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds-start_sim
            if sim_time >= args.sim_seconds:
                break
            world_ids.add(world.id)
            states,trace = traffic.update(snapshot,sim_time)
            trace_file.write(json.dumps({"frame":snapshot.frame,"sim_s":sim_time,"world_id":world.id,"actors":trace})+"\n")
            if all(row["speed"] < .1 for row in trace):
                stopped_since = sim_time if stopped_since is None else stopped_since
                longest_stopped = max(longest_stopped,sim_time-stopped_since)
            else:
                stopped_since = None
            if traffic.distance["ego"]-ego_mark >= 1.:
                ego_mark,ego_mark_s = traffic.distance["ego"],sim_time
            longest_ego_stall = max(longest_ego_stall,sim_time-ego_mark_s)
            pairs = list(states.items())
            for i,(aid,a) in enumerate(pairs):
                for bid,b in pairs[i+1:]:
                    if math.hypot(a["x"]-b["x"],a["y"]-b["y"]) > 17:
                        continue
                    overlap,gap = polygon_gap(a["polygon"],b["polygon"])
                    min_gap = min(min_gap,gap)
                    if overlap:
                        overlaps.append({"frame":snapshot.frame,"a":aid,"b":bid,"gap_m":gap})
            for row in trace:
                wp = native_map.get_waypoint(carla.Location(x=row["x"],y=row["y"],z=0),False,carla.LaneType.Driving)
                if wp is None:
                    departures.append({"frame":snapshot.frame,"actor":row["name"],"xy":[row["x"],row["y"]]})
                if row["speed"] > .3 and abs(math.remainder(row["yaw"]-row["expected_yaw"],2*math.pi)) > math.radians(70):
                    direction_errors.append({"frame":snapshot.frame,"actor":row["name"]})
            world.tick(25)
            ticks += 1
            if not streams["rgb"].empty():
                rgb = streams["rgb"].get(timeout=10)
                bundle = {"rgb":rgb}
                for name in ("semantic_segmentation","instance_segmentation","depth","follow"):
                    bundle[name] = wait_frame(streams[name],rgb.frame)
                received += 1
                latest = {name:image_surface(image) for name,image in bundle.items() if name in ("rgb","follow")}
                if received%5 == 0:
                    captures = {}
                    for name,image in bundle.items():
                        path = output/name/f"{image.frame:08d}.png"
                        captures[name] = {"file":str(path.relative_to(output)),"frame":image.frame,"timestamp":image.timestamp,
                            "sha256":save_native(image,path),"camera_to_world":image.transform.get_matrix()}
                    frame_file.write(json.dumps({"source_frame":rgb.frame,"world_id":world.id,
                        "sim_s":rgb.timestamp-start_sim,"district":trace[0]["district"],"streams":captures})+"\n")
                    saved += 1
            display.fill((16,27,30))
            if latest:
                display.blit(latest["rgb" if view == "driver" else "follow"],(0,58))
            district = trace[0]["district"]
            display.blit(font.render("ONE INDIAN TOWN  |  "+NAMES[district],True,(241,221,179)),(18,15))
            display.blit(map_base,(970,62))
            for row in trace:
                p = map_xy(row["x"],row["y"])
                color = (255,250,220) if row["name"] == "ego" else (244,143,106)
                pygame.draw.circle(display,color,(970+p[0],62+p[1]),4 if row["name"] == "ego" else 2)
            for i,(region,color) in enumerate(COLORS.items()):
                pygame.draw.rect(display,color,(983,355+i*26,13,13))
                display.blit(small.render(NAMES[region],True,(212,222,216)),(1005,351+i*26))
            info = [f"{len(traffic.drivers)} vehicles  |  {len(traffic.walkers)} people",
                "All areas share this same world", "TAB front / follow camera", "SPACE pause   S save picture", "ESC close", "",
                f"{sim_time:.1f}s   Ego: {trace[0]['speed']*3.6:.1f} km/h",
                f"Travelled {traffic.distance['ego']:.0f} m",
                f"Areas visited: {len(traffic.visited['ego'])}/5"]
            for i,line in enumerate(info):
                display.blit(small.render(line,True,(211,222,216)),(981,500+i*24))
            display.blit(small.render("Market, colony, divided road, village and irregular junction are connected. No scene switching.",True,(208,222,215)),(20,628))
            display.blit(small.render("Native mixed traffic and Hindi / English signs. Rule driver; no trained model control.",True,(182,204,195)),(20,663))
            pygame.display.flip()
            if ticks == 240 or (district not in saved_regions and sim_time > 8):
                pygame.image.save(display,str(output/"screenshots"/(district+".png")))
                saved_regions.add(district)
            if ticks%100 == 0:
                progress("running",sim_s=sim_time,ticks=ticks,saved_frames=saved,world_id=world.id,
                    ego_distance_m=traffic.distance["ego"],ego_districts=sorted(traffic.visited["ego"]),
                    collisions=len(traffic.collisions),overlaps=len(overlaps),departures=len(departures))
            if args.smoke and ticks == 80:
                before = world.get_snapshot().frame
                time.sleep(.15)
                pause_check = before == world.get_snapshot().frame
            if not args.smoke:
                time.sleep(max(0.,ticks*.05-(time.monotonic()-wall_start)))
        result.update(ticks=ticks,sim_seconds=world.get_snapshot().timestamp.elapsed_seconds-start_sim,
            wall_seconds=time.monotonic()-wall_start,world_ids=sorted(world_ids),native_vehicles=len(traffic.drivers),
            native_walkers=len(traffic.walkers),processed_camera_frames=received,saved_camera_sets=saved,
            collisions=traffic.collisions,vehicle_overlaps=overlaps,vehicle_centre_departures=departures,
            wrong_direction_events=direction_errors,sampled_min_vehicle_gap_m=min_gap if math.isfinite(min_gap) else None,
            travelled_m=traffic.distance,districts_visited={k:sorted(v) for k,v in traffic.visited.items()},
            ego_visited_all_five=len(traffic.visited["ego"]) == 5,
            any_vehicle_visited_all_five=any(len(v) == 5 for v in traffic.visited.values()),
            continuous_transitions=traffic.transitions,exit_reason=exit_reason,pause_check=pause_check,
            longest_all_vehicles_stopped_s=longest_stopped,longest_ego_stall_s=longest_ego_stall,
            junction_token_grants=traffic.token_grants,junction_watchdog_releases=traffic.token_events,
            all_world_generation_finished_before_traffic=True,training_allowed=False)
        result["passed"] = bool(ticks >= 200 and saved >= 5 and len(world_ids) == 1 and result["world_generate_calls"] == 1
            and not traffic.collisions and not overlaps and not departures and not direction_errors and traffic.distance["ego"] > 8
            and longest_stopped < 60 and longest_ego_stall < 120)
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
        write_json(output/"result.json",result)
        pygame.quit()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
