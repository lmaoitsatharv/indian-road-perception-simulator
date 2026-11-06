"""One live Indian town: every district and all traffic share one native world."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import multiprocessing
import json
import math
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback
from types import SimpleNamespace

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
from india_town_hud import TownHud,SIZE,PLACES,pretty,COLORS as HUD_COLORS,NAMES as HUD_NAMES
from india_png_worker import encode_bgra,encode_rgb
from india_world_scene import wait_frame,cleanup

ROOT = Path(__file__).resolve().parents[1]
SPEEDS = (.5,1.,2.,None)  # None: as fast as the server and laptop allow (about 1.8x on the dev laptop)
WEATHERS = ("clear","overcast","wet","dusk")
COLORS = {"market_street":(225,166,79),"divided_main_road":(98,172,219),"rural_road":(116,177,112),
    "residential_colony":(191,150,204),"irregular_junction":(227,115,104)}
NAMES = {"market_street":"Bazaar market","divided_main_road":"Divided main road","rural_road":"Village roads",
    "residential_colony":"Residential colony","irregular_junction":"Irregular junction"}


def minimap(native,network,size=(300,275),transparent=False):
    w,h = size
    surface = pygame.Surface(size,pygame.SRCALPHA)
    surface.fill((0,0,0,0) if transparent else (23,38,40,255))
    low = np.array([-360.,-330.])
    high = np.array([540.,230.])
    scale = min((w-25)/(high[0]-low[0]),(h-35)/(high[1]-low[1]))
    centre = (low+high)/2
    def xy(x,y):
        return int(w/2+(x-centre[0])*scale),int(h/2+(y-centre[1])*scale)
    lines = []
    for spec in network["streets"]:
        street = BY_ID[spec["id"]]
        points = []
        for s in np.arange(.1,spec["length"]-.1,2.):
            t = native.get_waypoint_xodr(street.id,street.lanes[0],float(s)).transform
            points.append(xy(t.location.x,t.location.y))
        lines.append((street,points))
    if transparent:
        for street,points in lines:
            pygame.draw.lines(surface,(8,12,14,255),False,points,9 if street.divided else 8)
    for street,points in lines:
        color = HUD_COLORS[street.district] if transparent else COLORS[street.district]
        pygame.draw.lines(surface,color,False,points,5 if street.divided else 4)
    for node,(x,y) in NODES.items():
        pygame.draw.circle(surface,(8,12,14) if transparent else (226,222,202),xy(x,-y),5 if transparent else 4)
        if transparent:
            pygame.draw.circle(surface,(236,232,214),xy(x,-y),3)
    return surface,xy


class EvidenceWriter(threading.Thread):
    """Encode and hash recorded PNGs off the display loop, in submission order.

    pygame holds the GIL for a whole PNG encode (about 50 ms per 960x540 frame),
    so a thread alone still froze the viewer once a second. Encoding runs in
    worker processes; this thread only waits on them and keeps the log ordered.
    """
    def __init__(self,frame_file):
        super().__init__(daemon=True)
        self.jobs = queue.Queue(maxsize=24)
        self.frame_file = frame_file
        self.saved,self.errors = 0,[]
        self.pool = ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn"))

    def run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                job()
            except Exception:
                self.errors.append(traceback.format_exc())

    def camera_set(self,output,images,row):
        paths = {name:output/name/f"{image.frame:08d}.png" for name,image in images.items()}
        def job():
            pending = {name:self.pool.submit(encode_bgra,str(paths[name]),image.raw_data,image.width,image.height)
                for name,image in images.items()}
            captures = {}
            for name,image in images.items():
                captures[name] = {"file":str(paths[name].relative_to(output)),"frame":image.frame,"timestamp":image.timestamp,
                    "sha256":pending[name].result(),"camera_to_world":image.matrix}
            self.frame_file.write(json.dumps({**row,"streams":captures})+"\n")
            self.saved += 1
        self.jobs.put(job)

    def picture(self,surface,path):
        raw,size = pygame.image.tobytes(surface,"RGB"),surface.get_size()
        self.jobs.put(lambda: self.pool.submit(encode_rgb,str(path),raw,*size).result())

    def close(self):
        self.jobs.put(None)
        self.join(timeout=120)
        self.pool.shutdown(wait=True)


def latest(pending,frame,wait=.4):
    """Newest image from an optional viewer camera, without failing the run."""
    image,deadline = None,time.monotonic()+wait
    while True:
        try:
            if image is None or image.frame < frame:
                image = pending.get(timeout=max(.005,deadline-time.monotonic()))
            else:
                image = pending.get_nowait()
        except queue.Empty:
            return image
        if time.monotonic() > deadline:
            return image


def detach(image):
    """Copy what a recorded image needs; the native buffer is not kept across threads."""
    return SimpleNamespace(raw_data=bytes(image.raw_data),width=image.width,height=image.height,frame=image.frame,
        timestamp=image.timestamp,matrix=image.transform.get_matrix())


def view_surface(image):
    return pygame.image.frombuffer(bytes(image.raw_data),(image.width,image.height),"BGRA").convert()


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
    for name in ("india_combined_map.py","india_combined_scenery.py","india_combined_traffic.py","india_combined_scene.py","india_town_hud.py","india_png_worker.py"):
        shutil.copy2(ROOT/"integration"/name,output/"source"/name)
    xml,network = build_combined()
    (output/"map.xodr").write_text(xml)
    write_json(output/"network.json",network)
    write_json(output/"map_check.json",validate_combined(xml,network))
    write_json(output/"config.json",{"schema":"one_connected_indian_town_v1","seed":args.seed,"weather":args.weather,
        "single_world":True,"sim_seconds":args.sim_seconds,"model_control":False,"rl_trained":False,
        "created_utc":datetime.now(timezone.utc).isoformat(),"preview_hz":20,"saved_camera_hz":1,
        "evidence_camera_hz":5,"follow_stream":"1280x720 chase display camera, saved at 1 Hz",
        "vehicle_pose_teleport":False,"original_models_unchanged":True})
    pygame.init()
    display = pygame.display.set_mode(SIZE,pygame.SCALED)
    pygame.display.set_caption("One Indian Town  -  connected CARLA world")
    font,small = pygame.font.SysFont("segoeui",34,bold=True),pygame.font.SysFont("segoeui",18)
    steps = ["generating","placing","checking","applying","spawning","running"]
    def progress(stage,**detail):
        write_json(output/"progress.json",{"stage":stage,**detail})
        if stage != "running":
            display.fill((12,18,22))
            done = next((i for i,key in enumerate(steps) if stage.startswith(key)),0)
            title = font.render("Building one connected Indian town",True,(240,226,190))
            display.blit(title,title.get_rect(center=(SIZE[0]//2,SIZE[1]//2-40)))
            label = small.render(stage.replace("_"," ").capitalize(),True,(176,190,186))
            display.blit(label,label.get_rect(center=(SIZE[0]//2,SIZE[1]//2+8)))
            bar = pygame.Rect(0,0,420,6)
            bar.center = (SIZE[0]//2,SIZE[1]//2+44)
            pygame.draw.rect(display,(40,50,54),bar,border_radius=3)
            pygame.draw.rect(display,(236,170,76),(bar.x,bar.y,int(bar.w*(done+1)/len(steps)),bar.h),border_radius=3)
            pygame.display.flip()
            pygame.event.pump()
    world,client,trace_file,frame_file,writer = None,None,None,None,None
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
        # The chase camera renders every tick for a smooth 20 Hz picture and
        # doubles as the saved "follow" stream. The driver view, street cameras
        # and other-vehicle cameras exist only while shown (open_spot), so the
        # laptop renders one display camera at a time. The 5 Hz evidence
        # cameras above are unchanged.
        views = {"follow":(carla.Transform(carla.Location(x=-7.0,z=3.0),carla.Rotation(pitch=-12)),
                carla.AttachmentType.Rigid,"90")}
        for name,(mount,attachment,fov) in views.items():
            bp = library.find("sensor.camera.rgb")
            for key,value in {"image_size_x":str(SIZE[0]),"image_size_y":str(SIZE[1]),"fov":fov,"sensor_tick":"0.0"}.items():
                bp.set_attribute(key,value)
            configure_rgb(bp)
            camera = world.spawn_actor(bp,mount,attach_to=traffic.ego.actor,attachment_type=attachment)
            owned.append(camera)
            sensors.append(camera)
            streams["view_"+name] = queue.Queue()
            camera.listen(streams["view_"+name].put)
        for name in ("rgb","semantic_segmentation","instance_segmentation","depth","follow"):
            (output/name).mkdir()
        (output/"screenshots").mkdir()
        map_base,map_xy = minimap(native_map,network)
        pygame.image.save(map_base,str(output/"town_plan.png"))
        hud_map,hud_xy = minimap(native_map,network,(270,232),transparent=True)
        hud = TownHud(hud_map,hud_xy,(SIZE[0]-298,26))
        # Viewer-only street cameras: one per district, looking along the longest
        # stretch of the tour there. Only a camera moves; no vehicle is teleported.
        tour,vantage = traffic.routes["tour"],{}
        for part in tour.parts:
            span = part["end_m"]-part["start_m"]
            if part["node"] is None and span > vantage.get(part["district"],(0,None))[0]:
                vantage[part["district"]] = (span,part)
        for district,(span,part) in list(vantage.items()):
            middle = part["start_m"]+span*.6
            look,eye = tour.at(middle),tour.at(middle-24.)
            ground = native_map.get_waypoint(carla.Location(x=float(eye[0]),y=float(eye[1]),z=0.))
            z = (ground.transform.location.z if ground else 0.)+9.5
            yaw = math.degrees(math.atan2(look[1]-eye[1],look[0]-eye[0]))
            vantage[district] = (carla.Transform(carla.Location(x=float(eye[0]),y=float(eye[1]),z=z),carla.Rotation(pitch=-21,yaw=yaw)),
                (float(look[0]),float(look[1])))
        missing = [d for d in PLACES if d not in vantage]
        if missing:
            raise RuntimeError("No street camera position for "+", ".join(missing))
        spot_bp = library.find("sensor.camera.rgb")
        for key,value in {"image_size_x":str(SIZE[0]),"image_size_y":str(SIZE[1]),"fov":"90","sensor_tick":"0.0"}.items():
            spot_bp.set_attribute(key,value)
        configure_rgb(spot_bp)
        spot = {"sensor":None,"kind":None,"value":None}
        def close_spot():
            if spot["sensor"] is not None:
                spot["sensor"].stop()
                spot["sensor"].destroy()
                owned.remove(spot["sensor"])
                sensors.remove(spot["sensor"])
            streams.pop("spot",None)
            spot.update(sensor=None,kind=None,value=None)
        def open_spot(kind,value):
            close_spot()
            if kind == "place":
                sensor = world.spawn_actor(spot_bp,vantage[value][0])
            elif kind == "driver":
                sensor = world.spawn_actor(spot_bp,carla.Transform(carla.Location(x=.4,z=1.55)),attach_to=traffic.ego.actor,
                    attachment_type=carla.AttachmentType.Rigid)
            else:
                driver = traffic.drivers[value]
                mount = carla.Transform(carla.Location(x=-(driver.half_length+4.8),z=2.2+driver.half_length*.36),carla.Rotation(pitch=-12))
                sensor = world.spawn_actor(spot_bp,mount,attach_to=driver.actor,attachment_type=carla.AttachmentType.Rigid)
            owned.append(sensor)
            sensors.append(sensor)
            streams["spot"] = queue.Queue()
            sensor.listen(streams["spot"].put)
            spot.update(sensor=sensor,kind=kind,value=value)
        trace_file = (output/"actor_trace.jsonl").open("w",encoding="utf-8")
        frame_file = (output/"camera_frames.jsonl").open("w",encoding="utf-8")
        writer = EvidenceWriter(frame_file)
        writer.start()
        start_sim = world.get_snapshot().timestamp.elapsed_seconds
        wall_start = time.monotonic()
        deadline = wall_start+args.seconds
        ticks,received,submitted = 0,0,0
        paused = False
        surface,last_trace,recent_follow = None,None,{}
        sim_time = 0.
        overlaps,departures,direction_errors = [],[],[]
        min_gap = float("inf")
        # Progress gate: contacts and departures alone cannot see a gridlock.
        stopped_since,longest_stopped = None,0.
        ego_mark,ego_mark_s,longest_ego_stall = 0.,0.,0.
        saved_regions = set()
        pause_check = None
        exit_reason = "duration"
        world_ids = {start_world_id}
        # Real-time pacing without catch-up bursts: a late tick moves the clock
        # instead of fast-forwarding the picture afterwards (also after a pause).
        next_tick,loop_ms,server_ms,last_loop,clock_resets = time.monotonic(),[],[],None,0
        speed_i,weather,show_hud,show_help,step,last_draw,rate = 1,args.weather,True,False,False,0.,1.
        viewer_events,pace = [],{}
        def camera_label():
            if spot["kind"] == "place":
                return "Street camera"
            if spot["kind"] == "vehicle":
                return "Following "+pretty(traffic.drivers[spot["value"]].name)
            return "Driver view" if spot["kind"] == "driver" else "Chase camera"
        def draw():
            place = spot["value"] if spot["kind"] == "place" else None
            focus = traffic.drivers[spot["value"]].name if spot["kind"] == "vehicle" else "ego"
            hud.draw(display,surface,trace=last_trace,sim_time=sim_time,wall_now=time.monotonic(),
                vehicles=len(traffic.drivers),people=len(traffic.walkers),visited=traffic.visited["ego"],
                travelled=traffic.distance["ego"],paused=paused,camera=camera_label(),focus=focus,place=place,
                place_xy=vantage[place][1] if place else None,speed=SPEEDS[speed_i],weather=weather,
                show_hud=show_hud,show_help=show_help)
            pygame.display.flip()
        def note(event,value):
            viewer_events.append({"sim_s":round(sim_time,2),"tick":ticks,"event":event,"value":value})
        while time.monotonic() < deadline:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_reason = "user_closed"
                if event.type != pygame.KEYDOWN:
                    continue
                key,wall = event.key,time.monotonic()
                place = PLACES.index(spot["value"]) if spot["kind"] == "place" else -1
                if key == pygame.K_ESCAPE:
                    if show_help:
                        show_help = False
                    else:
                        exit_reason = "user_closed"
                elif key == pygame.K_SPACE:
                    paused = not paused
                    next_tick,last_loop = wall,None
                elif key in (pygame.K_PERIOD,pygame.K_KP_PERIOD) and paused:
                    step = True
                elif key == pygame.K_TAB:
                    if spot["sensor"] is not None:
                        close_spot()
                    else:
                        open_spot("driver",0)
                    note("camera",spot["kind"] or "chase")
                elif key in (pygame.K_EQUALS,pygame.K_PLUS,pygame.K_KP_PLUS,pygame.K_MINUS,pygame.K_KP_MINUS,pygame.K_0,pygame.K_KP0):
                    if key in (pygame.K_0,pygame.K_KP0):
                        speed_i = 1
                    else:
                        speed_i = max(0,min(len(SPEEDS)-1,speed_i+(-1 if key in (pygame.K_MINUS,pygame.K_KP_MINUS) else 1)))
                    chosen = SPEEDS[speed_i]
                    next_tick,last_loop = wall,None
                    hud.notify("Real-time speed" if chosen == 1. else "Maximum speed" if chosen is None else f"Speed  {chosen:g}×",
                        (240,226,190),wall,1.6)
                    note("speed",chosen)
                elif pygame.K_1 <= key <= pygame.K_5 or pygame.K_KP1 <= key <= pygame.K_KP5 or key == pygame.K_l:
                    if key == pygame.K_l:
                        index = (place+1)%len(PLACES)
                    else:
                        index = key-(pygame.K_1 if pygame.K_1 <= key <= pygame.K_5 else pygame.K_KP1)
                    open_spot("place",PLACES[index])
                    hud.notify(HUD_NAMES[PLACES[index]]+"  ·  street camera",HUD_COLORS[PLACES[index]],wall,2.2)
                    note("street_camera",PLACES[index])
                elif key == pygame.K_v:
                    current = spot["value"] if spot["kind"] == "vehicle" else 0
                    index = (current+(-1 if event.mod & pygame.KMOD_SHIFT else 1))%len(traffic.drivers)
                    if index == 0:
                        close_spot()
                        hud.notify("Back to your car",(240,226,190),wall,1.6)
                    else:
                        open_spot("vehicle",index)
                        hud.notify("Following  "+pretty(traffic.drivers[index].name),(250,150,112),wall,1.8)
                    note("follow_vehicle",traffic.drivers[index].name)
                elif key == pygame.K_c:
                    close_spot()
                    hud.notify("Back to your car",(240,226,190),wall,1.6)
                    note("camera","chase")
                elif key == pygame.K_w:
                    weather = WEATHERS[(WEATHERS.index(weather)+1)%len(WEATHERS)]
                    world.set_weather(weather_parameters(weather,physical_atmosphere=True))
                    hud.notify("Weather  ·  "+weather,(150,190,240),wall,1.8)
                    note("weather",weather)
                elif key == pygame.K_r:
                    moved = traffic.reset_ego()
                    if moved is None:
                        hud.notify("No clear spot to reset the car",(240,112,96),wall,1.8)
                    else:
                        result["vehicle_pose_teleport"] = True
                        if spot["kind"] in ("place","vehicle"):
                            close_spot()
                        hud.notify("Car reset onto its lane" if moved < 1 else f"Car reset {moved:.0f} m ahead",(240,226,190),wall,1.8)
                    note("reset_car",moved)
                elif key == pygame.K_h:
                    show_hud = not show_hud
                elif key in (pygame.K_F1,pygame.K_SLASH):
                    show_help = not show_help
                elif key == pygame.K_f:
                    try:
                        pygame.display.toggle_fullscreen()
                    except pygame.error:
                        hud.notify("Fullscreen is not available here",(240,112,96),wall,1.8)
                elif key == pygame.K_s:
                    writer.picture(display.copy(),output/"screenshots"/f"manual_{ticks:07d}.png")
                    hud.snapshot(wall)
            if exit_reason == "user_closed" or (ROOT/"setup/carla/STOP_DEMO.json").exists():
                break
            if paused and not step:
                if last_trace is not None:
                    draw()
                time.sleep(.03)
                continue
            step = False
            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds-start_sim
            if sim_time >= args.sim_seconds:
                break
            world_ids.add(world.id)
            states,trace = traffic.update(snapshot,sim_time)
            last_trace = trace
            trace_file.write(json.dumps({"frame":snapshot.frame,"sim_s":sim_time,"world_id":world.id,"actors":trace,"jaywalkers":traffic.jaywalker_trace})+"\n")
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
                if wp is None and row["two_wheeler"]:
                    # Two-wheelers may ride the paved shoulder of village roads:
                    # inside the nearest driving lane plus that street's shoulder.
                    # (A point lookup finds neither lane in a band at their edge.)
                    near = native_map.get_waypoint(carla.Location(x=row["x"],y=row["y"],z=0),True,carla.LaneType.Driving)
                    if near is not None and near.road_id in BY_ID:
                        nt = near.transform
                        yaw = math.radians(nt.rotation.yaw)
                        side = abs(-(row["x"]-nt.location.x)*math.sin(yaw)+(row["y"]-nt.location.y)*math.cos(yaw))
                        if side <= near.lane_width/2+BY_ID[near.road_id].shoulder+.05:
                            wp = near
                if wp is None:
                    departures.append({"frame":snapshot.frame,"actor":row["name"],"xy":[row["x"],row["y"]]})
                if row["speed"] > .3 and abs(math.remainder(row["yaw"]-row["expected_yaw"],2*math.pi)) > math.radians(70):
                    direction_errors.append({"frame":snapshot.frame,"actor":row["name"]})
            server_start = time.monotonic()
            frame = world.tick(25)
            ticks += 1
            frames = {"follow":wait_frame(streams["view_follow"],frame)}
            server_ms.append((time.monotonic()-server_start)*1000)
            shown = frames["follow"] if spot["sensor"] is None else latest(streams["spot"],frame)
            rate = SPEEDS[speed_i]
            # Above 1x the picture refreshes at most 20 times a second (wall clock).
            due = (rate is not None and rate <= 1.) or paused or time.monotonic()-last_draw > 1/20
            if shown is not None and due:
                surface = view_surface(shown)
            recent_follow[frame] = frames["follow"]
            for old in [f for f in recent_follow if f < frame-8]:
                del recent_follow[old]
            if not streams["rgb"].empty():
                rgb = streams["rgb"].get(timeout=10)
                bundle = {"rgb":rgb}
                for name in ("semantic_segmentation","instance_segmentation","depth"):
                    bundle[name] = wait_frame(streams[name],rgb.frame)
                received += 1
                if received%5 == 0 and rgb.frame in recent_follow:
                    bundle["follow"] = recent_follow[rgb.frame]
                    writer.camera_set(output,{name:detach(image) for name,image in bundle.items()},
                        {"source_frame":rgb.frame,"world_id":world.id,"sim_s":rgb.timestamp-start_sim,"district":trace[0]["district"],
                        "weather":weather})
                    submitted += 1
            district = trace[0]["district"]
            if due:
                draw()
                last_draw = time.monotonic()
            # District pictures only show your own car, with the HUD on.
            if due and spot["sensor"] is None and show_hud and not show_help and (ticks == 240 or (district not in saved_regions and sim_time > 8)):
                writer.picture(display.copy(),output/"screenshots"/(district+".png"))
                saved_regions.add(district)
            if ticks%100 == 0:
                progress("running",sim_s=sim_time,ticks=ticks,saved_frames=writer.saved,world_id=world.id,
                    ego_distance_m=traffic.distance["ego"],ego_districts=sorted(traffic.visited["ego"]),
                    collisions=len(traffic.collisions),overlaps=len(overlaps),departures=len(departures))
            if args.smoke and ticks == 80:
                before = world.get_snapshot().frame
                time.sleep(.15)
                pause_check = before == world.get_snapshot().frame
            now = time.monotonic()
            # Loop timing is only a real-time quality measure at 1x speed.
            if last_loop is not None and rate == 1. and not paused:
                loop_ms.append((now-last_loop)*1000)
            if paused:
                next_tick = now
            elif not args.smoke:
                if rate is None:
                    next_tick = now
                else:
                    next_tick += .05/rate
                    if next_tick > now:
                        time.sleep(next_tick-now)
                    elif now-next_tick > .25:
                        next_tick = now
                        clock_resets += int(rate == 1.)
            end = time.monotonic()
            if last_loop is not None and not paused:
                spent = pace.setdefault("max" if rate is None else f"{rate:g}x",[0.,0.])
                spent[0] += .05
                spent[1] += end-last_loop
            last_loop = end
        writer.close()
        saved = writer.saved
        if writer.errors:
            result["errors"].extend(writer.errors)
        timing = np.array(loop_ms) if loop_ms else np.zeros(1)
        result["display_loop_ms"] = {"p50":float(np.percentile(timing,50)),"p95":float(np.percentile(timing,95)),
            "max":float(timing.max()),"over_100ms":int((timing > 100).sum()),"clock_resets":clock_resets,
            "submitted_camera_sets":submitted,"measured_at":"1x speed only"}
        result["viewer_events"] = viewer_events
        result["speed_achieved"] = {key:{"sim_s":round(sim,1),"wall_s":round(wall,1),"times_real_time":round(sim/wall,2)}
            for key,(sim,wall) in pace.items() if wall > 0}
        server = np.array(server_ms) if server_ms else np.zeros(1)
        result["server_tick_ms"] = {"p50":float(np.percentile(server,50)),"p95":float(np.percentile(server,95)),
            "max":float(server.max()),"over_100ms":int((server > 100).sum())}
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
            junction_token_grants=traffic.token_grants,ego_resets=traffic.ego_resets,lost_walkers=traffic.lost_walkers,junction_watchdog_releases=traffic.token_events,
            traffic_behaviour=traffic.junction_stats(),
            all_world_generation_finished_before_traffic=True,training_allowed=False)
        result["passed"] = bool(ticks >= 200 and saved >= 5 and not writer.errors and len(world_ids) == 1 and result["world_generate_calls"] == 1
            and not traffic.collisions and not overlaps and not departures and not direction_errors and traffic.distance["ego"] > 8
            and longest_stopped < 60 and longest_ego_stall < 120 and not traffic.lost_walkers)
    except Exception:
        result["errors"].append(traceback.format_exc())
        print(result["errors"][-1],flush=True)
    finally:
        if writer is not None and writer.is_alive():
            writer.close()
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
