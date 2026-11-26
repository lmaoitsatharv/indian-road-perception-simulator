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
from india_town_hud import TownHud,SIZE,PLACES,pretty,ai_view_surface,COLORS as HUD_COLORS,NAMES as HUD_NAMES,MapPicker
from india_png_worker import encode_bgra,encode_rgb
from india_world_scene import wait_frame,cleanup
from india_camera_geometry import CameraGeometry
from india_camera_driver import CameraDriver,DriverSettings
from india_route_planner import TownRoutes
from india_satnav import SatNav,parse_destination
from india_perception_client import PerceptionClient

ROOT = Path(__file__).resolve().parents[1]
SPEEDS = (.5,1.,2.,None)  # None: as fast as the server and laptop allow (about 1.8x on the dev laptop)
WEATHERS = ("clear","overcast","wet","dusk")
COLORS = {"market_street":(225,166,79),"divided_main_road":(98,172,219),"rural_road":(116,177,112),
    "residential_colony":(191,150,204),"irregular_junction":(227,115,104)}
NAMES = {"market_street":"Bazaar market","divided_main_road":"Divided main road","rural_road":"Village roads",
    "residential_colony":"Residential colony","irregular_junction":"Irregular junction"}



def single_gear(physics,top_speed_m_s=10.):
    """Replace a car's gearbox by one fixed ratio, so it never shifts (an EV-style single-speed drive).

    Keeps the lowest original gear (the most pulling power) that still reaches top_speed_m_s at the engine's
    max_rpm, and makes gear changes instant. Returns the new physics control and a record of the change."""
    gears = list(physics.forward_gears)
    radius = min(w.radius for w in physics.wheels)/100.          # wheel radius: cm -> m
    def reach(g):
        return physics.max_rpm*2*math.pi/60*radius/max(g.ratio*physics.final_ratio,1e-6)
    usable = [g for g in gears if reach(g) >= top_speed_m_s] or [max(gears,key=reach)]
    chosen = max(usable,key=lambda g:g.ratio)
    record = {"single_gear":True,"original_gears":[[round(g.ratio,3),round(g.down_ratio,3),round(g.up_ratio,3)] for g in gears],
        "original_gear_switch_time_s":round(physics.gear_switch_time,3),"final_ratio":round(physics.final_ratio,3),
        "max_rpm":round(physics.max_rpm,1),"wheel_radius_m":round(radius,3),"chosen_ratio":round(chosen.ratio,3),
        "top_speed_at_max_rpm_m_s":round(reach(chosen),2)}
    physics.forward_gears = [carla.GearPhysicsControl(ratio=chosen.ratio,down_ratio=chosen.down_ratio,up_ratio=chosen.up_ratio)]
    physics.gear_switch_time = 0.
    physics.use_gear_autobox = True
    return physics,record

def message_wait_changed(nav_state,turn):
    """True when the reason your car is waiting for its turn changed since last recorded."""
    key = (turn["node"],turn["why"])
    if nav_state.get("wait_key") == key:
        return False
    nav_state["wait_key"] = key
    return True


def navigation_summary(events,nav_state,metrics,stats):
    """result.json -> camera_driving.navigation: destinations reached, re-plans, junction turns."""
    destinations = []
    for event in events:
        if event["event"] == "destination_set":
            destinations.append({"name":event["name"],"set_s":event["sim_s"],"arrived_s":None,"short_m":None})
        elif event["event"] == "arrived" and destinations and destinations[-1]["arrived_s"] is None:
            destinations[-1].update(arrived_s=event["sim_s"],short_m=event.get("short_m"))
    times = np.array(nav_state["plan_ms"]) if nav_state["plan_ms"] else None
    return {"destinations":destinations,"reached":sum(d["arrived_s"] is not None for d in destinations),
        "replans":[{"sim_s":e["sim_s"],"why":e.get("why")} for e in events if e["event"] == "replanned"],
        "junction_turns":nav_state["turns"],
        "plan_ms":None if times is None else {"p50":float(np.percentile(times,50)),"p95":float(np.percentile(times,95)),"count":int(len(times))},
        "waiting_turn_s":round(metrics["reason_s"].get("waiting_turn",0.),2),"token_grants":stats.get("camera_car_token_grants",0)}


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
    parser.add_argument("--camera-drive",type=Path,help="Camera-only driving session (setup/run_camera_driving.py)")
    parser.add_argument("--destination",help="Camera-driving runs: drive to this place (a name, or x,y)")
    parser.add_argument("--destinations",help="Camera-driving test runs: places separated by ';', the next set 5 s after arriving")
    args,unknown = parser.parse_known_args()
    wanted_places = [w.strip() for w in ([args.destination] if args.destination else [])+(args.destinations.split(";") if args.destinations else [])
        if w.strip()]
    for wanted in wanted_places:
        try:
            parse_destination(wanted)
        except ValueError as error:
            parser.error(str(error))
    camera_mode = args.camera_drive is not None
    session = None
    if camera_mode:
        session = json.loads(args.camera_drive.read_text())
        if session.get("schema") != "camera_driving_session_v1" or session.get("model_control") is not True \
                or session.get("ground_truth_to_driver") is not False:
            raise ValueError("Not a camera-only driving session")
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
    for name in ("india_combined_map.py","india_combined_scenery.py","india_combined_traffic.py","india_combined_scene.py","india_town_hud.py","india_png_worker.py",
            "india_camera_geometry.py","india_camera_driver.py","india_perception_client.py",
            "india_route_planner.py","india_satnav.py","india_hybrid_astar.py"):
        shutil.copy2(ROOT/"integration"/name,output/"source"/name)
    xml,network = build_combined()
    (output/"map.xodr").write_text(xml)
    write_json(output/"network.json",network)
    write_json(output/"map_check.json",validate_combined(xml,network))
    write_json(output/"config.json",{"schema":"one_connected_indian_town_v1","seed":args.seed,"weather":args.weather,
        "single_world":True,"sim_seconds":args.sim_seconds,"model_control":camera_mode,"rl_trained":False,
        "camera_driving":None if not camera_mode else {"session":str(args.camera_drive),"controller":"camera_only_models",
            "ground_truth_to_driver":False,"safety_backstop":False,"auto_reset_s":session.get("auto_reset_s"),
            "latency_ticks":session["latency_ticks"],"camera":session["camera"],"driver":session.get("driver")},
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
    perception_client,decisions_file = None,None
    navigation_file = satnav = picker = nav_state = None
    owned,sensors,streams = [],[],{}
    result = {"passed":False,"errors":[],"single_world":True,"world_generate_calls":0,"cleanup_complete":False,
        "model_control":camera_mode,"rl_trained":False,"vehicle_pose_teleport":False}
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
        # Camera-driving runs skip the semantic evidence camera (it tags this generated road as
        # Building, so it adds nothing) to leave GPU time for the two models.
        evidence_kinds = ("rgb","instance_segmentation","depth") if camera_mode else ("rgb","semantic_segmentation","instance_segmentation","depth")
        for kind in evidence_kinds:
            bp = library.find("sensor.camera."+kind)
            # Camera-driving runs render these at the 1 Hz they are saved at (GPU time for the models).
            for name,value in {"image_size_x":"960","image_size_y":"540","fov":"95","sensor_tick":"1.0" if camera_mode else "0.20"}.items():
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
        for name in evidence_kinds+("follow",):
            (output/name).mkdir()
        (output/"screenshots").mkdir()
        geometry = camera_driver = None
        ai_state,pending_results,camera_log = {"view":None,"timing":"","big":False,"insets":{}},{},{"results":0,"unavailable":0,"resets":[],"views":0}
        # The rear and side cameras, each spawned while the driver wants it: config, mount, blueprint, sensor, last picture.
        aux,aux_turn = {},0
        def aux_camera(name,want,now):
            """Spawn a rear or side camera when the driver asks for it; remove it 5 s after it stops asking."""
            state = aux[name]
            if want:
                state["last_want"] = now
            if want and state["sensor"] is None:
                sensor = world.spawn_actor(state["bp"],state["transform"],attach_to=traffic.ego.actor,attachment_type=carla.AttachmentType.Rigid)
                owned.append(sensor)
                sensors.append(sensor)
                streams[name] = queue.Queue()
                sensor.listen(streams[name].put)
                state.update(sensor=sensor,latest=None)
                state["spawns"] += 1
            elif state["sensor"] is not None and not want and now-state["last_want"] > 5.:
                state["sensor"].stop()
                state["sensor"].destroy()
                owned.remove(state["sensor"])
                sensors.remove(state["sensor"])
                streams.pop(name,None)
                state.update(sensor=None,latest=None)
        if camera_mode:
            # The car's own front camera for the two models: 960x540, 95 degrees, the
            # evidence-camera mount (inside the road model's training range).
            cam = session["camera"]
            geometry = CameraGeometry(cam["width"],cam["height"],cam["fov"],tuple(cam["mount"]),cam["pitch"])
            ego = traffic.ego
            physics = ego.actor.get_physics_control()
            drivetrain = {"single_gear": False}
            if (session.get("vehicle") or {}).get("single_gear"):
                # The user's choice (12 Nov): no gearbox to shift. The car's automatic gearbox lost ~2 m/s in every
                # shift and hunted between gears when held at 6-6.8 m/s; one fixed ratio (EV-style) never shifts.
                physics,drivetrain = single_gear(physics,float(session["vehicle"].get("top_speed_m_s",10.)))
                ego.actor.apply_physics_control(physics)
                physics = ego.actor.get_physics_control()
                drivetrain["applied_gears"] = len(physics.forward_gears)
            forward = ego.actor.get_transform().get_forward_vector()
            along = [(w.position.x*forward.x+w.position.y*forward.y)/100 for w in physics.wheels]
            bbox = ego.actor.bounding_box
            # Rear camera (around passes, backing up and turning round): just behind the rear bumper, looking back.
            # Side cameras (the right one around passes, both while turning round): at the door pillars, looking
            # straight out. Same picture size; their boxes come from the same detector.
            aux_geometry = {}
            for name,yaw in (("rear",180.),("left",-90.),("right",90.)):
                cfg = cam.get(name)
                if not cfg:
                    continue
                if name == "rear":
                    at = (bbox.location.x-bbox.extent.x-.1,0.,float(cfg["z"]))
                else:
                    at = (bbox.location.x,bbox.location.y+(-1. if name == "left" else 1.)*(bbox.extent.y+.05),float(cfg["z"]))
                aux_geometry[name] = CameraGeometry(cam["width"],cam["height"],cfg.get("fov",cam["fov"]),at,cfg.get("pitch",0.),yaw_deg=yaw)
                aux[name] = {"cfg":cfg,"mount":at,"yaw":yaw,"sensor":None,"last_want":-1e9,"latest":None,"spawns":0,"active_s":0.,"sent":0}
            camera_driver = CameraDriver(geometry,half_length=bbox.extent.x,half_width=bbox.extent.y,
                front_m=bbox.location.x+bbox.extent.x,wheelbase=max(along)-min(along),max_steer=ego.max_steer,
                settings=DriverSettings.from_dict(session.get("driver")),rear_geometry=aux_geometry.get("rear"),
                side_geometry={name:g for name,g in aux_geometry.items() if name != "rear"})
            traffic.enable_camera_driver(camera_driver)
            bp = library.find("sensor.camera.rgb")
            for key,value in {"image_size_x":str(cam["width"]),"image_size_y":str(cam["height"]),"fov":str(cam["fov"]),
                    "sensor_tick":str(1./cam["hz"])}.items():
                bp.set_attribute(key,value)
            configure_rgb(bp)
            mount = carla.Transform(carla.Location(x=cam["mount"][0],y=cam["mount"][1],z=cam["mount"][2]),carla.Rotation(pitch=cam["pitch"]))
            sensor = world.spawn_actor(bp,mount,attach_to=ego.actor,attachment_type=carla.AttachmentType.Rigid)
            owned.append(sensor)
            sensors.append(sensor)
            streams["perception"] = queue.Queue()
            sensor.listen(streams["perception"].put)
            for name,state in aux.items():
                state["bp"] = library.find("sensor.camera.rgb")
                for key,value in {"image_size_x":str(cam["width"]),"image_size_y":str(cam["height"]),
                        "fov":str(state["cfg"].get("fov",cam["fov"])),"sensor_tick":str(1./cam["hz"])}.items():
                    state["bp"].set_attribute(key,value)
                configure_rgb(state["bp"])
                state["transform"] = carla.Transform(carla.Location(x=state["mount"][0],y=state["mount"][1],z=state["mount"][2]),
                    carla.Rotation(yaw=state["yaw"],pitch=state["cfg"].get("pitch",0.)))
            folder = output/"camera_driving"
            (folder/"views").mkdir(parents=True)
            write_json(folder/"session.json",{k:v for k,v in session.items() if k != "token"})
            write_json(folder/"vehicle.json",{"drivetrain":drivetrain,"half_length_m":bbox.extent.x,"half_width_m":bbox.extent.y,"front_m":bbox.location.x+bbox.extent.x,
                "wheelbase_m":max(along)-min(along),"max_steer_rad":ego.max_steer,"blueprint":ego.actor.type_id,
                "rear_camera_mount":list(aux["rear"]["mount"]) if "rear" in aux else None,
                "side_camera_mounts":{name:list(aux[name]["mount"]) for name in ("left","right") if name in aux}})
            decisions_file = (folder/"decisions.jsonl").open("w",encoding="utf-8")
            collision_file = (folder/"collisions.jsonl").open("w",encoding="utf-8",buffering=1)
            perception_client = PerceptionClient(session,folder/"transport_errors.jsonl")
            # Sat-nav: the car's exact pose and the town's lane graph give it a route to a picked place.
            def sample(road_id,lane,s):
                tf = native_map.get_waypoint_xodr(int(road_id),int(lane),float(s)).transform
                return tf.location.x,tf.location.y,math.radians(tf.rotation.yaw)
            bus = next((d for d in traffic.drivers if hasattr(d,"service_stop_s")),None)
            routes = TownRoutes(network,sample,10.+bus.service_stop_s if bus is not None else 180.)   # bus route starts 10 m into street 2
            satnav = SatNav(routes,front_m=bbox.location.x+bbox.extent.x)
            picker = MapPicker([(str(rid),[(float(x),float(y)) for x,y in np.vstack([v[::8,1:3],v[-1:,1:3]])]) for (rid,lane),v in routes.lanes.items()
                if lane > 0],{name:(pt.x,pt.y) for name,pt in routes.places.items()},SIZE)   # streets and junction connectors
            nav_state = {"open":False,"queue":list(wanted_places),"arrived_at":None,"written":0,"turns":[],"plan_ms":[],
                "pose":None,"hud":None,"maps":0}
            navigation_file = (folder/"navigation.jsonl").open("w",encoding="utf-8",buffering=1)
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
        unavailable_since,camera_overlaps,lane_offsets = None,[],[]
        stage_ms = {"ticks":0,"perception_and_plan":0.,"traffic_update":0.,"checks_and_trace":0.,"draw":0.}
        metrics = {"sim_s":0.,"off_road_s":0.,"wrong_side_s":0.,"junction_s":0.,"stopped_s":0.,"longest_stall_s":0.,
            "decision_s":{},"reason_s":{}}
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
                show_hud=show_hud,show_help=show_help,ai=ai_state if camera_mode else None,
                nav=nav_state["hud"] if camera_mode else None)
            if camera_mode and nav_state["open"] and nav_state["pose"] is not None:
                info = nav_state["hud"] or {}
                status = ("To "+satnav.destination_name+"  ·  "+str(info.get("next") or "")) if satnav.destination is not None else "No destination"
                picker.draw(display,nav_state["pose"][:2],info.get("route_xy"),info.get("pin"),status,nav_state["pose"][2])
            pygame.display.flip()
        def note(event,value):
            viewer_events.append({"sim_s":round(sim_time,2),"tick":ticks,"event":event,"value":value})
        while time.monotonic() < deadline:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_reason = "user_closed"
                if camera_mode and nav_state["open"] and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 \
                        and nav_state["pose"] is not None:
                    name = picker.hit_place(*event.pos)
                    target = name if name is not None else picker.to_world(*event.pos)
                    error = satnav.set_destination(target,*nav_state["pose"],sim_time)
                    if error:
                        hud.notify("Not on a street" if error == "Not on a street" else error[:60],(240,112,96),time.monotonic(),2.)
                    else:
                        nav_state.update(open=False,queue=[],arrived_at=None)
                        hud.notify("Destination  ·  "+satnav.destination_name,(236,180,230),time.monotonic(),2.4)
                    note("destination_click",target if isinstance(target,str) else [round(target[0],1),round(target[1],1)])
                    continue
                if event.type != pygame.KEYDOWN:
                    continue
                key,wall = event.key,time.monotonic()
                place = PLACES.index(spot["value"]) if spot["kind"] == "place" else -1
                if key == pygame.K_ESCAPE:
                    if show_help:
                        show_help = False
                    elif camera_mode and nav_state["open"]:
                        nav_state["open"] = False
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
                elif key == pygame.K_m and camera_mode:
                    nav_state["open"] = not nav_state["open"]
                    note("destination_map","open" if nav_state["open"] else "closed")
                elif key == pygame.K_x and camera_mode:
                    satnav.clear(sim_time)
                    nav_state.update(queue=[],arrived_at=None)
                    hud.notify("Destination cleared  ·  cruising",(240,226,190),wall,1.8)
                    note("destination_cleared",None)
                elif key == pygame.K_a and camera_mode:
                    ai_state["big"] = not ai_state["big"]
                    note("ai_view","full" if ai_state["big"] else "panel")
                elif key == pygame.K_r and camera_mode:
                    moved = traffic.reset_camera_car()
                    if moved is None:
                        hud.notify("No clear spot to reset the car",(240,112,96),wall,1.8)
                    else:
                        camera_driver.reset()
                        pending_results.clear()
                        result["vehicle_pose_teleport"] = True
                        camera_log["resets"].append({"sim_s":round(sim_time,2),"why":"viewer_key_R","moved_m":moved})
                        hud.notify("Car reset onto its lane",(240,226,190),wall,1.8)
                    note("reset_car",moved)
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
            stage_start = time.perf_counter()
            if camera_mode and ticks in pending_results:
                # Fixed latency: the frame captured `latency_ticks` steps ago steers from now on.
                frame_id,picture,extra = pending_results.pop(ticks)
                perception = perception_client.result(frame_id,timeout=session["result_timeout_s"])
                perception.rgb = picture   # the car's own picture (for the asphalt edge beside a shoulder)
                plan = camera_driver.observe(perception,sim_time)
                camera_log["results"] += 1
                if perception.ok:
                    unavailable_since = None
                    t = perception.timings
                    if extra is not None and (perception.rear_boxes is not None or perception.side_boxes):
                        ai_state["insets"][extra[0]] = (extra[1],sim_time)
                    # Each rear or side picture comes every 2-3 frames: show the last one (up to 1 s old) with
                    # what the driver last read from it.
                    insets = {}
                    for view,(shot,taken) in ai_state["insets"].items():
                        if sim_time-taken <= 1.:
                            found = camera_driver.rear_obstacles if view == "rear" else camera_driver.side_obstacles.get(view,(0.,[]))[1]
                            insets[view] = (shot,found or [])
                    ai_state["view"] = ai_view_surface(picture,perception.mask,plan,geometry,insets=insets)
                    ai_state["timing"] = (f"frame {frame_id}  ·  road {t['road_ms']:.0f} ms  ·  detector {t['detector_ms']:.0f} ms"
                        f"  ·  waited {t['wait_ms']:.0f} ms  ·  latency {session['latency_ticks']*50} ms")
                else:
                    camera_log["unavailable"] += 1
                    unavailable_since = unavailable_since or time.monotonic()
                    ai_state["timing"] = "PERCEPTION UNAVAILABLE: "+str(perception.error)[:70]
                    if time.monotonic()-unavailable_since > session["unavailable_limit_s"]:
                        raise RuntimeError("Perception unavailable for over %.0f s: %s" % (session["unavailable_limit_s"],perception.error))
                last = camera_driver.last_output or {}
                decisions_file.write(json.dumps({"frame":frame_id,"sim_s":round(sim_time,3),"ok":perception.ok,"error":perception.error,
                    "timings":perception.timings,"edge_rows":None if plan is None else sum(r[1] is not None for r in plan.rows),
                    "fit_rows":None if plan is None else plan.fit_rows,"wide":None if plan is None else plan.wide,
                    "lost":None if plan is None else plan.lost,"road_end_m":None if plan is None else plan.road_end_m,
                    "path_y_6_10_m":None if plan is None or plan.path is None else [round(float(np.interp(x,plan.path[:,0],plan.path[:,1])),3) for x in (6.,10.)],
                    "boxes":0 if perception.boxes is None else int(len(perception.boxes)),
                    "obstacles":[] if plan is None else [{k:(round(v,2) if isinstance(v,float) else v) for k,v in o.items() if k in ("class","conf","role","distance","x","y_left","y_right")}
                        for o in plan.obstacles if o["role"] != "ignored"],
                    "previous_decision":last.get("decision"),"previous_reason":last.get("reason"),"previous_detail":last.get("detail"),"previous_target":last.get("target"),"rl":last.get("rl"),
                    "reverse":bool(last.get("reverse")),"passing":None if plan is None else plan.passing,
                    "mode":None if plan is None else plan.nav_mode,"plan_found":None if plan is None else plan.plan_found,
                    "plan_ms":None if plan is None or plan.plan_ms is None else round(plan.plan_ms,1),
                    "goal":None if plan is None or plan.junction_goal is None else [round(v,2) for v in plan.junction_goal],
                    "junction_path":None if plan is None or plan.junction_path is None else
                        [[round(float(x),2),round(float(y),2)] for x,y in plan.junction_path[::4]],
                    "look":None if "lookahead_y" not in last else [round(float(last["lookahead_m"]),2),round(float(last["lookahead_y"]),2),
                        round(float(last["steer"]),3)],
                    "pass_note":None if plan is None else plan.pass_note,
                    "asphalt_rows":None if plan is None else len(plan.asphalt_rows),
                    "rear":None if plan is None or plan.rear is None else [{"class":o["class"],"behind_m":round(o["behind_m"],1),
                        "y":[round(o["y_left"],2),round(o["y_right"],2)],"box":[round(v) for v in o["box"]],
                        "conf":round(o["conf"],2)} for o in plan.rear],
                    "side":None if plan is None or not plan.side else {view:[{"class":o["class"],"distance_m":round(o["distance"],1),
                        "near":[[round(float(x),2),round(float(y),2)] for x,y in o["corners"][:2]],"box":[round(v) for v in o["box"]],
                        "conf":round(o["conf"],2)} for o in found] for view,found in plan.side.items()}})+"\n")
                if plan is not None and plan.plan_ms is not None:
                    nav_state["plan_ms"].append(plan.plan_ms)
                if plan is not None and plan.nav_mode == "junction" and nav_state["turns"]:
                    fallback = bool((camera_driver.junction or {}).get("fallback"))
                    nav_state["turns"][-1]["fallback_frames" if fallback else "planned_frames"] += 1
                if perception.ok and camera_log["results"]%10 == 0 and ai_state["view"] is not None:
                    writer.picture(ai_state["view"],output/"camera_driving"/"views"/f"{frame_id:08d}.png")
                    camera_log["views"] += 1
            if camera_mode:
                observed = snapshot.find(traffic.ego.actor.id)
                tf = observed.get_transform() if observed is not None else traffic.ego.actor.get_transform()
                nav_state["pose"] = (tf.location.x,tf.location.y,math.radians(tf.rotation.yaw))
                if satnav.arrived and nav_state["queue"]:
                    nav_state["arrived_at"] = nav_state["arrived_at"] or sim_time
                if nav_state["queue"] and (satnav.destination is None or (satnav.arrived and sim_time-nav_state["arrived_at"] >= 5.)):
                    error = satnav.set_destination(parse_destination(nav_state["queue"].pop(0)),*nav_state["pose"],sim_time)
                    nav_state["arrived_at"] = None
                    if error:
                        hud.notify(error[:60],(240,112,96),time.monotonic(),2.4)
                turn = traffic.camera_turn()
                if turn["granted"] is False and message_wait_changed(nav_state,turn):
                    navigation_file.write(json.dumps({"sim_s":round(sim_time,2),"event":"turn_wait","node":turn["node"],
                        "distance_m":round(turn["distance_m"],1),"why":turn["why"]})+"\n")
                message = satnav.update(*nav_state["pose"],sim_time,turn_granted=turn["granted"],
                    driver_arrived=(camera_driver.last_output or {}).get("reason") == "arrived",uturn_failed=camera_driver.uturn_failed)
                camera_driver.uturn_failed = False
                camera_driver.navigate(message)
                traffic.set_camera_route(satnav.route)
                nav_state["hud"] = satnav.hud()
                for event in satnav.events[nav_state["written"]:]:
                    navigation_file.write(json.dumps(event)+"\n")
                    if event["event"] == "junction_entered":
                        nav_state["turns"].append({"node":event["node"],"name":event.get("name"),"turn":event.get("turn"),
                            "sim_s":event["sim_s"],"planned_frames":0,"fallback_frames":0})
                    if satnav.route is not None and (event["event"] == "route_planned" or event.get("why") == "new_destination"):
                        # A picture of the map with the route, for the record.
                        shot = pygame.Surface(SIZE)
                        shot.fill((10,15,18))
                        info = nav_state["hud"]
                        picker.draw(shot,nav_state["pose"][:2],info["route_xy"],info["pin"],"To "+satnav.destination_name,nav_state["pose"][2])
                        nav_state["maps"] += 1
                        writer.picture(shot,output/"screenshots"/f"destination_map_{nav_state['maps']:02d}.png")
                nav_state["written"] = len(satnav.events)
            stage_update = time.perf_counter()
            states,trace = traffic.update(snapshot,sim_time)
            stage_checks = time.perf_counter()
            stage_ms["perception_and_plan"] += (stage_update-stage_start)*1000
            stage_ms["traffic_update"] += (stage_checks-stage_update)*1000
            stage_ms["ticks"] += 1
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
            if camera_mode:
                # Written as they happen, so a simulator crash cannot lose them.
                ego_id = traffic.ego.actor.id
                for event in traffic.collisions[camera_log.get("collisions_written",0):]:
                    if event["actor"] == "ego" or event["other_id"] == ego_id:
                        camera_log["ego_collision_events"] = camera_log.get("ego_collision_events",0)+1
                        collision_file.write(json.dumps({**event,"sim_s":round(sim_time,2)})+"\n")
                camera_log["collisions_written"] = len(traffic.collisions)
                row = trace[0]
                metrics["sim_s"] += .05
                metrics["off_road_s"] += .05*(not row["on_lane"])
                metrics["wrong_side_s"] += .05*bool(row["wrong_side"])
                metrics["junction_s"] += .05*(row["junction"] is not None)
                metrics["stopped_s"] += .05*(row["speed"] < .1)
                metrics["decision_s"][row["decision"]] = metrics["decision_s"].get(row["decision"],0.)+.05
                metrics["reason_s"][row["reason"]] = metrics["reason_s"].get(row["reason"],0.)+.05
                if row["on_lane"] and not row["wrong_side"] and row["junction"] is None and row["lane_offset_m"] is not None:
                    lane_offsets.append(abs(row["lane_offset_m"]))
                stall = sim_time-ego_mark_s
                metrics["longest_stall_s"] = max(metrics["longest_stall_s"],stall)
                # Waiting behind a vehicle (a junction queue) or working round one (passing, backing out)
                # is not being stuck: allow 3x longer (7 Nov run06 was reset the moment a pass began). Turning round
                # (waiting for a gap, then short moves forward and back) is not progress along its heading either.
                if row["reason"] in ("vehicle_ahead","passing","backing_out","backing_wait","shifting_gear","waiting_turn",
                        "uturn_wait","uturn_forward","uturn_reverse"):
                    camera_log["patient_s"] = sim_time
                # Recently waiting counts too: the reason flips the moment its turn comes (8 Nov smoke02 was reset
                # just as the motorcycle holding junction A cleared it).
                patient = sim_time-camera_log.get("patient_s",-1e9) < 5.
                limit_s = None if not session.get("auto_reset_s") else session["auto_reset_s"]*(3 if patient else 1)
                if row["reason"] == "arrived":
                    limit_s,ego_mark_s = None,sim_time   # parked at its destination, not stuck
                if limit_s and stall >= limit_s:
                    turning = camera_driver.uturn is not None and camera_driver.uturn["phase"] != "approach"
                    moved = traffic.reset_camera_car()
                    camera_log["resets"].append({"sim_s":round(sim_time,2),"why":f"no progress for {limit_s:.0f} s",
                        "moved_m":moved,"x":round(row["x"],2),"y":round(row["y"],2),"reason_before":row["reason"]})
                    if moved is not None:
                        camera_driver.reset()
                        # Stuck turning round: the re-plan from the lane goes round instead (18 Nov run: it was
                        # lifted back and planned the same U-turn again).
                        camera_driver.uturn_failed = turning
                        pending_results.clear()
                        result["vehicle_pose_teleport"] = True
                        hud.notify("Stuck: car lifted back onto its lane",(240,200,96),time.monotonic(),2.4)
                    ego_mark_s = sim_time
            pairs = list(states.items())
            for i,(aid,a) in enumerate(pairs):
                for bid,b in pairs[i+1:]:
                    if math.hypot(a["x"]-b["x"],a["y"]-b["y"]) > 17:
                        continue
                    overlap,gap = polygon_gap(a["polygon"],b["polygon"])
                    if camera_mode and traffic.ego.actor.id in (aid,bid):
                        if overlap:
                            camera_overlaps.append({"frame":snapshot.frame,"a":aid,"b":bid})
                        continue
                    min_gap = min(min_gap,gap)
                    if overlap:
                        overlaps.append({"frame":snapshot.frame,"a":aid,"b":bid,"gap_m":gap})
            for row in trace:
                if row.get("camera_driven"):
                    continue   # your car's road position is scored in the camera_driving block
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
            stage_ms["checks_and_trace"] += (time.perf_counter()-stage_checks)*1000
            server_start = time.monotonic()
            frame = world.tick(25)
            ticks += 1
            frames = {"follow":wait_frame(streams["view_follow"],frame)}
            if camera_mode:
                # The 10 Hz model camera: by now this tick's image has normally arrived too.
                image = None
                while True:
                    try:
                        image = streams["perception"].get_nowait()
                    except queue.Empty:
                        break
                for name,state in aux.items():
                    if state["sensor"] is not None:
                        state["active_s"] += .05
                        while True:
                            try:
                                state["latest"] = streams[name].get_nowait()
                            except queue.Empty:
                                break
                wanted = camera_driver.wanted_views if aux else []
                for name in aux:
                    aux_camera(name,name in wanted,sim_time)
                if image is not None:
                    picture = np.frombuffer(image.raw_data,np.uint8).reshape(image.height,image.width,4)[...,2::-1].copy()
                    camera_driver.captured(image.frame,traffic.camera_ego["speed"],traffic.camera_ego["yaw_rate"],
                        dt=.05 if image.frame == frame else 0.)
                    # One more picture a frame for the detector: the wanted rear and side views in turn (its link has
                    # time for two pictures a frame, not four). Not named `latest`: that would hide latest() below
                    # (TAB crash, 8 Nov).
                    extra = None
                    ready = [name for name in wanted if aux[name]["latest"] is not None and 0 <= image.frame-aux[name]["latest"].frame <= 2]
                    if ready:
                        name = ready[aux_turn%len(ready)]
                        aux_turn += 1
                        shot = aux[name]["latest"]
                        extra = (name,np.frombuffer(shot.raw_data,np.uint8).reshape(shot.height,shot.width,4)[...,2::-1].copy())
                        aux[name]["sent"] += 1
                    perception_client.submit(image.frame,picture,image.timestamp,extra=extra)
                    pending_results[ticks-1+session["latency_ticks"]] = (image.frame,picture,extra)
                    camera_log["late_frames"] = camera_log.get("late_frames",0)+int(image.frame != frame)
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
                for name in evidence_kinds[1:]:
                    bundle[name] = wait_frame(streams[name],rgb.frame)
                received += 1
                if received%(1 if camera_mode else 5) == 0 and rgb.frame in recent_follow:
                    bundle["follow"] = recent_follow[rgb.frame]
                    writer.camera_set(output,{name:detach(image) for name,image in bundle.items()},
                        {"source_frame":rgb.frame,"world_id":world.id,"sim_s":rgb.timestamp-start_sim,"district":trace[0]["district"],
                        "weather":weather})
                    submitted += 1
            district = trace[0]["district"]
            if due:
                draw_start = time.perf_counter()
                draw()
                stage_ms["draw"] += (time.perf_counter()-draw_start)*1000
                last_draw = time.monotonic()
            # District pictures only show your own car, with the HUD on.
            if due and spot["sensor"] is None and show_hud and not show_help and (ticks == 240 or (district not in saved_regions and sim_time > 8)):
                writer.picture(display.copy(),output/"screenshots"/(district+".png"))
                saved_regions.add(district)
            if ticks%100 == 0:
                progress("running",sim_s=sim_time,ticks=ticks,saved_frames=writer.saved,world_id=world.id,
                    ego_distance_m=traffic.distance["ego"],ego_districts=sorted(traffic.visited["ego"]),
                    collisions=len(traffic.collisions),overlaps=len(overlaps),departures=len(departures),
                    **({"camera_decision":trace[0]["decision"],"camera_reason":trace[0]["reason"],"camera_detail":trace[0]["detail"],
                        "camera_results":camera_log["results"],"camera_unavailable":camera_log["unavailable"],
                        "camera_resets":len(camera_log["resets"]),"off_road_s":round(metrics["off_road_s"],1),
                        "wrong_side_s":round(metrics["wrong_side_s"],1),"ego_collision_events":camera_log.get("ego_collision_events",0),
                        "camera_reset_log":camera_log["resets"],"camera_district_order":[t["district"] for t in traffic.transitions if t["actor"] == "ego"],
                        "late_frames":camera_log.get("late_frames",0),"rear_camera_on":aux.get("rear",{}).get("sensor") is not None,
                        "side_cameras_on":[name for name in ("left","right") if aux.get(name,{}).get("sensor") is not None],
                        "pass_events":camera_driver.events[-6:]} if camera_mode else {}))
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
        result["loop_stage_mean_ms"] = {k:round(v/max(1,stage_ms["ticks"]),2) for k,v in stage_ms.items() if k != "ticks"}
        result["loop_stage_mean_ms"]["server_tick_and_frames"] = round(float(server.mean()),2)
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
        if camera_mode:
            # Your car's crashes, off-road time and stalls are driving results, not software failures.
            ego_id = traffic.ego.actor.id
            mine = [c for c in traffic.collisions if c["actor"] == "ego" or c["other_id"] == ego_id]
            others = [c for c in traffic.collisions if c not in mine]
            events = []
            for c in sorted(mine,key=lambda c:c["frame"]):
                partner = c["other_id"] if c["actor"] == "ego" else next((d.actor.id for d in traffic.drivers if d.name == c["actor"]),None)
                if events and events[-1]["with_id"] == partner and c["frame"]-events[-1]["last_frame"] <= 40:
                    events[-1]["last_frame"] = c["frame"]
                    events[-1]["max_impulse"] = max(events[-1]["max_impulse"],c["impulse"])
                    continue
                events.append({"first_frame":c["frame"],"last_frame":c["frame"],"with_id":partner,
                    "with":c["other_type"] if c["actor"] == "ego" else c["actor"],"max_impulse":c["impulse"]})
            offsets = np.array(lane_offsets) if lane_offsets else np.zeros(1)
            moving = max(1e-6,metrics["sim_s"])
            result["camera_driving"] = {"controller":"camera_only_models","ground_truth_to_driver":False,"safety_backstop":False,
                "distance_m":traffic.distance["ego"],"districts_visited":sorted(traffic.visited["ego"]),
                "district_order":[t["district"] for t in traffic.transitions if t["actor"] == "ego"],
                "collisions":events,"collision_count":len(events),"contacts_sampled":len(camera_overlaps),
                "off_road_s":metrics["off_road_s"],"off_road_share":metrics["off_road_s"]/moving,
                "wrong_side_s":metrics["wrong_side_s"],"wrong_side_share":metrics["wrong_side_s"]/moving,
                "junction_s":metrics["junction_s"],"stopped_s":metrics["stopped_s"],"longest_stall_s":metrics["longest_stall_s"],
                "lane_centre_offset_m":{"median":float(np.median(offsets)),"p90":float(np.percentile(offsets,90)),"samples":len(lane_offsets),
                    "definition":"|distance| from the centre of the lane under the car, its own direction, outside junctions"},
                "decision_s":metrics["decision_s"],"reason_s":dict(sorted(metrics["reason_s"].items(),key=lambda kv:-kv[1])),
                "resets":camera_log["resets"],"reset_count":len(camera_log["resets"]),
                "perception_results":camera_log["results"],"perception_unavailable":camera_log["unavailable"],
                "views_saved":camera_log["views"],"transport":perception_client.summary(),
                "pass_events":[{"sim_s":s,"event":e,"detail":d} for s,e,d in camera_driver.events],
                "rear_camera":{"configured":"rear" in aux,"spawns":aux.get("rear",{}).get("spawns",0),
                    "active_s":round(aux.get("rear",{}).get("active_s",0.),1),"pictures_sent":aux.get("rear",{}).get("sent",0),
                    "mount":list(aux["rear"]["mount"]) if "rear" in aux else None},
                "side_cameras":{name:{"spawns":aux[name]["spawns"],"active_s":round(aux[name]["active_s"],1),
                    "pictures_sent":aux[name]["sent"],"mount":list(aux[name]["mount"])} for name in ("left","right") if name in aux},
                "reversed_m":traffic.camera_ego.get("reverse_m",0.),
                "navigation":navigation_summary(satnav.events,nav_state,metrics,traffic.stats),
                "others_yield_to_camera_car":{k:traffic.stats.get(k) for k in ("yield_to_camera_car_s","yield_to_camera_car_events")}}
            result["passed"] = bool(ticks >= 200 and saved >= 5 and not writer.errors and len(world_ids) == 1 and result["world_generate_calls"] == 1
                and not others and not overlaps and not departures and not direction_errors and longest_stopped < 60
                and not traffic.lost_walkers and camera_log["results"] > 0)
        else:
            result["passed"] = bool(ticks >= 200 and saved >= 5 and not writer.errors and len(world_ids) == 1 and result["world_generate_calls"] == 1
                and not traffic.collisions and not overlaps and not departures and not direction_errors and traffic.distance["ego"] > 8
                and longest_stopped < 60 and longest_ego_stall < 120 and not traffic.lost_walkers)
    except Exception:
        result["errors"].append(traceback.format_exc())
        print(result["errors"][-1],flush=True)
    finally:
        if writer is not None and writer.is_alive():
            writer.close()
        if perception_client is not None:
            perception_client.close()
        if navigation_file:
            navigation_file.close()
        if decisions_file:
            decisions_file.close()
            collision_file.close()
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
