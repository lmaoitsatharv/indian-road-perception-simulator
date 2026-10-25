"""Inspect native UV/material support instead of claiming texture application worked."""
import math
import queue
from pathlib import Path

import carla
import pygame

from carla_starter import write_json
from india_world_maps import build_world
from india_world_scenery import image_texture,ASSETS


def probe(client,output):
    from india_world_scene import save_native,wait_frame,cleanup
    world = client.generate_opendrive_world(build_world("market_street"),
        carla.OpendriveGenerationParameters(wall_height=0.,additional_width=0.,enable_pedestrian_navigation=False))
    settings = world.get_settings()
    settings.synchronous_mode,settings.fixed_delta_seconds = True,.05
    world.apply_settings(settings)
    weather = carla.WeatherParameters.ClearNoon
    weather.sun_altitude_angle = 55.
    world.set_weather(weather)
    candidates = ["/Engine/BasicShapes/Plane.Plane","/Game/Carla/Static/Other/SM_Plane.SM_Plane",
        "/Game/Carla/Static/Other/Banner/SM_bannerSingleShort.SM_bannerSingleShort",
        "/Game/Carla/Static/Static/SM_Cartel_Add_02.SM_Cartel_Add_02",
        "/Game/Carla/Static/Building/ConstructionSite/SM_concreteFloorT5_02.SM_concreteFloorT5_02",
        "/Game/Carla/Static/TrafficSign/PostSigns/SM_InterchangeSign_03_Cartel.SM_InterchangeSign_03_Cartel"]
    owned,sensors,records = [],[],[]
    try:
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for k,v in {"image_size_x":"640","image_size_y":"480","fov":"65"}.items():
            bp.set_attribute(k,v)
        camera = world.spawn_actor(bp,carla.Transform(carla.Location(z=20)))
        owned.append(camera)
        sensors.append(camera)
        pending = queue.Queue()
        camera.listen(pending.put)
        texture = image_texture(ASSETS/"textures/uv_grid.png")
        for i,path in enumerate(candidates):
            before = set(world.get_names_of_all_objects())
            bp = world.get_blueprint_library().find("static.prop.mesh")
            for k,v in {"mesh_path":path,"mass":"0","scale":"1","role_name":"indian_uv_probe"}.items():
                bp.set_attribute(k,v)
            actor = world.spawn_actor(bp,carla.Transform(carla.Location(x=i*70.,y=70.,z=10.)))
            owned.append(actor)
            names = sorted(set(world.get_names_of_all_objects())-before)
            world.tick(15)
            box = actor.bounding_box
            centre = actor.get_transform().transform(carla.Location(box.location))
            extents = [box.extent.x,box.extent.y,box.extent.z]
            normal = min(range(3),key=lambda j:extents[j])
            distance = max(extents)*3.8
            if normal == 0:
                cam = carla.Transform(carla.Location(x=centre.x+distance,y=centre.y,z=centre.z),carla.Rotation(yaw=180))
            elif normal == 1:
                cam = carla.Transform(carla.Location(x=centre.x,y=centre.y-distance,z=centre.z),carla.Rotation(yaw=90))
            else:
                cam = carla.Transform(carla.Location(x=centre.x,y=centre.y,z=centre.z+distance),carla.Rotation(pitch=-90))
            camera.set_transform(cam)
            for _ in range(4):
                frame = world.tick(15)
                image = wait_frame(pending,frame)
            save_native(image,output/f"asset_{i}_before.png")
            world.apply_color_texture_to_objects(names,carla.MaterialParameter.Diffuse,texture)
            for _ in range(4):
                frame = world.tick(15)
                image = wait_frame(pending,frame)
            save_native(image,output/f"asset_{i}_after.png")
            records.append({"index":i,"mesh":path,"native_extents_m":extents,"native_box_location_m":[box.location.x,box.location.y,box.location.z],
                "semantic_tags":list(actor.semantic_tags),"names":names,"camera_normal_axis":normal})
        write_json(output/"asset_probe.json",{"candidates":records,"purpose":"Inspect UV/material/readability before selecting native sign geometry"})
    finally:
        warnings = cleanup(client,world,owned,sensors)
        write_json(output/"result.json",{"passed":not warnings,"cleanup_warnings":warnings})
    return 0 if not warnings else 1
