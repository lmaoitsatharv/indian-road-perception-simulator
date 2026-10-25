"""Reusable installed CARLA assets for an illustrative neighbourhood street."""
from __future__ import annotations

import math
import carla


CONFIRMATION_FACADES = {
    "lowrise": ["/Game/Carla/Static/Building/SM_House12_1.SM_House12_1",
                "/Game/Carla/Static/Building/SM_Apartment01_1.SM_Apartment01_1"],
    "apartments": ["/Game/Carla/Static/Building/SM_Apartment02_1.SM_Apartment02_1",
                   "/Game/Carla/Static/Building/SM_Apartment01_1.SM_Apartment01_1"],
}


def add_scenery(world, owned, profile=None):
    if profile is not None and profile not in CONFIRMATION_FACADES:
        raise ValueError("Unknown confirmation scenery profile")
    library = world.get_blueprint_library()
    record = []

    def prop(identifier, x, y, yaw=0, z=0):
        bp = library.find(identifier)
        bp.set_attribute("role_name", "india_scenery")
        actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)))
        if actor is not None:
            owned.append(actor)
            record.append({"id": actor.id, "blueprint": identifier, "x": x, "y": y})
        return actor

    def mesh(path, x, y, scale=1, yaw=0, z=0):
        bp = library.find("static.prop.mesh")
        bp.set_attribute("role_name", "india_scenery")
        bp.set_attribute("mesh_path", path)
        bp.set_attribute("scale", str(scale))
        bp.set_attribute("mass", "0")
        actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)))
        if actor is not None:
            owned.append(actor)
        return actor

    ground = mesh("/Engine/BasicShapes/Plane.Plane", 0, 0, scale=360, z=-0.04)
    if ground is not None:
        record.append({"id": ground.id, "kind": "ground_plane"})
    facades = [
        "/Game/Carla/Static/Building/Apartment03/SM_Apartment03_v01_merged.SM_Apartment03_v01_merged",
        "/Game/Carla/Static/Building/Apartment04/SM_Apartment04_v03_Merged.SM_Apartment04_v03_Merged",
        "/Game/Carla/Static/Building/Apartment04/SM_Apartment04_v05_Merged.SM_Apartment04_v05_Merged",
    ]
    if profile:
        facades = CONFIRMATION_FACADES[profile]
    # Place mesh bounding boxes behind the sidewalk; no decorative wall is put in a traffic lane.
    for sign in (-1, 1):
        for index, x in enumerate((-115, -75, -35, 35, 75, 115)):
            scale = .45 if profile is None else (.55 if profile == "lowrise" else .35)
            actor = mesh(facades[index % len(facades)], x, sign * 30, scale=scale, yaw=180 if sign < 0 else 0)
            if actor is None:
                continue
            transform = actor.get_transform()
            vertices = actor.bounding_box.get_world_vertices(transform)
            min_x, max_x = min(v.x for v in vertices), max(v.x for v in vertices)
            min_y, max_y = min(v.y for v in vertices), max(v.y for v in vertices)
            min_z = min(v.z for v in vertices)
            transform.location.x += x - (min_x + max_x) / 2
            verge = 11.0 if sign < 0 and x > 12 else 8.5
            transform.location.y += (verge - min_y) if sign > 0 else (-verge - max_y)
            transform.location.z -= min_z
            actor.set_transform(transform)
            record.append({"id": actor.id, "kind": "generic_facade", "mesh": facades[index % len(facades)],
                           "profile": profile, "scale": scale,
                           "x": transform.location.x, "y": transform.location.y,
                           "note": "Existing generic architecture; not a calibrated Indian building"})
    for side in (-1, 1):
        for x in (-125, -95, -55, 55, 95, 125):
            prop("static.prop.coconutpalm", x, side * (10.0 if side < 0 and x > 12 else 7.7))
        for y in (-110, -65, 65, 110):
            prop("static.prop.aporosatree", side * 8, y)
    for x, side in [(-86, -1), (-50, 1), (48, -1), (96, 1)]:
        extra = 2.4 if side < 0 and x > 12 else 0
        prop("static.prop.kiosk_01", x, side * (8.0 + extra), yaw=0 if side > 0 else 180)
        prop("static.prop.foodcart", x + 4, side * (7.5 + extra), yaw=90)
        prop("static.prop.plasticchair", x + 6, side * (7.4 + extra))
    prop("static.prop.busstop", -64, -7.5, yaw=0)
    prop("static.prop.bench01", -61, -7.5, yaw=0)
    # Roadworks are placed on the outer verge, leaving the 2.2 m footpath open.
    for x in (70, 72, 74):
        prop("static.prop.trafficcone01", x, 7.2)
    return record
