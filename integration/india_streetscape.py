"""Raised paving beside the existing driving lanes; never used by the driver."""
import math

import carla
import numpy as np

from carla_starter import write_json
from india_surface import MESH

PAVING_RGB = (68, 60, 51)
PAVING_GRAIN_SEED = 20261021


def paving_pixels(size=128):
    """Muted paving blocks and mortar, with a repeatable fine aggregate."""
    yy, xx = np.indices((size, size))
    joints = (yy % 32 < 2) | ((xx + (yy // 32 % 2) * 16) % 32 < 2)
    grain = np.random.default_rng(PAVING_GRAIN_SEED).integers(-3, 4, (size, size, 1))
    pixels = np.clip(np.array(PAVING_RGB)[None, None, :] + grain, 0, 255).astype(np.uint8)
    pixels[joints] = [32, 30, 27]
    return pixels


def texture(pixels):
    height, width = pixels.shape[:2]
    result = carla.TextureColor(width, height)
    for y in range(height):
        for x in range(width):
            result.set(x, y, carla.Color(*map(int, pixels[y, x]), 255))
    return result


def paving_layout(active_arms=None):
    # The installed slab is 48 x 20 m. These strips match the existing 2.2 m
    # sidewalks. The east/south strip starts beyond the 2.4 m parking bay.
    result = []
    for axis in (0, 1):
        for direction in (-1, 1):
            for side in (-1, 1):
                offset = 8.0 if axis == 0 and direction == 1 and side == -1 else 5.6
                for distance in np.arange(14.65, 150.0, 4.9):
                    x, y = direction * float(distance), side * offset
                    if axis:
                        x, y = y, x
                    result.append({"x": x, "y": y, "yaw": 90. if axis else 0.,
                                   "scale": .11, "top_z": .16, "kind": "footpath"})
    if active_arms is not None:
        result = [t for t in result if t["yaw"] == 0 or
                  (t["y"] > 0 and "south" in active_arms) or (t["y"] < 0 and "north" in active_arms)]
        for side, name in ((1,"south"),(-1,"north")):
            if name not in active_arms:
                result += [{"x": x, "y": side*5.6, "yaw": 0., "scale": .11,
                            "top_z": .16, "kind": "closed_branch_footpath"} for x in (-2.5,2.5)]
    return result


def footprint_samples(tile):
    yaw = math.radians(tile["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    for u in np.linspace(-24*tile["scale"], 24*tile["scale"], 9):
        for v in np.linspace(-10*tile["scale"]+.03, 10*tile["scale"]-.03, 5):
            yield tile["x"]+u*c-v*s, tile["y"]+u*s+v*c


def apply_streetscape(world, owned, output, active_arms=None):
    road_map = world.get_map()
    tiles = paving_layout(active_arms)
    # Reject paving over an OpenDRIVE driving lane before placing any mesh.
    for tile in tiles:
        for x, y in footprint_samples(tile):
            if road_map.get_waypoint(carla.Location(x=x, y=y, z=0), False, carla.LaneType.Driving):
                raise ValueError("Footpath appearance overlaps a driving lane")
    before = set(world.get_names_of_all_objects())
    bp = world.get_blueprint_library().find("static.prop.mesh")
    bp.set_attribute("role_name", "india_footpath_appearance")
    bp.set_attribute("mesh_path", MESH)
    bp.set_attribute("mass", "0")
    actors, native, bounds = [], [], []
    for tile in tiles:
        bp.set_attribute("scale", str(tile["scale"]))
        actor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=tile["x"], y=tile["y"], z=tile["top_z"]-.25*tile["scale"]),
            carla.Rotation(yaw=tile["yaw"])))
        owned.append(actor)
        actor.set_collisions(False)
        actors.append(actor.id)
        native.append((actor, tile))
    # Newly spawned actor transforms/bounds are cached until world advancement.
    # No traffic actors or recording sensors exist during this static setup tick.
    world.tick(10)
    for actor, tile in native:
        vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
        limits = [[min(getattr(v, axis) for v in vertices), max(getattr(v, axis) for v in vertices)]
                  for axis in ("x", "y", "z")]
        if abs(limits[2][1]-tile["top_z"]) > .02:
            write_json(output / "streetscape_failed_bounds.json", {"tile": tile, "native_bounds": limits})
            raise ValueError(f"Installed footpath mesh has an unexpected native height: {limits[2]}")
        # Verify the native mesh bounds as well as the offline layout assumptions.
        for x in np.linspace(limits[0][0]+.03, limits[0][1]-.03, 9):
            for y in np.linspace(limits[1][0]+.03, limits[1][1]-.03, 5):
                if road_map.get_waypoint(carla.Location(x=float(x), y=float(y), z=0), False, carla.LaneType.Driving):
                    raise ValueError("Native footpath mesh bounds intrude into a driving lane")
        bounds.append({"actor_id": actor.id, "xyz_minmax_m": limits})
    names = sorted(set(world.get_names_of_all_objects())-before)
    if len(names) != len(actors):
        raise RuntimeError("Could not identify all owned footpath appearance meshes")
    world.apply_color_texture_to_objects(names, carla.MaterialParameter.Diffuse, texture(paving_pixels()))
    record = {"version": "raised-paving-v1", "mesh": MESH, "actor_ids": actors,
              "actor_names": names, "tiles": tiles, "top_height_m": .16,
              "texture_size": [128,128], "paving_rgb_texture": PAVING_RGB, "grain_seed": PAVING_GRAIN_SEED,
              "native_bounds": bounds, "native_bounds_driving_lane_overlap_samples": 0,
              "native_class": 3, "reference_class": "non-road; never part of the asphalt-instance set",
              "collisions_enabled": False, "driving_lane_overlap_samples": 0,
              "note": "Appearance only. Existing sidewalk collision and the traffic policy are retained."}
    write_json(output / "streetscape.json", record)
    return record
