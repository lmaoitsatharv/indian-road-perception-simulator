"""Collision-free asphalt appearance made from an installed, paintable flat mesh."""
import carla
import numpy as np

from carla_starter import write_json

MESH = "/Game/Carla/Static/Building/ConstructionSite/SM_concreteFloorT5_02.SM_concreteFloorT5_02"
ROAD_RGB = (17, 18, 20)
GRAIN_SEED = 20261020
CONFIRMATION_PALETTES = {"worn_warm": ((31,27,23),2026102301), "cool_aggregate": ((23,27,30),2026102302)}


def asphalt_texture(rgb=ROAD_RGB, seed=GRAIN_SEED):
    values = np.clip(np.array(rgb)[None, None, :] +
                     np.random.default_rng(seed).integers(-3, 4, (128, 128, 1)), 0, 255)
    texture = carla.TextureColor(128, 128)
    for y in range(128):
        for x in range(128):
            texture.set(x, y, carla.Color(*map(int, values[y, x]), 255))
    return texture


def apply_surface(world, owned, output, style, active_arms=None, palette=None):
    if palette is not None and palette not in CONFIRMATION_PALETTES:
        raise ValueError("Unknown confirmation asphalt palette")
    if style == "legacy":
        return {"style": style}
    if style != "asphalt":
        raise ValueError("Unknown road surface style")
    # Installed slab: 48 m by 20 m, local top at +0.25 m. At scale 0.45
    # its width matches the 9 m carriageway. Overlap pieces to cover cutouts.
    scale, top_z = .45, .015
    centers = list(range(18, 145, 6))
    tiles = [(side*x, 0., 0.) for side in (-1, 1) for x in centers]
    tiles += [(0., side*y, 90.) for side in (-1, 1) for y in centers]
    tiles += [(x, y, 0.) for x in (-1.2, 1.2) for y in (-7.5, 0., 7.5)]
    if active_arms is not None:
        tiles = [(x,y,yaw) for x,y,yaw in tiles
                 if (abs(y) < 1 or (y > 0 and "south" in active_arms) or (y < 0 and "north" in active_arms))]
    before = set(world.get_names_of_all_objects())
    bp = world.get_blueprint_library().find("static.prop.mesh")
    bp.set_attribute("role_name", "india_asphalt_appearance")
    bp.set_attribute("mesh_path", MESH)
    bp.set_attribute("scale", str(scale))
    bp.set_attribute("mass", "0")
    actors = []
    for x, y, yaw in tiles:
        actor = world.spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=top_z-.25*scale),
                                                    carla.Rotation(yaw=yaw)))
        owned.append(actor)
        actor.set_collisions(False)
        actors.append(actor.id)
    names = sorted(set(world.get_names_of_all_objects()) - before)
    if len(names) != len(actors):
        raise RuntimeError("Could not identify all owned asphalt surface meshes")
    rgb, seed = CONFIRMATION_PALETTES[palette] if palette else (ROAD_RGB, GRAIN_SEED)
    world.apply_color_texture_to_objects(names, carla.MaterialParameter.Diffuse, asphalt_texture(rgb, seed))
    record = {"style": style, "version": "asphalt-v2", "mesh": MESH,
              "actor_ids": actors, "actor_names": names, "tile_scale": scale,
              "tile_centers_xy_yaw": tiles, "surface_z_m": top_z,
              "collisions_enabled": False, "texture_size": [128, 128],
              "road_rgb_linear_texture": rgb, "grain_seed": seed, "confirmation_palette": palette,
              "note": "Appearance meshes only; original road collision, map and driving policy retained"}
    write_json(output / "surface_style.json", record)
    return record
