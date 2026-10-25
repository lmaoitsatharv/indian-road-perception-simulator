"""Auditable native scenery for the five new worlds; no old-map assumptions."""
from __future__ import annotations

import math
from pathlib import Path

import carla
import numpy as np
import pygame

from india_surface import MESH, asphalt_texture
from india_world_maps import PROFILES, road_specs

ASSETS = Path(__file__).resolve().parent / "indian_world_assets"
HOUSE = "/Game/Carla/Static/Building/SM_House12_1.SM_House12_1"
APARTMENT = "/Game/Carla/Static/Building/SM_Apartment01_1.SM_Apartment01_1"
BANNER = "/Game/Carla/Static/Other/Banner/SM_bannerSingleShort.SM_bannerSingleShort"


def image_texture(path):
    image = pygame.image.load(str(path))
    # Stream the original correctly oriented texture; never relabel native tags.
    values = pygame.surfarray.array3d(image)
    result = carla.TextureColor(image.get_width(), image.get_height())
    for x in range(image.get_width()):
        for y in range(image.get_height()):
            result.set(x, y, carla.Color(*map(int, values[x,y]), 255))
    return result


class WorldScenery:
    def __init__(self, world, owned, key, xml):
        self.world, self.owned, self.key = world, owned, key
        self.profile, self.map = PROFILES[key], world.get_map()
        self.roads = road_specs(xml)
        self.records, self.texture_groups, self.asphalt = [], {}, []

    def _record(self, actor, kind, project_class, collision, **detail):
        self.owned.append(actor)
        actor.set_collisions(collision)
        row = {"actor_id":actor.id,"type_id":actor.type_id,"kind":kind,
            "project_class":project_class,"collisions_enabled":collision,
            "native_semantic_tags":list(actor.semantic_tags),"project_label_requires_instance_override":False, **detail}
        self.records.append(row)
        return row

    def prop(self, identifier, transform, kind, project_class="other", *, optional=False):
        bp = self.world.get_blueprint_library().find(identifier)
        bp.set_attribute("role_name", "indian_world_"+kind)
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            if optional:
                self.records.append({"kind":kind,"blueprint":identifier,"spawned":False})
                return None
            raise RuntimeError("Required native scenery did not spawn: "+identifier)
        self._record(actor, kind, project_class, True, blueprint=identifier)
        return actor

    def mesh(self, path, transform, scale, kind, project_class, *, collision=True, texture=None):
        # Texture lookup is bracketed around each newly owned mesh. A role string
        # is not an Unreal object name, and cannot safely identify a material.
        before = set(self.world.get_names_of_all_objects()) if texture else None
        bp = self.world.get_blueprint_library().find("static.prop.mesh")
        bp.set_attribute("role_name", "indian_world_"+kind)
        bp.set_attribute("mesh_path", path)
        bp.set_attribute("scale", str(scale))
        bp.set_attribute("mass", "0")
        actor = self.world.spawn_actor(bp, transform)
        row = self._record(actor, kind, project_class, collision, mesh=path, scale=scale)
        if texture:
            names = sorted(set(self.world.get_names_of_all_objects())-before)
            if len(names) != 1:
                raise RuntimeError("Owned texture mesh identity is ambiguous")
            row["object_names"] = names
            self.texture_groups.setdefault(texture, []).extend(names)
            row["texture"] = texture
        return actor

    def outside(self, wp, distance):
        t = wp.transform
        # Offset left relative to travel, i.e. outside an LHT lane. Account for
        # another lane on a divided road, rather than placing props in that lane.
        yaw = math.radians(t.rotation.yaw)
        return carla.Location(x=t.location.x+math.sin(yaw)*distance,
            y=t.location.y-math.cos(yaw)*distance, z=.02)

    def board(self, wp, name, distance, *, shop=False):
        location = self.outside(wp, distance)
        location.z = 2.35 if shop else 2.10
        scale = 1.8 if shop else .9
        # Native banner material accepts runtime texture streaming. Its source
        # UV atlas is accounted for by the generated banner texture, rather than
        # using a tiled concrete-floor material for legible text.
        yaw = wp.transform.rotation.yaw
        actor = self.mesh(BANNER, carla.Transform(location,carla.Rotation(yaw=yaw+90,roll=90)),
            scale, "shop_board" if shop else "indian_sign", "building" if shop else "traffic_sign",
            texture="banner_"+name+".png")
        if not shop:
            pole = self.prop("static.prop.trafficcone01",carla.Transform(self.outside(wp,distance)),
                "sign_base", optional=True)
            # The cone is a separate visible object, not a traffic-sign pixel.
        return actor

    def _on_drive(self, x, y):
        return self.map.get_waypoint(carla.Location(x=float(x),y=float(y),z=0.),False,carla.LaneType.Driving)

    def surfaces(self):
        before = set(self.world.get_names_of_all_objects())
        checked = 0
        # Cover lane interiors only, leaving a small worn native edge. Corners
        # and edge samples must all remain inside a native driving lane, including
        # curves, variable widths and medians. Never paint a sidewalk as road.
        for road in self.roads:
            for lane in road["lanes"]:
                for s in np.arange(5.,road["length"]-4.,4.8):
                    wp = self.map.get_waypoint_xodr(road["id"],lane,float(s))
                    if wp is None:
                        raise ValueError("Missing appearance waypoint")
                    yaw = math.radians(wp.transform.rotation.yaw)
                    scale = (wp.lane_width-.30)/20.
                    samples = []
                    for u in (-24*scale,0.,24*scale):
                        for v in (-10*scale,0.,10*scale):
                            x = wp.transform.location.x+u*math.cos(yaw)-v*math.sin(yaw)
                            y = wp.transform.location.y+u*math.sin(yaw)+v*math.cos(yaw)
                            samples.append((x,y))
                    if any(not self._on_drive(x,y) for x,y in samples):
                        # Junction overlaps/cutouts can be irregular. Keep native
                        # material there instead of covering non-driving geometry.
                        continue
                    actor = self.mesh(MESH, carla.Transform(carla.Location(
                        x=wp.transform.location.x,y=wp.transform.location.y,z=.012-.25*scale),
                        carla.Rotation(yaw=wp.transform.rotation.yaw)), scale, "asphalt", "road",collision=False)
                    self.asphalt.append(actor.id)
                    checked += len(samples)
        names = sorted(set(self.world.get_names_of_all_objects())-before)
        if len(names) != len(self.asphalt):
            raise RuntimeError("Asphalt instance/object identity count differs")
        self.world.apply_color_texture_to_objects(names,carla.MaterialParameter.Diffuse,
            asphalt_texture((22,23,24),20261023))
        return {"asphalt_actor_ids":self.asphalt,"asphalt_object_names":names,
            "geometry_checked_samples":checked,"collision":"Native generated road; appearance collision disabled",
            "native_tags_preserved":True,"label_policy":"Instance-specific road override plus map/depth required; not raw Building tags"}

    def build(self):
        before = set(self.world.get_names_of_all_objects())
        self.mesh(MESH,carla.Transform(carla.Location(z=-.08-.25*35)),
            35.,"ground","terrain",collision=False)
        ground_names = sorted(set(self.world.get_names_of_all_objects())-before)
        self.world.apply_color_texture_to_objects(ground_names,carla.MaterialParameter.Diffuse,
            asphalt_texture((41,53,24) if self.profile.scenery == "rural" else (76,65,48),2026102302))
        self.mesh("/Engine/BasicShapes/Plane.Plane",carla.Transform(carla.Location(z=-.12)),
            600.,"ground","terrain",collision=True)
        surface = self.surfaces()
        if self.profile.median:
            for index,s in enumerate(np.arange(4.,self.profile.length-3.,4.5)):
                wp = self.map.get_waypoint_xodr(1,1,float(s))
                # Two half-median strips cover the native 2.2m divider. Their
                # visual collision is disabled; native raised median is retained.
                for sign in (1,-1):
                    mid = self.map.get_waypoint_xodr(1,sign,float(s))
                    self.mesh(MESH,carla.Transform(carla.Location(x=mid.transform.location.x,y=mid.transform.location.y,z=.17-.25*.052),
                        carla.Rotation(yaw=mid.transform.rotation.yaw)),.052,"median","median",collision=False)
            names = [a for a in self.records if a.get("kind") == "median"]
            # Keep native median geometry authoritative; named colour groups are
            # optional appearance only, so no object-name guessing is needed.
        index = 0
        for road in self.roads:
            if road["junction"] != -1:
                continue
            for sign in (1,-1):
                lane = sign*(self.profile.lanes_each_way+(1 if self.profile.median else 0))
                spacing = 23 if self.profile.scenery == "market" else 37 if self.profile.scenery != "rural" else 72
                for s in np.arange(24.,road["length"]-30,spacing):
                    wp = self.map.get_waypoint_xodr(road["id"],lane,float(s))
                    if wp is None:
                        raise ValueError("Missing scenery waypoint")
                    outer = wp.lane_width/2+self.profile.sidewalk+self.profile.shoulder
                    building_offset = outer+6.5
                    if self.profile.scenery == "rural":
                        building_offset += 5.
                    location = self.outside(wp,building_offset)
                    asset = APARTMENT if self.profile.scenery in ("market","urban") else HOUSE
                    house = self.mesh(asset,carla.Transform(location,carla.Rotation(yaw=wp.transform.rotation.yaw-90)),
                        .29 if asset == APARTMENT else .44,"facade","building")
                    self.records[-1]["align_to"] = {"road":road["id"],"lane":lane,"s":float(s),
                        "min_setback_from_driving_m":self.profile.sidewalk+self.profile.shoulder+.6}
                    if self.profile.scenery in ("market","junction","urban"):
                        self.board(wp,"shop_"+str(index%6),outer+1.0,shop=True)
                        if index%2 == 0:
                            self.prop("static.prop.kiosk_01",carla.Transform(self.outside(wp,outer+3.),
                                carla.Rotation(yaw=wp.transform.rotation.yaw)),"kiosk","building",optional=True)
                            self.prop("static.prop.foodcart",carla.Transform(self.outside(wp,outer+5.),
                                carla.Rotation(yaw=wp.transform.rotation.yaw)),"food_cart",optional=True)
                    else:
                        self.prop("static.prop.coconutpalm",carla.Transform(self.outside(wp,outer+8.0)),
                            "palm","vegetation",optional=True)
                    index += 1
                wp = self.map.get_waypoint_xodr(road["id"],lane,42.)
                outer = wp.lane_width/2+self.profile.sidewalk+self.profile.shoulder
                speed = 20 if self.profile.scenery in ("market","residential","junction") else 40 if self.profile.scenery == "urban" else 30
                self.board(wp,f"speed_{speed}",outer+.8)
                if sign == 1:
                    stop_wp = self.map.get_waypoint_xodr(road["id"],lane,min(83.,road["length"]-35))
                    self.prop("static.prop.busstop",carla.Transform(self.outside(stop_wp,outer+2.8),
                        carla.Rotation(yaw=stop_wp.transform.rotation.yaw)),"bus_stop","building")
                    self.board(stop_wp,"bus_stop",outer+.9)
        # World tick resolves dynamic native bounds after static spawn.
        self.world.tick(15)
        self.align_and_check()
        for name, objects in self.texture_groups.items():
            self.world.apply_color_texture_to_objects(objects,carla.MaterialParameter.Diffuse,image_texture(ASSETS/"textures"/name))
        actor_by_id = {a.id:a for a in self.owned}
        for row in self.records:
            if "actor_id" not in row:
                continue
            actor = actor_by_id[row["actor_id"]]
            t = actor.get_transform()
            row["native_semantic_tags"] = list(actor.semantic_tags)
            row["project_label_requires_instance_override"] = row["kind"] in ("asphalt","indian_sign","shop_board","ground","median")
            row["transform"] = {"xyz":[t.location.x,t.location.y,t.location.z],
                "pitch_yaw_roll":[t.rotation.pitch,t.rotation.yaw,t.rotation.roll]}
        return {"schema":"indian_world_scenery_v1","profile":self.key,"surface":surface,"objects":self.records,
            "raw_semantics_are_training_ready":False,
            "label_note":"Native raw streams remain native. Custom boards/asphalt require the saved exact instance-to-project-class mapping; full dataset reference audit is a separate step.",
            "authenticity":"Original Hindi/English textures on generic installed meshes; illustrative Indian setting"}

    def align_and_check(self):
        actor_by_id = {a.id:a for a in self.owned}
        for row in self.records:
            if "align_to" not in row:
                continue
            actor = actor_by_id[row["actor_id"]]
            spec = row["align_to"]
            wp = self.map.get_waypoint_xodr(spec["road"],spec["lane"],spec["s"])
            t = actor.get_transform()
            vertices = actor.bounding_box.get_world_vertices(t)
            # Centre the facade on its intended sample and move its nearest face
            # behind the footpath, using native mesh bounds rather than a guessed origin.
            forward = wp.transform.get_forward_vector()
            yaw = math.radians(wp.transform.rotation.yaw)
            outside = (math.sin(yaw),-math.cos(yaw))
            along = [(v.x-wp.transform.location.x)*forward.x+(v.y-wp.transform.location.y)*forward.y for v in vertices]
            lateral = [(v.x-wp.transform.location.x)*outside[0]+(v.y-wp.transform.location.y)*outside[1] for v in vertices]
            shift = wp.lane_width/2+spec["min_setback_from_driving_m"]-min(lateral)
            centre = (min(along)+max(along))/2
            t.location.x += shift*outside[0]-centre*forward.x
            t.location.y += shift*outside[1]-centre*forward.y
            t.location.z -= min(v.z for v in vertices)
            actor.set_transform(t)
        self.world.tick(15)
        failures = []
        for row in self.records:
            if row.get("kind") in ("ground","asphalt","median") or "actor_id" not in row:
                continue
            actor = actor_by_id[row["actor_id"]]
            vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
            row["world_bounds_xyz"] = [[min(getattr(v,a) for v in vertices),max(getattr(v,a) for v in vertices)] for a in ("x","y","z")]
            # Sample the oriented local body, not its larger world-axis rectangle.
            box,t = actor.bounding_box,actor.get_transform()
            samples = 0
            for x in np.linspace(-box.extent.x,box.extent.x,5):
                for y in np.linspace(-box.extent.y,box.extent.y,5):
                    for z in (-box.extent.z,0.,box.extent.z):
                        p = t.transform(carla.Location(x=float(x)+box.location.x,y=float(y)+box.location.y,z=float(z)+box.location.z))
                        if self._on_drive(p.x,p.y):
                            failures.append({"id":actor.id,"kind":row["kind"],"xy":[p.x,p.y]})
                        samples += 1
            row["driving_lane_overlap_samples_checked"] = samples
        if failures:
            raise ValueError("Scenery intrudes into driving geometry: "+str(failures[:5]))
