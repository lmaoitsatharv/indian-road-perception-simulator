"""All five scenery districts alive together in the connected native town."""
import math
from types import SimpleNamespace

import carla
import numpy as np

from india_surface import MESH,asphalt_texture
from india_world_scenery import WorldScenery,HOUSE,APARTMENT,ASSETS,image_texture
from india_combined_map import BY_ID


class CombinedScenery(WorldScenery):
    def __init__(self,world,owned,xml,network):
        super().__init__(world,owned,"market_street",xml)
        self.network = network
        self.current_street = None

    def _record(self,actor,kind,project_class,collision,**detail):
        row = super()._record(actor,kind,project_class,collision,**detail)
        if self.current_street is not None:
            row.update(street_id=self.current_street.id,district=self.current_street.district)
        return row

    def build(self,progress=lambda stage:None):
        progress("placing_ground")
        before = set(self.world.get_names_of_all_objects())
        self.mesh(MESH,carla.Transform(carla.Location(x=60,y=-60,z=-.08-.25*75)),75.,"ground","terrain",collision=False)
        names = sorted(set(self.world.get_names_of_all_objects())-before)
        self.world.apply_color_texture_to_objects(names,carla.MaterialParameter.Diffuse,asphalt_texture((65,60,41),20261024))
        self.mesh("/Engine/BasicShapes/Plane.Plane",carla.Transform(carla.Location(x=60,y=-60,z=-.12)),
            1800.,"ground","terrain",collision=True)
        progress("placing_connected_road_surfaces")
        surface = self.surfaces()
        index = 0
        for spec in self.network["streets"]:
            street = BY_ID[spec["id"]]
            self.current_street = street
            self.profile = SimpleNamespace(sidewalk=street.sidewalk,shoulder=street.shoulder)
            progress("placing_"+street.district+"_street_"+str(street.id))
            spacing = {"market_street":25.,"residential_colony":55.,"divided_main_road":70.,
                "rural_road":110.,"irregular_junction":65.}[street.district]
            for sign in (1,-1):
                lane = sign*street.lanes[-1]
                for s in np.arange(38.,spec["length"]-40.,spacing):
                    wp = self.map.get_waypoint_xodr(street.id,lane,float(s))
                    outer = wp.lane_width/2+street.sidewalk+street.shoulder
                    house = self.mesh(APARTMENT if street.district in ("market_street","divided_main_road") else HOUSE,
                        carla.Transform(self.outside(wp,outer+7.),carla.Rotation(yaw=wp.transform.rotation.yaw-90)),
                        .29 if street.district in ("market_street","divided_main_road") else .44,"facade","building")
                    self.records[-1]["align_to"] = {"road":street.id,"lane":lane,"s":float(s),
                        "min_setback_from_driving_m":street.sidewalk+street.shoulder+.85}
                    if street.district == "market_street":
                        self.board(wp,"shop_"+str(index%6),outer+1.1,shop=True)
                        if index%2 == 0:
                            self.prop("static.prop.kiosk_01",carla.Transform(self.outside(wp,outer+3.4),
                                carla.Rotation(yaw=wp.transform.rotation.yaw)),"kiosk","building")
                            cart_wp = self.map.get_waypoint_xodr(street.id,lane,float(s)+7.)
                            self.prop("static.prop.foodcart",carla.Transform(self.outside(cart_wp,outer+3.)),"food_cart")
                    elif street.district in ("residential_colony","rural_road"):
                        tree_wp = self.map.get_waypoint_xodr(street.id,lane,min(spec["length"]-35.,float(s)+15.))
                        self.prop("static.prop.coconutpalm",carla.Transform(self.outside(tree_wp,outer+8.)),"palm","vegetation")
                    elif index%2 == 0:
                        self.board(wp,"shop_"+str(index%6),outer+1.1,shop=True)
                    index += 1
                if spec["length"] > 115:
                    wp = self.map.get_waypoint_xodr(street.id,lane,55. if sign > 0 else spec["length"]-55.)
                    outer = wp.lane_width/2+street.sidewalk+street.shoulder
                    self.board(wp,"speed_"+str(40 if street.divided else 30 if street.shoulder else 20),outer+1.1)
                    if sign > 0:
                        stop = self.map.get_waypoint_xodr(street.id,lane,spec["length"]*.53)
                        self.prop("static.prop.busstop",carla.Transform(self.outside(stop,outer+3.3),
                            carla.Rotation(yaw=stop.transform.rotation.yaw)),"bus_stop","building")
                        self.board(stop,"bus_stop",outer+1.1)
                        if street.district in ("residential_colony","rural_road"):
                            label = "school" if street.sidewalk else "village"
                            wp = self.map.get_waypoint_xodr(street.id,lane,spec["length"]*.72)
                            self.board(wp,label,outer+1.1)
            if street.divided:
                before = set(self.world.get_names_of_all_objects())
                for i,s in enumerate(np.arange(5.,spec["length"]-4.,4.8)):
                    for side in (1,-1):
                        wp = self.map.get_waypoint_xodr(street.id,side,float(s))
                        self.mesh(MESH,carla.Transform(carla.Location(x=wp.transform.location.x,y=wp.transform.location.y,z=.17-.25*.052),
                            carla.Rotation(yaw=wp.transform.rotation.yaw)),.052,"median","median",collision=False)
                objects = sorted(set(self.world.get_names_of_all_objects())-before)
                self.world.apply_color_texture_to_objects(objects,carla.MaterialParameter.Diffuse,asphalt_texture((65,55,25),17))
        self.current_street = None
        progress("checking_all_native_scenery_bounds")
        self.world.tick(20)
        self.align_and_check()
        progress("applying_indian_sign_artwork")
        for name,objects in self.texture_groups.items():
            self.world.apply_color_texture_to_objects(objects,carla.MaterialParameter.Diffuse,image_texture(ASSETS/"textures"/name))
        actor_by_id = {a.id:a for a in self.owned}
        for row in self.records:
            if "actor_id" not in row:
                continue
            actor = actor_by_id[row["actor_id"]]
            t = actor.get_transform()
            row["native_semantic_tags"] = list(actor.semantic_tags)
            row["project_label_requires_instance_override"] = row["kind"] in ("asphalt","ground","median","shop_board","indian_sign")
            row["transform"] = {"xyz":[t.location.x,t.location.y,t.location.z],
                "pitch_yaw_roll":[t.rotation.pitch,t.rotation.yaw,t.rotation.roll]}
        return {"schema":"combined_india_scenery_v1","single_world":True,"objects":self.records,"surface":surface,
            "raw_semantics_are_training_ready":False,"label_policy":"Native evidence unchanged; exact owned instance corrections required",
            "districts_present":sorted({row.get("district") for row in self.records if row.get("district")}),
            "authenticity":"Original Hindi/English artwork; generic installed vehicle/building meshes; no live autorickshaw"}
