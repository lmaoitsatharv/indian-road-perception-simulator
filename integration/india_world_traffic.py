"""Native physics traffic for the new road profiles, using explicit waypoint routes.

Reuses the established low-level vehicle controller without altering it. No
perception model, trained policy, vehicle teleporting or replacement old driver.
"""
from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET

import carla
import numpy as np

from india_behavior import Driver, Route, body, polygon_gap
from india_world_maps import PROFILES, road_specs

ROSTER = [("ego","vehicle.nissan.micra","car"),
    ("scooter","vehicle.vespa.zx125","motorcycle"),
    ("motorcycle","vehicle.yamaha.yzf","motorcycle"),
    ("local_bus","vehicle.mitsubishi.fusorosa","bus"),
    ("delivery_van","vehicle.mercedes.sprinter","other_vehicle"),
    ("hatchback","vehicle.citroen.c3","car"),
    ("cycle","vehicle.bh.crossbike","bicycle")]


class WorldDriver(Driver):
    """Shorter geometric lookahead for the new narrower junction turns only."""
    world_steer = 0.

    def control(self,state):
        control = super().control(state)
        if self.route.incoming == "road":
            return control
        lookahead = 2.2+.35*state["speed"]
        target = self.route.at(self.s+lookahead)
        alpha = math.atan2(target[1]-state["y"],target[0]-state["x"])-state["yaw"]
        steer = math.atan2(2*self.wheelbase*math.sin(alpha),lookahead)/self.max_steer
        self.world_steer += float(np.clip(steer-self.world_steer,-.07,.07))
        control.steer = float(np.clip(self.world_steer,-.7,.7))
        return control


def path_for(native, parts, incoming="road", outgoing="road"):
    points, junction_end = [], 0
    for rid,lane,start,end in parts:
        for s in np.linspace(start,end,max(2,int(abs(end-start)/.35)+1)):
            wp = native.get_waypoint_xodr(rid,lane,float(s))
            if wp is None:
                raise ValueError(f"Missing native route {rid}/{lane}/{s}")
            t = wp.transform
            p = [t.location.x,t.location.y,math.radians(t.rotation.yaw)]
            if not points or math.hypot(points[-1][0]-p[0],points[-1][1]-p[1]) > .02:
                points.append(p)
        if rid >= 100:
            junction_end = len(points)-1
    points = np.array(points)
    gaps = np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1)
    if gaps.max() > 1.0:
        raise ValueError(f"Disconnected route {incoming}->{outgoing}: {gaps.max()}m")
    distances = np.r_[0., np.cumsum(gaps)]
    return Route(incoming,outgoing,points,distances,float(distances[junction_end]))


def routes_for(key, xml, native):
    profile, specs = PROFILES[key],road_specs(xml)
    if not profile.arms:
        lanes = [i for i in specs[0]["lanes"] if i > 0]
        result = {}
        for lane in lanes:
            result[f"forward_{lane}"] = path_for(native,[(1,lane,3.,profile.length-3.)])
            result[f"reverse_{lane}"] = path_for(native,[(1,-lane,profile.length-3.,3.)])
        return result
    names = {r["name"]:r for r in specs}
    result = {}
    for connector in specs:
        if connector["junction"] == -1:
            continue
        a,b = connector["name"].split("_to_")
        ra,rb = names[a],names[b]
        result[a+"_to_"+b] = path_for(native,
            [(ra["id"],1,3.,ra["length"]-.02),(connector["id"],1,.01,connector["length"]-.01),
             (rb["id"],-1,rb["length"]-.02,3.)],a,b)
    return result


class WorldTraffic:
    def __init__(self, world, client, owned, key, xml, seed):
        self.world,self.client,self.owned,self.key = world,client,owned,key
        self.profile = PROFILES[key]
        self.routes = routes_for(key,xml,world.get_map())
        self.drivers,self.walkers,self.records,self.collisions = [],[],[],[]
        self.sensor_actors = []
        self.rng = random.Random(seed)
        self.holder = None
        keys = list(self.routes)
        outer = self.profile.lanes_each_way+(1 if self.profile.median else 0)
        forward = "west_to_east" if self.profile.arms else f"forward_{outer}"
        reverse = "east_to_west" if self.profile.arms else f"reverse_{outer}"
        for index,(name,identifier,category) in enumerate(ROSTER):
            route_key = forward if index in (0,1,3,6) else reverse
            if self.profile.lanes_each_way == 2 and index == 6:
                route_key = "forward_2"
            if self.profile.arms and index in (4,5):
                side = [k for k in keys if not k.startswith(("west_","east_"))]
                route_key = side[(index-4)%len(side)]
            route = self.routes[route_key]
            # Longitudinal spacing is checked against the actual native body.
            initial = {0:10.,1:40.,2:12.,3:83.,4:52.,5:94.,6:111.}[index]
            if initial > route.distances[-1]-20:
                initial = route.distances[-1]-22
            position = route.at(initial)
            bp = world.get_blueprint_library().find(identifier)
            bp.set_attribute("role_name","indian_world_"+name)
            if bp.has_attribute("color"):
                colors = bp.get_attribute("color").recommended_values
                if colors:
                    bp.set_attribute("color",colors[self.rng.randrange(len(colors))])
            actor = world.try_spawn_actor(bp,carla.Transform(carla.Location(x=position[0],y=position[1],z=.6),
                carla.Rotation(yaw=math.degrees(position[2]))))
            if actor is None:
                raise RuntimeError("Required traffic actor failed to spawn: "+name)
            owned.append(actor)
            speed = min(self.profile.speed_mps,3. if category == "bicycle" else self.profile.speed_mps)
            driver = WorldDriver(actor,route,name,speed,s=initial)
            driver.index = int(np.searchsorted(route.distances,initial))
            driver.goal_s = route.distances[-1]-7.
            self.drivers.append(driver)
            self.records.append({"actor_id":actor.id,"name":name,"blueprint":identifier,"project_class":category,
                "route":route_key,"initial_route_s_m":initial,"native_semantic_tags":list(actor.semantic_tags),
                "collisions_enabled":True,"motion":"Native physics; throttle/brake/steer"})
        for _ in range(35):
            world.tick(15)
        for driver in self.drivers:
            driver.calibrate()
            driver.actor.apply_control(carla.VehicleControl(brake=1.))
            sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),
                carla.Transform(),attach_to=driver.actor)
            owned.append(sensor)
            self.sensor_actors.append(sensor)
            sensor.listen(lambda event,name=driver.name:self.collisions.append({"frame":event.frame,"actor":name,
                "other_id":event.other_actor.id,"other_type":event.other_actor.type_id,
                "impulse":math.sqrt(event.normal_impulse.x**2+event.normal_impulse.y**2+event.normal_impulse.z**2)}))
        self.spawn_walkers()
        self.ego = self.drivers[0]
        self.initial_positions = {d.name:d.actor.get_location() for d in self.drivers}
        self.max_travel = {d.name:0. for d in self.drivers}

    def spawn_walkers(self):
        if not self.profile.sidewalk:
            return
        native = self.world.get_map()
        # Short individual sidewalk walks; no new crossing-behaviour claim.
        library = sorted(self.world.get_blueprint_library().filter("walker.pedestrian.*"),key=lambda bp:bp.id)
        from india_behavior import Pedestrian
        rid = 1
        lane = self.profile.lanes_each_way+(1 if self.profile.median else 0)+1
        length = self.profile.length if not self.profile.arms else self.profile.arms[0][2]
        for i in range(4):
            start = 22.+i*18
            end = min(start+10,length-15)
            wp = native.get_waypoint_xodr(rid,lane,start)
            target = native.get_waypoint_xodr(rid,lane,end)
            if wp is None or target is None:
                raise ValueError("Sidewalk pedestrian has no native waypoint")
            bp = library[(i*7)%len(library)]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible","false")
            transform = wp.transform
            transform.location.z += .5
            actor = self.world.try_spawn_actor(bp,transform)
            if actor is None:
                raise RuntimeError("Sidewalk walker did not spawn")
            self.owned.append(actor)
            self.walkers.append(Pedestrian(actor,"walker_"+str(i),(wp.transform.location.x,wp.transform.location.y),
                (target.transform.location.x,target.transform.location.y),.75+.1*i))
            self.records.append({"actor_id":actor.id,"name":"walker_"+str(i),"blueprint":bp.id,
                "project_class":"person","native_semantic_tags":list(actor.semantic_tags),"collisions_enabled":True})

    def update(self, snapshot, sim_time):
        states = {}
        for driver in self.drivers:
            actor_state = snapshot.find(driver.actor.id)
            if actor_state is None:
                raise RuntimeError("Traffic actor disappeared")
            state = body(driver.actor,actor_state)
            velocity = actor_state.get_velocity()
            state["speed"] = math.hypot(velocity.x,velocity.y)
            driver.index,driver.s = driver.route.project(state["x"],state["y"],driver.index)
            states[driver.actor.id] = state
        if self.profile.arms:
            if self.holder is not None:
                held = next(d for d in self.drivers if d.actor.id == self.holder)
                if held.s > held.route.junction_end+held.half_length+5:
                    self.holder = None
            waiting = [d for d in self.drivers if d.s < d.route.junction_end and
                d.s > self.junction_start(d)-30]
            if self.holder is None and waiting:
                self.holder = min(waiting,key=lambda d:(self.junction_start(d)-d.s,d.actor.id)).actor.id
        controls,trace = [],[]
        for driver in self.drivers:
            state = states[driver.actor.id]
            target,reason = driver.desired,"follow_route"
            remaining = driver.goal_s-driver.s
            target = min(target,math.sqrt(max(0.,2.*1.7*(remaining-2))))
            if remaining < 2:
                target,reason = 0.,"route_finished"
            if self.profile.arms and self.junction_start(driver)-18 < driver.s < driver.route.junction_end+5:
                target = min(target,1.8 if driver.name in ("local_bus","delivery_van") else 2.4)
            if self.profile.arms and driver.actor.id != self.holder and driver.s < self.junction_start(driver):
                distance = self.junction_start(driver)-driver.s-driver.half_length-3
                target = min(target,math.sqrt(max(0.,2*1.5*distance)))
                if target < driver.desired:
                    reason = "junction_wait"
            # Same-lane and crossing-body prediction: simulator ground truth only.
            for other in self.drivers:
                if other is driver:
                    continue
                observed = states[other.actor.id]
                dx,dy = observed["x"]-state["x"],observed["y"]-state["y"]
                longitudinal = dx*math.cos(state["yaw"])+dy*math.sin(state["yaw"])
                lateral = abs(-dx*math.sin(state["yaw"])+dy*math.cos(state["yaw"]))
                if longitudinal <= 0 or lateral > driver.half_width+other.half_width+.4:
                    continue
                gap = longitudinal-driver.half_length-other.half_length
                safe = max(0.,(gap-3.)/1.8)
                if safe < target:
                    target,reason = safe,"vehicle_headway"
                if gap < max(1.3,state["speed"]*.7):
                    target,reason = 0.,"emergency_body_gap"
            driver.target,driver.reason = target,reason
            control = driver.control(state)
            controls.append(carla.command.ApplyVehicleControl(driver.actor.id,control))
            self.max_travel[driver.name] = max(self.max_travel[driver.name],
                driver.actor.get_location().distance(self.initial_positions[driver.name]))
            reference = driver.route.at(driver.s)
            lateral_error = -(state["x"]-reference[0])*math.sin(reference[2])+(state["y"]-reference[1])*math.cos(reference[2])
            trace.append({"id":driver.actor.id,"name":driver.name,"x":state["x"],"y":state["y"],"yaw":state["yaw"],
                "expected_yaw":float(reference[2]),"lateral_error_m":float(lateral_error),
                "speed":state["speed"],"route_s":driver.s,"target":target,"reason":reason,
                "throttle":control.throttle,"brake":control.brake,"steer":control.steer})
        for walker in self.walkers:
            controls.append(carla.command.ApplyWalkerControl(walker.actor.id,walker.control(sim_time)))
        replies = self.client.apply_batch_sync(controls,False)
        if any(r.has_error() for r in replies):
            raise RuntimeError("Native actor control failed")
        return states,trace

    def junction_start(self, driver):
        name = driver.route.incoming
        return next(a[2] for a in self.profile.arms if a[0] == name)-3.
