"""Persistent traffic across every district of one native connected town."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
import random

import carla
import numpy as np

from india_behavior import Driver,Pedestrian,body,wrap
from india_combined_map import BY_ID,NODES,TOUR,road_between,outgoing_lane,incoming_lane


class LoopPath:
    def __init__(self,native,network,nodes,reverse=False,outer=False):
        if nodes[0] != nodes[-1]:
            raise ValueError("Traffic paths must close within the same world")
        self.nodes = tuple(nodes)
        self.network = network
        self.points,self.regions,self.parts = [],[],[]
        self.junctions = []
        specs = {s["id"]:s for s in network["streets"]}
        for i in range(len(nodes)-1):
            a,b = nodes[i:i+2]
            street = road_between(a,b)
            lane = outgoing_lane(street,a,street.lanes[-1] if outer else street.lanes[0])
            length = specs[street.id]["length"]
            s0,s1 = (.02,length-.02) if lane > 0 else (length-.02,.02)
            self._part(native,street.id,lane,s0,s1,street.district,None,street.cruise)
            outgoing = road_between(b,nodes[(i+2)%(len(nodes)-1)])
            to = outgoing_lane(outgoing,b,outgoing.lanes[-1] if outer else outgoing.lanes[0])
            c = next(c for c in network["connectors"] if c["node"] == b and c["incoming"] == street.id and
                c["from_lane"] == lane and c["outgoing"] == outgoing.id and c["to_lane"] == to)
            self._part(native,c["id"],1,.02,c["length"]-.02,street.district,b,2.6)
        self.points = np.asarray(self.points,dtype=float)
        gaps = np.linalg.norm(np.diff(np.vstack([self.points[:,:2],self.points[0,:2]]),axis=0),axis=1)
        if gaps.max() > .65:
            raise ValueError("Connected loop has a discontinuous lane join: "+str(gaps.max()))
        self.distances = np.r_[0.,np.cumsum(gaps[:-1])]
        self.length = float(gaps.sum())
        self.max_gap = float(gaps.max())
        self.junction_end = self.length
        self.incoming = "combined"
        for part in self.parts:
            part["start_m"] = float(self.distances[part["index_start"]])
            part["end_m"] = float(self.distances[part["index_end"]])
            if part["node"]:
                self.junctions.append(part)
        self.starts = [p["start_m"] for p in self.parts]

    def _part(self,native,rid,lane,a,b,district,node,cruise):
        begin = len(self.points)
        for s in np.linspace(a,b,max(3,int(abs(b-a)/.35)+1)):
            wp = native.get_waypoint_xodr(rid,lane,float(s))
            if wp is None:
                raise ValueError(f"Missing connected path waypoint: {rid}/{lane}/{s}")
            t = wp.transform
            self.points.append([t.location.x,t.location.y,math.radians(t.rotation.yaw)])
            self.regions.append((district,node,rid,lane))
        self.parts.append({"road_id":rid,"lane":lane,"district":district,"node":node,"cruise":cruise,
            "index_start":begin,"index_end":len(self.points)-1})

    def at(self,s,offset=0.):
        s %= self.length
        i = min(int(np.searchsorted(self.distances,s)),len(self.points)-1)
        if i == 0:
            p = self.points[0].copy()
        else:
            f = (s-self.distances[i-1])/max(self.distances[i]-self.distances[i-1],1e-9)
            p = self.points[i-1].copy()
            p[:2] += f*(self.points[i,:2]-p[:2])
            p[2] += f*wrap(self.points[i,2]-p[2])
        p[:2] += offset*np.array([-math.sin(p[2]),math.cos(p[2])])
        return p

    def project(self,x,y,last=0):
        indexes = np.arange(last-10,last+130)%len(self.points)
        i = int(indexes[np.argmin(np.sum((self.points[indexes,:2]-[x,y])**2,axis=1))])
        return i,float(self.distances[i])

    def part_at(self,s):
        return self.parts[max(0,bisect_right(self.starts,s%self.length)-1)]

    def next_junction(self,s):
        local = s%self.length
        candidates = []
        for junction in self.junctions:
            start,end = junction["start_m"],junction["end_m"]
            if local <= end+8:
                distance = start-local
            else:
                distance = start+self.length-local
            candidates.append((distance,junction))
        return min(candidates,key=lambda pair:pair[0])


class TownDriver(Driver):
    turn_steer = 0.

    def control(self,state):
        control = super().control(state)
        lookahead = 2.5+.4*state["speed"]
        p = self.route.at(self.s+lookahead)
        alpha = math.atan2(p[1]-state["y"],p[0]-state["x"])-state["yaw"]
        steer = math.atan2(2*self.wheelbase*math.sin(alpha),lookahead)/self.max_steer
        self.turn_steer += float(np.clip(steer-self.turn_steer,-.07,.07))
        control.steer = float(np.clip(self.turn_steer,-.7,.7))
        return control


class UrbanBusPath(LoopPath):
    """The unusually wide installed bus serves the divided corridor only."""
    def __init__(self,native,network):
        self.nodes = ("A","B")
        self.network = network
        self.points,self.regions,self.parts,self.junctions = [],[],[],[]
        length = next(s["length"] for s in network["streets"] if s["id"] == 2)
        self._part(native,2,3,10.,length-18.,"divided_main_road",None,3.5)
        self.points = np.asarray(self.points,dtype=float)
        gaps = np.linalg.norm(np.diff(self.points[:,:2],axis=0),axis=1)
        self.distances = np.r_[0.,np.cumsum(gaps)]
        self.length = float(self.distances[-1])
        self.max_gap = float(gaps.max())
        self.incoming = "combined"
        self.junction_end = self.length
        self.parts[0].update(start_m=0.,end_m=self.length)
        self.starts = [0.]

    def at(self,s,offset=0.):
        return super().at(float(np.clip(s,0.,self.length-1e-5)),offset)

    def project(self,x,y,last=0):
        indexes = np.arange(max(0,last-10),min(len(self.points),last+130))
        i = int(indexes[np.argmin(np.sum((self.points[indexes,:2]-[x,y])**2,axis=1))])
        return i,float(self.distances[i])

    def next_junction(self,s):
        return 100000.,{"node":"B","start_m":100000.,"end_m":100001.}


VEHICLES = [("vehicle.nissan.micra","car"),("vehicle.citroen.c3","car"),
    ("vehicle.vespa.zx125","motorcycle"),("vehicle.yamaha.yzf","motorcycle"),
    ("vehicle.mitsubishi.fusorosa","bus"),("vehicle.mercedes.sprinter","other_vehicle"),
    ("vehicle.bh.crossbike","bicycle")]


class CombinedTraffic:
    def __init__(self,world,client,owned,network,seed):
        self.world,self.client,self.owned,self.network = world,client,owned,network
        self.rng = random.Random(seed)
        native = world.get_map()
        reverse = tuple(reversed(TOUR))
        self.routes = {"tour":LoopPath(native,network,TOUR),"counter_tour":LoopPath(native,network,reverse),
            "colony_loop":LoopPath(native,network,("E","G","C","D","E")),
            "village_loop":LoopPath(native,network,("C","H","D","C")),
            "tour_outer":LoopPath(native,network,TOUR),"urban_bus_service":UrbanBusPath(native,network)}
        self.drivers,self.walkers,self.records,self.sensors,self.collisions = [],[],[],[],[]
        self.reservations,self.waiting_since = {},{}
        self.distance,self.visited,self.transitions = {},{},[]
        placements = []
        # All districts receive moving traffic at startup. Extra loops connect
        # back into the same road graph, rather than isolated scene copies.
        for name,count,phase in (("tour",10,20.),("counter_tour",7,45.),("colony_loop",3,30.),("village_loop",3,50.),("tour_outer",2,110.)):
            route = self.routes[name]
            for i in range(count):
                route = self.routes["urban_bus_service"] if len(self.drivers) == 24 else self.routes[name]
                initial = (phase+i*route.length/count)%route.length
                if len(self.drivers) == 24:
                    initial = 48.
                # Keep initial bodies off a junction and clear of already
                # placed bodies. Changing initial spawn is not live teleporting.
                for _ in range(80):
                    p = route.at(initial)
                    if route.part_at(initial)["node"] is None and all(math.dist(p[:2],q[:2]) > 19 for q in placements):
                        break
                    initial = (initial+13.)%route.length
                else:
                    raise RuntimeError("Could not find a clear native traffic spawn")
                placements.append(p)
                number = len(self.drivers)
                identifier,category = VEHICLES[number%len(VEHICLES)]
                if category == "bus":
                    identifier,category = "vehicle.citroen.c3","car"
                if number == 24:
                    identifier,category = "vehicle.mitsubishi.fusorosa","bus"
                role = "ego" if number == 0 else category+"_"+str(number)
                bp = world.get_blueprint_library().find(identifier)
                bp.set_attribute("role_name","combined_india_"+role)
                if bp.has_attribute("color"):
                    values = bp.get_attribute("color").recommended_values
                    if values:
                        bp.set_attribute("color",self.rng.choice(values))
                actor = world.try_spawn_actor(bp,carla.Transform(carla.Location(x=p[0],y=p[1],z=.65),
                    carla.Rotation(yaw=math.degrees(p[2]))))
                if actor is None:
                    raise RuntimeError("Required combined-town vehicle failed to spawn: "+role)
                owned.append(actor)
                speed = 3.2 if category == "bicycle" else 7.8
                driver = TownDriver(actor,route,role,speed,s=initial)
                driver.index = int(np.searchsorted(route.distances,initial))
                driver.goal_s = float("inf")
                driver.last_local_s = initial
                if number == 24:
                    driver.service_stop_s = min(route.length-40.,initial+150.)
                self.drivers.append(driver)
                self.distance[role] = 0.
                self.visited[role] = set()
                self.records.append({"actor_id":actor.id,"name":role,"blueprint":identifier,"project_class":category,
                    "route":"urban_bus_service" if number == 24 else name,"initial_route_s_m":initial,
                    "service_stop_route_s_m":getattr(driver,"service_stop_s",None),
                    "collisions_enabled":True,"live_pose_teleport":False})
        for _ in range(35):
            world.tick(20)
        for driver in self.drivers:
            driver.calibrate()
            driver.actor.apply_control(carla.VehicleControl(brake=1.))
            row = next(r for r in self.records if r["actor_id"] == driver.actor.id)
            row["bbox_extent_xy_m"] = [driver.half_length,driver.half_width]
            sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=driver.actor)
            owned.append(sensor)
            self.sensors.append(sensor)
            sensor.listen(lambda event,name=driver.name:self.collisions.append({"frame":event.frame,"actor":name,
                "other_id":event.other_actor.id,"other_type":event.other_actor.type_id,
                "impulse":math.sqrt(event.normal_impulse.x**2+event.normal_impulse.y**2+event.normal_impulse.z**2)}))
        self.spawn_walkers()
        self.ego = self.drivers[0]

    def spawn_walkers(self):
        native = self.world.get_map()
        library = sorted(self.world.get_blueprint_library().filter("walker.pedestrian.*"),key=lambda bp:bp.id)
        for spec in self.network["streets"]:
            street = BY_ID[spec["id"]]
            if not street.sidewalk:
                continue
            count = 6 if street.district == "market_street" else 3
            for i in range(count):
                side = 1 if i%2 == 0 else -1
                lane = side*(street.lanes[-1]+1)
                s = 45.+i*(spec["length"]-105)/max(1,count-1)
                a = native.get_waypoint_xodr(street.id,lane,s)
                b = native.get_waypoint_xodr(street.id,lane,s+10.)
                bp = library[(len(self.walkers)*7)%len(library)]
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible","false")
                t = a.transform
                t.location.z += .5
                actor = self.world.try_spawn_actor(bp,t)
                if actor is None:
                    raise RuntimeError("Required sidewalk walker failed to spawn")
                self.owned.append(actor)
                name = "walker_"+str(len(self.walkers))
                self.walkers.append(Pedestrian(actor,name,(a.transform.location.x,a.transform.location.y),
                    (b.transform.location.x,b.transform.location.y),.8+.1*(i%3)))
                self.records.append({"actor_id":actor.id,"name":name,"blueprint":bp.id,"project_class":"person",
                    "district":street.district,"collisions_enabled":True})

    def update(self,snapshot,sim_time):
        states,near = {},{}
        for driver in self.drivers:
            observed = snapshot.find(driver.actor.id)
            if observed is None:
                raise RuntimeError("A persistent traffic actor disappeared")
            state = body(driver.actor,observed)
            v = observed.get_velocity()
            state["speed"] = math.hypot(v.x,v.y)
            driver.index,local = driver.route.project(state["x"],state["y"],driver.index)
            delta = (local-driver.last_local_s+driver.route.length/2)%driver.route.length-driver.route.length/2
            driver.s += delta
            driver.last_local_s = local
            self.distance[driver.name] += max(0.,delta)
            part = driver.route.part_at(driver.s)
            if part["district"] not in self.visited[driver.name]:
                self.visited[driver.name].add(part["district"])
                self.transitions.append({"sim_s":sim_time,"actor":driver.name,"district":part["district"],
                    "distance_m":self.distance[driver.name],"frame":snapshot.frame})
            states[driver.actor.id] = state
            distance,junction = driver.route.next_junction(driver.s)
            near[driver.actor.id] = (distance,junction)
            if distance < 35 and distance > -(junction["end_m"]-junction["start_m"])-7:
                self.waiting_since.setdefault((junction["node"],driver.actor.id),sim_time)
        # One native reservation per junction. Hold until the complete vehicle
        # clears the connector. FIFO queue avoids starving side streets.
        for node in NODES:
            holder = self.reservations.get(node)
            if holder is not None:
                distance,junction = near[holder]
                driver = next(d for d in self.drivers if d.actor.id == holder)
                if junction["node"] != node or distance < -(junction["end_m"]-junction["start_m"])-driver.half_length-4:
                    self.reservations.pop(node,None)
                    self.waiting_since.pop((node,holder),None)
            if node not in self.reservations:
                waiting = [d for d in self.drivers if near[d.actor.id][1]["node"] == node and -5 < near[d.actor.id][0] < 35]
                if waiting:
                    chosen = min(waiting,key=lambda d:(self.waiting_since.get((node,d.actor.id),sim_time),near[d.actor.id][0],d.actor.id))
                    self.reservations[node] = chosen.actor.id
        commands,trace = [],[]
        for driver in self.drivers:
            state = states[driver.actor.id]
            part = driver.route.part_at(driver.s)
            target,reason = min(driver.desired,part["cruise"]),"district_speed"
            if hasattr(driver,"service_stop_s"):
                remaining = driver.service_stop_s-driver.s
                target = min(target,math.sqrt(max(0.,2*1.5*(remaining-2.))))
                if remaining < 2.2:
                    target,reason = 0.,"urban_bus_stop"
            distance,junction = near[driver.actor.id]
            if distance < 18 and distance > -(junction["end_m"]-junction["start_m"])-6:
                target = min(target,1.8 if driver.half_length > 2.5 else 2.7)
            if distance > -2 and distance < 40 and self.reservations.get(junction["node"]) != driver.actor.id:
                stop_distance = distance-driver.half_length-5
                target = min(target,math.sqrt(max(0.,2*1.5*stop_distance)))
                reason = "junction_wait"
            for other in self.drivers:
                if other is driver:
                    continue
                observed = states[other.actor.id]
                # Longitudinal headway applies to co-directional traffic. At a
                # curved junction exit, projecting an opposite-lane waiting body
                # into the holder's current tangent creates a false leader and
                # locks the whole junction. Exclusive native junction tokens
                # handle crossing streams; legal LHT oncoming lanes stay separate.
                if math.cos(observed["yaw"]-state["yaw"]) < .35:
                    continue
                other_part = other.route.part_at(other.s)
                if part["road_id"] == other_part["road_id"] and part["lane"] != other_part["lane"]:
                    continue
                dx,dy = observed["x"]-state["x"],observed["y"]-state["y"]
                longitudinal = dx*math.cos(state["yaw"])+dy*math.sin(state["yaw"])
                lateral = abs(-dx*math.sin(state["yaw"])+dy*math.cos(state["yaw"]))
                if longitudinal <= 0 or lateral > driver.half_width+other.half_width+.18:
                    continue
                # Opposing lanes have physically separate centre lines; same
                # direction/merging bodies use a larger gap at these junctions.
                gap = longitudinal-driver.half_length-other.half_length
                safe = max(0.,(gap-4.)/1.9)
                if safe < target:
                    target,reason = safe,"vehicle_headway"
                if gap < max(1.4,state["speed"]*.8):
                    target,reason = 0.,"emergency_gap"
            driver.target,driver.reason = target,reason
            control = driver.control(state)
            commands.append(carla.command.ApplyVehicleControl(driver.actor.id,control))
            reference = driver.route.at(driver.s)
            error = -(state["x"]-reference[0])*math.sin(reference[2])+(state["y"]-reference[1])*math.cos(reference[2])
            trace.append({"id":driver.actor.id,"name":driver.name,"x":state["x"],"y":state["y"],"yaw":state["yaw"],
                "ex":state["ex"],"ey":state["ey"],"speed":state["speed"],"route_s":driver.s,
                "distance_m":self.distance[driver.name],"district":part["district"],"road_id":part["road_id"],
                "lane":part["lane"],"junction":part["node"],"expected_yaw":float(reference[2]),"lateral_error_m":float(error),
                "target":target,"reason":reason,"throttle":control.throttle,"brake":control.brake,"steer":control.steer})
        for walker in self.walkers:
            commands.append(carla.command.ApplyWalkerControl(walker.actor.id,walker.control(sim_time)))
        replies = self.client.apply_batch_sync(commands,False)
        if any(reply.has_error() for reply in replies):
            raise RuntimeError("Native combined-town actor control failed")
        return states,trace
