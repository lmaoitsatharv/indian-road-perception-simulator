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
        p = self.route.at(self.s+lookahead,self.offset)
        alpha = math.atan2(p[1]-state["y"],p[0]-state["x"])-state["yaw"]
        steer = math.atan2(2*self.wheelbase*math.sin(alpha),lookahead)/self.max_steer
        self.turn_steer += float(np.clip(steer-self.turn_steer,-.07,.07))
        control.steer = float(np.clip(self.turn_steer,-.7,.7))
        return control


TWO_WHEELERS = ("motorcycle","bicycle")
# Up to this many vehicles may hold one junction at once, on movements whose
# paths through it never come within CONFLICT_M of each other.
MAX_HOLDERS = 3
CONFLICT_M = 3.6
JAYWALKERS = {"market_street":3,"divided_main_road":2,"residential_colony":1,"irregular_junction":1,"rural_road":1}


def walker_segments(street,length):
    """Sidewalk walking segments: (side, start s); each walker paces 10 m."""
    if not street.sidewalk:
        return []
    count = 6 if street.district == "market_street" else 3
    return [(1 if i%2 == 0 else -1,45.+i*(length-105)/max(1,count-1)) for i in range(count)]


def jaywalk_spots(street,length):
    """Mid-block crossing places, clear of sidewalk walkers and junction mouths."""
    busy = [s+5. for _,s in walker_segments(street,length)]+[length*.53]
    spots = []
    for k in range(JAYWALKERS[street.district]):
        s = length*(k+1)/(JAYWALKERS[street.district]+1)
        for _ in range(30):
            if all(abs(s-b) > 12. for b in busy+spots) and 40. < s < length-40.:
                break
            s += 7.
        else:
            continue
        spots.append(s)
    return spots


class JayWalker:
    """Crosses the street mid-block, Indian style, when the gap looks acceptable.
    Vehicles yield to a crossing jaywalker (CombinedTraffic.update)."""
    def __init__(self,actor,name,a,b,speed,rng):
        self.actor,self.name,self.ends,self.speed,self.rng = actor,name,(a,b),speed,rng
        self.side,self.crossing,self.crossings = 0,False,0
        self.next_try = rng.uniform(5.,25.)
        self.position = a

    def gap_ok(self,states):
        (ax,ay),(bx,by) = self.ends
        mx,my = (ax+bx)/2,(ay+by)/2
        for v in states:
            d = math.hypot(v["x"]-mx,v["y"]-my)
            if d < 12.:
                return False
            toward = ((mx-v["x"])*math.cos(v["yaw"])+(my-v["y"])*math.sin(v["yaw"]))/d
            if toward > .5 and d < 12.+v["speed"]*2.8:
                return False
        return True

    def control(self,sim_time,states):
        x,y = self.position
        if self.crossing:
            tx,ty = self.ends[1-self.side]
            dx,dy = tx-x,ty-y
            norm = math.hypot(dx,dy)
            if norm < .3:
                self.crossing,self.side = False,1-self.side
                self.next_try = sim_time+self.rng.uniform(8.,28.)
                self.crossings += 1
                return carla.WalkerControl(speed=0)
            return carla.WalkerControl(direction=carla.Vector3D(dx/norm,dy/norm,0),speed=self.speed)
        if sim_time >= self.next_try:
            if self.gap_ok(states):
                self.crossing = True
            else:
                self.next_try = sim_time+.5
        # Drift back onto the waiting spot if a step overshot it.
        tx,ty = self.ends[self.side]
        if math.hypot(tx-x,ty-y) > .6:
            norm = math.hypot(tx-x,ty-y)
            return carla.WalkerControl(direction=carla.Vector3D((tx-x)/norm,(ty-y)/norm,0),speed=.6)
        return carla.WalkerControl(speed=0)


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


# Indian town traffic is mostly two-wheelers: six in every ten vehicles here.
VEHICLES = [("vehicle.nissan.micra","car"),("vehicle.vespa.zx125","motorcycle"),("vehicle.yamaha.yzf","motorcycle"),
    ("vehicle.citroen.c3","car"),("vehicle.vespa.zx125","motorcycle"),("vehicle.bh.crossbike","bicycle"),
    ("vehicle.mercedes.sprinter","other_vehicle"),("vehicle.yamaha.yzf","motorcycle"),("vehicle.nissan.micra","car"),
    ("vehicle.gazelle.omafiets","bicycle")]
FOOTPATH_WALKERS = {"market_street":8,"divided_main_road":4,"residential_colony":3,"irregular_junction":3,"rural_road":2}


class FootpathWalker:
    """Walks a long route along the footpath (or a village yard edge), turning
    back at its ends and now and then pausing, as people stop at a shop."""
    def __init__(self,actor,name,path,speed,start,rng):
        self.actor,self.name,self.path,self.speed,self.rng = actor,name,path,speed,rng
        self.i,self.step,self.pause_until = start,1,0.
        self.position = path[start]

    def control(self,sim_time,states):
        if sim_time < self.pause_until:
            return carla.WalkerControl(speed=0)
        x,y = self.position
        tx,ty = self.path[self.i]
        if math.hypot(tx-x,ty-y) < .6:
            if not 0 <= self.i+self.step < len(self.path):
                self.step = -self.step
            self.i += self.step
            if self.rng.random() < .04:
                self.pause_until = sim_time+self.rng.uniform(2.,7.)
            tx,ty = self.path[self.i]
        dx,dy = tx-x,ty-y
        norm = max(math.hypot(dx,dy),1e-6)
        return carla.WalkerControl(direction=carla.Vector3D(dx/norm,dy/norm,0),speed=self.speed)


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
        self.stats = {"shared_junction_grants":0,"max_simultaneous_holders":0,"two_wheeler_filtering_s":0.,
            "car_passing_two_wheeler_s":0.,"lane_change_overtaking_s":0.,"pedestrian_yield_vehicle_s":0.,"jaywalker_crossings":0}
        self.hold_check,self.token_events,self.token_grants = {},[],0
        self.distance,self.visited,self.transitions = {},{},[]
        placements = []
        # All districts receive moving traffic at startup. Extra loops connect
        # back into the same road graph, rather than isolated scene copies.
        for name,count,phase in (("tour",12,20.),("counter_tour",9,45.),("colony_loop",4,30.),("village_loop",4,50.),("tour_outer",2,110.)):
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
                speed = 3.2 if category == "bicycle" else 8.4 if category == "motorcycle" else 7.8
                driver = TownDriver(actor,route,role,speed,s=initial)
                driver.kind,driver.filter_until,driver.squeeze,driver.mode = category,-1.,0.,None
                driver.overtake_until,driver.overtake_target = -1.,0.
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
                "other_xyz":[round(event.other_actor.get_location().x,2),round(event.other_actor.get_location().y,2),round(event.other_actor.get_location().z,2)],
                "impulse":math.sqrt(event.normal_impulse.x**2+event.normal_impulse.y**2+event.normal_impulse.z**2)}))
        self.spawn_walkers()
        self.spawn_jaywalkers()
        self.by_id = {d.actor.id:d for d in self.drivers}
        self.conflicts = self._junction_conflicts(native)
        self.ego = self.drivers[0]
        self.native = native
        self.ego_resets = 0
        self.lost_walkers = []

    def reset_ego(self):
        """Viewer key R: put your car back on its lane centre, stopped, moving
        ahead along its route to a spot clear of junctions, vehicles and
        people. A live pose teleport, recorded as one. Returns metres moved
        ahead, or None when no clear spot is found."""
        ego,route = self.ego,self.ego.route
        cars = [d.actor.get_location() for d in self.drivers[1:]]
        people = [w.actor.get_location() for w in self.walkers]
        for ahead in np.arange(0.,300.,3.):
            s = ego.s+ahead
            if any(route.part_at(s+d)["node"] is not None for d in (-ego.half_length-3.,0.,ego.half_length+6.)):
                continue
            p = route.at(s)
            if all(math.hypot(q.x-p[0],q.y-p[1]) > 10. for q in cars) and all(math.hypot(q.x-p[0],q.y-p[1]) > 3. for q in people):
                break
        else:
            return None
        wp = self.native.get_waypoint(carla.Location(x=float(p[0]),y=float(p[1]),z=0.))
        z = (wp.transform.location.z if wp else 0.)+.35
        ego.actor.set_target_velocity(carla.Vector3D())
        ego.actor.set_target_angular_velocity(carla.Vector3D())
        ego.actor.set_transform(carla.Transform(carla.Location(x=float(p[0]),y=float(p[1]),z=z),carla.Rotation(yaw=math.degrees(p[2]))))
        ego.actor.apply_control(carla.VehicleControl(brake=1.))
        local = s%route.length
        ego.s,ego.last_local_s = s,local
        ego.index = min(int(np.searchsorted(route.distances,local)),len(route.points)-1)
        for name,value in (("steer",0.),("speed_integral",0.),("smoothed_accel",0.),("previous_speed",0.),("offset",0.),
                ("mode",None),("overtake_until",-1.),("filter_until",-1.),("squeeze",0.),("stop_until",None),
                ("stuck_started",None),("wait_started",None),("pass_state","follow"),("pass_leader",None)):
            setattr(ego,name,value)
        for held in self.reservations.values():
            held.pop(ego.actor.id,None)
        for table in (self.waiting_since,self.hold_check):
            for key in [k for k in table if k[1] == ego.actor.id]:
                table.pop(key)
        self.records[0]["live_pose_teleport"] = True
        self.ego_resets += 1
        return float(ahead)

    def _junction_conflicts(self,native):
        """Which movements through the same junction may run at the same time."""
        paths = {}
        for c in self.network["connectors"]:
            points = []
            for s in np.linspace(.05,c["length"]-.05,max(4,int(c["length"])+1)):
                wp = native.get_waypoint_xodr(c["id"],1,float(s))
                if wp is not None:
                    points.append((wp.transform.location.x,wp.transform.location.y))
            if len(points) >= 4:
                paths[c["id"]] = (c["node"],np.asarray(points))
        conflicts = {}
        for a,(node_a,pa) in paths.items():
            for b,(node_b,pb) in paths.items():
                if a < b and node_a == node_b:
                    gap = float(np.min(np.linalg.norm(pa[:,None,:]-pb[None,:,:],axis=2)))
                    conflicts[(a,b)] = conflicts[(b,a)] = gap < CONFLICT_M
        self.conflict_summary = {"pairs":len(conflicts)//2,"compatible_pairs":sum(not v for v in conflicts.values())//2,
            "conflict_distance_m":CONFLICT_M,"max_holders":MAX_HOLDERS}
        return conflicts

    def spawn_jaywalkers(self):
        native = self.world.get_map()
        library = sorted(self.world.get_blueprint_library().filter("walker.pedestrian.*"),key=lambda bp:bp.id)
        self.jaywalkers = []
        for spec in self.network["streets"]:
            street = BY_ID[spec["id"]]
            for s in jaywalk_spots(street,spec["length"]):
                ends = []
                for side in (1,-1):
                    if street.sidewalk:
                        wp = native.get_waypoint_xodr(street.id,side*(street.lanes[-1]+1),s)
                        ends.append(None if wp is None else (wp.transform.location.x,wp.transform.location.y))
                    else:
                        # No footpath: wait in the yard beyond the paved shoulder,
                        # clear of two-wheelers riding on it.
                        wp = native.get_waypoint_xodr(street.id,side*street.lanes[-1],s)
                        if wp is None:
                            ends.append(None)
                            continue
                        yaw = math.radians(wp.transform.rotation.yaw)
                        out = wp.lane_width/2+street.shoulder+.9
                        ends.append((wp.transform.location.x+math.sin(yaw)*out,wp.transform.location.y-math.cos(yaw)*out))
                if None in ends:
                    continue
                bp = library[(len(self.walkers)*11+3)%len(library)]
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible","false")
                actor = self.world.try_spawn_actor(bp,carla.Transform(carla.Location(x=ends[0][0],y=ends[0][1],z=.7)))
                if actor is None:
                    continue
                self.owned.append(actor)
                name = "jaywalker_"+str(len(self.jaywalkers))
                walker = JayWalker(actor,name,ends[0],ends[1],1.15+.1*(len(self.jaywalkers)%3),random.Random(len(self.jaywalkers)*31+7))
                self.jaywalkers.append(walker)
                self.walkers.append(walker)
                self.records.append({"actor_id":actor.id,"name":name,"blueprint":bp.id,"project_class":"person",
                    "district":street.district,"behaviour":"mid_block_jaywalker","crossing_s_m":s,"collisions_enabled":True})

    def _slow_leader(self,driver,state,states,reach):
        """A slower or queued co-directional body directly ahead."""
        c,s = math.cos(state["yaw"]),math.sin(state["yaw"])
        for other in self.drivers:
            if other is driver:
                continue
            o = states[other.actor.id]
            if math.cos(o["yaw"]-state["yaw"]) < .35:
                continue
            dx,dy = o["x"]-state["x"],o["y"]-state["y"]
            lon,lat = dx*c+dy*s,abs(-dx*s+dy*c)
            gap = lon-driver.half_length-other.half_length
            if lon > 0 and gap < reach and lat < driver.half_width+other.half_width+.4 and \
                    (o["speed"] < driver.desired*.7 or other.target < 1.) and o["speed"] < state["speed"]+.6:
                return other
        return None

    def _side_clear(self,driver,state,states,step):
        """Moving sideways by step (left positive) keeps room to bodies alongside."""
        c,s = math.cos(state["yaw"]),math.sin(state["yaw"])
        for other in self.drivers:
            if other is driver:
                continue
            o = states[other.actor.id]
            dx,dy = o["x"]-state["x"],o["y"]-state["y"]
            lon,lat = dx*c+dy*s,-dx*s+dy*c
            if abs(lon) > driver.half_length+other.half_length+2.:
                continue
            after = lat-step
            if abs(after) < driver.half_width+other.half_width+.2 and abs(after) < abs(lat):
                return False
        return True

    def _plan_offsets(self,states,near,sim_time):
        """Lateral position inside the lane. Two-wheelers keep left and filter past
        slow traffic on a paved shoulder or the lane line of the divided road;
        cars squeeze past a two-wheeler keeping left when oncoming is clear.
        Everyone is back on the lane centre near a junction."""
        merge_blocked = set()
        for driver in self.drivers:
            if driver.kind == "bus":
                continue
            state = states[driver.actor.id]
            part = driver.route.part_at(driver.s)
            distance,_ = near[driver.actor.id]
            street = BY_ID.get(part["road_id"]) if part["node"] is None else None
            desired,mode = 0.,None
            if street is not None and distance > 28.:
                edge = street.width/2
                overtake = None
                if street.divided:
                    # Dual carriageway: change into the other same-direction
                    # lane around a stopped or crawling vehicle (the bus at its
                    # stop), never to jump a queue waiting for the junction.
                    other_lane = street.width*(-1 if abs(part["lane"]) == 2 else 1)
                    leader = self._slow_leader(driver,state,states,15.)
                    if leader is not None and states[leader.actor.id]["speed"] < 2.5 and distance > 60. and                             leader.reason not in ("junction_wait","vehicle_headway","emergency_gap","merging"):
                        # Pass on whichever side moves away from it: into the
                        # other lane if it sits in ours, back to our lane centre
                        # if it overhangs from the other lane (the wide bus).
                        position = self._lane_position(driver,state,states[leader.actor.id])
                        target = other_lane if abs(position) < street.width/2 else 0.
                        if self._lane_clear(driver,state,states,target):
                            driver.overtake_until,driver.overtake_target = sim_time+3.,target
                    if sim_time < driver.overtake_until:
                        overtake = driver.overtake_target
                if overtake is not None:
                    desired,mode = overtake,"overtaking"
                elif driver.kind in TWO_WHEELERS:
                    # Route offsets are positive toward the centre line (CARLA's
                    # left-handed frame), so the LHT kerb side is negative.
                    kerb = -(edge-driver.half_width-.05)
                    shoulder = -(edge+street.shoulder/2) if street.shoulder else None
                    if street.divided and abs(part["lane"]) == 2:
                        desired = 0.  # inner lane: ride its centre, the left lane is beside it
                    elif driver.kind == "bicycle":
                        desired = shoulder if shoulder is not None else kerb
                    else:
                        desired = .45*kerb
                    option = shoulder if shoulder is not None else (edge-.08)*(-1 if abs(part["lane"]) == 2 else 1) if street.divided else None
                    if option is not None and abs(option-desired) > .3:
                        leader = self._slow_leader(driver,state,states,18.) if distance > 45. else None
                        if leader is not None:
                            # Filter only past something actually in the way,
                            # and only on the side that leads away from it.
                            position = self._lane_position(driver,state,states[leader.actor.id])
                            if abs(option-position) > abs(driver.offset-position):
                                driver.filter_until = sim_time+2.5
                        if sim_time < driver.filter_until:
                            desired,mode = option,"filtering"
                else:
                    leader = self._slow_two_wheeler(driver,state,states,edge,street,distance)
                    if leader is False:
                        driver.filter_until = min(driver.filter_until,sim_time)
                    elif leader is not None:
                        driver.squeeze,driver.filter_until = leader,sim_time+2.
                    if sim_time < driver.filter_until:
                        desired,mode = driver.squeeze,"passing"
            # Hard bounds: cars never move kerb-wards; two-wheelers stay within
            # their lane, the lane line or the paved shoulder.
            if street is not None:
                if street.divided:
                    low,high = -street.width,street.width
                else:
                    low,high = 0. if driver.kind not in TWO_WHEELERS else -(street.width/2+street.shoulder),street.width/2
                desired = float(np.clip(desired,low,high))
            step = float(np.clip(desired-driver.offset,-.05,.05))
            if abs(step) > 1e-4:
                if self._side_clear(driver,state,states,step):
                    driver.offset += step
                elif distance <= 28.:
                    merge_blocked.add(driver.actor.id)
            driver.mode = mode if mode and abs(driver.offset) > .3 else None
            if driver.mode == "filtering":
                self.stats["two_wheeler_filtering_s"] += .05
            elif driver.mode == "passing":
                self.stats["car_passing_two_wheeler_s"] += .05
            elif driver.mode == "overtaking":
                self.stats["lane_change_overtaking_s"] += .05
        return merge_blocked

    def _lane_position(self,driver,state,other):
        """Where another body sits across this vehicle's lane (0 = lane centre)."""
        dx,dy = other["x"]-state["x"],other["y"]-state["y"]
        return -dx*math.sin(state["yaw"])+dy*math.cos(state["yaw"])+driver.offset

    def _lane_clear(self,driver,state,states,target):
        """The lane this vehicle would move into is free alongside and close behind."""
        c,s = math.cos(state["yaw"]),math.sin(state["yaw"])
        shift = target-driver.offset
        for other in self.drivers:
            if other is driver:
                continue
            o = states[other.actor.id]
            if math.cos(o["yaw"]-state["yaw"]) < .35:
                continue
            dx,dy = o["x"]-state["x"],o["y"]-state["y"]
            lon,lat = dx*c+dy*s,-dx*s+dy*c
            if -20. < lon < 14. and abs(lat-shift) < driver.half_width+other.half_width+.4:
                return False
        return True

    def _slow_two_wheeler(self,driver,state,states,edge,street,distance):
        """Offset that lets this car pass a slower two-wheeler keeping left; None
        when there is none, False when oncoming traffic forbids passing now."""
        c,s = math.cos(state["yaw"]),math.sin(state["yaw"])
        target = None
        for other in self.drivers:
            if other is driver:
                continue
            o = states[other.actor.id]
            dx,dy = o["x"]-state["x"],o["y"]-state["y"]
            lon,lat = dx*c+dy*s,-dx*s+dy*c
            if math.cos(o["yaw"]-state["yaw"]) < -.5:
                # Oncoming traffic: never start or keep a squeeze toward it.
                if not street.divided and 0 < lon < 55. and abs(lat) < 4.5:
                    return False
                continue
            if other.kind not in TWO_WHEELERS or math.cos(o["yaw"]-state["yaw"]) < .35:
                continue
            reach = lon-driver.half_length-other.half_length
            if -(driver.half_length+other.half_length+1.5) < lon and reach < 20. and o["speed"] < driver.desired-1.:
                position = lat+driver.offset
                if position > -.5:
                    continue
                squeeze = position+driver.half_width+other.half_width+.3
                if squeeze <= .05:
                    continue  # already clear of it (for example it rides the shoulder)
                overshoot = squeeze+driver.half_width-edge
                if overshoot > (-.05 if street.divided else .15):
                    continue
                if reach > 0 and distance < 60.:
                    continue
                target = squeeze if target is None else max(target,squeeze)
        return target

    def junction_stats(self):
        return {**self.stats,"jaywalker_crossings":sum(w.crossings for w in self.jaywalkers),
            "jaywalkers":len(self.jaywalkers),"conflict_map":self.conflict_summary}

    def spawn_walkers(self):
        native = self.world.get_map()
        library = sorted(self.world.get_blueprint_library().filter("walker.pedestrian.*"),key=lambda bp:bp.id)
        rng = random.Random(4242)
        for spec in self.network["streets"]:
            street = BY_ID[spec["id"]]
            length = spec["length"]
            count = FOOTPATH_WALKERS[street.district]
            for k in range(count):
                side = 1 if k%2 == 0 else -1
                start = 28.+(k+rng.random()*.6)*(length-56.)/count
                end = min(length-25.,start+rng.uniform(45.,85.))
                jitter = rng.uniform(-.3,.3)
                path = []
                for s in np.arange(start,end,4.):
                    if street.sidewalk:
                        wp = native.get_waypoint_xodr(street.id,side*(street.lanes[-1]+1),float(s))
                        out = jitter
                    else:
                        # Village roads: along the yard edge beyond the shoulder.
                        wp = native.get_waypoint_xodr(street.id,side*street.lanes[-1],float(s))
                        out = None if wp is None else wp.lane_width/2+street.shoulder+.55
                    if wp is None:
                        continue
                    yaw = math.radians(wp.transform.rotation.yaw)
                    path.append((wp.transform.location.x+math.sin(yaw)*out,wp.transform.location.y-math.cos(yaw)*out))
                if len(path) < 4:
                    continue
                first = rng.randrange(len(path)-1)
                bp = library[(len(self.walkers)*7)%len(library)]
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible","false")
                actor = self.world.try_spawn_actor(bp,carla.Transform(carla.Location(x=path[first][0],y=path[first][1],z=.75)))
                if actor is None:
                    continue
                self.owned.append(actor)
                name = "walker_"+str(len(self.walkers))
                self.walkers.append(FootpathWalker(actor,name,path,rng.uniform(1.,1.45),first,random.Random(k*13+street.id)))
                self.records.append({"actor_id":actor.id,"name":name,"blueprint":bp.id,"project_class":"person",
                    "district":street.district,"behaviour":"footpath_walker","route_m":round(end-start,1),
                    "collisions_enabled":True})

    def _front_of_approach(self,driver,distance,states,ignore=()):
        """No co-directional body sits between this vehicle and the junction entry
        (a holder on the very same movement may: this vehicle follows it through)."""
        state = states[driver.actor.id]
        room = distance-driver.half_length+.5
        for other in self.drivers:
            if other is driver or other.actor.id in ignore:
                continue
            observed = states[other.actor.id]
            if math.cos(observed["yaw"]-state["yaw"]) < .35:
                continue
            dx,dy = observed["x"]-state["x"],observed["y"]-state["y"]
            longitudinal = dx*math.cos(state["yaw"])+dy*math.sin(state["yaw"])
            lateral = abs(-dx*math.sin(state["yaw"])+dy*math.cos(state["yaw"]))
            gap = longitudinal-driver.half_length-other.half_length
            if longitudinal > 0 and lateral <= driver.half_width+other.half_width+.18 and gap < room:
                return False
        return True

    def _exit_clear(self,driver,approach,states):
        """The lane just past the junction on this vehicle's route has room for it."""
        distance,junction = approach
        start = driver.s+distance+(junction["end_m"]-junction["start_m"])
        need = 2*driver.half_length+4
        samples = [driver.route.at(start+k) for k in np.arange(0.,need+.01,1.5)]
        for other in self.drivers:
            if other is driver:
                continue
            observed = states[other.actor.id]
            if observed["speed"] > 2.:
                continue
            for p in samples:
                if math.cos(observed["yaw"]-p[2]) < .35:
                    continue
                dx,dy = p[0]-observed["x"],p[1]-observed["y"]
                longitudinal = abs(dx*math.cos(observed["yaw"])+dy*math.sin(observed["yaw"]))
                lateral = abs(-dx*math.sin(observed["yaw"])+dy*math.cos(observed["yaw"]))
                if longitudinal <= observed["ex"]+.5 and lateral <= observed["ey"]+driver.half_width:
                    return False
        return True

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
            # Queue entry only while approaching. Re-adding a vehicle that has
            # just released the token stamped it with an old time for its next
            # lap, so it jumped the queue over vehicles already waiting.
            if -5 < distance < 35:
                self.waiting_since.setdefault((junction["node"],driver.actor.id),sim_time)
        crossing,self.jaywalker_trace = [],[]
        for walker in self.walkers:
            # Spawned after the last setup tick, so absent from the very first
            # snapshot; ask the actor itself then.
            observed = snapshot.find(walker.actor.id) or walker.actor
            location,v = observed.get_transform().location,observed.get_velocity()
            walker.position = (location.x,location.y)
            walker.last_xyz = [round(location.x,2),round(location.y,2),round(location.z,2)]
            if isinstance(walker,JayWalker):
                self.jaywalker_trace.append([walker.name,round(location.x,2),round(location.y,2),round(location.z,2),walker.crossing,walker.side])
            if getattr(walker,"crossing",False):
                crossing.append((location.x,location.y,v.x,v.y))
        merge_blocked = self._plan_offsets(states,near,sim_time)
        # Junction reservations. Hold until the complete vehicle clears the
        # connector. Up to MAX_HOLDERS vehicles share a junction when their
        # movements never come close (opposing straights, kerb-side turns), or
        # follow one another on the very same movement. FIFO with head-of-line
        # blocking keeps side streets from starving. Only the front vehicle of
        # an approach with room past the junction may take a token; a token
        # given to a queued follower can never be used and locked the whole
        # town (29 Oct 900 s run).
        for node in NODES:
            held = self.reservations.setdefault(node,{})
            for holder in list(held):
                distance,junction = near[holder]
                driver = self.by_id[holder]
                if junction["node"] != node or distance < -(junction["end_m"]-junction["start_m"])-driver.half_length-4:
                    held.pop(holder)
                    self.waiting_since.pop((node,holder),None)
                    self.hold_check.pop((node,holder),None)
                else:
                    # Watchdog: a holder still before the junction that has not
                    # moved for 10 s gives the token back and rejoins the queue.
                    watched,s0,t0 = self.hold_check.get((node,holder),(None,0.,0.))
                    if watched is None or abs(driver.s-s0) > .5:
                        self.hold_check[(node,holder)] = (holder,driver.s,sim_time)
                    elif sim_time-t0 > 10 and distance > 0:
                        held.pop(holder)
                        self.hold_check.pop((node,holder),None)
                        self.waiting_since[(node,holder)] = sim_time
                        self.token_events.append({"sim_s":sim_time,"node":node,"actor":driver.name,"event":"watchdog_release"})
            waiting = sorted((d for d in self.drivers if d.actor.id not in held and near[d.actor.id][1]["node"] == node
                and -5 < near[d.actor.id][0] < 35),
                key=lambda d:(self.waiting_since.get((node,d.actor.id),sim_time),near[d.actor.id][0],d.actor.id))
            # Head-of-line blocking: once the oldest front vehicle cannot go,
            # only vehicles following a holder on its very movement may still
            # join (a platoon), and only while that head has waited < 12 s.
            blocked,head_wait = False,0.
            for chosen in waiting:
                if len(held) >= MAX_HOLDERS:
                    break
                distance,junction = near[chosen.actor.id]
                road = junction["road_id"]
                if chosen.kind in TWO_WHEELERS and abs(chosen.offset) > .3:
                    continue
                same = road in held.values()
                if blocked and (not same or head_wait > 12.):
                    continue
                if not self._front_of_approach(chosen,distance,states,{a for a,r in held.items() if r == road}):
                    continue
                if not same and any(self.conflicts.get((road,r),True) for r in held.values()):
                    blocked,head_wait = True,sim_time-self.waiting_since.get((node,chosen.actor.id),sim_time)
                    continue
                if not self._exit_clear(chosen,(distance,junction),states):
                    continue
                held[chosen.actor.id] = road
                self.hold_check[(node,chosen.actor.id)] = (chosen.actor.id,chosen.s,sim_time)
                self.token_grants += 1
                if len(held) > 1:
                    self.stats["shared_junction_grants"] += 1
                self.stats["max_simultaneous_holders"] = max(self.stats["max_simultaneous_holders"],len(held))
        holders = {holder:node for node,held in self.reservations.items() for holder in held}
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
            if distance > -2 and distance < 40 and driver.actor.id not in self.reservations.get(junction["node"],{}):
                stop_distance = distance-driver.half_length-5
                target = min(target,math.sqrt(max(0.,2*1.5*stop_distance)))
                reason = "junction_wait"
            c,s = math.cos(state["yaw"]),math.sin(state["yaw"])
            for other in self.drivers:
                if other is driver:
                    continue
                observed = states[other.actor.id]
                heading = math.cos(observed["yaw"]-state["yaw"])
                if heading < -.5:
                    if part["node"] is not None or (abs(driver.offset) < .3 and abs(other.offset) < .3):
                        continue
                    # Head-on guard, only while someone squeezes or filters.
                    # Centred LHT traffic never needs it; a vehicle still
                    # turning out of a junction once mistook a waiting
                    # oncoming van for an obstacle and locked junction B.
                    dx,dy = observed["x"]-state["x"],observed["y"]-state["y"]
                    lon,lat = dx*c+dy*s,abs(-dx*s+dy*c)
                    gap = lon-driver.half_length-other.half_length
                    if lon > 0 and gap < 25. and lat < driver.half_width+other.half_width+.35:
                        safe = max(0.,(gap-4.)/1.9)
                        if safe < target:
                            target,reason = safe,"oncoming_vehicle"
                        if gap < 3.:
                            target,reason = 0.,"emergency_gap"
                    continue
                # Longitudinal headway applies to co-directional traffic. At a
                # curved junction exit, projecting an opposite-lane waiting body
                # into the holder's current tangent creates a false leader and
                # locks the whole junction. Junction tokens handle crossing
                # streams; legal LHT oncoming lanes stay separate.
                if heading < .35:
                    continue
                other_part = other.route.part_at(other.s)
                if part["road_id"] == other_part["road_id"] and part["lane"] != other_part["lane"] \
                        and abs(driver.offset) < .3 and abs(other.offset) < .3:
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
            for wx,wy,wvx,wvy in crossing:
                dx,dy = wx-state["x"],wy-state["y"]
                lon,lat = dx*c+dy*s,-dx*s+dy*c
                gap = lon-driver.half_length
                if lon <= 0 or gap > 32.:
                    continue
                moving_in = lat*(-wvx*s+wvy*c) < 0 and abs(lat) < 10.
                if abs(lat) < driver.half_width+1.3 or moving_in:
                    safe = math.sqrt(max(0.,2*2.4*(gap-3.)))
                    if safe < target:
                        target,reason = safe,"pedestrian_crossing"
                    if gap < 2.:
                        target,reason = 0.,"emergency_pedestrian"
            if driver.actor.id in merge_blocked and target > 0:
                # Let the body alongside go first, then merge in behind it.
                target,reason = min(target,1.),"merging"
            if reason == "district_speed" and driver.mode:
                reason = driver.mode
            if reason in ("pedestrian_crossing","emergency_pedestrian"):
                self.stats["pedestrian_yield_vehicle_s"] += .05
            driver.target,driver.reason = target,reason
            control = driver.control(state)
            commands.append(carla.command.ApplyVehicleControl(driver.actor.id,control))
            reference = driver.route.at(driver.s)
            error = -(state["x"]-reference[0])*math.sin(reference[2])+(state["y"]-reference[1])*math.cos(reference[2])
            trace.append({"id":driver.actor.id,"name":driver.name,"x":state["x"],"y":state["y"],"yaw":state["yaw"],
                "ex":state["ex"],"ey":state["ey"],"speed":state["speed"],"route_s":driver.s,
                "distance_m":self.distance[driver.name],"district":part["district"],"road_id":part["road_id"],
                "lane":part["lane"],"junction":part["node"],"expected_yaw":float(reference[2]),"lateral_error_m":float(error),
                "target":target,"reason":reason,"throttle":control.throttle,"brake":control.brake,"steer":control.steer,
                "holds_junction":holders.get(driver.actor.id),"lane_offset_m":driver.offset,
                "two_wheeler":driver.kind in TWO_WHEELERS,"kind":driver.kind})
        vehicles = list(states.values())
        for walker in self.walkers:
            commands.append(carla.command.ApplyWalkerControl(walker.actor.id,walker.control(sim_time,vehicles)))
        replies = self.client.apply_batch_sync(commands,False)
        if any(reply.has_error() for reply in replies):
            failed = [(cmd.actor_id,reply.error) for cmd,reply in zip(commands,replies) if reply.has_error()]
            walkers = {w.actor.id:w for w in self.walkers}
            # CARLA occasionally drops a pedestrian from its registry (seen once
            # with a bazaar jaywalker at 180 s). Record it and carry on rather
            # than closing the viewer; any vehicle failure still stops the run.
            if all(aid in walkers and "could not be found in the registry" in error for aid,error in failed):
                for aid,error in failed:
                    lost = walkers[aid]
                    self.walkers.remove(lost)
                    if lost.actor in self.owned:
                        self.owned.remove(lost.actor)
                    self.lost_walkers.append({"sim_s":round(sim_time,2),"name":lost.name,"actor_id":aid,"error":error,
                        "last_xyz":getattr(lost,"last_xyz",None),"crossing":getattr(lost,"crossing",None)})
            else:
                raise RuntimeError("Native combined-town actor control failed: "+str(failed[:5]))
        return states,trace
