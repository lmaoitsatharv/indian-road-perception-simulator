"""One connected Indian LHT town with five districts and two extra loops.

All roads, lane links and junctions belong to a single OpenDRIVE world. Geometry
is original/illustrative, in OpenDRIVE coordinates; CARLA reflects the y axis.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
import xml.etree.ElementTree as ET

import numpy as np

from india_road import element,add_lane,make_road

RADIUS = 23.
NODES = {"A":(-220.,-160.),"B":(180.,-160.),"C":(290.,90.),"D":(10.,300.),
         "E":(-300.,150.),"G":(-70.,45.),"H":(480.,230.)}
NODE_NAMES = {"A":"Bazaar gate","B":"Village gate","C":"Irregular junction",
              "D":"Colony junction","E":"Market junction","G":"Residential square","H":"Outskirts bend"}


@dataclass(frozen=True)
class Street:
    id: int
    a: str
    b: str
    district: str
    width: float
    sidewalk: float
    shoulder: float = 0.
    divided: bool = False
    bend: float = 0.
    cruise: float = 5.

    @property
    def lanes(self):
        return (2,3) if self.divided else (1,)

    @property
    def length_chord(self):
        return math.dist(NODES[self.a],NODES[self.b])-2*RADIUS

    @property
    def heading(self):
        return math.atan2(NODES[self.b][1]-NODES[self.a][1],NODES[self.b][0]-NODES[self.a][0])


STREETS = [Street(1,"E","A","market_street",2.95,1.8,bend=7.,cruise=4.4),
    Street(2,"A","B","divided_main_road",3.25,2.2,divided=True,cruise=8.),
    Street(3,"B","C","rural_road",2.8,0.,shoulder=1.35,bend=12.,cruise=6.8),
    Street(4,"C","D","irregular_junction",3.2,1.6,cruise=5.2),
    Street(5,"D","E","residential_colony",3.05,2.,bend=-8.,cruise=5.),
    Street(6,"E","G","residential_colony",2.85,1.8,cruise=4.8),
    Street(7,"G","C","residential_colony",2.95,1.8,bend=6.,cruise=5.),
    Street(8,"C","H","rural_road",2.75,0.,shoulder=1.3,bend=-7.,cruise=6.),
    Street(9,"H","D","rural_road",2.8,0.,shoulder=1.3,bend=10.,cruise=6.5)]
BY_ID = {s.id:s for s in STREETS}
TOUR = ("E","A","B","C","D","E")


def native_junction(node):
    return 1000+list(NODES).index(node)


def adjacent(node):
    return [s for s in STREETS if node in (s.a,s.b)]


def road_between(a,b):
    return next(s for s in STREETS if {s.a,s.b} == {a,b})


def incoming_lane(street,node,lane=None):
    return (1 if street.b == node else -1)*(lane or street.lanes[0])


def outgoing_lane(street,node,lane=None):
    return -incoming_lane(street,node,lane)


def lane_offset(street,lane):
    if street.divided:
        value = 1.1+street.width*(abs(lane)-2+.5)
    else:
        value = street.width/2
    return math.copysign(value,lane)


def endpoint(street,node,lane):
    heading = street.heading
    direction = 1 if node == street.a else -1
    x = NODES[node][0]+direction*RADIUS*math.cos(heading)
    y = NODES[node][1]+direction*RADIUS*math.sin(heading)
    offset = lane_offset(street,lane)
    return x-offset*math.sin(heading),y+offset*math.cos(heading)


def cubic_length(curve):
    t = (np.arange(2400)+.5)/2400
    du = curve["bU"]+2*curve["cU"]*t+3*curve["dU"]*t*t
    dv = curve["bV"]+2*curve["cV"]*t+3*curve["dV"]*t*t
    return float(np.hypot(du,dv).mean())


def add_street(root,street):
    h = street.heading
    x,y = NODES[street.a]
    x,y = x+RADIUS*math.cos(h),y+RADIUS*math.sin(h)
    chord = street.length_chord
    pieces = []
    if street.bend:
        for index,offset in enumerate((street.bend,-street.bend)):
            curve = dict(aU=0.,bU=chord/2,cU=0.,dU=0.,aV=0.,bV=0.,cV=3*offset,dV=-2*offset)
            length = cubic_length(curve)
            px = x+(chord/2*math.cos(h)-street.bend*math.sin(h) if index else 0)
            py = y+(chord/2*math.sin(h)+street.bend*math.cos(h) if index else 0)
            pieces.append((px,py,length,curve))
    else:
        pieces.append((x,y,chord,None))
    length = sum(p[2] for p in pieces)
    road,section = make_road(root,street.id,street.district+"_"+street.a+street.b,length,-1,x,y,h)
    link = road.find("link")
    element(link,"predecessor",elementType="junction",elementId=native_junction(street.a))
    element(link,"successor",elementType="junction",elementId=native_junction(street.b))
    plan = road.find("planView")
    plan.clear()
    distance = 0.
    for px,py,l,curve in pieces:
        geo = element(plan,"geometry",s=distance,x=px,y=py,hdg=h,length=l)
        element(geo,"paramPoly3",**curve,pRange="normalized") if curve else element(geo,"line")
        distance += l
    road.find("type/speed").set("max",str(round(street.cruise*3.6)))
    for side,sign in (("left",1),("right",-1)):
        if street.divided:
            median = add_lane(section,side,sign,"median",1.1)
            element(median,"height",sOffset=0,inner=.16,outer=.16)
        for index in street.lanes:
            lane = add_lane(section,side,index*sign,"driving",street.width)
            if street.divided and index == 2:
                lane.find("roadMark").set("type","broken")
                lane.find("roadMark").set("laneChange","both")
        if street.sidewalk:
            add_lane(section,side,sign*(street.lanes[-1]+1),"sidewalk",street.sidewalk)
        if street.shoulder:
            add_lane(section,side,sign*(street.lanes[-1]+1),"shoulder",street.shoulder)
    add_lane(section,"center",0,"none",0)
    return length


def build_combined():
    root = ET.Element("OpenDRIVE")
    header = element(root,"header",revMajor=1,revMinor=4,name="Connected_Indian_Town_LHT_v1",version="1.0",
        date="2026-10-23",north=550,south=-400,east=650,west=-500,vendor="Tesla project synthetic connected town")
    element(header,"geoReference").text = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    lengths = {s.id:add_street(root,s) for s in STREETS}
    connectors = []
    rid = 100
    for node in NODES:
        junction = ET.Element("junction",{"id":str(native_junction(node)),"name":NODE_NAMES[node]})
        for incoming in adjacent(node):
            for outgoing in adjacent(node):
                if incoming.id == outgoing.id:
                    continue
                for from_index in incoming.lanes:
                    for to_index in outgoing.lanes:
                        a = incoming_lane(incoming,node,from_index)
                        b = outgoing_lane(outgoing,node,to_index)
                        x,y = endpoint(incoming,node,a)
                        x1,y1 = endpoint(outgoing,node,b)
                        heading = incoming.heading+(math.pi if a < 0 else 0)
                        end_heading = outgoing.heading+(math.pi if b < 0 else 0)
                        dx = (x1-x)*math.cos(heading)+(y1-y)*math.sin(heading)
                        dy = -(x1-x)*math.sin(heading)+(y1-y)*math.cos(heading)
                        delta = math.remainder(end_heading-heading,2*math.pi)
                        tangent = min(2.1*RADIUS,max(18.,math.hypot(dx,dy)*.86))
                        curve = dict(aU=0.,bU=tangent,cU=3*dx-2*tangent-tangent*math.cos(delta),
                            dU=-2*dx+tangent+tangent*math.cos(delta),aV=0.,bV=0.,
                            cV=3*dy-tangent*math.sin(delta),dV=-2*dy+tangent*math.sin(delta))
                        length = cubic_length(curve)
                        road,section = make_road(root,rid,f"{node}_{incoming.id}_{a}_to_{outgoing.id}_{b}",
                            length,native_junction(node),x,y,heading,curve)
                        element(road.find("link"),"predecessor",elementType="road",elementId=incoming.id,
                            contactPoint="end" if incoming.b == node else "start")
                        element(road.find("link"),"successor",elementType="road",elementId=outgoing.id,
                            contactPoint="start" if outgoing.a == node else "end")
                        # Reference curve is the lane centre. Cancel the usual
                        # positive-lane half-width so unequal approach widths and
                        # median offsets meet exactly at their actual lane centres.
                        lanes = road.find("lanes")
                        offset = ET.Element("laneOffset",{"s":"0","a":str(-incoming.width/2),
                            "b":str(-(outgoing.width-incoming.width)/(2*length)),"c":"0","d":"0"})
                        lanes.insert(0,offset)
                        lane = add_lane(section,"left",1,"driving",incoming.width,predecessor=a,successor=b)
                        lane.find("width").set("b",str((outgoing.width-incoming.width)/length))
                        add_lane(section,"center",0,"none",0)
                        connection = element(junction,"connection",id=rid,incomingRoad=incoming.id,
                            connectingRoad=rid,contactPoint="start")
                        element(connection,"laneLink",**{"from":a,"to":1})
                        connectors.append({"id":rid,"node":node,"incoming":incoming.id,"from_lane":a,
                            "outgoing":outgoing.id,"to_lane":b,"length":length})
                        rid += 1
        root.append(junction)
    ET.indent(root,space="  ")
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(root,encoding="unicode")+'\n'
    return xml,{"schema":"connected_indian_town_v1","rule":"LHT","world_load_count_expected":1,
        "nodes":NODES,"node_names":NODE_NAMES,"streets":[dict(vars(s),length=lengths[s.id]) for s in STREETS],
        "connectors":connectors,"tour_nodes":TOUR,"road_length_m":sum(lengths.values()),
        "sha256":hashlib.sha256(xml.encode()).hexdigest()}


def validate_combined(xml,metadata):
    import carla
    native = carla.Map("connected_indian_town_audit",xml)
    assert all(r.get("rule") == "LHT" for r in ET.fromstring(xml).findall("road"))
    graph = {n:{s.b if s.a == n else s.a for s in adjacent(n)} for n in NODES}
    for start in NODES:
        reached,queue = {start},deque([start])
        while queue:
            for n in graph[queue.popleft()]-reached:
                reached.add(n)
                queue.append(n)
        assert reached == set(NODES),"Disconnected region"
    join_gaps = []
    for c in metadata["connectors"]:
        incoming,outgoing = BY_ID[c["incoming"]],BY_ID[c["outgoing"]]
        for rid,lane,s,expected_s in ((incoming.id,c["from_lane"],
            metadata["streets"][incoming.id-1]["length"]-.01 if incoming.b == c["node"] else .01,.01),
            (outgoing.id,c["to_lane"],.01 if outgoing.a == c["node"] else metadata["streets"][outgoing.id-1]["length"]-.01,c["length"]-.01)):
            a = native.get_waypoint_xodr(rid,lane,s)
            b = native.get_waypoint_xodr(c["id"],1,expected_s)
            assert a and b,(c,"missing waypoint")
            gap = a.transform.location.distance(b.transform.location)
            assert gap < .25,(c,"disconnected lane centres",gap)
            heading_error = abs(math.remainder(a.transform.rotation.yaw-b.transform.rotation.yaw,360))
            assert heading_error < 3,(c,"heading mismatch",heading_error)
            join_gaps.append(gap)
    samples = 0
    for spec in metadata["streets"]:
        street = BY_ID[spec["id"]]
        for sign in (1,-1):
            for index in street.lanes:
                lane = sign*index
                for fraction in (.15,.5,.85):
                    s = fraction*spec["length"]
                    wp = native.get_waypoint_xodr(street.id,lane,s)
                    assert wp and wp.lane_type == carla.LaneType.Driving
                    nxt = wp.next(.5)
                    assert nxt and (nxt[0].s > s) == (lane > 0)
                    samples += 1
                wp = native.get_waypoint_xodr(street.id,lane,spec["length"]-.2 if sign > 0 else .2)
                assert wp.next(1.),(street.id,lane,"unconnected lane")
    return {"passed":True,"single_connected_world":True,"districts":sorted({s.district for s in STREETS}),
        "nodes":len(NODES),"streets":len(STREETS),"directed_connectors":len(metadata["connectors"]),
        "street_length_m":metadata["road_length_m"],"native_join_checks":len(join_gaps),
        "max_native_join_gap_m":max(join_gaps),"native_lht_samples":samples,"map_sha256":metadata["sha256"]}
