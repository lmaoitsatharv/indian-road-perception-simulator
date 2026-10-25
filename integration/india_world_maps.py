"""Five distinct synthetic Indian LHT road settings, independent of the V4 map.

Coordinates in specifications are OpenDRIVE coordinates. Read runtime poses from
CARLA waypoints (Unreal reflects the OpenDRIVE y axis). No real city is claimed.
"""
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from india_road import element, add_lane, make_road


@dataclass(frozen=True)
class WorldProfile:
    key: str
    title: str
    length: float
    width: float
    width_change: float
    sidewalk: float
    shoulder: float
    median: float
    lanes_each_way: int
    speed_mps: float
    scenery: str
    geometries: tuple
    arms: tuple = ()


PROFILES = {
    "market_street": WorldProfile("market_street", "Bazaar market street", 260, 2.85, .25,
        1.8, 0, 0, 1, 4.0, "market", ((65, 0.), (100, .0025), (95, -.0025))),
    "residential_colony": WorldProfile("residential_colony", "Residential colony", 125, 3.1, 0,
        2.0, 0, 0, 1, 4.5, "residential", (),
        (("west", math.pi, 125., 3.1), ("east", 0., 150., 3.1), ("side_lane", 1.38, 95., 2.65))),
    "divided_main_road": WorldProfile("divided_main_road", "Divided urban main road", 360, 3.25, 0,
        2.2, 0, 1.1, 2, 7.0, "urban", ((90, 0.), (180, .0012), (90, 0.))),
    "rural_road": WorldProfile("rural_road", "Rural village approach", 390, 2.75, -.15,
        0, 1.35, 0, 1, 5.5, "rural", ((100, .003), (160, -.003), (130, .0018))),
    "irregular_junction": WorldProfile("irregular_junction", "Skewed village junction", 125, 3.2, 0,
        1.5, 0, 0, 1, 4.0, "junction", (),
        (("west", 3.08, 135., 3.2), ("east", -.08, 155., 3.65),
         ("north_east", 1.12, 110., 2.85), ("south", -1.72, 100., 2.7))),
}


def _header(profile):
    root = ET.Element("OpenDRIVE")
    h = element(root, "header", revMajor=1, revMinor=4, name="Indian_"+profile.key+"_LHT_v1",
        version="1.0", date="2026-10-23", north=500, south=-500, east=500, west=-500,
        vendor="Tesla project original synthetic world")
    element(h, "geoReference").text = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    return root


def _outer_lanes(section, profile, width, length, *, two_way=True):
    for side, sign in (("left", 1), ("right", -1)):
        if not two_way and sign == -1:
            continue
        index = 1
        if profile.median:
            lane = add_lane(section, side, sign, "median", profile.median)
            element(lane, "height", sOffset=0, inner=.16, outer=.16)
            index += 1
        for _ in range(profile.lanes_each_way):
            lane = add_lane(section, side, index*sign, "driving", width)
            if profile.width_change:
                lane.find("width").set("b", str(profile.width_change/length))
            if profile.lanes_each_way > 1 and index == 2:
                lane.find("roadMark").set("type", "broken")
                lane.find("roadMark").set("laneChange", "both")
            index += 1
        if profile.shoulder:
            add_lane(section, side, index*sign, "shoulder", profile.shoulder)
        elif profile.sidewalk:
            add_lane(section, side, index*sign, "sidewalk", profile.sidewalk)
    add_lane(section, "center", 0, "none", 0)


def _curved_road(root, profile):
    road, section = make_road(root, 1, profile.key, profile.length, -1, -profile.length/2, 0., 0.)
    plan = road.find("planView")
    plan.clear()
    x, y, heading, distance = -profile.length/2, 0., 0., 0.
    for length, curvature in profile.geometries:
        geometry = element(plan, "geometry", s=distance, x=x, y=y, hdg=heading, length=length)
        if curvature:
            element(geometry, "arc", curvature=curvature)
            x += (math.sin(heading+curvature*length)-math.sin(heading))/curvature
            y -= (math.cos(heading+curvature*length)-math.cos(heading))/curvature
            heading += curvature*length
        else:
            element(geometry, "line")
            x += length*math.cos(heading)
            y += length*math.sin(heading)
        distance += length
    assert abs(distance-profile.length) < 1e-8
    _outer_lanes(section, profile, profile.width, profile.length)


def _junction(root, profile):
    radius = 16. if len(profile.arms) == 4 else 13.
    arms = []
    for rid, (name, outward, length, width) in enumerate(profile.arms, 1):
        x, y = (radius+length)*math.cos(outward), (radius+length)*math.sin(outward)
        heading = outward+math.pi
        road, section = make_road(root, rid, name, length, -1, x, y, heading)
        element(road.find("link"), "successor", elementType="junction", elementId=1000)
        _outer_lanes(section, profile, width, length)
        arms.append((rid, name, outward, length, width))
    junction = element(root, "junction", id=1000, name=profile.key)
    connector = 100
    for ra, na, ha, la, wa in arms:
        for rb, nb, hb, lb, wb in arms:
            if ra == rb:
                continue
            x, y = radius*math.cos(ha), radius*math.sin(ha)
            x1, y1 = radius*math.cos(hb), radius*math.sin(hb)
            heading, end_heading = ha+math.pi, hb
            dx = (x1-x)*math.cos(heading)+(y1-y)*math.sin(heading)
            dy = -(x1-x)*math.sin(heading)+(y1-y)*math.cos(heading)
            delta = math.remainder(end_heading-heading, 2*math.pi)
            tangent = radius*1.8
            curve = dict(aU=0., bU=tangent, cU=3*dx-2*tangent-tangent*math.cos(delta),
                dU=-2*dx+tangent+tangent*math.cos(delta), aV=0., bV=0.,
                cV=3*dy-tangent*math.sin(delta), dV=-2*dy+tangent*math.sin(delta))
            # Integrate the actual polynomial speed; avoid pretending the chord is arc length.
            length = sum(math.hypot(curve["bU"]+2*curve["cU"]*((i+.5)/2000)+3*curve["dU"]*((i+.5)/2000)**2,
                2*curve["cV"]*((i+.5)/2000)+3*curve["dV"]*((i+.5)/2000)**2)/2000 for i in range(2000))
            road, section = make_road(root, connector, na+"_to_"+nb, length, 1000, x, y, heading, curve)
            element(road.find("link"), "predecessor", elementType="road", elementId=ra, contactPoint="end")
            element(road.find("link"), "successor", elementType="road", elementId=rb, contactPoint="end")
            lane = add_lane(section, "left", 1, "driving", wa, predecessor=1, successor=-1)
            lane.find("width").set("b", str((wb-wa)/length))
            add_lane(section, "center", 0, "none", 0)
            connection = element(junction, "connection", id=connector, incomingRoad=ra,
                connectingRoad=connector, contactPoint="start")
            element(connection, "laneLink", **{"from": 1, "to": 1})
            connector += 1


def build_world(key):
    profile = PROFILES[key]
    root = _header(profile)
    if profile.arms:
        _junction(root, profile)
    else:
        _curved_road(root, profile)
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(root, encoding="unicode")+'\n'


def road_specs(xml):
    return [{"id": int(r.get("id")), "length": float(r.get("length")), "name": r.get("name"),
        "junction": int(r.get("junction")), "lanes": [int(l.get("id")) for l in r.findall("lanes/laneSection/*/lane")
        if l.get("type") == "driving"]} for r in ET.fromstring(xml).findall("road")]


def validate_world(key, xml):
    import carla
    native = carla.Map("audit_"+key, xml)
    root = ET.fromstring(xml)
    assert all(r.get("rule") == "LHT" for r in root.findall("road"))
    sampled, widths = 0, []
    for road in road_specs(xml):
        for lane in road["lanes"]:
            for fraction in (.1, .35, .65, .9):
                wp = native.get_waypoint_xodr(road["id"], lane, fraction*road["length"])
                assert wp is not None and wp.lane_type == carla.LaneType.Driving, (key, road, lane)
                widths.append(wp.lane_width)
                sampled += 1
        if road["junction"] == -1:
            for lane in road["lanes"]:
                s = .3*road["length"]
                wp = native.get_waypoint_xodr(road["id"], lane, s)
                ahead = wp.next(.5)
                assert ahead and ahead[0].road_id == wp.road_id
                assert (ahead[0].s > s) == (lane > 0), (key, "LHT direction", lane)
    if PROFILES[key].arms:
        for road in road_specs(xml):
            if road["junction"] != -1:
                continue
            wp = native.get_waypoint_xodr(road["id"], 1, road["length"]-1)
            assert len(wp.next(2)) == len(PROFILES[key].arms)-1, (key, "branch count")
    return {"passed": True, "key": key, "sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "rule": "LHT", "road_count": len(root.findall("road")), "sampled_driving_waypoints": sampled,
        "lane_width_range_m": [min(widths), max(widths)], "junction_approaches": len(PROFILES[key].arms),
        "median_width_m": PROFILES[key].median*2, "sidewalk_width_m": PROFILES[key].sidewalk,
        "shoulder_width_m": PROFILES[key].shoulder, "lanes_each_way": PROFILES[key].lanes_each_way}
