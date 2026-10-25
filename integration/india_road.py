"""A small, explicit left-hand OpenDRIVE junction, with no imported dataset data."""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

LANE_WIDTH = 4.5
SIDEWALK = 2.2
RADIUS = 12.0
ARM_LENGTH = 140.0
ARMS = {
    "west": (1, -RADIUS - ARM_LENGTH, 0.0, 0.0),
    "east": (2, RADIUS + ARM_LENGTH, 0.0, math.pi),
    "south": (3, 0.0, -RADIUS - ARM_LENGTH, math.pi / 2),
    "north": (4, 0.0, RADIUS + ARM_LENGTH, -math.pi / 2),
}


def element(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attributes.items()})


def add_lane(section, side, lane_id, kind, width, predecessor=None, successor=None):
    group = section.find(side)
    if group is None:
        group = element(section, side)
    lane = element(group, "lane", id=lane_id, type=kind, level="false")
    links = element(lane, "link")
    if predecessor is not None:
        element(links, "predecessor", id=predecessor)
    if successor is not None:
        element(links, "successor", id=successor)
    if lane_id:
        element(lane, "width", sOffset=0, a=width, b=0, c=0, d=0)
    if kind == "driving" or lane_id == 0:
        element(lane, "roadMark", sOffset=0, type="broken" if lane_id == 0 else "solid",
                weight="standard", color="white", width=0.12, laneChange="both" if lane_id == 0 else "none")
    if kind == "sidewalk":
        element(lane, "height", sOffset=0, inner=0.15, outer=0.15)
    return lane


def make_road(root, road_id, name, length, junction, x, y, heading, curve=None):
    road = element(root, "road", name=name, length=f"{length:.8f}", id=road_id, junction=junction, rule="LHT")
    element(road, "link")
    road_type = element(road, "type", s=0, type="town")
    element(road_type, "speed", max=30, unit="km/h")
    plan = element(road, "planView")
    geom = element(plan, "geometry", s=0, x=f"{x:.8f}", y=f"{y:.8f}", hdg=f"{heading:.10f}", length=f"{length:.8f}")
    if curve is None:
        element(geom, "line")
    else:
        element(geom, "paramPoly3", **curve, pRange="normalized")
    elevation = element(road, "elevationProfile")
    element(elevation, "elevation", s=0, a=0, b=0, c=0, d=0)
    lanes = element(road, "lanes")
    section = element(lanes, "laneSection", s=0)
    return road, section


LAYOUT_ARMS = {"crossroad": tuple(ARMS), "t_north": ("west", "east", "north"),
               "t_south": ("west", "east", "south"), "straight": ("west", "east")}


def junction_xml(layout="crossroad", *, junction_radius=RADIUS) -> str:
    if not 10.0 <= junction_radius <= 14.0:
        raise ValueError("Confirmation junction radius must stay within the audited 10..14 m extent")
    active = {key: value for key, value in ARMS.items() if key in LAYOUT_ARMS[layout]}
    if junction_radius != RADIUS:
        active = {key: (rid, x - (junction_radius-RADIUS)*math.cos(h),
                        y - (junction_radius-RADIUS)*math.sin(h), h)
                  for key,(rid,x,y,h) in active.items()}
    root = ET.Element("OpenDRIVE")
    header = element(root, "header", revMajor=1, revMinor=4, name="Indian_neighbourhood_LHT_v1", version="1.0",
            date="2026-10-19", north=160, south=-160, east=160, west=-160, vendor="Tesla project synthetic scenario")
    element(header, "geoReference").text = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    for name, (road_id, x, y, heading) in active.items():
        road, section = make_road(root, road_id, name, ARM_LENGTH, -1, x, y, heading)
        element(road.find("link"), "successor", elementType="junction", elementId=1000)
        add_lane(section, "left", 1, "driving", LANE_WIDTH)
        add_lane(section, "left", 2, "sidewalk", SIDEWALK)
        add_lane(section, "center", 0, "none", 0)
        add_lane(section, "right", -1, "driving", LANE_WIDTH)
        if name == "east":
            add_lane(section, "right", -2, "parking", 2.4)
            add_lane(section, "right", -3, "sidewalk", SIDEWALK)
        else:
            add_lane(section, "right", -2, "sidewalk", SIDEWALK)
    junction = ET.Element("junction", {"name": "neighbourhood_unsignalised", "id": "1000"})
    connector = 100
    for incoming, (road_a, x_a, y_a, h_a) in active.items():
        for outgoing, (road_b, x_b, y_b, h_b) in active.items():
            if incoming == outgoing:
                continue
            x0, y0 = x_a + ARM_LENGTH * math.cos(h_a), y_a + ARM_LENGTH * math.sin(h_a)
            x1, y1 = x_b + ARM_LENGTH * math.cos(h_b), y_b + ARM_LENGTH * math.sin(h_b)
            end_h = h_b + math.pi
            dx = (x1 - x0) * math.cos(h_a) + (y1 - y0) * math.sin(h_a)
            dy = -(x1 - x0) * math.sin(h_a) + (y1 - y0) * math.cos(h_a)
            delta = math.atan2(math.sin(end_h - h_a), math.cos(end_h - h_a))
            tangent = 2 * junction_radius if abs(delta) < 1e-6 else junction_radius * 1.65685424949
            curve = {"aU": 0, "bU": tangent, "cU": 3 * dx - 2 * tangent - tangent * math.cos(delta),
                     "dU": -2 * dx + tangent + tangent * math.cos(delta),
                     "aV": 0, "bV": 0, "cV": 3 * dy - tangent * math.sin(delta),
                     "dV": -2 * dy + tangent * math.sin(delta)}
            p = np.linspace(0, 1, 1001)
            du = curve["bU"] + 2 * curve["cU"] * p + 3 * curve["dU"] * p * p
            dv = 2 * curve["cV"] * p + 3 * curve["dV"] * p * p
            length = float(np.trapz(np.hypot(du, dv), p))
            road, section = make_road(root, connector, f"{incoming}_to_{outgoing}", length, 1000, x0, y0, h_a, curve)
            element(road.find("link"), "predecessor", elementType="road", elementId=road_a, contactPoint="end")
            element(road.find("link"), "successor", elementType="road", elementId=road_b, contactPoint="end")
            add_lane(section, "left", 1, "driving", LANE_WIDTH, predecessor=1, successor=-1)
            add_lane(section, "center", 0, "none", 0)
            connection = element(junction, "connection", id=connector, incomingRoad=road_a,
                                 connectingRoad=connector, contactPoint="start")
            element(connection, "laneLink", **{"from": 1, "to": 1})
            connector += 1
    root.append(junction)
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def validate_offline(xml: str) -> dict:
    import carla
    road_map = carla.Map("Indian_neighbourhood_LHT_v1", xml)
    checks = []
    for name, (road_id, _, _, heading) in ARMS.items():
        incoming = road_map.get_waypoint_xodr(road_id, 1, ARM_LENGTH - 5)
        outgoing = road_map.get_waypoint_xodr(road_id, -1, ARM_LENGTH - 5)
        assert incoming is not None and outgoing is not None
        expected_yaw = -math.degrees(heading)
        incoming_error = abs(math.remainder(incoming.transform.rotation.yaw - expected_yaw, 360))
        outgoing_error = abs(math.remainder(outgoing.transform.rotation.yaw - expected_yaw - 180, 360))
        assert incoming_error < 0.01 and outgoing_error < 0.01, (name, incoming_error, outgoing_error)
        continuation = incoming.next(10)
        assert len(continuation) == 3, (name, "junction branches", len(continuation))
        exits = incoming.next(50)
        assert {w.road_id for w in exits} == {value[0] for key, value in ARMS.items() if key != name}, (name, "outgoing roads")
        assert all(w.lane_id == -1 for w in exits), (name, "outgoing lane")
        checks.append({"approach": name, "incoming_lane": 1, "outgoing_lane": -1,
                       "incoming_yaw": incoming.transform.rotation.yaw, "outgoing_yaw": outgoing.transform.rotation.yaw,
                       "junction_branches": len(continuation), "left_hand_verified": True})
    return {"passed": True, "rule": "LHT", "approaches": checks, "roads": 16,
            "lane_width_m": LANE_WIDTH, "sidewalk_width_m": SIDEWALK}


if __name__ == "__main__":
    import json
    root = Path(__file__).resolve().parents[1]
    target = root / "integration/carla_india"
    target.mkdir(exist_ok=True)
    xml = junction_xml()
    verification = validate_offline(xml)
    (target / "neighbourhood_lht.xodr").write_text(xml, encoding="utf-8")
    (target / "map_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
