"""Validate genuinely different OpenDRIVE topologies, all with left-hand lanes."""
import xml.etree.ElementTree as ET
import carla
from india_road import ARMS, LAYOUT_ARMS, ARM_LENGTH


def validate_layout(xml, layout):
    active = LAYOUT_ARMS[layout]
    road_map = carla.Map("dataset_" + layout, xml)
    root = ET.fromstring(xml)
    roads = root.findall("road")
    assert len(roads) == len(active) ** 2
    assert all(r.attrib["rule"] == "LHT" for r in roads)
    for name in active:
        road_id = ARMS[name][0]
        for lane in (1, -1):
            wp = road_map.get_waypoint_xodr(road_id, lane, ARM_LENGTH-5)
            assert wp is not None and wp.lane_type == carla.LaneType.Driving
        incoming = road_map.get_waypoint_xodr(road_id, 1, ARM_LENGTH-5)
        assert len(incoming.next(10)) == len(active)-1
    return {"passed": True, "layout": layout, "active_arms": list(active), "roads": len(roads),
            "rule": "LHT", "independent_topology_check": "Lane existence, branch counts and all road rules"}
