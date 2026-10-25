"""Turn observed controller decisions into frame-linked text. Never controls actors."""
from copy import deepcopy


def human_name(value):
    return (value or "traffic").replace("_", " ")


def describe_ego(traffic, states, sim_s, source_frame):
    driver = traffic.ego
    passing = deepcopy(traffic.pass_explanation)
    reason, state = driver.reason, states[driver.actor.id]
    motion = {"cruise": "Following the route", "headway": "Keeping distance from traffic ahead",
              "junction_approach": "Approaching the junction", "junction_yield": "Waiting for junction priority",
              "pedestrian_yield": "Yielding to a crossing pedestrian", "emergency_pedestrian": "Braking for a nearby pedestrian",
              "emergency_vehicle": "Braking for a nearby vehicle", "route_goal": "Destination reached",
              "parked": "Parked", "bus_stop": "Waiting at the bus stop"}.get(reason, human_name(reason))
    if driver.finished:
        action = "STOPPED" if state["speed"] < .15 else "STOPPING"
        motion = "Destination reached; R replays" if action == "STOPPED" else "Braking at the destination"
    elif reason.startswith("emergency"):
        action = "BRAKE"
    elif reason == "pedestrian_yield":
        action = "YIELD"
    elif reason == "junction_yield":
        action = "WAIT"
    elif driver.pass_state == "passing":
        action = "OVERTAKE"
    elif driver.pass_state == "returning":
        action = "RETURN LEFT"
    elif reason == "headway":
        action = "FOLLOW"
    else:
        action = "DRIVE"
        if reason == "cruise" and driver.target < driver.desired and driver.s > driver.goal_s - 32:
            motion = "Slowing for the destination"
    pass_reason = passing["reason"]
    blockers = [b for b in passing["blockers"] if b["reason"] == pass_reason]
    blocker = blockers[-1] if blockers else {}
    if driver.finished:
        pass_text = "Route complete | Press R to replay at the selected speed" if action == "STOPPED" else "Stopping at the route end"
    elif pass_reason == "junction_too_close":
        pass_text = f"Pass blocked: junction too close | available {blocker['available_m']:.1f} m / required {blocker['required_m']:.1f} m"
    elif pass_reason == "junction_zone":
        pass_text = f"Pass blocked: junction zone | {passing['permitted_section_remaining_m']:.1f} m until passing is permitted"
    elif pass_reason == "oncoming_gap":
        pass_text = f"Pass blocked: oncoming traffic | gap {blocker['available_m']:.1f} m / required {blocker['required_m']:.1f} m"
    elif pass_reason == "pedestrian_conflict":
        pass_text = "Pass blocked: a pedestrian may cross during the manoeuvre"
    elif pass_reason == "body_clearance":
        label = "side move" if blocker.get("metric") == "lateral_offset" else "side separation"
        pass_text = f"Pass blocked: body clearance | {label} available {blocker['available_m']:.2f} m / needed {blocker['required_m']:.2f} m"
    elif pass_reason == "return_gap":
        pass_text = "Return delayed: merge gap occupied by " + ", ".join(human_name(n) for n in passing["blocking_actors"])
    elif pass_reason == "passing":
        pass_text = "Passing " + human_name(passing["leader"]) + "; waiting for enough clearance to return left"
    elif pass_reason == "returning":
        pass_text = "Pass: returning to the left after clearing the lead vehicle"
    elif pass_reason == "overtake_completed":
        pass_text = "Pass completed; back on the left"
    else:
        pass_text = "Pass: no eligible slow vehicle within the passing trigger"
    speed, target = float(state["speed"]), float(driver.target)
    lines = [f"{action} | {motion} | speed {speed * 3.6:.1f} / target {target * 3.6:.1f} km/h", pass_text]
    return {"schema_version": 1, "source_frame": int(source_frame), "sim_s": float(sim_s),
            "actor": driver.name, "source": "simulator_ground_truth_rules", "controller_reason": reason,
            "action": action, "speed_mps": speed, "target_mps": target,
            "pass_state": driver.pass_state, "passing": passing, "hud_lines": lines}
