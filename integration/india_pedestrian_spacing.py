"""Conservative entry gap for a crossing person and people already on the footpath."""
import math


def entry_blockers(person, walkers, states):
    """Reserve the first two metres of a crossing before leaving the kerb.

    People already crossing keep priority. This only delays a not-yet-active
    crossing; it does not freeze someone in the carriageway.
    """
    ax, ay = person.a
    dx, dy = person.b[0]-ax, person.b[1]-ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return []
    ux, uy = dx/length, dy/length
    blocked = []
    horizon = min(3., 2./max(person.speed, .1))
    for other in walkers:
        if other is person or other.completed:
            continue
        state = states[other.actor.id]
        target = other.a if other.reverse else other.b
        ox, oy = target[0]-state["x"], target[1]-state["y"]
        distance = math.hypot(ox, oy)
        speed = other.speed if other.active else 0.
        minimum = float("inf")
        for step in range(13):
            t = horizon*step/12
            travel = min(distance, speed*t)
            x = state["x"] + ox/max(distance, 1e-6)*travel
            y = state["y"] + oy/max(distance, 1e-6)*travel
            px, py = ax+ux*person.speed*t, ay+uy*person.speed*t
            minimum = min(minimum, math.hypot(x-px, y-py))
        if minimum < 1.2:
            blocked.append(other.name)
    return blocked
