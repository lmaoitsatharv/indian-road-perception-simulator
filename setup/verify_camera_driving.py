"""Offline checks for camera-only driving (no CARLA server, no GPU).

  python setup/verify_camera_driving.py            # geometry, synthetic roads, closed loop, protocol, isolation
  python setup/verify_camera_driving.py --replay   # also replay the 120 recorded town frames (needs the carla
                                                   # package for the offline map; run with .venv-carla)

Synthetic road masks are drawn with the same camera model the driver uses, so a
failure here is a planning/geometry/control bug, not a model-accuracy question.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integration"))
from india_camera_geometry import CameraGeometry, MASK_SIZE
from india_camera_driver import CameraDriver, CLASSES, Perception, DriverSettings

GEOMETRY = CameraGeometry()
CAR = {"half_length": 1.9, "half_width": .92, "front_m": 1.9, "wheelbase": 2.45, "max_steer": math.radians(70)}
# The rear camera the town fits: just behind the rear bumper, 1.3 m up, looking back.
REAR = CameraGeometry(mount=(-CAR["half_length"] - .1, 0., 1.3), pitch_deg=-20., yaw_deg=180.)
# The side cameras: at the door pillars, 1.3 m up, looking straight out and 15 degrees down.
SIDES = {"left": CameraGeometry(mount=(0., -CAR["half_width"] - .05, 1.3), pitch_deg=-15., yaw_deg=-90.),
         "right": CameraGeometry(mount=(0., CAR["half_width"] + .05, 1.3), pitch_deg=-15., yaw_deg=90.)}
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""), flush=True)


# ---------------------------------------------------------------- synthetic camera
rows, cols = np.mgrid[0:MASK_SIZE, 0:MASK_SIZE]
PIXEL_U = (cols + .5) * GEOMETRY.width / MASK_SIZE
PIXEL_V = (rows + .5) * GEOMETRY.height / MASK_SIZE
GROUND_X, GROUND_Y = GEOMETRY.ground_from_pixel(PIXEL_U, PIXEL_V)
ON_GROUND = np.isfinite(GROUND_X)
_FULL = {}


def render_rgb(brightness, pose):
    """The camera picture (grey levels) for brightness(x, y) on flat ground in world coordinates."""
    if not _FULL:
        u, v = np.meshgrid(np.arange(GEOMETRY.width) + .5, np.arange(GEOMETRY.height) + .5)
        _FULL["x"], _FULL["y"] = GEOMETRY.ground_from_pixel(u, v)
    px, py, psi = pose
    c, s = math.cos(psi), math.sin(psi)
    gx, gy = np.nan_to_num(_FULL["x"]), np.nan_to_num(_FULL["y"])
    value = np.where(np.isfinite(_FULL["x"]), brightness(px + gx * c - gy * s, py + gx * s + gy * c), 200.)
    return np.repeat(np.clip(value, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


def render(road, objects=(), pose=(0., 0., 0.)):
    """road(x, y) -> bool in world coordinates; objects: (x0, x1, y0, y1, height, class) in the car frame.
    Returns the 512x512 mask and detector-style boxes for the car at `pose`."""
    px, py, psi = pose
    c, s = math.cos(psi), math.sin(psi)
    wx = px + np.nan_to_num(GROUND_X) * c - np.nan_to_num(GROUND_Y) * s
    wy = py + np.nan_to_num(GROUND_X) * s + np.nan_to_num(GROUND_Y) * c
    mask = (road(wx, wy) & ON_GROUND).astype(np.uint8)
    boxes = []
    for obj in objects:
        x0, x1, y0, y1, height, name = obj[:6]
        bottom = obj[6] if len(obj) > 6 else 0.   # ground clearance of the body (a bus's bumper is raised)
        corners = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (bottom, height)])
        u, v, valid = GEOMETRY.project(corners[:, 0], corners[:, 1], corners[:, 2])
        if not valid.any():
            continue
        forward, _, _ = GEOMETRY.to_camera(corners[:, 0], corners[:, 1], corners[:, 2])
        safe = np.maximum(forward, .05)
        _, right, up = GEOMETRY.to_camera(corners[:, 0], corners[:, 1], corners[:, 2])
        u_all = GEOMETRY.cx + GEOMETRY.f * right / safe
        v_all = GEOMETRY.cy - GEOMETRY.f * up / safe
        # Hide the road only where the body really covers it (its outline: convex hull of the corners).
        hull = convex_hull(np.stack([u_all * MASK_SIZE / GEOMETRY.width, v_all * MASK_SIZE / GEOMETRY.height], axis=1))
        inside = np.ones(mask.shape, bool)
        for (ax, ay), (bx, by) in zip(hull, np.roll(hull, -1, axis=0)):
            inside &= (bx - ax) * (rows + .5 - ay) - (by - ay) * (cols + .5 - ax) >= 0
        mask[inside] = 0
        u = np.clip(u_all, 0, GEOMETRY.width - 1)
        v = np.clip(v_all, 0, GEOMETRY.height - 1)
        box = [u.min(), v.min(), u.max(), v.max()]
        boxes.append(box + [.9, CLASSES.index(name)])
    return mask, np.array(boxes, np.float32).reshape(-1, 6)


def convex_hull(points):
    """Monotone-chain hull, counter-clockwise in (u, v) image coordinates."""
    pts = sorted(map(tuple, np.asarray(points, float)))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def straight(left=-1.5, right=4.5):
    return lambda x, y: (y >= left) & (y <= right)


def ring(radius, turn, inner=1.5, outer=4.5):
    """Road curving left (turn=-1) or right (turn=+1); the car starts on its lane centre."""
    cy = turn * radius
    def road(x, y):
        rho = np.hypot(x, y - cy)
        if turn < 0:   # centre on the left: kerb is nearer the centre
            return (rho >= radius - inner) & (rho <= radius + outer)
        return (rho <= radius + inner) & (rho >= radius - outer)
    return road, (lambda x, y: (np.hypot(x, y - cy) - radius) * -turn)


def fresh_driver(settings=None, rear=False, sides=False):
    return CameraDriver(GEOMETRY, settings=settings, rear_geometry=REAR if rear else None,
                        side_geometry=SIDES if sides else None, **CAR)


def view_boxes(g, objects):
    """Detector-style boxes from any camera `g` for car-frame boxes (x0, x1, y0, y1, height, class[, bottom]). Points
    along the box's edges are projected, so a long vehicle close beside the camera, its corners out of view, still
    shows."""
    boxes = []
    k = np.linspace(0., 1., 13)
    for obj in objects:
        x0, x1, y0, y1, height, name = obj[:6]
        bottom = obj[6] if len(obj) > 6 else 0.
        ring = np.vstack([np.stack([x0 + (x1 - x0) * k, np.full(13, y0)], 1), np.stack([np.full(13, x1), y0 + (y1 - y0) * k], 1),
                          np.stack([x1 + (x0 - x1) * k, np.full(13, y1)], 1), np.stack([np.full(13, x0), y1 + (y0 - y1) * k], 1)])
        points = np.vstack([np.c_[ring, np.full(len(ring), z)] for z in (bottom, height)])
        forward, right, up = g.to_camera(points[:, 0], points[:, 1], points[:, 2])
        ahead = forward > .05
        if not ahead.any():
            continue
        u = g.cx + g.f * right[ahead] / forward[ahead]
        v = g.cy - g.f * up[ahead] / forward[ahead]
        if u.max() < 0 or u.min() > g.width - 1 or v.max() < 0 or v.min() > g.height - 1:
            continue
        if not ahead.all():   # it reaches round behind the camera: on out to the side of the picture
            u = np.r_[u, 0. if (right[~ahead] < 0).any() else u.min(), g.width - 1. if (right[~ahead] > 0).any() else u.max()]
        u, v = np.clip(u, 0, g.width - 1), np.clip(v, 0, g.height - 1)
        boxes.append([u.min(), v.min(), u.max(), v.max(), .9, CLASSES.index(name)])
    return np.array(boxes, np.float32).reshape(-1, 6)


def extra_view(driver, objects, turn):
    """The one extra camera picture of a frame, as the town sends it: the driver's wanted views in turn.
    `objects`: car-frame boxes (car_frame_all). Returns (rear boxes or None, {view: boxes} or None)."""
    wanted = driver.wanted_views
    if not wanted:
        return None, None
    view = wanted[turn % len(wanted)]
    if view == "rear":
        return rear_boxes(objects), None
    return None, {view: view_boxes(SIDES[view], objects)}


def rear_boxes(objects):
    """Detector-style boxes from the rear camera for car-frame boxes (x0, x1, y0, y1, height, class)."""
    g, boxes = REAR, []
    for obj in objects:
        x0, x1, y0, y1, height, name = obj[:6]
        bottom = obj[6] if len(obj) > 6 else 0.
        if x0 > g.mount[0] - 1.:
            continue
        x1 = min(x1, g.mount[0] + .35)
        corners = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (bottom, height)])
        _, _, valid = g.project(corners[:, 0], corners[:, 1], corners[:, 2])
        if not valid.any():
            continue
        forward, right, up = g.to_camera(corners[:, 0], corners[:, 1], corners[:, 2])
        safe = np.maximum(forward, .05)
        u = np.clip(g.cx + g.f * right / safe, 0, g.width - 1)
        v = np.clip(g.cy - g.f * up / safe, 0, g.height - 1)
        boxes.append([u.min(), v.min(), u.max(), v.max(), .9, CLASSES.index(name)])
    return np.array(boxes, np.float32).reshape(-1, 6)


def drive_static(mask, boxes, steps=6, speed=0.):
    """Feed one view repeatedly to a stationary car; returns the driver and its last output."""
    driver = fresh_driver()
    out = None
    for k in range(steps):
        driver.captured(k, 0., 0.)
        driver.observe(Perception(k, True, mask=mask, boxes=boxes), k * .1)
        out = driver.control(speed, 0., k * .1)
    return driver, out


# ---------------------------------------------------------------- checks
def geometry_checks():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(3, 25, 500), rng.uniform(-8, 8, 500)
    u, v, valid = GEOMETRY.project(x, y)
    gx, gy = GEOMETRY.ground_from_pixel(u[valid], v[valid])
    error = np.hypot(gx - x[valid], gy - y[valid]).max()
    check("geometry: ground -> pixel -> ground round trip", error < .02, f"max error {error*100:.2f} cm over {valid.sum()} points")
    tilted = CameraGeometry(pitch_deg=-3.)
    u, v, valid = tilted.project(x, y)
    gx, gy = tilted.ground_from_pixel(u[valid], v[valid])
    check("geometry: round trip with 3 degree pitch", np.hypot(gx - x[valid], gy - y[valid]).max() < .02)
    near = GEOMETRY.nearest_visible_ground()
    check("geometry: nearest visible ground ahead of the car", 3. < near < 5., f"{near:.2f} m")


def synthetic_checks():
    mask, boxes = render(straight())
    driver, out = drive_static(mask, boxes)
    edges = [r[1] for r in driver.plan.rows if r[1] is not None]
    check("straight road: left kerb found", len(edges) >= 18 and max(abs(e + 1.5) for e in edges) <= .1,
          f"{len(edges)} rows, worst {max(abs(e + 1.5) for e in edges):.3f} m")
    y8 = float(np.interp(8., driver.plan.path[:, 0], driver.plan.path[:, 1]))
    check("straight road: target line on the lane centre", abs(y8) < .1 and abs(out["steer"]) < .02,
          f"y(8 m) {y8:+.3f} m, steer {out['steer']:+.3f}, reason {out['reason']}")
    check("straight road: cruising", out["reason"] == "cruise" and out["decision"] == "PROCEED", out["reason"])

    mask, boxes = render(straight(-2.2, 3.8))
    driver, out = drive_static(mask, boxes)
    y8 = float(np.interp(8., driver.plan.path[:, 0], driver.plan.path[:, 1]))
    check("car 0.7 m right of its lane: steers left", abs(y8 + .7) < .12 and out["steer"] < 0, f"y(8 m) {y8:+.3f}, steer {out['steer']:+.3f}")

    for radius, turn, label in ((30., -1, "left"), (40., 1, "right")):
        road, _ = ring(radius, turn)
        mask, boxes = render(road)
        driver, out = drive_static(mask, boxes)
        true10 = turn * (radius - math.sqrt(radius ** 2 - 100.))
        y10 = float(np.interp(10., driver.plan.path[:, 0], driver.plan.path[:, 1]))
        check(f"{label} curve R={radius:.0f}: target follows the lane", abs(y10 - true10) < .25 and np.sign(out["steer"]) == turn,
              f"y(10 m) {y10:+.2f} vs {true10:+.2f}, steer {out['steer']:+.3f}")

    base = straight()
    opening = lambda x, y: base(x, y) | ((x >= 10) & (x <= 17) & (y <= -1.5))
    mask, boxes = render(opening)
    driver, out = drive_static(mask, boxes)
    check("side street opening on the left: wide road area, slows", driver.plan.wide and out["target"] <= 3.5 + 1e-6,
          f"wide {driver.plan.wide}, target {out['target']:.2f}, reason {out['reason']}")

    mask, boxes = render(straight(), [(12., 16., -.9, .9, 1.5, "car")])
    driver, out = drive_static(mask, boxes)
    check("car 12 m ahead in the lane: follows it (slower)", out["reason"] == "vehicle_ahead" and out["target"] < 6.,
          f"target {out['target']:.2f}, {out['detail']}")
    mask, boxes = render(straight(), [(5.6, 9.6, -.9, .9, 1.5, "car")])
    driver, out = drive_static(mask, boxes)
    check("car 5.6 m ahead: stops", out["target"] == 0. and out["decision"] == "STOP", f"{out['reason']}: {out['detail']}")
    mask, boxes = render(straight(), [(4.1, 8.1, -.9, .9, 1.5, "car")])
    driver, out = drive_static(mask, boxes)
    out = driver.control(3., 0., .6)   # still rolling at 3 m/s
    check("car 4.1 m ahead while moving: emergency brake", out["reason"] == "emergency_vehicle" and out["brake"] == 1., out["detail"])

    mask, boxes = render(straight(), [(8., 8.5, -2.0, -1.6, 1.7, "person")])
    driver, out = drive_static(mask, boxes)
    check("person at the kerb edge 8 m ahead: slows to 2.5 m/s", out["reason"] == "person_near_path" and abs(out["target"] - 2.5) < 1e-6,
          f"{out['reason']}: {out['detail']}")
    mask, boxes = render(straight(), [(9., 9.5, -.3, .1, 1.7, "person")])
    driver, out = drive_static(mask, boxes)
    check("person on the path 9 m ahead: slowing to stop 5 m short", out["reason"] == "person_ahead" and out["target"] < 4.,
          f"target {out['target']:.2f}, {out['detail']}")
    mask, boxes = render(straight(), [(6.5, 7., -.3, .1, 1.7, "person")])
    driver, out = drive_static(mask, boxes)
    check("person on the path 6.5 m ahead: stops", out["target"] == 0. and out["decision"] == "STOP", out["detail"])
    mask, boxes = render(straight(), [(8., 8.5, -2.6, -2.2, 1.7, "person")])
    driver, out = drive_static(mask, boxes)
    check("person on the footpath 1.3 m beside the car's side: ignored", out["reason"] == "cruise", out["reason"])
    mask, boxes = render(straight(), [(8., 8.5, 3.2, 3.6, 1.7, "person")])
    driver, out = drive_static(mask, boxes)
    check("person across the road (3.2 m right) ignored", out["reason"] == "cruise", out["reason"])

    mask, boxes = render(straight())
    fake = np.array([[180., 40., 820., 535., .82, CLASSES.index("bus")]], np.float32)   # frame-wide box over open road
    driver, out = drive_static(mask, fake)
    check("phantom frame-wide 'bus' over visible road: ignored", out["reason"] == "cruise" and len(driver.plan.rejected) == 1,
          f"{out['reason']}, rejected {len(driver.plan.rejected)}")
    # 8 Nov town run: a frame-wide phantom 'bus' was thrown out for 14 s at a junction, then kept once
    # the car turned and half its base showed sand (not road); the car stopped and was reset. A box
    # judged a phantom a moment ago in the same place stays one while road still shows under it.
    half = mask.copy()
    half[:, :MASK_SIZE // 2] = 0
    driver = fresh_driver()
    for k in range(10):
        driver.captured(k, 0., 0.)
        driver.observe(Perception(k, True, mask=mask if k < 5 else half, boxes=fake), k * .1)
        out = driver.control(0., 0., k * .1)
    _, fresh = drive_static(half, fake)
    check("the same phantom after the car turns, half its base now not road: still ignored; new there: kept",
          out["reason"] not in ("emergency_vehicle", "vehicle_ahead") and len(driver.plan.rejected) == 1 and fresh["target"] == 0.,
          f"after the turn: {out['reason']}, rejected {len(driver.plan.rejected)}; never seen before: {fresh['reason']}")
    # 8 Nov town runs: facing the open desert in a junction, a 'bus' box 3-4 m away over plain sky and
    # sand (asphalt under its base the road model does not call road). A bus that close hides the sky
    # for 100+ rows above the horizon; on the saved views real ones measured luma std 46-63 there,
    # the phantoms about 7.
    def sky_and_ground(x, y):
        return np.full(np.shape(x), 120.)
    def textured(rgb, box):
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        stripes = np.where((np.arange(y1, y2 + 1)[:, None] // 12 + np.arange(x1, x2 + 1)[None, :] // 40) % 2, 60, 210)
        rgb[y1:y2 + 1, x1:x2 + 1] = stripes[..., None].astype(np.uint8)
        return rgb
    def drive_frames(mask, boxes, rgb, steps=6):
        driver, out = fresh_driver(), None
        for k in range(steps):
            driver.captured(k, 0., 0.)
            driver.observe(Perception(k, True, mask=mask, boxes=boxes, rgb=rgb), k * .1)
            out = driver.control(5., 0., k * .1)
        return driver, out
    rgb = render_rgb(sky_and_ground, (0., 0., 0.))   # sky (above the horizon) 200, ground 120
    mask, _ = render(straight())
    desert = np.array([[330., 0., 900., 400., .89, CLASSES.index("bus")]], np.float32)
    mask[int(330 * MASK_SIZE / 540):int(410 * MASK_SIZE / 540), int(330 * MASK_SIZE / 960):int(900 * MASK_SIZE / 960)] = 0
    phantom_driver, phantom = drive_frames(mask, desert, rgb)
    # 9 Nov (destinations15): a real sky grows brighter towards the horizon and has clouds higher up. A box from the
    # top of the picture tested the whole sky and missed a phantom 'bus' 1.8 m away (90 s stop, reset). Only the band
    # just above the horizon, where a near body would surely be, is judged.
    cloudy = rgb.copy()
    rows_above = np.arange(270)[:, None]
    sky = 120. + 90. * rows_above / 270. + np.where((rows_above < 120) & ((np.arange(960)[None, :] // 25 + rows_above // 9) % 3 == 0), 25., 0.)
    sky = sky + 40. * np.arange(960)[None, :] / 960.   # and brighter towards the sun side (destinations17: ±15 luma)
    cloudy[:270] = np.clip(sky, 0, 255).astype(np.uint8)[..., None]
    full = np.array([[60., 0., 780., 539., .95, CLASSES.index("bus")]], np.float32)   # cut at the top and bottom
    sand = mask.copy()   # sand and unlabelled asphalt under its base (not road), as in the town
    sand[int(460 * MASK_SIZE / 540):, int(60 * MASK_SIZE / 960):int(780 * MASK_SIZE / 960)] = 0
    cloudy_driver, cloudy_out = drive_frames(sand, full, cloudy)
    check("a 'bus' box from the top of the picture over a graded, cloudy sky: still ignored",
          cloudy_out["reason"] not in ("emergency_vehicle", "vehicle_ahead") and len(cloudy_driver.plan.rejected) == 1,
          f"{cloudy_out['reason']}, rejected {len(cloudy_driver.plan.rejected)}")
    mask, boxes = render(straight(), [(6., 16., -1.3, 1.3, 3.0, "bus")])
    bus_driver, bus = drive_frames(mask, boxes, textured(render_rgb(sky_and_ground, (0., 0., 0.)), boxes[0]))
    mask, boxes = render(straight(), [(5.6, 9.6, -.9, .9, 1.5, "car")])
    _, car = drive_frames(mask, boxes, rgb)   # a car is lower than the camera: plain sky above it is normal
    # Final review (9 Nov): a plain truck side (no texture) passed the plain-band test too. The band must also look
    # like the sky beside the box, on the same rows.
    mask, boxes = render(straight(), [(6., 13., -1.2, 1.2, 2.6, "truck")])
    panel = render_rgb(sky_and_ground, (0., 0., 0.))
    x1, y1, x2, y2 = (int(v) for v in boxes[0][:4])
    panel[y1:y2 + 1, x1:x2 + 1] = 90
    truck_driver, truck = drive_frames(mask, boxes, panel)
    check("'bus' 4 m away over plain sky, not-road under its base: ignored; a real bus, a plain-sided truck and a car under plain sky: kept",
          phantom["reason"] not in ("emergency_vehicle", "vehicle_ahead") and len(phantom_driver.plan.rejected) == 1
          and bus["target"] == 0. and not bus_driver.plan.rejected and car["target"] == 0.
          and truck["target"] == 0. and not truck_driver.plan.rejected,
          f"over sky: {phantom['reason']}, rejected {len(phantom_driver.plan.rejected)}; real bus: {bus['reason']}; "
          f"plain truck: {truck['reason']}, rejected {len(truck_driver.plan.rejected)}; car: {car['reason']}")
    mask, boxes = render(straight(), [(5.6, 9.6, -.9, .9, 1.5, "car")])
    driver, out = drive_static(mask, boxes)
    check("real car close ahead (road hidden under it): kept", not driver.plan.rejected and out["target"] == 0., out["reason"])
    # 9 Nov (destinations19): stopped 0.7 m behind a waiting car that filled the picture, the detector lost it for
    # 1.1 s; a sighting was kept 0.3 s, so the car pulled away into it. A vehicle seen close ahead is kept longer.
    mask, boxes = render(straight(), [(2.6, 6.6, -.9, .9, 1.5, "car")])
    driver, gone = fresh_driver(), []
    for k in range(40):
        driver.captured(k, 0., 0.)
        driver.observe(Perception(k, True, mask=mask, boxes=boxes if (k < 10 or k >= 34) else np.zeros((0, 6), np.float32)), k * .05)
        out = driver.control(0., 0., k * .05)
        if 10 <= k < 34:
            gone.append(out["target"])
    check("a car stopped just ahead, lost by the detector for 1.2 s: the car stays stopped",
          max(gone) == 0., f"highest target speed while it was not seen: {max(gone):.2f} m/s")
    # 14 Nov: a man walking beside the village road was also boxed as a "rider" for a frame or two; that
    # sighting was kept 2.5 s like a car filling the picture, so the car stopped after passing him, he walked
    # on ahead, and the same happened again. Only a four-wheeler sighting is kept that long.
    mask, boxes = render(straight(), [(2.9, 3.5, -.3, .3, 1.7, "rider")])
    driver, gone = fresh_driver(), []
    for k in range(40):
        driver.captured(k, 0., 0.)
        driver.observe(Perception(k, True, mask=mask, boxes=boxes if k < 10 else np.zeros((0, 6), np.float32)), k * .05)
        out = driver.control(0., 0., k * .05)
        if k >= 20:
            gone.append(out["reason"])
    check("a rider seen just ahead then gone for 0.5 s: no longer held as a vehicle",
          not any(r.startswith(("emergency_vehicle", "vehicle_ahead")) for r in gone), ", ".join(sorted(set(gone))))
    # A car close at the front-left, at an angle: its box is cut off at the picture's edge and takes in
    # road in front of it. It is not judged as a phantom (7 Nov run06 threw one out).
    mask, _ = render(straight())
    side = np.array([[0., 150., 300., 420., .8, CLASSES.index("car")]], np.float32)
    driver, out = drive_static(mask, side)
    check("car cut off at the picture's side, road visible under its box: kept", not driver.plan.rejected,
          f"rejected {len(driver.plan.rejected)}; {out['reason']}")
    wide = np.array([[250., 0., 959., 535., .8, CLASSES.index("bus")]], np.float32)   # run11: phantom over 3/4 of the picture
    driver, out = drive_static(mask, wide)
    check("phantom 'bus' over three quarters of the picture, cut at its edge: still ignored",
          out["reason"] == "cruise" and len(driver.plan.rejected) == 1, f"{out['reason']}, rejected {len(driver.plan.rejected)}")
    # Passing is for buses and trucks only: a stopped motorcycle is waited behind, however long.
    bike = (10., 11.8, -.5, .3, 1.4, "motorcycle")
    log = []
    simulate(straight(-1.6, 4.9), lambda x, y: y, seconds=40., world_objects=[bike], log=log)
    check("motorcycle stopped ahead for 30 s (a junction queue): waits behind it, never pulls out",
          max(y for t, x, y, v, why, psi in log) < .3 and max(x for t, x, y, v, why, psi in log) < 10.,
          f"max right offset {max(y for t, x, y, v, why, psi in log):.2f} m")
    road = straight()
    banded = lambda x, y: road(x, y) & ~((x > 9.) & (x < 9.25))
    driver, out = drive_static(*render(banded))
    check("thin non-road band across the road: no stop", out["reason"] == "cruise" and driver.plan.road_end_m is None,
          f"{out['reason']}, road end {driver.plan.road_end_m}")
    driver, out = drive_static(*render(lambda x, y: (y >= .8) & (y <= 6.8)), steps=8)
    y8 = float(np.interp(8., driver.plan.path[:, 0], driver.plan.path[:, 1]))
    check("car 0.8 m off the road's left edge: finds the road again", driver.plan.fit_rows >= 10 and abs(y8 - 2.3) < .25,
          f"fit rows {driver.plan.fit_rows}, y(8 m) {y8:+.2f}")

    driver, out = drive_static(*render(straight()))
    for k in range(12):
        driver.observe(Perception(100 + k, False, error="worker offline"), .6 + k * .05)
        out = driver.control(4., 0., .6 + k * .05)
    check("perception lost: brakes to a stop", out["reason"] == "perception_unavailable" and out["target"] == 0., out["reason"])
    first = fresh_driver().control(0., 0., 0.)
    check("before the first result: stays stopped", first["reason"] == "starting" and first["brake"] > 0, first["reason"])

    mask = np.zeros((MASK_SIZE, MASK_SIZE), np.uint8)
    driver, out = drive_static(*render(straight()))
    for k in range(100):
        t = .6 + k * .1
        driver.captured(1000 + k, 2., 0.)
        driver.observe(Perception(1000 + k, True, mask=mask, boxes=np.zeros((0, 6), np.float32)), t)
        out = driver.control(2., 0., t)
    check("no road in view for 10 s: stops", out["target"] == 0. and out["reason"] in ("lost_road_stop", "road_ends"), out["reason"])


def car_frame_all(world_objects, pose):
    """World axis-aligned boxes -> car-frame bounds, without clipping (for the rear camera)."""
    px, py, psi = pose
    c, s = math.cos(psi), math.sin(psi)
    out = []
    for obj in world_objects:
        x0, x1, y0, y1, height, name = obj[:6]
        corners = np.array([(x, y) for x in (x0, x1) for y in (y0, y1)]) - [px, py]
        local = np.stack([corners[:, 0] * c + corners[:, 1] * s, -corners[:, 0] * s + corners[:, 1] * c], axis=1)
        out.append((local[:, 0].min(), local[:, 0].max(), local[:, 1].min(), local[:, 1].max(), height, name) + tuple(obj[6:]))
    return out


class Mover:
    """A vehicle driving along x at a fixed lateral position (negative speed: towards the car).
    It gives way like the town's other drivers: it stops while the car's body stands in its
    lane just ahead of it."""

    def __init__(self, x, y, speed, length, width, height, name, bottom=0.):
        self.x, self.y, self.speed, self.length, self.width = x, y, speed, length, width
        self.height, self.name, self.bottom, self.v = height, name, bottom, abs(speed)

    def box(self):
        return (self.x - self.length / 2, self.x + self.length / 2, self.y - self.width / 2, self.y + self.width / 2,
                self.height, self.name, self.bottom)

    def step(self, car_polygon, dt=.05):
        from india_behavior import polygon_gap, rectangle
        sign = 1. if self.speed > 0 else -1.
        front = self.x + sign * self.length / 2
        reach = 6. + 2.5 * self.v
        lane = rectangle(front + sign * reach / 2, self.y, 0., reach / 2, self.width / 2 + .4)
        blocked = polygon_gap(lane, car_polygon)[0]
        self.v = max(0., self.v - 6. * dt) if blocked else min(abs(self.speed), self.v + 2. * dt)
        self.x += sign * self.v * dt


class Walker:
    """A pedestrian at (x, y) who starts walking at (vx, vy) once the car's centre is within `trigger`
    metres before x, and does not stop for the car (like the town's scripted jaywalkers)."""

    def __init__(self, x, y, vx, vy, trigger):
        self.x, self.y, self.vx, self.vy, self.trigger, self.walking = x, y, vx, vy, trigger, False

    def box(self):
        return (self.x - .3, self.x + .3, self.y - .3, self.y + .3, 1.7, "person")

    def step(self, car_polygon, dt=.05):
        centre = float(np.mean(np.asarray(car_polygon)[:, 0]))
        self.walking = self.walking or self.x - centre <= self.trigger
        if self.walking:
            self.x += self.vx * dt
            self.y += self.vy * dt


def pedestrian_checks():
    """9 Nov (destinations08/09): jaywalker_3 crossed from the right at 1.15 m/s. The car saw them 2.3 m
    ahead of its front, 1.2 m clear of its side, only slowed (they were 'near the path'), lost them from
    view beside its bumper and drove on into them at 1.9 m/s."""
    road = straight()
    cases = {"crossing from the right": Walker(40., 3.9, 0., -1.15, 12.7),
             "walking along the road edge": Walker(40., 2.4, -1.2, 0., 30.),
             "standing at the road edge": Walker(40., 2.4, 0., 0., 30.)}
    found = {}
    for name, walker in cases.items():
        log, info = [], {}
        simulate(road, lambda x, y: y, seconds=25., log=log, movers=[walker], info=info, start=(0., 0., 0.))
        found[name] = (round(info["mover_gaps"][0], 2), round(log[-1][1], 1), round(min(v for t, x, y, v, why, psi in log if t > 8.), 2))
    crossing, along, standing = found.values()
    check("jaywalker crossing in front: the car stops clear of them, then drives on; people walking or standing at the edge do not stop it",
          crossing[0] > .3 and crossing[1] > 45. and along[0] > .3 and along[2] > 1. and standing[0] > .3 and standing[2] > 1.,
          "(closest gap m, final x, slowest speed after 8 s): " + "; ".join(f"{k}: {v}" for k, v in found.items()))


def car_frame_objects(world_objects, pose):
    """World axis-aligned boxes -> axis-aligned boxes around them in the car frame."""
    px, py, psi = pose
    c, s = math.cos(psi), math.sin(psi)
    out = []
    for obj in world_objects:
        x0, x1, y0, y1, height, name = obj[:6]
        corners = np.array([(x, y) for x in (x0, x1) for y in (y0, y1)]) - [px, py]
        local = np.stack([corners[:, 0] * c + corners[:, 1] * s, -corners[:, 0] * s + corners[:, 1] * c], axis=1)
        if local[:, 0].max() > 1.:
            out.append((max(local[:, 0].min(), 1.), local[:, 0].max(), local[:, 1].min(), local[:, 1].max(), height, name) + tuple(obj[6:]))
    return out


def simulate(road, error_of, seconds=30., start=(0., 0., 0.), latency=2, world_objects=(), log=None,
             rear=False, rear_dead=False, movers=(), info=None, brightness=None, settings=None, sides=False):
    """Kinematic bicycle in a synthetic world, perceiving only its own rendered camera(s).
    The rear camera runs whenever the driver asks for it (as in the town); rear_dead: it never answers.
    sides: the side cameras too (one extra view a frame, in turn, as in the town)."""
    from india_behavior import polygon_gap, rectangle
    driver = fresh_driver(settings, rear=rear, sides=sides)
    x, y, psi = start
    vel, r = 0., 0.   # signed: negative while reversing
    asphalt_frames = 0
    due, errors, speeds = {}, [], []
    notes, gaps, rear_frames = set(), {}, 0
    for it in range(int(seconds / .05)):
        t = it * .05
        if it in due:
            frame, mask, boxes, back, picture, side = due.pop(it)
            driver.observe(Perception(frame, True, mask=mask, boxes=boxes, rear_boxes=back, rgb=picture, side_boxes=side), t)
            asphalt_frames += bool(driver.plan is not None and driver.plan.asphalt_rows)
            if driver.plan is not None and driver.plan.pass_note:
                notes.add(driver.plan.pass_note)
        out = driver.control(abs(vel), r, t)
        delta = out["steer"] * CAR["max_steer"]
        r = vel * math.tan(delta) / CAR["wheelbase"]
        psi += r * .05
        x += vel * math.cos(psi) * .05
        y += vel * math.sin(psi) * .05
        gear = -1. if out.get("reverse") else 1.
        speed = abs(vel)
        if speed < 1e-6 or np.sign(vel) == gear:
            speed = max(0., speed + (3.5 * out["throttle"] - 7. * out["brake"] - .03 * speed) * .05)
            vel = gear * speed
        else:   # throttle against the motion acts as a brake
            vel = np.sign(vel) * max(0., speed - (3.5 * out["throttle"] + 7. * out["brake"]) * .05)
        body = rectangle(x, y, psi, CAR["half_length"], CAR["half_width"])
        for k, mover in enumerate(movers):
            mover.step(body)
            x0, x1, y0, y1 = mover.box()[:4]
            gap = polygon_gap(body, rectangle((x0 + x1) / 2, (y0 + y1) / 2, 0., (x1 - x0) / 2, (y1 - y0) / 2))
            gaps[k] = min(gaps.get(k, 1e9), 0. if gap[0] else gap[1])
        objects = list(world_objects) + [m.box() for m in movers]
        if it % 2 == 0:
            mask, boxes = render(road, car_frame_objects(objects, (x, y, psi)), pose=(x, y, psi))
            back = side = None
            if rear and not rear_dead:
                back, side = extra_view(driver, car_frame_all(objects, (x, y, psi)), it // 2)
                rear_frames += back is not None
            driver.captured(it, abs(vel), r)
            picture = None if brightness is None else render_rgb(brightness, (x, y, psi))
            due[it + latency] = (it, mask, boxes, back, picture, side)
        if log is not None:
            log.append((t, x, y, abs(vel), out["reason"], psi))
        if t > 4.:
            errors.append(float(error_of(np.array(x), np.array(y))))
            speeds.append(abs(vel))
    if info is not None:
        info.update(driver=driver, notes=notes, mover_gaps=gaps, rear_frames=rear_frames, asphalt_frames=asphalt_frames)
    return np.array(errors), np.array(speeds)


def closed_loop_checks():
    for radius, turn, label in ((50., -1, "left"), (50., 1, "right")):
        road, error_of = ring(radius, turn)
        errors, speeds = simulate(road, error_of)
        check(f"closed loop, {label} curve R={radius:.0f} for 30 s: stays within 0.3 m of the lane centre",
              np.abs(errors).max() <= .3 and speeds.mean() > 3.,
              f"max |error| {np.abs(errors).max():.3f} m, mean speed {speeds.mean():.2f} m/s")
    errors, speeds = simulate(straight(), lambda x, y: y, start=(0., .6, 0.))
    check("closed loop, straight road starting 0.6 m off: settles on the lane centre",
          np.abs(errors[-200:]).max() <= .15 and np.abs(errors).max() <= .65,
          f"final max |error| {np.abs(errors[-200:]).max():.3f} m")
    # The CARLA bus (10.3 m x 3.94 m, raised bumper) stopped in the left lane of the divided road
    # (6.5 m carriageway), as at the town's bus stop: stop short, wait, pass on the right.
    bus = (60., 70.3, .14 - 1.97, .14 + 1.97, 3.2, "bus", .45)
    road = straight(-1.6, 4.9)
    log = []
    simulate(road, lambda x, y: y, seconds=45., world_objects=[bus], log=log)
    from india_behavior import polygon_gap, rectangle
    bus_body = rectangle((bus[0] + bus[1]) / 2, .14, 0., (bus[1] - bus[0]) / 2, 1.97)
    gaps = [polygon_gap(rectangle(x, y, psi, CAR["half_length"], CAR["half_width"]), bus_body) for t, x, y, v, why, psi in log]
    closest = min(g for overlap, g in gaps) if not any(overlap for overlap, g in gaps) else 0.
    first_stop = next((bus[0] - (x + CAR["front_m"]) for t, x, y, v, why, psi in log if v < .05 and t > 5), None)
    passed = max(x for t, x, y, v, why, psi in log) > bus[1] + 2.
    check("closed loop, stopped bus ahead: stops short, then passes on the right",
          first_stop is not None and 2.5 <= first_stop <= 8. and closest >= .2 and passed,
          f"first stop {first_stop if first_stop is not None else float('nan'):.2f} m short; closest body gap {closest:.2f} m; "
          f"passed {passed}; max right offset {max(y for t, x, y, v, why, psi in log):.2f} m")
    pass_checks(bus, road, bus_body)
    shoulder_checks()


def shoulder_checks():
    """Village road: 2.8 m lanes with a 1.35 m paved shoulder each side, the shoulder brighter than
    the asphalt. The road model counts the shoulder as road once the car is near it (7 Nov probe);
    here it always does, the worst case. The car starts 1.2 m left, as it leaves a junction."""
    road = straight(-2.75, 5.55)
    def village(x, y):
        return np.where((y >= -1.4) & (y <= 4.2), 95., np.where((y >= -2.75) & (y <= 5.55), 165., 150.))
    info = {}
    errors, speeds = simulate(road, lambda x, y: y, seconds=30., start=(0., -1.2, 0.), brightness=village, info=info)
    check("village road, shoulder counted as road: finds the asphalt edge and keeps to the lane",
          np.abs(errors[-200:]).max() <= .3 and info["asphalt_frames"] > 0,
          f"final max |error| {np.abs(errors[-200:]).max():.2f} m; asphalt edge used in {info['asphalt_frames']} frames")
    errors, _ = simulate(road, lambda x, y: y, seconds=30., start=(0., -1.2, 0.), brightness=village,
                         settings=DriverSettings(asphalt_edge=0.))
    check("village road without the asphalt edge: rides on the shoulder (the 6-7 Nov town behaviour)",
          np.median(errors[-200:]) < -.9, f"final median error {np.median(errors[-200:]):+.2f} m")
    # Roadside carts on the shoulder beside the car: cut off at the bottom and the side of the picture.
    # The road model shows road up to them, so they are not on the path (7 Nov run05 stopped for one
    # for 30 s). A car cutting in at the front-left still stops it.
    carts = [(2.6, 4.6, -3.0, -1.6, 1.8, "truck"), (2.2, 4.2, -3.2, -1.7, 1.8, "other_vehicle"), (3.0, 6.0, -2.9, -1.5, 2.2, "truck"),
             # wheels in view, box cut only at the side (7 Nov run11, run12)
             (4.5, 6.5, -4.0, -1.6, 1.8, "other_vehicle"), (4.0, 6.0, -4.5, -1.7, 1.8, "truck")]
    lane = straight(-1.4, 4.2)   # the car holds its lane (the shoulder not counted as road)
    reasons = []
    for cart in carts:
        mask, boxes = render(lane, [cart])
        _, out = drive_static(mask, boxes, speed=5.)
        reasons.append(out["reason"])
    mask, boxes = render(lane, [(2.4, 6.8, -1.6, .2, 1.5, "car")])
    _, cut_in = drive_static(mask, boxes, speed=5.)
    mask, boxes = render(lane, [(4.5, 6.5, -4.0, -.6, 1.6, "car")])   # cut at the side, wheels in view, into the path
    _, cut_in2 = drive_static(mask, boxes, speed=5.)
    check("roadside carts beside the car (cut-off boxes): not on its path; a car cutting in still is",
          all(r == "cruise" for r in reasons) and cut_in["reason"].startswith("emergency")
          and cut_in2["reason"] in ("emergency_vehicle", "vehicle_ahead") and cut_in2["target"] < 1.,
          f"carts: {', '.join(reasons)}; car cutting in: {cut_in['reason']}, {cut_in2['reason']}")
    # Vehicles waiting in the opposite lane, seen at an angle: the box's inner edge is their far corner,
    # which read 1-2 m nearer the path at their near face (8 Nov town runs stopped 90 s beside a van
    # 2.4 m clear of the car, twice). The road model shows road at the box's base on that side.
    oncoming = [(6., 10.5, 1.9, 3.7, 1.5, "car"), (8., 12.5, 1.9, 3.7, 1.5, "car"), (6., 12., 1.9, 3.9, 2.4, "truck"),
                (7.7, 13.7, 3.36, 5.34, 2.4, "truck")]
    found = []
    for other in oncoming:
        mask, boxes = render(lane, [other])
        driver, out = drive_static(mask, boxes, speed=5.)
        found.append((out["reason"], round(min(o["y_left"] for o in driver.plan.obstacles), 2)))
    mask, boxes = render(lane, [(6., 10.5, .6, 2.4, 1.5, "car")])   # over the lane line, into the car's path
    _, over = drive_static(mask, boxes, speed=5.)
    check("vehicles waiting in the opposite lane: not on the path; one over the lane line still is",
          all(reason == "cruise" for reason, _ in found) and over["reason"] in ("emergency_vehicle", "vehicle_ahead"),
          f"opposite lane (reason, inner edge): {found}; over the line: {over['reason']}")
    # A kerb street: the sidewalk is brighter than the asphalt too, but it is not road, so nothing changes.
    def kerb(x, y):
        return np.where((y >= -1.5) & (y <= 4.5), 95., 160.)
    info = {}
    errors, _ = simulate(straight(), lambda x, y: y, start=(0., .6, 0.), brightness=kerb, info=info)
    check("kerb street with a bright sidewalk: unchanged, settles on the lane centre",
          np.abs(errors[-200:]).max() <= .15 and info["asphalt_frames"] == 0,
          f"final max |error| {np.abs(errors[-200:]).max():.3f} m; asphalt edge used in {info['asphalt_frames']} frames")


def bus_gap(log, bus_body):
    from india_behavior import polygon_gap, rectangle
    gaps = [polygon_gap(rectangle(x, y, psi, CAR["half_length"], CAR["half_width"]), bus_body) for t, x, y, v, why, psi in log]
    return 0. if any(overlap for overlap, g in gaps) else min(g for overlap, g in gaps)


def pass_checks(bus, road, bus_body):
    """Passing with the rear camera, and the cases that must stop or undo a pass."""
    # 1. Nothing around: the rear camera comes on while waiting, and the pass goes ahead.
    log, info = [], {}
    simulate(road, lambda x, y: y, seconds=45., world_objects=[bus], log=log, rear=True, info=info)
    passed = max(x for t, x, y, v, why, psi in log) > bus[1] + 2.
    events = [e[1] for e in info["driver"].events]
    check("pass with the rear camera, lane clear: rear view runs only around the pass, then passes",
          passed and bus_gap(log, bus_body) >= .2 and 0 < info["rear_frames"] < 300 and "pass_started" in events,
          f"passed {passed}; rear frames {info['rear_frames']} of 450; events {events}")
    # 2. A car coming up behind in the passing lane: wait until it has gone by and is well ahead.
    log, info = [], {}
    behind = Mover(-60., 3.4, 5., 4.4, 1.8, 1.5, "car")
    simulate(road, lambda x, y: y, seconds=70., world_objects=[bus], log=log, rear=True, movers=[behind], info=info)
    started = next((e[0] for e in info["driver"].events if e[1] == "pass_started"), None)
    ego_at = next((x for t, x, y, v, why, psi in log if started is not None and t >= started), None)
    car_at = None if started is None else -60. + 5. * started
    waited_mirror = any("behind in the passing lane" in n for n in info["notes"])
    waited_ahead = any("ahead in the passing lane" in n for n in info["notes"])
    passed = max(x for t, x, y, v, why, psi in log) > bus[1] + 2.
    check("car coming up behind in the passing lane: waits (rear camera), lets it by, then passes",
          started is not None and waited_mirror and car_at - ego_at > 20. and info["mover_gaps"][0] > .5 and passed
          and bus_gap(log, bus_body) >= .2,
          f"pass started at {started if started is not None else float('nan'):.1f} s with the car "
          f"{(car_at - ego_at) if started is not None else float('nan'):.0f} m ahead; mirror wait {waited_mirror}, "
          f"ahead wait {waited_ahead}; closest to it {info['mover_gaps'][0]:.2f} m; passed {passed}")
    # 3. A bus coming the other way during the pass (the same class as the one being passed, which
    #    was ignored before 7 Nov): stop, give up, back out, let it by, then pass.
    log, info = [], {}
    # Timed to come into view just after the pass starts (about 14 s, the car at x = 51 m).
    oncoming = Mover(182., 3.3, -6., 10., 2.6, 3.2, "bus", .45)
    simulate(road, lambda x, y: y, seconds=110., world_objects=[bus], log=log, rear=True, movers=[oncoming], info=info)
    events = [e[1] for e in info["driver"].events]
    reversed_m = sum(abs(log[i + 1][1] - log[i][1]) for i in range(len(log) - 1) if log[i + 1][1] < log[i][1] - 1e-4)
    passed = max(x for t, x, y, v, why, psi in log) > bus[1] + 2.
    check("bus coming the other way mid-pass: stops for it, backs out, lets it by, passes later",
          "pass_abandoned" in events and "backed_out" in events and info["mover_gaps"][0] >= .5 and passed
          and bus_gap(log, bus_body) >= .2 and oncoming.x < bus[0] - 20.,
          f"events {events}; reversed {reversed_m:.1f} m; closest to the oncoming bus {info['mover_gaps'][0]:.2f} m; "
          f"to the parked bus {bus_gap(log, bus_body):.2f} m; passed {passed}")
    # 4. No rear view: never moves out.
    log, info = [], {}
    simulate(road, lambda x, y: y, seconds=30., world_objects=[bus], log=log, rear=True, rear_dead=True, info=info)
    moved_out = max(y for t, x, y, v, why, psi in log) > 1.
    check("rear camera not answering: does not start a pass",
          not moved_out and any("rear camera" in n for n in info["notes"]),
          f"max right offset {max(y for t, x, y, v, why, psi in log):.2f} m; notes {sorted(info['notes'])[:2]}")
    # 5. Backing up stops short of a car that has pulled up behind.
    driver = fresh_driver(rear=True)
    mask, boxes = render(road, [])
    driver.captured(0, 0., 0.)
    driver.observe(Perception(0, True, mask=mask, boxes=boxes, rear_boxes=np.zeros((0, 6), np.float32)), 0.)
    xs = np.arange(-8., .01, .5)
    driver.backing = {"path": np.stack([xs, -1.2 * (1 - np.cos(np.pi * (xs + 8) / 16))], axis=1), "why": "test",
                      "start_index": 0, "since": 0., "best": 9., "best_time": 0.}
    car = (-8.5, -4.1, -1.5, .3, 1.5, "car")
    driver.captured(1, 0., 0.)
    driver.observe(Perception(1, True, mask=mask, boxes=boxes, rear_boxes=rear_boxes([car])), .1)
    stop = driver.control(0., 0., .1)
    driver.captured(2, 0., 0.)
    driver.observe(Perception(2, True, mask=mask, boxes=boxes, rear_boxes=np.zeros((0, 6), np.float32)), .2)
    go = driver.control(0., 0., .2)
    go = driver.control(0., 0., .25)
    check("backing up: stops for a car 2 m behind, reverses when it has gone",
          stop["target"] == 0. and stop["reason"] == "backing_wait" and go["reverse"] and go["throttle"] > 0.,
          f"with the car: {stop['reason']} ({stop['detail']}); without: reverse {go['reverse']}, throttle {go['throttle']:.2f}")
    # 6. With the side cameras the town fits: the rear and right ones send a picture a frame in turn (5 Hz each).
    results = []
    for movers, seconds in (([], 45.), ([Mover(-60., 3.4, 5., 4.4, 1.8, 1.5, "car")], 70.)):
        log, info = [], {}
        simulate(road, lambda x, y: y, seconds=seconds, world_objects=[bus], log=log, rear=True, sides=True, movers=movers, info=info)
        results.append((max(x for t, x, y, v, why, psi in log) > bus[1] + 2., bus_gap(log, bus_body),
                        info["mover_gaps"].get(0, 9.), next((e[0] for e in info["driver"].events if e[1] == "pass_started"), None)))
    check("passing with the side cameras too (rear and right in turn): passes with the lane clear; lets a car coming "
          "up behind go by first", all(p and g >= .2 and m > .5 for p, g, m, _ in results),
          "; ".join(f"passed {p}, bus gap {g:.2f} m, closest to the car {m:.2f} m, pass at {s} s" for p, g, m, s in results))
    # 7. The right camera before pulling out: a car moving alongside in the passing lane (never seen behind: out of a
    #    side lane) is waited for; one standing there is not (the car pulls out ahead of it, as before). A vehicle seen
    #    behind a moment ago is assumed alongside while anything shows there, not once the lane beside is empty.
    def mirror_with(cars, blind=False):
        driver = fresh_driver(rear=True, sides=True)
        mask, boxes = render(road, [])
        for k in range(12):
            driver.captured(k, 0., 0.)
            driver.observe(Perception(k, True, mask=mask, boxes=boxes, rear_boxes=np.zeros((0, 6), np.float32),
                                      side_boxes={"right": view_boxes(SIDES["right"], [c(k * .1) for c in cars])}), k * .1)
        driver.blind_since = 1.1 if blind else None
        return driver._mirror(3.4, 1.15)
    moving = mirror_with([lambda t: (-6. + 4. * t, -1.6 + 4. * t, 2.5, 4.3, 1.5, "car")])   # 4 m/s, half past the camera
    standing = mirror_with([lambda t: (-2.5, 1.9, 2.5, 4.3, 1.5, "car")])
    guess_empty, guess_there = mirror_with([], blind=True), mirror_with([lambda t: (-2.5, 1.9, 2.5, 4.3, 1.5, "car")], blind=True)
    check("right camera before pulling out: waits for a car moving alongside, not one standing there; the 'seen behind' "
          "guess holds only while something shows beside the car",
          moving is not None and "alongside" in moving and standing is None and guess_empty is None
          and guess_there is not None and "seen behind" in guess_there,
          f"moving: {moving}; standing: {standing}; guess, lane empty: {guess_empty}; guess, something there: {guess_there}")


def wedge_checks():
    """Wedged: the throttle on and the car not moving, against something no camera shows (26 Nov run: a kerb on the
    inside of a sharp left, 29 s of pushing until the town's stall reset)."""
    none = np.zeros((0, 6), np.float32)

    def drive(driver, mask, boxes, speeds):
        """Feed 10 Hz frames and 20 Hz control at the given speeds (signed by the car's own gear); returns outputs."""
        outs = []
        for speed in speeds:
            k = driver.wedge_test_k
            driver.wedge_test_k = k + 1
            t = k * .1
            driver.captured(k, speed, 0.)
            driver.observe(Perception(k, True, mask=mask, boxes=boxes, rear_boxes=none), t)
            outs.append(driver.control(speed, 0., t))
            outs.append(driver.control(speed, 0., t + .05))
        return outs
    mask, boxes = render(straight(), [])
    driver = fresh_driver(rear=True)
    driver.wedge_test_k = 0
    drive(driver, mask, boxes, [3.] * 30)                        # 3 s at 3 m/s: a way it came to back along
    held = drive(driver, mask, boxes, [0.] * 30)                 # then held where it is, whatever the throttle
    first = next((i for i, o in enumerate(held) if o["reverse"]), None)
    x0, backed = driver.pose[0], None
    for _ in range(60):                                          # reversing at 0.8 m/s until it has backed off
        drive(driver, mask, boxes, [.8])
        if "unwedged" in [e[1] for e in driver.events]:
            backed = x0 - driver.pose[0]
            break
    backed = -1. if backed is None else backed
    events = [e[1] for e in driver.events]
    after = drive(driver, mask, boxes, [0.] * 10)
    check("wedged (throttle on, not moving, nothing seen): backs off the way it came after 2 s, then drives on",
          first is not None and 2. <= first * .05 <= 2.6 and "wedged" in events and "unwedged" in events
          and 1.1 <= backed <= 1.8 and not after[-1]["reverse"] and after[-1]["throttle"] > 0.,
          f"backing from {None if first is None else round(first * .05, 2)} s held; backed {backed:.2f} m; events {events}; "
          f"then reverse {after[-1]['reverse']}, throttle {after[-1]['throttle']:.2f}")
    # Waiting behind a stopped car (target 0) is not being wedged, however long.
    mask, boxes = render(straight(), [(6., 10.5, -.9, .9, 1.5, "car")])   # 3.3 m ahead: it stays where it is
    driver = fresh_driver(rear=True)
    driver.wedge_test_k = 0
    outs = drive(driver, mask, boxes, [0.] * 80)
    check("standing behind a stopped car for 8 s is not taken for being wedged",
          not any(o["reverse"] for o in outs) and "wedged" not in [e[1] for e in driver.events],
          f"last: {outs[-1]['reason']}; events {[e[1] for e in driver.events]}")


def isolation_check():
    forbidden = {"carla", "world", "get_world", "get_map", "snapshot", "route", "traffic", "actor", "waypoint",
                 "get_transform", "get_location", "native", "drivers"}
    allowed_imports = {"__future__", "dataclasses", "math", "numpy", "india_camera_geometry", "india_hybrid_astar", "heapq", "time",
                       "india_rl_speed"}
    # The learned speed's module also reads its policy file (hashlib, json, pathlib); nothing else.
    for name in ("india_camera_driver.py", "india_camera_geometry.py", "india_hybrid_astar.py", "india_rl_speed.py"):
        if name == "india_rl_speed.py":
            allowed_imports = allowed_imports | {"hashlib", "json", "pathlib"}
        tree = ast.parse((ROOT / "integration" / name).read_text(encoding="utf-8"))
        imports, words = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Name):
                words.add(node.id)
            elif isinstance(node, ast.Attribute):
                words.add(node.attr)
            elif isinstance(node, ast.arg):
                words.add(node.arg)
        bad = (imports - allowed_imports) | (words & forbidden)
        check(f"isolation: {name} has no simulator access", not bad, "found " + ", ".join(sorted(bad)) if bad else "")


def viewer_names_check():
    """8 Nov: the viewer crashed when TAB showed the spot camera: main() assigned a local `latest`
    (the rear picture), hiding the module's latest() it calls there."""
    import symtable
    path = ROOT / "integration" / "india_combined_scene.py"
    top = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    functions = {s.get_name() for s in top.get_symbols() if s.is_assigned() and any(
        c.get_name() == s.get_name() and c.get_type() == "function" for c in top.get_children())}
    clashes = []
    def walk(table):
        for child in table.get_children():
            if child.get_type() == "function":
                clashes.extend(f"{child.get_name()}: {s.get_name()}" for s in child.get_symbols()
                               if s.get_name() in functions and s.is_local() and not s.is_parameter())
            walk(child)
    walk(top)
    check("viewer: no function hides one of the scene's own functions with a local variable", not clashes, ", ".join(clashes))


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def protocol_check():
    from setup.camera_driving_service import serve
    from india_perception_client import PerceptionClient
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        session = {"session": "s" * 32, "token": "t" * 64, "service_max_seconds": 30, "stop_file": str(scratch / "stop.json"),
                   "road": {"port": free_port(), "folder": str(scratch / "road")},
                   "detector": {"port": free_port(), "folder": str(scratch / "detector")}}
        (scratch / "session.json").write_text(json.dumps(session))
        fake = lambda folder: ({"weights": 1}, {"fake": True})
        road_infer = lambda model, rgb: ({"inference_ms": 1.}, (rgb[:512, :512, 0] > 100).astype(np.uint8).tobytes() + bytes(512 * 512))
        det_infer = lambda model, rgb: ({"inference_ms": 1., "boxes": 1}, np.array([[1, 2, 3, 4, .9, 2]], np.float32).tobytes())
        threads = [threading.Thread(target=serve, args=(scratch / "session.json", role, fake, infer, lambda m: "same"), daemon=True)
                   for role, infer in (("road", road_infer), ("detector", det_infer))]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 10
        while not all((scratch / role / "worker_ready.json").exists() for role in ("road", "detector")):
            if time.monotonic() > deadline:
                check("protocol: stand-in workers start", False)
                return
            time.sleep(.05)
        client = PerceptionClient(session)
        image = np.zeros((540, 960, 3), np.uint8)
        image[:200, :300] = 200
        client.submit(7, image, 1.)
        answer = client.result(7, timeout=3.)
        good = answer.ok and answer.mask.shape == (512, 512) and answer.mask[:100, :100].all() and not answer.mask[300:, 400:].any() \
            and answer.boxes.shape == (1, 6) and int(answer.boxes[0, 5]) == 2
        check("protocol: frame round trip through both workers", good, answer.error or "")
        client.submit(8, image, 1.05, rear=image[::-1].copy())
        answer = client.result(8, timeout=3.)
        check("protocol: rear picture goes to the detector too", answer.ok and answer.rear_boxes is not None
              and answer.rear_boxes.shape == (1, 6), answer.error or str(answer.timings.get("rear_error", "")))
        client.submit(10, image, 1.08, extra=("left", image[:, ::-1].copy()))
        answer = client.result(10, timeout=3.)
        check("protocol: a side picture goes to the detector too, answered under its own view",
              answer.ok and answer.rear_boxes is None and set(answer.side_boxes or {}) == {"left"}
              and answer.side_boxes["left"].shape == (1, 6) and client.sides["left"]["answered"] == 1,
              answer.error or str(answer.timings.get("left_error", "")))
        (scratch / "stop.json").write_text("{}")
        for thread in threads:
            thread.join(timeout=5)
        client.submit(9, image, 1.1)
        answer = client.result(9, timeout=1.5)
        check("protocol: stopped workers make the frame unavailable", not answer.ok, answer.error or "")
        client.close()
        status = [json.loads((scratch / role / "worker_status.json").read_text()) for role in ("road", "detector")]
        check("protocol: workers stop cleanly on their marker", all(s["status"] == "completed" and s["model_weights_unchanged"] for s in status))


def replay_check():
    """The driver's target line on 120 recorded town frames vs the true keep-left lane centre (map)."""
    import carla
    import pygame
    probe = ROOT / "runs/camera_driving_probe_20261106_01"
    meta = json.loads((probe / "truth.json").read_text())
    run = ROOT / meta["run"]
    detections = json.loads((probe / "detections.json").read_text())["sizes"]["1280_fp16"]["frames"]
    boxes_by_frame = {d["frame"]: np.array([b + [c, k] for b, c, k in zip(d["boxes"], d["conf"], d["cls"])], np.float32).reshape(-1, 6)
                      for d in detections}
    poses = {json.loads(l)["source_frame"]: json.loads(l)["streams"]["rgb"]["camera_to_world"] for l in (run / "camera_frames.jsonl").open()}
    town = carla.Map("replay", (run / "map.xodr").read_text())
    errors = {}
    for row in meta["frames"]:
        frame = row["frame"]
        surface = pygame.image.load(str(probe / "road_masks" / f"mask_{frame:08d}.png"))
        mask = (pygame.surfarray.array2d(surface).T > 0).astype(np.uint8) if surface.get_bitsize() == 8 else \
            (pygame.surfarray.array3d(surface)[..., 0].T > 0).astype(np.uint8)
        driver = fresh_driver()
        driver.captured(frame, 0., 0.)
        picture = pygame.surfarray.array3d(pygame.image.load(str(run / row["rgb"]))).swapaxes(0, 1)
        plan = driver.observe(Perception(frame, True, mask=mask, boxes=boxes_by_frame.get(frame), rgb=picture), 0.)
        if plan is None or plan.lost:
            errors.setdefault(row["district"] if not row["in_junction"] else "junction", []).append(None)
            continue
        pose = np.array(poses[frame])
        fwd, right = pose[:3, 0], pose[:3, 1]
        origin = pose[:3, 3] - 1.35 * fwd - .25 * right
        for ahead in (6., 10.):
            point = origin + ahead * fwd
            wp = town.get_waypoint(carla.Location(x=float(point[0]), y=float(point[1]), z=0.), True, carla.LaneType.Driving)
            heading = math.atan2(fwd[1], fwd[0])
            candidates = [w for w in (wp, wp.get_left_lane(), wp.get_right_lane()) if w is not None and w.lane_type == carla.LaneType.Driving]
            same = [w for w in candidates if math.cos(math.radians(w.transform.rotation.yaw) - heading) > 0]
            if not same:
                continue
            lane = min(same, key=lambda w: w.transform.location.distance(carla.Location(*map(float, point))))
            while True:   # keep left: the leftmost lane in this direction
                left = lane.get_left_lane()
                if left is None or left.lane_type != carla.LaneType.Driving or left.lane_id * lane.lane_id <= 0:
                    break
                lane = left
            centre = lane.transform.location
            true_y = (np.array([centre.x, centre.y, 0.]) - origin) @ right
            mine = float(np.interp(ahead, plan.path[:, 0], plan.path[:, 1]))
            key = (row["district"] if not row["in_junction"] else "junction", ahead)
            errors.setdefault(key, []).append(mine - true_y)
    summary = {}
    for key, values in sorted((k, v) for k, v in errors.items() if isinstance(k, tuple)):
        e = np.abs(np.array(values))
        summary[f"{key[0]}@{key[1]:.0f}m"] = {"n": len(e), "median_m": float(np.median(e)), "p90_m": float(np.percentile(e, 90)),
                                              "median_signed_m": float(np.median(values))}
        print(f"   replay {key[0]:20s} {key[1]:4.0f} m  n={len(e):3d}  |error| median {np.median(e):.2f} m  p90 {np.percentile(e, 90):.2f} m"
              f"  signed median {np.median(values):+.2f} m", flush=True)
    urban = [abs(v) for k, vals in errors.items() if isinstance(k, tuple) and k[0] in ("market_street", "residential_colony", "irregular_junction", "junction")
             for v in vals]
    check("replay: target line within 0.4 m median on market/colony/junction streets", np.median(urban) <= .4,
          f"median {np.median(urban):.2f} m over {len(urban)} samples")
    (probe / "replay_result.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    geometry_checks()
    synthetic_checks()
    closed_loop_checks()
    pedestrian_checks()
    wedge_checks()
    isolation_check()
    viewer_names_check()
    protocol_check()
    if args.replay:
        replay_check()
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + ("" if not failed else "; failed: " + "; ".join(failed)))
    raise SystemExit(1 if failed else 0)
