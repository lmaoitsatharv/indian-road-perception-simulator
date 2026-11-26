"""Camera-only driver: road-model mask + detector boxes -> steering, throttle, brake.

The car "sees" only its own front camera through two trained models, plus its
own speedometer and yaw-rate gyro and its own size/steering limits. This module
has no handle to the simulator world, map, routes, junctions or other actors;
setup/verify_camera_driving.py checks its source for that.

Lane keeping follows the left kerb (India drives on the left): the target line
is a fixed distance right of the road model's left road edge, fitted over the
next 4-14 m. Following the left kerb through a junction takes the leftmost exit.
Where a paved shoulder lies beside the lane, the road model counts it as road
once the car is near it, so the left edge also checks the picture for the
bright-to-dark step where the asphalt starts (7 Nov village-road probe).
Detected objects are placed on flat ground and slow or stop the car when they
sit on its path. See docs/CAMERA_DRIVING_DESIGN_20261106.md.

Passing a stopped bus or truck: the car first checks the passing lane ahead
(detector) and behind (a rear camera, run through the same detector only around
passes), then follows one planned S-curve out past the stopped vehicle. Only
that vehicle is ignored while passing (matched by where it appears in the
picture); anything else on the path stops the car, and if it stays stopped the
car gives up and backs out the way it came, watching the rear camera.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from india_camera_geometry import CameraGeometry, GroundGrid, MASK_SIZE
from india_hybrid_astar import disc_offsets, disc_radius, occupancy as astar_occupancy, plan as astar_plan
import india_rl_speed as rl

DT = .05
MASK_SIZE_ROWS = MASK_SIZE_COLS = MASK_SIZE
CLASSES = ("person", "rider", "car", "motorcycle", "bicycle", "autorickshaw", "bus", "truck", "animal",
           "traffic_light", "traffic_sign", "other_vehicle")
LENGTH_M = {"person": .6, "animal": 1.5, "rider": 1.8, "motorcycle": 1.8, "bicycle": 1.8, "autorickshaw": 2.8,
            "car": 4., "other_vehicle": 4., "truck": 7., "bus": 10.}
# A box's bottom edge sits at the wheels; the body reaches this much closer.
# The bus value was measured on 6 Nov (box-bottom estimate about 2 m too far).
OVERHANG_M = {"car": .8, "other_vehicle": .8, "truck": 1.2, "bus": 2., "autorickshaw": .5,
              "motorcycle": .3, "bicycle": .3, "rider": .3}
# Typical width seen from behind or in front (about 85% of the real body, so a width-based
# distance errs short). A raised bumper makes the box-bottom distance read far, up to 2x
# for the CARLA bus on 6 Nov probe frames; the nearer of the two estimates is used.
# The CARLA bus (fusorosa) is 3.94 m wide, far wider than an Indian bus; 3.0 still errs short.
REAR_WIDTH_M = {"car": 1.5, "other_vehicle": 1.7, "autorickshaw": 1.2, "bus": 3.0, "truck": 2.1}
LARGE = ("bus", "truck")
FOUR_WHEELERS = ("car", "other_vehicle", "truck", "bus", "autorickshaw")
# Typical height of the tall classes: one this close rises well above the horizon, hiding the sky.
TALL_M = {"bus": 3.0, "truck": 2.5}


def _rect_gap(o, px, py):
    """Distance from a point to an obstacle's footprint rectangle (0 inside)."""
    dx = max(o["x"] - px, 0., px - (o["x"] + o["length"]))
    dy = max(o["y_left"] - py, 0., py - o["y_right"])
    return math.hypot(dx, dy)


def _occ_free(occ, px, py):
    """A point is free on a Hybrid A* occupancy grid (behind its first row counts as free, as in plan())."""
    i = int(math.floor((px - occ.xs[0] + occ.cell / 2) / occ.cell))
    j = int(math.floor((py - occ.ys[0] + occ.cell / 2) / occ.cell))
    if not 0 <= j < occ.blocked.shape[1] or i >= occ.blocked.shape[0]:
        return False
    return i < 0 or not occ.blocked[i, j]


def _iou(a, b):
    """Overlap of two image boxes (x1, y1, x2, y2) over their union."""
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.
PEOPLE = ("person", "animal")
IGNORED = ("traffic_light", "traffic_sign")
REASONS = {
    "cruise": "Cruising on the road it sees",
    "speed_limit": "At the posted speed limit",
    "junction_ahead": "Easing down before the junction",
    "learned_slow": "Learned speed: slower than allowed",
    "starting": "Waiting for the first camera result",
    "wide_road_area": "Wide road area: slowing down",
    "uncertain_road": "Road edge unclear: slowing down",
    "lost_road": "Left kerb lost: creeping on",
    "lost_road_stop": "Lost the road: stopped",
    "curve": "Slowing for the curve",
    "vehicle_ahead": "Vehicle ahead on its path",
    "person_ahead": "Person ahead on its path",
    "person_near_path": "Person close to its path",
    "road_ends": "Road ahead not visible",
    "passing": "Passing a stopped vehicle on its right",
    "backing_out": "Backing out of a blocked pass",
    "backing_wait": "Waiting to back up",
    "wedged": "Stuck: not moving with the throttle on",
    "shifting_gear": "Stopping to change direction",
    "junction_turn": "Turning at the junction (planned path)",
    "junction_fallback": "Turning at the junction (route line)",
    "waiting_turn": "Waiting for its turn at the junction",
    "arriving": "Arriving at the destination",
    "arrived": "Arrived",
    "uturn_ahead": "Slowing to turn round",
    "uturn_wait": "Waiting for a gap to turn round",
    "uturn_forward": "Turning round: forward",
    "uturn_reverse": "Turning round: backing up",
    "no_route": "No route to the destination",
    "emergency_vehicle": "Braking hard for a vehicle",
    "emergency_person": "Braking hard for a person",
    "perception_unavailable": "Perception unavailable: stopping",
}


@dataclass
class DriverSettings:
    cruise: float = 6.
    wide_area: float = 3.5
    lost_road: float = 2.5
    near_person: float = 2.5
    keep_left_m: float = 1.5
    lateral_accel: float = 2.
    brake_decel: float = 2.5
    vehicle_gap: float = 4.
    large_vehicle_gap: float = 6.   # buses and trucks: room to see and steer around one that has stopped
    person_gap: float = 5.
    road_end_gap: float = 3.
    emergency_m: float = 2.5
    vehicle_margin: float = .3
    person_margin: float = .5       # a person this close to the car's side is on its path: brake
    person_near_m: float = 1.       # within this of the car's side (beyond the margin): slow down
    person_near_range: float = 15.
    hold_s: float = .3
    crossing_speed: float = .5      # a person moving towards the path at least this fast (m/s) is crossing
    near_vehicle_m: float = 4.      # a vehicle hit this close ahead of the front is kept ...
    near_vehicle_hold_s: float = 2.5   # ... this long (the detector can lose one that fills the picture)
    lost_after_s: float = 1.5
    lost_stop_after_s: float = 4.
    unavailable_after_s: float = .5
    blend: float = .5
    pass_after_stopped_s: float = 3.   # look for a way past a stopped bus or truck after waiting this long
    pass_small: float = 0.             # 1: also pass stopped cars and two-wheelers (off: they are usually queuing;
                                       # run02 and 7 Nov run06 both started passing a motorcycle waiting at a junction)
    pass_after_stopped_small_s: float = 15.   # ...and then only after this long
    pass_window_s: float = 5.          # ...and keep looking for this long once moving again
    passing: float = 2.5               # speed while moving out around a stopped vehicle
    pass_clearance: float = .7         # planned gap to its side while passing (about 0.4 m in the 6 Nov synthetic check)
    pass_clearance_large: float = 1.3  # buses and trucks: their width is uncertain from a box (the CARLA bus is 3.9 m)
    pass_clearance_large_min: float = .9   # ...stepping down to this if the road further out is not visibly free
    pass_look_beyond_m: float = 10.    # traffic in the passing lane up to this far past the stopped vehicle also waits
    pass_abort_after_s: float = 3.     # stopped this long mid-pass: give up and back out the way it came
    pass_retry_after_s: float = 6.     # after giving up, wait at least this long before trying again
    mirror_range_m: float = 30.        # rear camera: a vehicle this close behind in the passing lane waits
    blind_spot_s: float = 8.           # a vehicle seen close behind in the passing lane may be alongside, out of
                                       # both cameras' view, for this long unless it shows up ahead first
    backing: float = 1.                # reverse speed when backing out of a pass
    rear_gap: float = 3.               # stop backing this far short of anything behind
    # Wedged: the throttle on and the car not moving, against a kerb or something else the cameras do not show
    # (26 Nov run: it cut the inside of a sharp left at Market junction and pushed on a kerb for 29 s, until the
    # town's stall reset lifted it). It backs off the way it came and tries again.
    wedged_after_s: float = 2.
    wedged_back_m: float = 1.5
    wedged_tries: int = 3              # back-offs a minute at most; then it stops and waits
    min_confidence: float = .25
    asphalt_edge: float = 1.           # 1: also find the lane edge where a brighter paved shoulder meets the asphalt
    asphalt_ratio: float = 1.25        # the shoulder band is at least this much brighter than the asphalt beside it
    asphalt_hold_s: float = 1.5        # keep a found asphalt edge (odometry) this long between frames that miss it
    # Destinations (the sat-nav's route; docs/CAMERA_DRIVING_DESTINATIONS_DESIGN_20261108.md)
    junction_speed: float = 3.5        # along a Hybrid A* path through a junction
    junction_fallback_speed: float = 2.5   # along the route line when no path was found
    junction_replan_s: float = .5
    junction_lookahead: float = 4.     # pure pursuit along the junction path
    junction_fallback_give_up_s: float = 10.
    arrive_slow_m: float = 30.         # within this of the destination: slow, and keep closer to the kerb
    arrive_speed: float = 3.
    arrive_keep_left_m: float = 1.1
    arrive_stop_m: float = 1.
    hold_gap_m: float = 3.             # aim to stop this far short of a junction until it is the car's turn; the speed
                                       # controller runs about 1.2 m past a stopping curve, so it stops ~1.8 m short
    no_pass_near_destination_m: float = 40.
    # U-turns on a two-way street, driven as a three-point turn (or more moves) between the road's edges
    uturn_speed: float = .9            # creeping speed of each move
    uturn_steer: float = 1.            # full lock for the turn (normal driving keeps within 0.7)
    uturn_edge_m: float = .35          # end a move this far from the road's edge (any corner of the car)
    uturn_clear_ahead_m: float = 35.   # no moving vehicle this far ahead (either lane) before starting
    uturn_clear_people_m: float = 12.  # nor a person this close ahead
    uturn_clear_s: float = 1.          # ...for this long
    uturn_reverse_m: float = 4.        # back up at most this far in one move
    uturn_give_up_s: float = 80.       # then give the turn up where it stands (the sat-nav re-plans); under the
                                       # town's 90 s stall reset, which would otherwise come first (18 Nov run)
    uturn_rear_seen_m: float = 1.5     # rear camera, turning round: a box nearer than this counts only if something was
                                       # seen coming (read farther out), or the camera has just started...
    uturn_rear_reveal_m: float = .5    # ...and the car has not since driven this far forward (a real one would then show)
    uturn_coming_m: float = 20.        # hold a move while anything on the road this near is coming towards the car...
    uturn_coming_speed: float = 1.2    # ...at this speed or more (any camera: across the street, the side ones see along it)
    # Learned speed (docs/RL_SPEED_DESIGN_20261112.md): a trained policy picks a fraction of the shield's
    # speed, the shield being every safety constraint above plus the sat-nav's posted limit.
    rl_speed: float = 0.               # 1: on (needs models/rl_speed/policy.npz)
    rl_speed_limits: float = 1.        # with rl_speed: keep to the sat-nav's posted limits (0: the cruise speed is the limit)
    rl_limit_hold: float = 1.          # with rl_speed: raise the controller's set-point so the car holds a posted limit
                                       # (it otherwise settles 0.5-0.95 m/s under its target; never while at the limit)
    rl_junction_entry: float = 1.      # with rl_speed: once its turn is granted and the road is not wide or unclear, reach
                                       # the junction speed at the junction's entry instead of by the start of the 20 m
                                       # junction zone. Not while waiting for the turn: in town drive s42 a fast approach
                                       # to a held turn at a wide junction mouth stopped 1.2 m over the centre line

    @classmethod
    def from_dict(cls, values):
        known = {k: float(v) for k, v in (values or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Perception:
    """One processed camera frame from the two GPU workers."""
    frame: int
    ok: bool
    mask: np.ndarray | None = None          # 512x512 uint8, 1 = road
    boxes: np.ndarray | None = None         # (N, 6): x1 y1 x2 y2 confidence class
    timings: dict = field(default_factory=dict)
    error: str | None = None
    probability: np.ndarray | None = None   # 512x512 uint8 road probability x255 (display only)
    rear_boxes: np.ndarray | None = None    # (N, 6) detector boxes from the rear camera, when it was asked for
    side_boxes: dict | None = None          # {"left" or "right": (N, 6) boxes} from the side cameras that ran this frame
    rgb: np.ndarray | None = None           # the front camera picture itself (540x960x3), for the asphalt edge


@dataclass
class Plan:
    """What the driver concluded from one frame, in that frame's vehicle coordinates."""
    frame: int
    sim_time: float
    rows: list                 # (x, yL or None, yR or None, start_y or None)
    fit_rows: int
    path: np.ndarray | None    # (n, 2) target line points
    wide: bool
    road_end_m: float | None   # along-path distance from the car's front
    obstacles: list            # dicts: class, conf, box, x, y_left, y_right, role, distance
    lost: bool
    road_end_x: float | None = None   # forward position (vehicle frame) where the road stops
    passing: bool = False             # a pass around a stopped vehicle is under way
    rejected: list = field(default_factory=list)   # detector boxes contradicted by the road mask
    rear: list | None = None          # rear-camera detections (vehicle frame; x < 0), when the rear view ran
    pass_note: str = ""               # why a pass is waiting, or how the last one ended
    asphalt_rows: list = field(default_factory=list)   # forward distances whose left edge moved to the asphalt edge
    nav_mode: str = "free"            # "free" (no destination), "lane" or "junction"
    junction_path: np.ndarray | None = None   # the path steered through a junction (this frame)
    junction_goal: tuple | None = None
    plan_found: bool | None = None    # this frame's Hybrid A* result, when it planned
    plan_ms: float | None = None
    side: dict | None = None          # side-camera detections this frame: {view: dicts with "corners" (vehicle frame)}


def to_vehicle(points, pose):
    px, py, psi = pose
    dx, dy = points[:, 0] - px, points[:, 1] - py
    c, s = math.cos(psi), math.sin(psi)
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=1)


def iou(a, b):
    """Overlap of two image boxes (x1, y1, x2, y2) as intersection over union."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def to_odometry(points, pose):
    px, py, psi = pose
    c, s = math.cos(psi), math.sin(psi)
    return np.stack([px + points[:, 0] * c - points[:, 1] * s, py + points[:, 0] * s + points[:, 1] * c], axis=1)


class CameraDriver:
    def __init__(self, geometry: CameraGeometry, *, half_length, half_width, front_m, wheelbase, max_steer,
                 settings: DriverSettings | None = None, names=CLASSES, rear_geometry: CameraGeometry | None = None,
                 side_geometry: dict | None = None):
        self.geometry, self.settings = geometry, settings or DriverSettings()
        self.rear_geometry = rear_geometry   # None: no rear camera (passes then skip the mirror check)
        self.side_geometry = dict(side_geometry or {})   # {"left"/"right": CameraGeometry}: cameras looking out sideways
        self.grid = GroundGrid(geometry)
        self.half_length, self.half_width, self.front = float(half_length), float(half_width), float(front_m)
        self.wheelbase, self.max_steer = float(wheelbase), float(max_steer)
        self.names = tuple(names)
        self.governor = rl.SpeedGovernor() if self.settings.rl_speed else None
        self.reset()

    # ---- state -----------------------------------------------------------------
    def reset(self):
        """Forget everything seen (used after the simulator lifts the car back onto a lane)."""
        self.pose = (0., 0., 0.)
        self.captures = {}
        self.path = None            # odometry-frame target line
        self.path_time = -1e9
        self.edge_time = None
        self.hits = []              # (sim_time, kind, odometry xy, label)
        self.people = []            # tracks of people seen: {"points": [(time, odometry x, y)], "seen": time}
        self.flags = {"wide": False, "uncertain": True, "time": -1e9}
        self.last_ok_time = None
        self.started = False
        self.speed = 0.
        self.steer = 0.
        self.accel_command = self.speed_integral = self.smoothed_accel = 0.
        self.previous_speed = self.previous_target = 0.
        self.plan = None
        self.last_output = None
        self.unavailable_frames = 0
        self.rejected = []
        self.phantoms = []          # (time, box) of wide boxes thrown out as phantoms, the last 2 s
        self.stopped_since = None
        self.pass_until = -1e9
        self.pass_hold = None       # committed pass: the stopped vehicle's right side (odometry) to keep clear of
        self.pass_cooldown_until = -1e9
        self.pass_note, self.pass_note_time = "", -1e9
        self.backing = None         # backing out of an abandoned pass (or off a kerb): the path (odometry) to retrace
        self.pushing_since = None   # throttle on and not moving since (sim_time), for the wedged check
        self.trail = []             # where the car's centre has been (odometry), every 0.25 m, the last 5 m
        self.wedge_log = []         # times it backed off being wedged
        self.reversing = False      # reverse gear commanded (the car knows its own gear)
        self.rear_obstacles, self.rear_time = None, None
        self.rear_hint = False
        self.blind_since = None     # a vehicle was last seen close behind in the passing lane (may be alongside)
        self.asphalt = None         # last asphalt edge found beside a shoulder: {"line": odometry points, "time"}
        self.nav = None             # the sat-nav's latest message (guide kept in odometry), None without a destination
        self.junction = None        # {"path": odometry (M, 2), "fallback", "planned_at", "blocked_since"} in a junction
        self.uturn = None           # a U-turn ahead or under way: {"phase", "frame" (odometry x, y, yaw), edges, ...}
        self.uturn_done_at = -1e9
        self.uturn_failed = False   # the last U-turn was given up (until the sat-nav has re-planned)
        self.seen = None            # (sim_time, every obstacle of the last frame: dicts with "corners" in odometry)
        self.seen_log = []          # (sim_time, odometry centres of the vehicles seen), the last 3 s
        self.rear_log = []          # the same for the rear camera
        self.rear_seen_at = None    # car pose (odometry) when something was last read close behind, or the rear camera started
        self.side_obstacles = {}    # view -> (sim_time, side-camera detections, with "odo" corners)
        self.track_log = []         # (sim_time, view, [(class, person, odometry points)]) from every camera, the last 3 s
        self.stay_behind = None     # after giving up a pass: hold position behind that vehicle (last seen time)
        # (sim_time, event, detail): passes started, given up, backed out, finished. Kept across
        # resets (7 Nov run05 lost its pass record when the car was lifted back onto its lane).
        self.events = getattr(self, "events", [])
        if self.events:
            self.events.append((round(float(getattr(self, "sim_time", 0.)), 2), "reset", "lifted back onto its lane"))
        self.sim_time = 0.
        self.rl_last = None         # the learned speed's last choice (cap, its reason, fraction), for the records
        self.over_since = None      # learned speed: when the car went over the posted limit (holding is then paused)
        self.hold_paused_until = -1e9
        if getattr(self, "governor", None) is not None:
            self.governor.reset()

    @property
    def wants_rear(self):
        """The rear camera is needed: waiting behind a stopped vehicle it may pass, passing, or backing up."""
        return self.rear_geometry is not None and (self.rear_hint or self.pass_hold is not None or self.backing is not None
                                                   or (self.uturn is not None and self.uturn["phase"] != "approach"))

    @property
    def wanted_views(self):
        """The cameras besides the front one the driver needs now: the rear one, the right one (the passing lane
        beside the car) around passes and backing up, and both side ones while turning round (the street then runs
        past the car's sides)."""
        turning = self.uturn is not None and self.uturn["phase"] != "approach"
        views = ["rear"] if self.wants_rear else []
        if "right" in self.side_geometry and (turning or self.rear_hint or self.pass_hold is not None or self.backing is not None):
            views.append("right")
        if "left" in self.side_geometry and turning:
            views.append("left")
        return views

    def navigate(self, message):
        """The sat-nav's message for this control step (None: no destination, drive as before).
        Keys: mode ("lane"|"junction"), guide (route line ahead, this vehicle frame), junction,
        hold_at_m (from the car's front to the junction start, when not its turn), arrive_in_m."""
        if message is None:
            self.nav = self.junction = None
            if self.uturn is not None and self.uturn["phase"] == "approach":
                self.uturn = None   # (a turn under way is finished: the car is across the road)
            return
        nav = dict(message)
        guide = np.asarray(nav.get("guide") if nav.get("guide") is not None else np.zeros((0, 2)), float).reshape(-1, 2)
        nav["guide_odo"] = to_odometry(guide, self.pose) if len(guide) else None
        self.nav = nav
        turn = nav.get("uturn")
        if turn is not None and (self.uturn is None or self.uturn["phase"] == "approach") \
                and self.sim_time - self.uturn_done_at > 10.:
            x, y, yaw = turn["pose"]
            origin = to_odometry(np.array([[x, y]], float), self.pose)[0]
            self.uturn = {"phase": "approach", "frame": (float(origin[0]), float(origin[1]), self.pose[2] + float(yaw)),
                          "left": float(turn["left_m"]), "right": float(turn["right_m"]), "across": float(turn["across_m"]),
                          "in_m": float(turn["in_m"])}
        elif turn is None and self.uturn is not None and self.uturn["phase"] == "approach":
            self.uturn = None

    def odometry(self, speed, yaw_rate, dt=DT):
        """Dead reckoning from the car's own speedometer and gyro (and its own gear)."""
        px, py, psi = self.pose
        psi += yaw_rate * dt
        signed = -speed if self.reversing else speed
        self.pose = (px + signed * math.cos(psi) * dt, py + signed * math.sin(psi) * dt, psi)
        self.speed = float(speed)
        if not self.trail or math.hypot(self.pose[0] - self.trail[-1][0], self.pose[1] - self.trail[-1][1]) >= .25:
            self.trail = self.trail[-19:] + [self.pose[:2]]
        if self.uturn is not None and self.uturn.get("axle") is not None:
            # Turning round on full lock the car pivots about its rear axle: track its centre that way for the
            # turn's edge checks (the centre-line dead reckoning above drifts 0.1 m or more per move there).
            ax, ay, apsi = self.uturn["axle"]
            half = self.wheelbase / 2
            rx, ry = ax - half * math.cos(apsi), ay - half * math.sin(apsi)
            apsi += yaw_rate * dt
            rx, ry = rx + signed * math.cos(apsi) * dt, ry + signed * math.sin(apsi) * dt
            self.uturn["axle"] = (rx + half * math.cos(apsi), ry + half * math.sin(apsi), apsi)

    def captured(self, frame, speed, yaw_rate, dt=DT):
        """Remember where the car was when its camera took this frame (one step ahead of the last odometry)."""
        px, py, psi = self.pose
        psi += yaw_rate * dt
        signed = -speed if self.reversing else speed
        self.captures[frame] = (px + signed * math.cos(psi) * dt, py + signed * math.sin(psi) * dt, psi)
        for old in [f for f in self.captures if f < frame - 40]:
            del self.captures[old]

    # ---- perception -> plan -----------------------------------------------------
    def observe(self, perception: Perception, sim_time):
        if not perception.ok or perception.mask is None:
            self.unavailable_frames += 1
            self.plan = None
            return None
        pose = self.captures.pop(perception.frame, self.pose)
        self.last_ok_time, self.started = sim_time, True
        plan = self._plan(perception, pose, sim_time)
        self.plan = plan
        return plan

    def _previous_y(self, pose):
        """The last target line, carried into the frame at `pose`, as a function of x."""
        if self.path is None:
            return lambda x: np.zeros_like(np.asarray(x, float))
        points = to_vehicle(self.path, pose)
        order = np.argsort(points[:, 0])
        xs, ys = points[order, 0], points[order, 1]
        return lambda x: np.interp(x, xs, ys)

    def _scan(self, road, row, start, step, max_gap=3):
        """Walk from `start` along a grid row; return the last road column before a real edge,
        or None when the camera's view ends first."""
        valid = self.grid.valid[row]
        width = road.shape[1]
        last, j, gap = start, start, 0
        while True:
            j += step
            if j < 0 or j >= width or not valid[j]:
                return None
            if road[row, j]:
                last, gap = j, 0
            else:
                gap += 1
                if gap > max_gap:
                    return last

    def _start_column(self, road, row, y, reacquire=False):
        """A road cell to scan from: the previous path, or up to 0.5 m either side of it
        (mask specks); after a frame without a kerb fit, up to 2.5 m either side, to find
        the road again. (Rows blocked by a detected vehicle are skipped by the caller, so
        this never steers around one; passing is a separate, explicit manoeuvre.)"""
        grid = self.grid
        col = grid.col_of(y)
        if road[row, col]:
            return col
        reach = int(round((2.5 if reacquire else .5) / grid.cell))
        for k in range(1, reach + 1):
            if col + k < road.shape[1] and road[row, col + k]:
                return col + k
            if col - k >= 0 and road[row, col - k]:
                return col - k
        return None

    def _asphalt_step(self, road, luma, row, left):
        """Inside the road model's left edge, the bright-to-dark step where a paved shoulder
        meets the asphalt: 0.4-1.8 m in, road on both sides, 0.5 m bands. Its lateral position, or None."""
        best, where = 0., None
        for k in range(left + 4, min(left + 19, road.shape[1] - 6)):
            if not road[row, k - 5:k + 5].all():
                continue
            ratio = float(luma[row, k - 5:k].mean()) / max(float(luma[row, k:k + 5].mean()), 1.)
            if ratio > best:
                best, where = ratio, k
        if where is None or best < self.settings.asphalt_ratio:
            return None
        return float(self.grid.ys[where] - self.grid.cell / 2)

    def _fit(self, rows):
        """Robust quadratic (or line) through (x, target y), near rows weighted up."""
        data = np.array([(x, y) for x, y in rows], float)
        if len(data) < 2:
            return None, 0
        weights = np.where(data[:, 0] <= 9., 1., .4)
        def solve(mask):
            n = int(mask.sum())
            if n < 2:
                return None
            span = data[mask, 0].max() - data[mask, 0].min()
            degree = 2 if n >= 4 and span >= 4. else 1
            return np.polyfit(data[mask, 0], data[mask, 1], degree, w=weights[mask])
        near = data[:, 0] <= 9.
        keep = near if near.sum() >= 3 else np.ones(len(data), bool)
        coef = solve(keep)
        if coef is None:
            return None, 0
        keep = np.abs(np.polyval(coef, data[:, 0]) - data[:, 1]) <= 1.2
        for _ in range(2):
            coef2 = solve(keep)
            if coef2 is None:
                break
            coef = coef2
            keep = np.abs(np.polyval(coef, data[:, 0]) - data[:, 1]) <= .8
        if len(coef) == 2:
            coef = np.r_[0., coef]
        self._fit_reach = float(data[keep, 0].max()) if keep.any() else float(data[:, 0].max())
        return coef, int(keep.sum())

    def _phantom(self, mask, x1, y1, x2, y2, now=None):
        """A wide box with mostly road at its base: something standing on the road hides the
        road there, so the detector's box disagrees with the road model (6 Nov: a frame-wide
        'bus' over an empty junction). Narrow boxes (people, two-wheelers) are not judged.
        A box thrown out in the last 2 s in about the same place stays out while a quarter of its
        base still shows road (8 Nov: the car turned and half the base became sand)."""
        g = self.geometry
        if mask is None or (x2 - x1) < .15 * g.width:
            return False
        if (x1 <= 1 or x2 >= g.width - 1) and (x2 - x1) < .45 * g.width:
            # Cut off at a side: something close beside the car, often at an angle, so its
            # bounding box takes in road in front of it (7 Nov run06 threw out a real car here,
            # about a quarter of the picture wide). A box spanning most of the picture is still
            # judged: run11 had a phantom 'bus' over three quarters of it, cut at the right edge.
            return False
        w, h = x2 - x1, y2 - y1
        r0, r1 = (np.array([y2 - .12 * h, y2]) * MASK_SIZE_ROWS / g.height).astype(int)
        c0, c1 = (np.array([x1 + .25 * w, x2 - .25 * w]) * MASK_SIZE_COLS / g.width).astype(int)
        strip = mask[max(0, r0):max(r0 + 1, min(MASK_SIZE_ROWS, r1)), max(0, c0):max(c0 + 1, min(MASK_SIZE_COLS, c1))]
        if strip.size == 0:
            return False
        road = float(strip.mean())
        if now is None:
            return road > .6
        self.phantoms = [(t, b) for t, b in self.phantoms if now - t <= 2.]
        phantom = road > .6 or (road >= .25 and any(_iou(b, (x1, y1, x2, y2)) >= .6 for _, b in self.phantoms))
        if phantom:
            self.phantoms.append((now, (x1, y1, x2, y2)))
        return phantom

    def _base_contact(self, mask, x1, y1, x2, y2):
        """The image columns (u_lo, u_hi) where the road model sees no road along the box's base:
        where the object stands on the ground. None when it sees road all along. A vehicle in the
        next lane seen at an angle has its far corner as the box's inner edge, with road visible
        under that part of the box at its base (8 Nov town runs)."""
        g = self.geometry
        h = y2 - y1
        r0, r1 = (np.array([y2 - .12 * h, y2]) * MASK_SIZE_ROWS / g.height).astype(int)
        c0 = max(0, int(x1 * MASK_SIZE_COLS / g.width))
        c1 = min(MASK_SIZE_COLS, int(math.ceil(x2 * MASK_SIZE_COLS / g.width)))
        band = mask[max(0, r0):max(r0 + 1, min(MASK_SIZE_ROWS, r1)), c0:c1]
        if band.size == 0:
            return None
        hidden = np.r_[False, band.mean(axis=0) < .5, False]
        edges = np.flatnonzero(np.diff(hidden.astype(int)))
        runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 3]   # at least 3 mask columns
        if not runs:
            return None
        return (c0 + runs[0][0]) * g.width / MASK_SIZE_COLS, (c0 + runs[-1][1]) * g.width / MASK_SIZE_COLS

    def _sky_through(self, rgb, name, x1, y1, x2, y2):
        """A bus or truck box whose own distance puts the body well above the horizon, yet the picture
        there is plain sky: nothing that tall stands there (8 Nov: phantom 'bus' boxes 3-4 m away while
        facing the open desert in junctions; on the saved views real ones measured luma std 46-63 in
        that band, the phantoms about 7)."""
        g = self.geometry
        if rgb is None or name not in TALL_M or (x2 - x1) < .15 * g.width:
            return False
        gx, gy = g.ground_from_pixel(np.array([(x1 + x2) / 2]), np.array([min(y2, g.height - .5)]))
        if not np.isfinite(gx).all():
            return False
        _, top, _ = g.project(gx, gy, np.array([TALL_M[name]]))
        _, horizon, _ = g.project(np.array([1e4]), np.array([g.mount[1]]), np.array([g.mount[2]]))
        # Only the band just above the horizon (at most 100 rows): higher up a real sky darkens and has clouds
        # (9 Nov: a box from the top of the picture tested the whole sky and kept a phantom 1.8 m away).
        r0, r1 = int(max(y1 + 4, top[0] + 8, horizon[0] - 100, 0)), int(min(horizon[0] - 8, y2))
        if r1 - r0 < 30:
            return False
        w = x2 - x1
        band = np.asarray(rgb[r0:r1, int(x1 + .2 * w):int(x2 - .2 * w), :3], float).mean(axis=2)
        if band.shape[1] < 10:
            return False
        def plain(strip):
            return strip.std() < 15. and np.abs(np.diff(strip, axis=1)).mean() + np.abs(np.diff(strip, axis=0)).mean() < 4.
        if not plain(band):
            return False
        # ... and at one of its sides the picture runs on unchanged across the box's edge: the sky goes on
        # through it. A plain truck side is plain too, but differs from the sky beside it (final review, 9 Nov).
        # The edge, not the middle, is compared: the sky brightens towards the sun, ±15 luma across a wide box
        # (destinations17).
        margin = int(max(20, .1 * w))
        for outside, inside in (((int(x1) - margin, int(x1) - 2), (int(x1) + 2, int(x1) + margin)),
                                ((int(x2) + 2, int(x2) + margin), (int(x2) - margin, int(x2) - 2))):
            o0, o1 = max(0, outside[0]), min(g.width, outside[1])
            if o1 - o0 < 10:
                continue
            beside = np.asarray(rgb[r0:r1, o0:o1, :3], float).mean(axis=2)
            within = np.asarray(rgb[r0:r1, inside[0]:inside[1], :3], float).mean(axis=2)
            if plain(beside) and plain(within) and abs(beside.mean() - within.mean()) < 8.:
                return True
        return False

    def _near_road_edge(self, road, step, x_from=None):
        """Where the visible road ends on one side (step -1 left, +1 right) in the rows from x_from
        (default: the nearest the camera sees) to 1.3 m beyond, scanning out from straight ahead;
        the median over rows, or None."""
        grid, edges = self.grid, []
        near = max(self.geometry.nearest_visible_ground(), x_from or 0.)
        for x in np.arange(near + .1, near + 1.31, .2):
            r, start = grid.row_of(x), grid.col_of(0.)
            if not road[r, start]:
                return None   # something straight ahead: no claim about the side
            j = self._scan(road, r, start, step)
            if j is not None:
                edges.append(float(grid.ys[j] + step * grid.cell / 2))
        return float(np.median(edges)) if len(edges) >= 3 else None

    def _boxes_to_obstacles(self, boxes, mask=None, road=None, now=None, rgb=None):
        g, out = self.geometry, []
        self.rejected = []
        if boxes is None:
            return out

        for x1, y1, x2, y2, conf, cls in np.asarray(boxes, float).reshape(-1, 6):
            name = self.names[int(cls)] if 0 <= int(cls) < len(self.names) else "other_vehicle"
            if conf < self.settings.min_confidence or name in IGNORED:
                continue
            if self._phantom(mask, x1, y1, x2, y2, now) or self._sky_through(rgb, name, x1, y1, x2, y2):
                self.rejected.append({"class": name, "conf": float(conf), "box": [float(x1), float(y1), float(x2), float(y2)]})
                continue
            bottom = min(y2, g.height - .5)
            gx, gy = g.ground_from_pixel(np.array([x1, x2, (x1 + x2) / 2]), np.array([bottom] * 3))
            if not np.isfinite(gx).all() or gx[2] > 45:
                continue
            near_x = float(gx[2]) - OVERHANG_M.get(name, 0.)
            y_left, y_right = float(min(gy[0], gy[1])), float(max(gy[0], gy[1]))
            u_lo, u_hi = x1, x2
            if mask is not None and name in FOUR_WHEELERS and y2 < g.height - 1:
                # Its sides where it meets the ground: road visible at the base is not under it.
                contact = self._base_contact(mask, x1, y1, x2, y2)
                if contact is not None:
                    cx_, cy_ = g.ground_from_pixel(np.array(contact), np.array([bottom] * 2))
                    if np.isfinite(cy_).all():
                        u_lo, u_hi = contact
                        y_left, y_right = max(y_left, float(cy_.min())), min(y_right, float(cy_.max()))
            side = y_right   # its right side as seen at its near face (used to plan a pass)
            cut = x1 <= 1 or x2 >= g.width - 1 or y2 >= g.height - 1
            if name in REAR_WIDTH_M and not cut and (u_hi - u_lo) < 1.6 * (y2 - y1):
                # Camera-forward distance to its near face, from that face's width where it meets the
                # ground (a box taking in the side too read a van 2 m nearer and 1.5 m further in).
                forward = g.f * REAR_WIDTH_M[name] / max(1., u_hi - u_lo)
                if g.mount[0] + forward < near_x:
                    near_x = g.mount[0] + forward
                    side = g.mount[1] + (u_hi - g.cx) * forward / g.f
                    # The footprint keeps both lateral estimates (the cautious union).
                    y_left = min(y_left, g.mount[1] + (u_lo - g.cx) * forward / g.f)
                    y_right = max(y_right, side)
            if road is not None and (x1 <= 1 or x2 >= g.width - 1):
                # Cut off at a side: it is beside the car, seen at an angle, so its box joins its
                # far corner's column to its near corner's row (or its base is out of view) and
                # places it too far in: a roadside cart read 0.5-0.9 m inside the car's path
                # (7 Nov run05, run11, run12). Where the road model sees unbroken road from straight
                # ahead out to an edge, at the object's own distance, the object stands beyond it.
                step = -1 if x1 <= 1 else 1
                base = None if y2 >= g.height - 1 else float(gx[2])
                edge = self._near_road_edge(road, step, base)
                if edge is not None and step < 0 and edge < y_right:
                    y_right = side = edge
                elif edge is not None and step > 0 and edge > y_left:
                    y_left = edge
            out.append({"class": name, "conf": float(conf), "box": [float(x1), float(y1), float(x2), float(y2)],
                        "x": near_x, "y_left": y_left, "y_right": y_right, "side_right": side,
                        "length": LENGTH_M.get(name, 4.), "person": name in PEOPLE, "role": "ignored", "distance": None})
        return out

    def _rear_obstacles(self, boxes):
        """Rear-camera boxes on flat ground behind the car (x < 0). The near face is the
        nearer of the box-bottom and box-width estimates, as for the front camera."""
        g = self.rear_geometry
        if g is None or boxes is None:
            return None
        out = []
        for x1, y1, x2, y2, conf, cls in np.asarray(boxes, float).reshape(-1, 6):
            name = self.names[int(cls)] if 0 <= int(cls) < len(self.names) else "other_vehicle"
            if conf < self.settings.min_confidence or name in IGNORED:
                continue
            bottom = min(y2, g.height - .5)
            gx, gy = g.ground_from_pixel(np.array([x1, x2, (x1 + x2) / 2]), np.array([bottom] * 3))
            if not np.isfinite(gx).all() or gx.min() < -45:
                continue
            near_x = float(gx.max()) + OVERHANG_M.get(name, 0.)   # its front reaches nearer the car
            y_left, y_right = float(gy.min()), float(gy.max())
            cut = x1 <= 1 or x2 >= g.width - 1 or y2 >= g.height - 1
            if name in REAR_WIDTH_M and not cut and (x2 - x1) < 1.6 * (y2 - y1):
                forward = g.f * REAR_WIDTH_M[name] / max(1., x2 - x1)
                wx, _ = g.vehicle_from_camera(forward, 0.)
                if float(wx) > near_x:
                    near_x = float(wx)
                    _, ya = g.vehicle_from_camera(forward, (x1 - g.cx) * forward / g.f)
                    _, yb = g.vehicle_from_camera(forward, (x2 - g.cx) * forward / g.f)
                    y_left, y_right = min(y_left, float(ya), float(yb)), max(y_right, float(ya), float(yb))
            out.append({"class": name, "conf": float(conf), "box": [float(x1), float(y1), float(x2), float(y2)],
                        "x": near_x, "y_left": y_left, "y_right": y_right, "person": name in PEOPLE,
                        "ground_x": float(gx.max()), "cut_bottom": bool(y2 >= g.height - 1),
                        "behind_m": max(0., (self.front - 2 * self.half_length) - near_x), "role": "behind"})
        return out

    def _side_obstacles(self, view, boxes):
        """Side-camera boxes on flat ground. The ground under the box's bottom corners (vehicle frame) is its near
        edge; the thing is taken to reach on 2 m (a person 0.5 m) away from the camera."""
        g, out = self.side_geometry[view], []
        camera = np.array(g.mount[:2], float)
        for x1, y1, x2, y2, conf, cls in np.asarray(boxes, float).reshape(-1, 6):
            name = self.names[int(cls)] if 0 <= int(cls) < len(self.names) else "other_vehicle"
            if conf < self.settings.min_confidence or name in IGNORED:
                continue
            bottom = min(y2, g.height - .5)
            gx, gy = g.ground_from_pixel(np.array([x1, x2]), np.array([bottom, bottom]))
            if not (np.isfinite(gx).all() and np.isfinite(gy).all()):
                continue
            near = np.stack([gx, gy], axis=1)
            away = near - camera
            reach = np.hypot(away[:, 0], away[:, 1])
            if reach.max() > 45.:
                continue
            far = near + away / np.maximum(reach, 1e-6)[:, None] * (.5 if name in PEOPLE else 2.)
            out.append({"class": name, "conf": float(conf), "box": [float(x1), float(y1), float(x2), float(y2)],
                        "person": name in PEOPLE, "view": view, "corners": np.vstack([near, far[::-1]]),
                        "distance": float(reach.min()), "cut_bottom": bool(y2 >= g.height - 1)})
        return out

    def _passed_rect(self, pose):
        """Where the stopped vehicle being passed should appear in the front picture now: its
        planned footprint (odometry) and height, carried to `pose` and projected (clipped)."""
        hold, g = self.pass_hold, self.geometry
        corners = to_vehicle(hold["footprint"], pose)
        points = []
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            n = max(2, int(math.hypot(*(b - a)) / .25) + 1)
            points.extend(a + (b - a) * q for q in np.linspace(0., 1., n))
        points = np.array(points)
        xs, ys = np.r_[points[:, 0], points[:, 0]], np.r_[points[:, 1], points[:, 1]]
        zs = np.r_[np.zeros(len(points)), np.full(len(points), hold["height"])]
        forward, right, up = g.to_camera(xs, ys, zs)
        ahead = forward > .3
        if not ahead.any():
            return None
        u = np.clip(g.cx + g.f * right[ahead] / forward[ahead], 0, g.width - 1)
        v = np.clip(g.cy - g.f * up[ahead] / forward[ahead], 0, g.height - 1)
        return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]

    @staticmethod
    def _is_passed(box, rect, track):
        """This box is the vehicle being passed: it continues that vehicle's box from the last
        frames, or it sits where that vehicle should appear (and stands on the same ground)."""
        if track is not None and iou(box, track) >= .3:
            return True
        return (rect is not None and iou(box, rect) >= .25
                and abs(box[3] - rect[3]) <= .2 * (rect[3] - rect[1]) + 10.)

    def _traffic_on(self, plan_path, others, reach):
        """Something detected on or beside the planned pass path, out to `reach` m ahead
        (the path continues straight beyond its end): a short description, else None."""
        s = self.settings
        for o in others:
            if o["x"] > reach:
                continue
            margin = s.person_margin + s.person_near_m if o["person"] else s.vehicle_margin + .5
            for x in np.arange(max(o["x"], .5), o["x"] + o["length"] + .01, .5):
                y = float(np.interp(x, plan_path[:, 0], plan_path[:, 1]))
                if o["y_left"] - self.half_width - margin <= y <= o["y_right"] + self.half_width + margin:
                    return f"{o['class'].replace('_', ' ')} {max(0., o['x'] - self.front):.0f} m ahead in the passing lane"
        return None

    def _mirror(self, lateral, sim_time):
        """Rear-camera check before moving out to `lateral`: a vehicle behind in the lane the car
        will move into (not one queuing straight behind it). None when clear."""
        if self.rear_geometry is None:
            return None
        if self.rear_obstacles is None or self.rear_time is None or sim_time - self.rear_time > .6:
            return "waiting for the rear camera"
        guessed = self.blind_since is not None and sim_time - self.blind_since < self.settings.blind_spot_s
        right = self.side_obstacles.get("right")
        if right is not None and sim_time - right[0] <= .6:
            # The right camera shows the passing lane beside the car. A vehicle moving there is waited for. One seen
            # behind a moment ago is still assumed alongside while anything shows there (a bus filling the picture
            # looks still), but no longer once the lane beside the car is empty. One standing there, not seen
            # coming, does not stop the car pulling out ahead of it.
            tail = self.front - 2 * self.half_length
            beside = []
            for o in right[1]:
                v = to_vehicle(o["odo"], self.pose)
                if not o["person"] and v[:, 1].max() > self.half_width + .3 and v[:, 1].min() < lateral + self.half_width + .8 \
                        and v[:, 0].min() < self.front + 4. and v[:, 0].max() > tail - 4.:
                    beside.append(o)
                    if self._moving(o["odo"], sim_time):
                        return f"{o['class'].replace('_', ' ')} alongside in the passing lane"
            if guessed and beside:
                return "a vehicle may be alongside in the passing lane (seen behind, not yet ahead)"
        elif guessed:
            return "a vehicle may be alongside in the passing lane (seen behind, not yet ahead)"
        for o in self.rear_obstacles:
            if o["person"] or o["behind_m"] > self.settings.mirror_range_m:
                continue
            if o["y_right"] > self.half_width + .3 and o["y_left"] < lateral + self.half_width + .8:
                return f"{o['class'].replace('_', ' ')} {o['behind_m']:.0f} m behind in the passing lane"
        return None

    def _event(self, sim_time, kind, detail=""):
        self.events.append((round(float(sim_time), 2), kind, detail))
        self.pass_note, self.pass_note_time = detail or kind.replace("_", " "), sim_time

    def _abandon_pass(self, pose, sim_time, why):
        """Give up a pass that has stopped: back out the way it came if it has moved out."""
        hold = self.pass_hold
        start = to_vehicle(hold["path"][:1], pose)[0]
        moved = math.hypot(*start) > .4
        if moved:
            # Retrace the pass path, then carry on straight back along the heading it started with
            # (up to 6 m) until the car is straight again: following a point on the path brings the
            # car back to where it began but turned (24 degrees in the 7 Nov synthetic check).
            path, psi0 = hold["path"], hold["start_pose"][2]
            direction = np.array([math.cos(psi0), math.sin(psi0)])
            extension = np.array([path[0] - direction * k for k in np.arange(6., .01, -.5)])
            full = np.vstack([extension, path])
            self.backing = {"path": full, "start_index": len(extension), "heading": psi0, "why": why, "since": sim_time,
                            "best": float(np.hypot(*to_vehicle(full[:1], pose)[0])), "best_time": sim_time}
        # Steer straight along the lane it left (not the abandoned pass path) and stay behind the
        # stopped vehicle: creeping closer would make the next try steeper (7 Nov synthetic check).
        path, psi0 = hold["path"], hold["start_pose"][2]
        direction = np.array([math.cos(psi0), math.sin(psi0)])
        self.path = np.array([path[0] + direction * k for k in np.arange(.5, 16.01, .5)])
        self.path_time = sim_time
        self.stay_behind = sim_time
        self.pass_hold = None
        self.pass_cooldown_until = sim_time + self.settings.pass_retry_after_s
        self.pass_until = -1e9   # a fresh wait before the next try
        self._event(sim_time, "pass_abandoned", f"gave up passing: {why} on its path" + ("; backing out" if moved else ""))

    def _plan(self, perception, pose, sim_time):
        s, grid = self.settings, self.grid
        road = grid.road(perception.mask)
        previous = self._previous_y(pose)
        allow_pass = sim_time < self.pass_until
        reacquire = self.plan is None or self.plan.fit_rows < 4
        obstacles = self._boxes_to_obstacles(perception.boxes, perception.mask, road, sim_time, perception.rgb)
        shown = obstacles   # every front detection, for display (the one being passed included)
        vehicles = [o for o in obstacles if not o["person"]]
        rear = self._rear_obstacles(perception.rear_boxes)
        if rear is not None:
            started = self.rear_time is None or sim_time - self.rear_time > 1.
            if started or any(self._read_behind(o) for o in rear):
                self.rear_seen_at = tuple(float(v) for v in pose)
            self.rear_obstacles, self.rear_time = rear, sim_time
            self._track(sim_time, "rear", [(o["class"], o["person"], to_odometry(np.array(
                [[o["x"], o["y_left"]], [o["x"], o["y_right"]], [o["x"] - LENGTH_M.get(o["class"], 4.), o["y_right"]],
                 [o["x"] - LENGTH_M.get(o["class"], 4.), o["y_left"]]]), pose)) for o in rear])
            centres = [to_odometry(np.array([[o["x"] - LENGTH_M.get(o["class"], 4.) / 2, (o["y_left"] + o["y_right"]) / 2]]), pose)[0]
                       for o in rear if not o["person"]]
            self.rear_log = [e for e in self.rear_log if sim_time - e[0] <= 3.] + [(sim_time, centres)]
            # Something close behind in the lane to the right will be alongside next, where neither
            # camera sees it; remember it until it shows up ahead (or long enough to have gone by).
            if any(not o["person"] and o["behind_m"] < 15. and o["y_right"] > self.half_width + .3
                   and o["y_left"] < self.half_width + 4.5 for o in rear):
                self.blind_since = sim_time
        if self.blind_since is not None and any(
                not o["person"] and o["x"] < 15. and o["y_left"] > self.half_width - .3 and o["y_left"] < self.half_width + 4.5
                and o["x"] > self.front - 1. for o in obstacles) and sim_time - self.blind_since > .3:
            self.blind_since = None   # it is in front now (the lane check ahead takes over)
        sides = {}
        for view, boxes in (perception.side_boxes or {}).items():
            if view not in self.side_geometry or boxes is None:
                continue
            found = self._side_obstacles(view, boxes)
            for o in found:
                o["odo"] = to_odometry(o["corners"], pose)
            self.side_obstacles[view] = (sim_time, found)
            self._track(sim_time, view, [(o["class"], o["person"], o["odo"]) for o in found])
            sides[view] = found
        rows, targets, steps = [], [], []
        luma = None
        if s.asphalt_edge and perception.rgb is not None:
            luma = grid.brightness(perception.rgb)
        behind_vehicle = False
        for x in np.arange(4., 14.001, .5):
            r = grid.row_of(x)
            y_prev = float(previous(x))
            # A vehicle on the path: follow it, never steer round it here, and treat the road
            # beyond it as unseen (its image hides that road).
            behind_vehicle = behind_vehicle or any(o["x"] <= x and o["y_left"] - .3 <= y_prev <= o["y_right"] + .3 for o in vehicles)
            if behind_vehicle:
                rows.append((float(x), None, None, None))
                continue
            start = self._start_column(road, r, y_prev, reacquire)
            if start is None:
                rows.append((float(x), None, None, None))
                continue
            left = self._scan(road, r, start, -1)
            right = self._scan(road, r, start, +1)
            y_left = None if left is None else float(grid.ys[left] - grid.cell / 2)
            y_right = None if right is None else float(grid.ys[right] + grid.cell / 2)
            rows.append((float(x), y_left, y_right, float(grid.ys[start])))
            if luma is not None and left is not None and 6. <= x <= 12.:
                step = self._asphalt_step(road, luma, r, left)
                if step is not None and step - y_left >= .6:
                    steps.append((float(x), step))
        # A paved shoulder the road model counted as road: when at least 4 rows agree on an
        # asphalt edge 0.6-1.8 m inside its left edge, that is the lane's edge; remember it
        # (odometry) for a moment, as the step is not found in every frame.
        if len(steps) >= 4:
            data = np.array(steps)
            line = np.polyfit(data[:, 0], data[:, 1], 1)
            if (np.abs(np.polyval(line, data[:, 0]) - data[:, 1]) <= .3).sum() >= 4:
                along = np.arange(4., 14.01, .5)
                self.asphalt = {"line": to_odometry(np.stack([along, np.polyval(line, along)], axis=1), pose), "time": sim_time}
        corrected = []
        if self.asphalt is not None and sim_time - self.asphalt["time"] <= s.asphalt_hold_s:
            guide = to_vehicle(self.asphalt["line"], pose)
            guide = guide[np.argsort(guide[:, 0])]
            for i, (x, y_left, y_right, start) in enumerate(rows):
                if y_left is not None and guide[0, 0] <= x <= guide[-1, 0] + 2.:
                    edge = float(np.interp(x, guide[:, 0], guide[:, 1]))
                    if .6 <= edge - y_left <= 1.8:
                        rows[i] = (x, edge, y_right, start)
                        corrected.append(x)
        elif self.asphalt is not None:
            self.asphalt = None
        arriving = self.nav is not None and self.nav.get("arrive_in_m") is not None and self.nav["arrive_in_m"] <= s.arrive_slow_m
        keep = s.arrive_keep_left_m if arriving else s.keep_left_m
        for x, y_left, y_right, start in rows:
            if y_left is not None:
                targets.append((float(x), y_left + keep))
        coef, used = self._fit(targets)
        xs = np.arange(.5, 16.01, .5)
        lost = coef is None
        if not lost:
            # Beyond the farthest kerb row used, continue straight on (no curve extrapolation).
            reach = self._fit_reach + 1.
            slope = float(np.polyval(np.polyder(coef), reach))
            fitted = np.where(xs <= reach, np.polyval(coef, xs), np.polyval(coef, reach) + slope * (xs - reach))
            if self.path is not None and sim_time - self.path_time < .5:
                fitted = s.blend * previous(xs) + (1 - s.blend) * fitted
            path = np.stack([xs, fitted], axis=1)
            self.path, self.path_time, self.edge_time = to_odometry(path, pose), sim_time, sim_time
        elif self.path is not None:
            path = to_vehicle(self.path, pose)
            path = path[np.argsort(path[:, 0])]
        else:
            path = np.stack([xs, np.zeros_like(xs)], axis=1)

        def path_y(x):
            """Target line at forward distance x; straight extension beyond its end."""
            if x > path[-1, 0]:
                slope = (path[-1, 1] - path[-2, 1]) / max(1e-6, path[-1, 0] - path[-2, 0])
                return float(path[-1, 1] + (x - path[-1, 0]) * slope)
            return float(np.interp(x, path[:, 0], path[:, 1]))
        wide = False
        for x, y_left, y_right, start in rows:
            if not 5. <= x <= 12. or start is None:
                continue
            if y_left is None or y_left < path_y(x) - 4.5 or (y_right is not None and y_right - y_left > 9.):
                wide = True
        # After waiting behind a stopped vehicle, move out around its right side if the road there
        # is visibly free and nothing is in the passing lane ahead or behind (a tall vehicle hides
        # the road beside and behind it from the kerb rule). Once started, the pass is held until
        # the car is past it; only that vehicle is ignored meanwhile.
        if self.pass_hold is not None:
            hold = self.pass_hold
            end = to_vehicle(hold["line"][1:], pose)[0]
            if end[0] < -self.half_length - 2. or sim_time - hold["created"] > 60.:
                self._event(sim_time, "pass_done" if end[0] < -self.half_length - 2. else "pass_timed_out",
                            f"passed the stopped {hold['class'].replace('_', ' ')}")
                self.pass_hold = None
            else:
                path = to_vehicle(hold["path"], pose)
                path = path[np.argsort(path[:, 0])]
                self.path, self.path_time = hold["path"], sim_time
                rect = self._passed_rect(pose)
                track = hold["track"] if sim_time - hold["track_time"] <= 1. else None
                best, kept = None, []
                for o in obstacles:
                    if not o["person"] and self._is_passed(o["box"], rect, track):
                        o["role"] = "passing"
                        score = iou(o["box"], track if track is not None else rect)
                        if best is None or score > best[0]:
                            best = (score, o["box"])
                    else:
                        kept.append(o)
                if best is not None:
                    hold["track"], hold["track_time"] = best[1], sim_time
                obstacles = kept
        passing = self.pass_hold is not None
        if wide:
            self.wide_seen = sim_time
        waited = 0. if self.stopped_since is None else sim_time - self.stopped_since
        near_junction = sim_time - getattr(self, "wide_seen", -1e9) < 3.
        # Following a route through a junction: steer by a Hybrid A* path on this grid, aimed at the
        # route's exit (or by the route line itself when no path is found), until the kerb is back.
        nav = self.nav
        mode = "free" if nav is None else nav.get("mode", "lane")
        plan_found = plan_ms = goal = None
        if self.junction is not None and mode == "lane" and used >= 6 and nav.get("guide_odo") is not None:
            # Back on a street with a kerb to follow, once the car has settled on the route's lane (the
            # kerb follower's longer look-ahead took 10 m more to close a 0.4 m exit offset).
            near = to_vehicle(nav["guide_odo"], pose)
            k = int(np.argmin(np.hypot(near[:, 0], near[:, 1])))
            ahead = near[min(k + 4, len(near) - 1)] - near[k]
            heading = abs(math.atan2(ahead[1], ahead[0])) if np.hypot(*ahead) > .5 else 0.
            if abs(float(near[k, 1])) <= .3 and heading <= math.radians(5.):   # on the lane and lined up with it
                self.junction = None
        guided = (nav is not None and not passing and nav.get("guide_odo") is not None and len(nav["guide_odo"]) >= 2
                  and (mode == "junction" or self.junction is not None))
        if guided:
            guide_v = to_vehicle(nav["guide_odo"], pose)
            j = self.junction or {"path": None, "fallback": False, "planned_at": -1e9, "blocked_since": None}
            if mode == "junction":
                current = None if j["path"] is None else to_vehicle(j["path"], pose)
                blocked_now = current is not None and bool(self._guided_hits(current, obstacles))
                if current is None or blocked_now or sim_time - j["planned_at"] >= s.junction_replan_s:
                    occ = astar_occupancy(road, grid.valid, grid.xs, grid.ys, obstacles, self.half_width, self.half_length)
                    goal = self._junction_goal(guide_v, occ, obstacles)
                    if goal is not None and goal[0] < 6.:
                        goal = None   # only a few metres in view: the turn itself is not seen (8 Nov smoke run, B)
                    result = None
                    if goal is not None:
                        result = astar_plan(occ, goal, guide_v, self.wheelbase, self.max_steer)
                        plan_ms = result.ms
                    plan_found = result is not None and result.reason == "found"
                    # A failed re-plan keeps the last plan only while it still reaches well ahead of the car;
                    # driving past the end of a short straight plan missed the bend at B.
                    reaches = current is not None and self._remaining(current) >= s.junction_lookahead
                    if plan_found:
                        j.update(path=to_odometry(result.path[:, :2], pose), fallback=False)
                    elif current is None or blocked_now or j["fallback"] or not reaches:
                        j.update(path=nav["guide_odo"].copy(), fallback=True)
                    # (otherwise a failed re-plan keeps driving the last good plan: switching to the tighter
                    # route line mid-turn ran the car 0.5 m wide in the 8 Nov synthetic left turn)
                    j["planned_at"] = sim_time
            else:
                j["path"] = nav["guide_odo"].copy()   # out of the junction, kerb not yet back: the route line
            self.junction = j
            path = to_vehicle(j["path"], pose)
            self.path, self.path_time = j["path"], sim_time
        ahead = [] if guided else [
            o for o in obstacles if not o["person"] and o["x"] < 15. and
            o["y_left"] - self.half_width - s.vehicle_margin <= path_y(max(o["x"], .5)) <= o["y_right"] + self.half_width + s.vehicle_margin]
        candidate = None if passing or not ahead else min(ahead, key=lambda o: o["x"])
        if self.stay_behind is not None:
            if candidate is not None:
                self.stay_behind = sim_time
            elif sim_time - self.stay_behind > 2.:
                self.stay_behind = None   # it has gone
        # The rear camera runs while waiting behind something it may pass.
        self.rear_hint = (candidate is not None and self.backing is None and self.speed < .5 and
                          ((candidate["class"] in LARGE and waited >= 1.) or
                           (s.pass_small and waited >= s.pass_after_stopped_small_s - 3.)))
        turning_round = self.uturn is not None and self.uturn["phase"] != "approach"
        if allow_pass and not passing and not near_junction and not guided and self.backing is None and not turning_round \
                and sim_time >= self.pass_cooldown_until:
            blocker = candidate
            # Near the destination, pass only when it lies well beyond the stopped vehicle.
            to_go = None if nav is None else nav.get("arrive_in_m")
            if blocker is not None and to_go is not None and to_go <= s.no_pass_near_destination_m \
                    and to_go <= blocker["x"] + blocker["length"] + 8.:
                blocker = None
            # A bus or truck standing in the lane (the bus stop) is passed after a short wait; anything
            # else only after a long one, since it is far more likely to be queuing (6 Nov run 2: a
            # motorcycle waiting at a junction was "passed" into the oncoming lane).
            if blocker is not None and blocker["class"] not in LARGE and (not s.pass_small or waited < s.pass_after_stopped_small_s):
                blocker = None
            if blocker is not None and blocker["x"] - self.front < 3.:
                if "too close" not in self.pass_note or sim_time - self.pass_note_time > 2.:
                    self.pass_note, self.pass_note_time = "waiting to pass: too close behind it to pull out", sim_time
                blocker = None
            if blocker is not None:
                x0, x1 = blocker["x"], blocker["x"] + blocker["length"]
                large = blocker["class"] in LARGE
                clearances = np.arange(s.pass_clearance_large if large else s.pass_clearance,
                                       (s.pass_clearance_large_min if large else s.pass_clearance) - .01, -.1)

                def s_curve(lateral):
                    """From where the car stands, out to `lateral`, reached just before its rear."""
                    return lambda x: lateral * float(np.clip(x / max(1., x0 - .5), 0., 1.)) ** 2 * \
                        (3 - 2 * float(np.clip(x / max(1., x0 - .5), 0., 1.)))
                # Out to its right side plus a clearance, the largest whose route is visibly free road.
                lateral = next((blocker["side_right"] + self.half_width + c for c in clearances
                                if 0. < blocker["side_right"] + self.half_width + c <= 4.5
                                and self._corridor_free(road, s_curve(blocker["side_right"] + self.half_width + c),
                                                        max(self.front, .5), x0 + 1.)), None)
                note = "waiting to pass: no visible road beside it" if lateral is None else ""
                if lateral is not None:
                    # Plan the whole pass once (the vehicle is stopped): the S-curve, then parallel past it.
                    pass_xs = np.arange(0., x1 + 6.01, .5)
                    curve = s_curve(lateral)
                    plan_path = np.stack([pass_xs, [curve(float(q)) for q in pass_xs]], axis=1)
                    same = [o for o in obstacles if o is blocker or (not o["person"] and iou(o["box"], blocker["box"]) >= .3)]
                    others = [o for o in obstacles if all(o is not b for b in same)]
                    note = self._traffic_on(plan_path, others, x1 + 6. + s.pass_look_beyond_m) or self._mirror(lateral, sim_time)
                    note = "waiting to pass: " + note if note else ""
                if not note:
                    # The same side estimate the clearance was planned from (the cautious union
                    # reads a bus's corner ~1 m too far out and would freeze the car mid-pass).
                    side = np.array([[x0, blocker["side_right"]], [x1, blocker["side_right"]]])
                    footprint = np.array([[x0, blocker["y_left"]], [x1, blocker["y_left"]],
                                          [x1, blocker["side_right"]], [x0, blocker["side_right"]]])
                    self.pass_hold = {"line": to_odometry(side, pose), "path": to_odometry(plan_path, pose),
                                      "footprint": to_odometry(footprint, pose), "height": 3.4 if large else 2.,
                                      "class": blocker["class"], "created": sim_time, "start_pose": tuple(pose),
                                      "track": blocker["box"], "track_time": sim_time}
                    for o in same:
                        o["role"] = "passing"
                    obstacles = others
                    path = plan_path
                    self.path, self.path_time = self.pass_hold["path"], sim_time
                    passing = True
                    self.stay_behind = None
                    self._event(sim_time, "pass_started", f"passing the stopped {blocker['class'].replace('_', ' ')} on its right")
                elif note != self.pass_note or sim_time - self.pass_note_time > 2.:
                    self.pass_note, self.pass_note_time = note, sim_time
        # Road ends: the car's corridor is mostly not road for 0.3 m and stays mostly blocked
        # over the next 1.5 m (a thin band such as a junction seam or a shadow is ignored).
        half = self.half_width + s.vehicle_margin
        blocked = np.full(len(grid.xs), False)
        known = np.full(len(grid.xs), False)
        for r, x in enumerate(grid.xs):
            if x < self.front:
                continue
            y = float(path_y(x))
            lo, hi = grid.col_of(y - half), grid.col_of(y + half)
            valid = grid.valid[r, lo:hi + 1]
            if valid.mean() < .5:
                continue
            known[r] = True
            blocked[r] = road[r, lo:hi + 1][valid].mean() < .6
        road_end, run = None, 0
        beyond = int(round(1.5 / grid.cell))
        for r in ([] if guided else range(len(grid.xs))):
            run = run + 1 if blocked[r] else 0
            if run >= 3:
                after = slice(r + 1, r + 1 + beyond)
                seen = known[after].sum()
                if seen < beyond // 2 or blocked[after][known[after]].mean() >= .6:
                    road_end = float(grid.xs[r] - 2 * grid.cell)   # forward position where the road stops
                    break
        new_hits = []
        self._track_people(obstacles, pose, sim_time)
        for obstacle in ([] if not guided else obstacles):
            hit = self._guided_hits(path, [obstacle])
            if hit:
                hx, hy = hit[0][1]
                obstacle["role"], obstacle["distance"] = "in_path", max(0., hx - self.front)
                new_hits.append((sim_time, "person" if obstacle["person"] else "vehicle",
                                 to_odometry(np.array([[hx, hy]]), pose)[0], obstacle["class"]))
            elif obstacle["person"] and self._crossing(obstacle, self._guided_y(path, obstacle["x"]), pose, sim_time):
                obstacle["role"], obstacle["distance"] = "in_path", max(0., obstacle["x"] - self.front)
                hy = self._guided_y(path, obstacle["x"])
                new_hits.append((sim_time, "person", to_odometry(np.array([[obstacle["x"], hy]]), pose)[0], obstacle["class"]))
            elif obstacle["person"] and self._guided_hits(path, [obstacle], extra=s.person_near_m):
                obstacle["role"], obstacle["distance"] = "near_path", max(0., obstacle["x"] - self.front)
                new_hits.append((sim_time, "person_near", to_odometry(np.array([[obstacle["x"], obstacle["y_left"]]]), pose)[0],
                                 obstacle["class"]))
        for obstacle in ([] if guided else obstacles):
            margin = s.person_margin if obstacle["person"] else s.vehicle_margin
            span = np.arange(max(obstacle["x"], .5), obstacle["x"] + obstacle["length"] + .01, .25)
            hit = None
            for x in span:
                y = float(path_y(x))
                if obstacle["y_left"] - self.half_width - margin <= y <= obstacle["y_right"] + self.half_width + margin:
                    hit = x
                    break
            if hit is not None:
                obstacle["role"], obstacle["distance"] = "in_path", max(0., hit - self.front)
                point = to_odometry(np.array([[hit, float(path_y(hit))]]), pose)[0]
                new_hits.append((sim_time, "person" if obstacle["person"] else "vehicle", point, obstacle["class"]))
            elif obstacle["person"] and self._crossing(obstacle, float(path_y(max(obstacle["x"], .5))), pose, sim_time):
                # Walking towards the path, it will be there as the car comes by (9 Nov: a jaywalker seen
                # 2.3 m ahead, 1.2 m clear of the car's side, only slowed the car; it walked into its side).
                obstacle["role"], obstacle["distance"] = "in_path", max(0., obstacle["x"] - self.front)
                point = to_odometry(np.array([[obstacle["x"], float(path_y(max(obstacle["x"], .5)))]]), pose)[0]
                new_hits.append((sim_time, "person", point, obstacle["class"]))
            elif obstacle["person"] and obstacle["x"] < s.person_near_range:
                y = float(path_y(max(obstacle["x"], .5)))
                gap = max(obstacle["y_left"] - (y + self.half_width), (y - self.half_width) - obstacle["y_right"])
                if gap <= s.person_near_m:
                    obstacle["role"], obstacle["distance"] = "near_path", max(0., obstacle["x"] - self.front)
                    point = to_odometry(np.array([[obstacle["x"], y]]), pose)[0]
                    new_hits.append((sim_time, "person_near", point, obstacle["class"]))
        if road_end is not None:
            point = to_odometry(np.array([[road_end, float(path_y(road_end))]]), pose)[0]
            new_hits.append((sim_time, "road_end", point, "road"))
        def keep(h):
            if h[1] == "road_end":
                return False
            if sim_time - h[0] <= s.hold_s:
                return True
            # A vehicle seen close ahead stays a while: right behind one that fills the picture the detector can
            # lose it for a second (9 Nov: it pulled away into a waiting car 0.7 m ahead). Not while passing,
            # when the stopped vehicle is deliberately ignored. Four-wheelers only: a person walking beside the road
            # was boxed as a "rider" now and then, and holding that stopped the car after passing him (14 Nov).
            return (h[1] == "vehicle" and h[3] in FOUR_WHEELERS and self.pass_hold is None and sim_time - h[0] <= s.near_vehicle_hold_s
                    and float(to_vehicle(np.asarray(h[2])[None], pose)[0, 0]) - self.front <= s.near_vehicle_m)
        self.hits = [h for h in self.hits if keep(h)] + new_hits
        if self.pass_hold is not None:
            # Stopped mid-pass (something on its path, the road ahead not visible, or no room to edge
            # round the corner): give up after a few seconds and back out the way it came.
            hold = self.pass_hold
            if self.speed < .3 and sim_time - hold["created"] > 2.:
                hold.setdefault("stalled_since", sim_time)
                if sim_time - hold["stalled_since"] >= s.pass_abort_after_s:
                    why = next((o["class"].replace("_", " ") for o in obstacles if o["role"] == "in_path"), None)
                    self._abandon_pass(pose, sim_time, why or ("road not visible" if road_end is not None else "no room"))
                    passing = False
            else:
                hold.pop("stalled_since", None)
        self.flags = {"wide": wide, "uncertain": used < 6, "time": sim_time, "passing": passing}
        self._remember_seen(obstacles, pose, sim_time)
        note = self.pass_note if sim_time - self.pass_note_time <= 3. else ""
        return Plan(perception.frame, sim_time, rows, used, path, wide,
                    None if road_end is None else max(0., road_end - self.front), shown, lost, road_end, passing,
                    list(self.rejected), rear, note, corrected, mode,
                    None if not guided else path, goal, plan_found, plan_ms, sides or None)

    def _track_people(self, obstacles, pose, sim_time):
        """Follow each person seen from frame to frame in odometry coordinates (the car's own motion
        taken out), keeping the last second of positions."""
        tracks = [k for k in self.people if sim_time - k["seen"] <= 1.]
        for o in obstacles:
            if not o["person"]:
                continue
            here = to_odometry(np.array([[o["x"] + .3, (o["y_left"] + o["y_right"]) / 2]]), pose)[0]
            best = min(tracks, key=lambda k: math.dist(k["points"][-1][1:], here), default=None)
            if best is None or best["seen"] == sim_time or math.dist(best["points"][-1][1:], here) > 1.2:
                best = {"points": [], "seen": sim_time}
                tracks.append(best)
            best["points"] = [q for q in best["points"] if sim_time - q[0] <= 1.] + [(sim_time, float(here[0]), float(here[1]))]
            best["seen"] = sim_time
            o["track"] = best
        self.people = tracks

    def _crossing(self, o, path_y_there, pose, sim_time):
        """A person walking towards the path who will be in the car's corridor while the car goes by."""
        s, track = self.settings, o.get("track")
        if track is None or len(track["points"]) < 3 or track["points"][-1][0] - track["points"][0][0] < .3:
            return False
        t, ox, oy = (np.array(c) for c in zip(*track["points"]))
        vx, vy = np.polyfit(t - t[-1], ox, 1)[0], np.polyfit(t - t[-1], oy, 1)[0]
        c, sn = math.cos(pose[2]), math.sin(pose[2])
        lateral = -vx * sn + vy * c                      # its speed across the car's heading (right positive)
        offset = (o["y_left"] + o["y_right"]) / 2 - path_y_there
        if abs(lateral) < s.crossing_speed or offset * lateral >= 0. or o["x"] - self.front > s.person_near_range:
            return False
        band = self.half_width + s.person_margin + .3      # its centre within this of the path: in the car's way
        enter = max(0., (abs(offset) - band) / abs(lateral))
        leave = (abs(offset) + band) / abs(lateral)
        speed = max(self.speed, 1.)
        arrive = max(0., o["x"] - self.front) / speed
        gone = arrive + (2 * self.half_length + 1.) / speed
        return enter <= gone and leave >= arrive

    @staticmethod
    def _guided_y(path, x):
        """The lateral position of a path (path order) where it is nearest to forward distance x."""
        ahead = path[path[:, 0] >= .5]
        if not len(ahead):
            return 0.
        return float(ahead[int(np.argmin(np.abs(ahead[:, 0] - x))), 1])

    def _guided_hits(self, path, obstacles, extra=0.):
        """Obstacles whose footprint (widened by the car's half-width and the margin, plus `extra`)
        holds a point of a path given in path order (a turn is not a function of x): a list of
        (obstacle, first such point)."""
        s, out = self.settings, []
        ahead = path[path[:, 0] >= .5]
        for o in obstacles:
            margin = (s.person_margin if o["person"] else s.vehicle_margin) + self.half_width + extra
            inside = ((ahead[:, 0] >= o["x"]) & (ahead[:, 0] <= o["x"] + o["length"])
                      & (ahead[:, 1] >= o["y_left"] - margin) & (ahead[:, 1] <= o["y_right"] + margin))
            if inside.any():
                out.append((o, ahead[int(np.argmax(inside))]))
        return out

    def _junction_goal(self, guide, occ=None, obstacles=()):
        """The farthest route-line point the camera sees (3-22 m ahead, within 8 m to the side) where the
        car's whole body (its discs, along the route's heading) is free, with that heading; None when none
        is. (The far rows are coarse, about 1.2 m of ground per mask row at 20 m, so a lane-centre goal
        there can fall in the widened edge of the road.) A vehicle or person on the route ahead ends the
        search: the goal stays behind it, so the car follows it rather than planning round it (9 Nov: a
        goal right behind a bicycle, where the car's centre fitted but not its body, made Hybrid A* swing
        the car off a left bend into the sand)."""
        grid, best = self.grid, None
        offsets, radius = disc_offsets(self.half_length), disc_radius(self.half_width, self.half_length)
        ahead = [o for o in obstacles if o["x"] >= self.front - .5]
        for i, (x, y) in enumerate(guide):
            a, b = guide[max(0, i - 1)], guide[min(len(guide) - 1, i + 1)]
            if np.allclose(a, b):
                continue
            heading = math.atan2(b[1] - a[1], b[0] - a[0])
            discs = [(x + d * math.cos(heading), y + d * math.sin(heading)) for d in offsets]
            if x >= self.front and any(_rect_gap(o, px, py) < radius for o in ahead for px, py in discs):
                break
            if not (3. <= x <= 22. and abs(y) <= 8. and grid.valid[grid.row_of(x), grid.col_of(y)]):
                continue
            if occ is not None and not all(_occ_free(occ, px, py) for px, py in discs):
                continue
            best = (float(x), float(y), heading)
        return best

    @staticmethod
    def _remaining(path):
        """How far a path (path order) runs on from its point nearest the car."""
        i = int(np.argmin(np.hypot(path[:, 0], path[:, 1])))
        return float(np.hypot(*np.diff(path[i:], axis=0).T).sum()) if len(path) - i > 1 else 0.

    @staticmethod
    def _along(path, distance):
        """The point `distance` metres along a path (path order) from its point nearest the car,
        continuing straight past its end."""
        i = int(np.argmin(np.hypot(path[:, 0], path[:, 1])))
        rest = path[i:]
        if len(rest) < 2:
            return rest[-1] if len(rest) else np.array([distance, 0.])
        steps = np.hypot(*np.diff(rest, axis=0).T)
        total = np.r_[0., np.cumsum(steps)]
        if distance <= total[-1]:
            return np.array([np.interp(distance, total, rest[:, 0]), np.interp(distance, total, rest[:, 1])])
        d = rest[-1] - rest[-2]
        return rest[-1] + d / max(np.hypot(*d), 1e-6) * (distance - total[-1])

    def _corridor_free(self, road, y_of, x_from, x_to):
        """The car's body (plus 0.2 m) would sit on visible road from x_from to x_to."""
        grid, half, checked = self.grid, self.half_width + .2, 0
        for r in range(grid.row_of(x_from), grid.row_of(x_to) + 1):
            y = y_of(float(grid.xs[r]))
            lo, hi = grid.col_of(y - half), grid.col_of(y + half)
            valid = grid.valid[r, lo:hi + 1]
            if valid.mean() < .5:
                continue
            checked += 1
            if road[r, lo:hi + 1][valid].mean() < .85:
                return False
        return checked >= 3

    # ---- control ---------------------------------------------------------------
    def control(self, speed, yaw_rate, sim_time):
        """One 20 Hz control step. Returns a dict with throttle/brake/steer and the decision."""
        s = self.settings
        self.odometry(speed, yaw_rate)
        self.sim_time = sim_time
        if speed < .5:
            self.stopped_since = sim_time if self.stopped_since is None else self.stopped_since
            if sim_time - self.stopped_since >= s.pass_after_stopped_s:
                self.pass_until = sim_time + s.pass_window_s
        else:
            self.stopped_since = None
        if self.backing is not None:
            return self._back_up(speed, sim_time)
        if self.uturn is not None and self.uturn["phase"] != "approach":
            return self._turn_around(speed, sim_time)
        if self._wedged(speed, sim_time):
            self.wedge_log = [t for t in self.wedge_log if sim_time - t < 60.]
            if len(self.wedge_log) >= s.wedged_tries:
                self.steer += float(np.clip(-self.steer, -.07, .07))
                return self._output(0., .7, 0., "wedged", f"stuck: backed off {len(self.wedge_log)} times in a minute; waiting", 0., 0.)
            self._back_off(sim_time)
            return self._back_up(speed, sim_time)
        target, reason, detail = s.cruise, "cruise", ""
        learned = self.governor is not None
        if learned and self.nav is not None and s.rl_speed_limits and self.nav.get("legal_cap") is not None:
            # The posted limit (and slowing in time for a lower one ahead) replaces the fixed cruise.
            posted = float(self.nav["limit"])
            # (below the car's gear-change band on the 5.6-7.3 m/s streets: india_rl_speed.cruise_limit)
            target, reason = float(self.nav.get("cruise_cap", self.nav["legal_cap"])), "speed_limit"
            detail = f"limit {posted * 3.6:.0f} km/h" + ("" if target >= posted - 1e-6 else
                                                        f", {self.nav['next_limit'] * 3.6:.0f} km/h ahead")
        def limit(value, why, text=""):
            nonlocal target, reason, detail
            # An emergency also takes over an equal (zero) target set for another reason,
            # so the car brakes hard rather than coasting to a stop.
            if value < target or (why.startswith("emergency") and value <= target and not reason.startswith("emergency")):
                target, reason, detail = max(0., value), why, text
        available = self.last_ok_time is not None and sim_time - self.last_ok_time <= s.unavailable_after_s
        if not self.started:
            limit(0., "starting")
        elif not available:
            limit(0., "perception_unavailable")
        path = None if self.path is None else to_vehicle(self.path, self.pose)
        lost_for = sim_time - (self.edge_time if self.edge_time is not None else -1e9)
        nav = self.nav
        guided = nav is not None and self.junction is not None and self.junction.get("path") is not None
        if self.started and available and nav is not None:
            # Arrival first, so it names the stop at the destination.
            to_go = nav.get("arrive_in_m")
            if to_go is not None:
                if to_go <= s.arrive_stop_m:
                    limit(0., "arrived")
                elif to_go <= s.arrive_slow_m:
                    # (The learned speed keeps only the stopping curve to the pin; the 3 m/s crawl is a pace choice.)
                    stop_curve = math.sqrt(2 * s.brake_decel * max(0., to_go - .5))
                    limit(stop_curve if learned else min(s.arrive_speed, stop_curve), "arriving", f"arriving in {to_go:.0f} m")
            if nav.get("hold_at_m") is not None:
                limit(math.sqrt(2 * s.brake_decel * max(0., nav["hold_at_m"] - s.hold_gap_m)), "waiting_turn",
                      f"junction {nav['hold_at_m']:.0f} m ahead")
            if self.uturn is not None:
                # Stop where the route turns round (the controller runs about a metre past a stopping curve).
                to_turn = self.uturn["in_m"]
                limit(math.sqrt(2 * s.brake_decel * max(0., to_turn - .3)), "uturn_ahead", f"turning round in {max(0., to_turn):.0f} m")
                if speed < .3 and to_turn <= 2.:
                    self.uturn.update(phase="wait", started=sim_time, clear_since=None, moves=0, travelled=0.,
                                      theta=0., last_yaw=self.pose[2], last_xy=self.pose[:2], axle=tuple(self.pose))
                    self._event(sim_time, "uturn_started", "stopped to turn round")
            junction_m = nav.get("junction_m")
            entering = (learned and s.rl_junction_entry and junction_m is not None and junction_m > self.front
                        and nav.get("hold_at_m") is None and not (self.flags["wide"] or self.flags["uncertain"]))
            if learned and junction_m is not None and (entering or (not guided and junction_m > rl.JUNCTION_ZONE_M)):
                # Ease down to the junction speed rather than braking hard as the junction zone starts: by the
                # junction's entry (the car's front at its start), or by the start of the 20 m zone.
                v_in = (s.junction_fallback_speed if guided and self.junction.get("fallback") else s.junction_speed)
                room = junction_m - self.front if entering else junction_m - rl.JUNCTION_ZONE_M
                limit(math.sqrt(v_in ** 2 + 2 * rl.JUNCTION_EASE * max(0., room)), "junction_ahead", f"junction {junction_m:.0f} m ahead")
            if guided and not entering:
                fallback = self.junction.get("fallback")
                limit(s.junction_fallback_speed if fallback else s.junction_speed,
                      "junction_fallback" if fallback else "junction_turn",
                      "" if not nav.get("junction") else f"{nav['junction'].get('turn', '')} at {nav['junction'].get('name', 'the junction')}")
        if self.started and available and self.pass_hold is None and not guided:
            # (A planned pass does not follow the kerb, which the stopped vehicle often hides; its
            # path, the obstacle checks and the road-end check guide it. 7 Nov run12 stopped
            # mid-pass on "lost the road" at dusk and gave the pass up.)
            if lost_for > s.lost_stop_after_s:
                limit(0., "lost_road_stop")
            elif lost_for > s.lost_after_s:
                limit(s.lost_road, "lost_road")
        if self.started and available:
            if self.stay_behind is not None and any(kind == "vehicle" for _, kind, _, _ in self.hits):
                limit(0., "vehicle_ahead", "waiting behind it to try passing again")
            if self.pass_hold is not None:
                limit(s.passing, "passing", "passing a stopped vehicle on its right")
                # Edge forward only while its rear right corner is clear of the car's body
                # straight ahead (or still 0.6 m away); the car keeps turning right meanwhile.
                (x0, y0), _ = to_vehicle(self.pass_hold["line"], self.pose)
                if x0 > self.front - .5 and y0 > -self.half_width - .15:
                    room = x0 - self.front - .6
                    limit(math.sqrt(2 * s.brake_decel * room) if room > 0 else 0., "passing", "edging out around a stopped vehicle")
            if self.flags["wide"]:
                limit(s.wide_area, "wide_road_area")
            elif self.flags["uncertain"]:
                limit(s.wide_area, "uncertain_road")
        lookahead = 3.5 if self.pass_hold is not None else float(np.clip(4. + .5 * speed, 5., 11.))
        if guided:
            # Through a junction the path turns, so look ahead along it rather than straight ahead.
            lookahead, y_look = map(float, self._along(to_vehicle(self.junction["path"], self.pose), s.junction_lookahead))
            lookahead = max(lookahead, .5)
        elif path is not None and path[:, 0].max() > 1.:
            order = np.argsort(path[:, 0])
            y_look = float(np.interp(lookahead, path[order, 0], path[order, 1]))
        else:
            y_look = 0.
        distance = math.hypot(lookahead, y_look)
        alpha = math.atan2(y_look, lookahead)
        curvature = 2 * math.sin(alpha) / distance
        if path is not None and abs(curvature) > 1e-3:
            limit(math.sqrt(s.lateral_accel / abs(curvature)), "curve")
        for when, kind, point, label in self.hits:
            if self.started and available:
                ahead = float(to_vehicle(point[None], self.pose)[0, 0]) - self.front
                if ahead < -1.:
                    continue
                ahead = max(0., ahead)
                pretty = label.replace("_", " ")
                if kind == "vehicle":
                    gap = s.large_vehicle_gap if label in ("bus", "truck") else s.vehicle_gap
                    if ahead < s.emergency_m:
                        limit(0., "emergency_vehicle", f"{pretty} {ahead:.1f} m ahead")
                    limit(math.sqrt(2 * s.brake_decel * max(0., ahead - gap)), "vehicle_ahead", f"{pretty} {ahead:.1f} m ahead")
                elif kind == "person":
                    if ahead < s.emergency_m:
                        limit(0., "emergency_person", f"{pretty} {ahead:.1f} m ahead")
                    limit(math.sqrt(2 * s.brake_decel * max(0., ahead - s.person_gap)), "person_ahead", f"{pretty} {ahead:.1f} m ahead")
                elif kind == "person_near":
                    limit(s.near_person, "person_near_path", f"{pretty} {ahead:.0f} m ahead, beside the path")
                elif kind == "road_end":
                    limit(math.sqrt(2 * s.brake_decel * max(0., ahead - s.road_end_gap)), "road_ends", f"road not visible {ahead:.1f} m ahead")
        if self.reversing:
            # Still in reverse after backing out: stop before driving forward.
            if speed > .15:
                limit(0., "shifting_gear", "stopping before driving forward")
            else:
                self.reversing = False
                self.speed_integral = self.accel_command = 0.
        to_go = None if nav is None else nav.get("arrive_in_m")
        if (to_go is not None and to_go <= 25. and reason != "arrived" and speed < .5 and self.stopped_since is not None
                and sim_time - self.stopped_since >= 5.
                and any(kind == "vehicle" and self._at_destination(point, label, to_go) for _, kind, point, label in self.hits)):
            target, reason, detail = 0., "arrived", f"arrived {to_go:.0f} m short (a stopped vehicle is at the destination)"
        if reason == "vehicle_ahead" and self.pass_note and sim_time - self.pass_note_time <= 3.:
            detail = f"{detail} · {self.pass_note}"
        set_point = target
        if learned:
            target, reason, detail, set_point = self._learned_speed(target, reason, detail, speed, sim_time, guided, lost_for)
        steer = math.atan(2 * self.wheelbase * math.sin(alpha) / max(distance, 1e-3)) / self.max_steer
        self.steer += float(np.clip(steer - self.steer, -.07, .07))
        self.steer = float(np.clip(self.steer, -.7, .7))
        throttle, brake = self._pedals(speed, set_point, reason)
        return self._output(throttle, brake, target, reason, detail, y_look, lookahead)

    def _learned_speed(self, cap, why, detail, speed, sim_time, guided, lost_for):
        """The learned policy's share of the shield's speed `cap` (set by `why`). Returns target, reason, detail and the
        speed controller's set-point (the target, raised to hold a posted limit when the policy proceeds at it)."""
        s, nav = self.settings, self.nav or {}
        vehicle_m = person_m = math.inf
        near = False
        for _, kind, point, _ in self.hits:
            ahead = float(to_vehicle(np.asarray(point)[None], self.pose)[0, 0]) - self.front
            if ahead < -1.:
                continue
            ahead = max(0., ahead)
            if kind == "vehicle":
                vehicle_m = min(vehicle_m, ahead)
            elif kind == "person":
                person_m = min(person_m, ahead)
            elif kind == "person_near":
                near = near or ahead < s.person_near_range
        count, clear = 0, math.inf
        for track in self.people:
            if sim_time - track["seen"] > s.hold_s or not track["points"]:
                continue
            x, y = to_vehicle(np.array([track["points"][-1][1:]]), self.pose)[0]   # the person's centre
            ahead = float(x) - self.front
            if -1. <= ahead <= rl.PEOPLE_RANGE_M:
                count += 1
                clear = min(clear, abs(float(y)) - self.half_width - .3)
        if s.rl_speed_limits and nav.get("limit") is not None:
            posted, next_posted = float(nav["limit"]), float(nav.get("next_limit") or nav["limit"])
            next_m = nav.get("next_limit_m")
        else:
            posted, next_posted, next_m = s.cruise, s.cruise, None
        hold = nav.get("hold_at_m")
        ctx = {"v": speed, "accel": self.smoothed_accel, "cap": cap, "limit": posted, "next_limit": next_posted,
               "next_limit_m": math.inf if next_m is None else float(next_m), "veh_m": vehicle_m,
               "ped_m": person_m, "near_ped": float(near),
               "hold_m": math.inf if hold is None else float(hold), "junction": float(guided),
               "junction_m": math.inf if nav.get("junction_m") is None else float(nav["junction_m"]),
               "arrive_m": math.inf if nav.get("arrive_in_m") is None else float(nav["arrive_in_m"]),
               "curv": self._guide_curvature(), "stopped_s": 0. if self.stopped_since is None else sim_time - self.stopped_since,
               "area": float(self.flags["wide"] or self.flags["uncertain"] or lost_for > s.lost_after_s),
               "cap_cat": rl.category(why), "people_n": count, "people_clear": clear}
        fraction = self.governor.choose(ctx, sim_time)
        self.rl_last = {"cap": round(float(cap), 3), "cap_reason": why, "fraction": fraction}
        if fraction >= 1. or why.startswith("emergency") or cap <= 0.:
            set_point = cap
            # Safety net for the limit hold: over the posted limit by 0.3 m/s for 1 s (the pedal model is off), stop
            # raising the set-point for 30 s.
            if speed > posted + .3:
                self.over_since = sim_time if self.over_since is None else self.over_since
                if sim_time - self.over_since >= 1. and sim_time >= self.hold_paused_until:
                    self.hold_paused_until = sim_time + 30.
                    self._event(sim_time, "limit_hold_paused", f"{speed:.1f} m/s in a {posted:.1f} m/s limit")
            else:
                self.over_since = None
            aim = float(nav.get("cruise_limit", posted)) if s.rl_speed_limits and nav.get("limit") is not None else posted
            if (fraction >= 1. and why == "speed_limit" and s.rl_limit_hold and abs(cap - aim) < 1e-9
                    and sim_time >= self.hold_paused_until):
                # (faded out smoothly near the limit, never switched: see india_rl_speed.hold_weight)
                set_point = cap + float(rl.hold_offset(cap) * rl.hold_weight(speed, aim))
                if set_point > cap:
                    self.rl_last["set_point"] = round(set_point, 3)
            return cap, why, detail, set_point
        target = fraction * cap
        text = f"learned: {fraction:.0%} of {cap:.1f} m/s ({REASONS.get(why, why).lower()})"
        # Within 10% of the shield the shield's reason still says what holds the car back.
        return target, why if fraction >= .9 else "learned_slow", text if fraction < .9 else f"{detail or REASONS.get(why, why)} · {text}", target

    def _guide_curvature(self):
        """Largest |curvature| of the sat-nav's line over the next 30 m (1/m), 0 without one."""
        guide = None if self.nav is None else self.nav.get("guide_odo")
        if guide is None or len(guide) < 4:
            return 0.
        g = to_vehicle(guide, self.pose)
        step = np.hypot(*np.diff(g, axis=0).T)
        keep = step > 1e-3
        if keep.sum() < 3:
            return 0.
        d = np.diff(g, axis=0)[keep]
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        step = step[keep]
        s_mid = np.cumsum(step) - step / 2
        k = np.abs(np.diff(heading) / np.maximum(np.diff(s_mid), 1e-3))
        if len(k) >= 3:
            k = np.convolve(k, np.ones(3) / 3., mode="same")
        return float(k.max()) if len(k) else 0.

    def _at_destination(self, point, label, to_go):
        """A stopped vehicle (hit at `point`, odometry) stands at the destination: the pin lies under it or in
        the gap the car keeps behind it. Not a queue further back (final review, 9 Nov: any vehicle within
        25 m of the pin counted, so a queue 15-20 m before a junction place ended the trip early)."""
        s = self.settings
        ahead = float(to_vehicle(np.asarray(point)[None], self.pose)[0, 0]) - self.front
        gap = s.large_vehicle_gap if label in ("bus", "truck") else s.vehicle_gap
        pin = to_go - self.front
        return ahead - gap - 1.5 <= pin <= ahead + LENGTH_M.get(label, 4.) + 3.

    def _output(self, throttle, brake, target, reason, detail, y_look, lookahead):
        s = self.settings
        proceed = s.cruise
        if self.governor is not None and self.nav is not None and s.rl_speed_limits and self.nav.get("limit") is not None:
            proceed = float(self.nav["limit"])
        if target >= .9 * proceed:
            decision = "PROCEED"
        elif target < .3:
            decision = "STOP"
        else:
            decision = "SLOW"
        self.last_output = {"throttle": throttle, "brake": brake, "steer": self.steer, "target": target, "reason": reason,
                            "decision": decision, "detail": detail or REASONS.get(reason, reason), "lookahead_y": y_look,
                            "lookahead_m": lookahead, "reverse": self.reversing}
        if self.governor is not None:
            self.last_output["rl"] = self.rl_last
        return self.last_output

    def _wedged(self, speed, sim_time):
        """The throttle has been on for wedged_after_s with the car not moving (wanting to: target 0.3 m/s or
        more), so something the cameras do not show holds it: a kerb, a step, the car's own wheel in a gap."""
        out = self.last_output or {}
        last, self.wedge_checked = getattr(self, "wedge_checked", -1e9), sim_time
        if out.get("throttle", 0.) >= .4 and out.get("target", 0.) >= .3 and speed < .15:
            if self.pushing_since is None or sim_time - last > .2:   # (not checked meanwhile: start again)
                self.pushing_since = sim_time
            return sim_time - self.pushing_since >= self.settings.wedged_after_s
        self.pushing_since = None
        return False

    def _back_off(self, sim_time):
        """Back wedged_back_m the way the car came (its trail; straight back if it has none), with the rear
        camera's checks, then drive on and plan again."""
        s = self.settings
        here = np.array(self.pose[:2], float)
        points, run = [here], 0.
        for xy in reversed(self.trail[:-1]):
            step = float(np.hypot(*(np.asarray(xy) - points[-1])))
            if step < 1e-3:
                continue
            points.append(np.asarray(xy, float))
            run += step
            if run >= s.wedged_back_m:
                break
        if run < .5:   # no way it came to follow: straight back
            back = np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])
            points = [here - back * d for d in np.arange(0., s.wedged_back_m + .01, .25)]
        path = np.array(points[::-1])   # from the far end back to here
        self.backing = {"path": path, "start_index": 0, "why": "wedged", "since": sim_time, "kind": "wedged",
                        "best": float(np.hypot(*(path[0] - here))), "best_time": sim_time}
        self.wedge_log.append(sim_time)
        self.pushing_since = None
        self._event(sim_time, "wedged", "not moving with the throttle on: backing off the way it came")

    def _back_up(self, speed, sim_time):
        """Reverse along the abandoned pass path to where the pass began, slowly, stopping for
        anything the rear camera sees on the way back. Pure pursuit on a point behind the car."""
        s, back = self.settings, self.backing
        path = to_vehicle(back["path"], self.pose)   # from the far end of the straight extension forward
        k = back["start_index"]
        start, end = path[k], path[0]
        heading_error = math.remainder(back.get("heading", self.pose[2]) - self.pose[2], 2 * math.pi)   # vs the heading it started with
        remaining = float(math.hypot(*end))
        wedged = back.get("kind") == "wedged"
        why = "backing off: stuck against something it cannot see" if wedged else f"backing out: {back['why']} on its path"
        if (start[0] > .3 and abs(heading_error) < math.radians(2.5) and not wedged) or end[0] > -.25 or remaining < .3:
            self.backing = None
            if wedged:
                self._event(sim_time, "unwedged", "backed off; trying again")
                return self._output(0., .7, 0., "backing_wait", "backed off; trying again", 0., 0.)
            self._event(sim_time, "backed_out", f"back behind it, turned {math.degrees(heading_error):+.0f} degrees")
            self.steer += float(np.clip(-self.steer, -.07, .07))
            return self._output(0., .7, 0., "backing_wait", "back in its lane; waiting before trying again", 0., 0.)
        if remaining < back["best"] - .2:
            back["best"], back["best_time"] = remaining, sim_time
        elif sim_time - back["best_time"] > 20.:
            self.backing = None
            self._event(sim_time, "backing_given_up", "could not back out (blocked behind)")
            return self._output(0., .7, 0., "backing_wait", "could not back out: blocked behind", 0., 0.)
        target, reason, detail = s.backing, "backing_out", why
        available = self.last_ok_time is not None and sim_time - self.last_ok_time <= s.unavailable_after_s
        fresh = self.rear_obstacles is not None and self.rear_time is not None and sim_time - self.rear_time <= .6
        if not available or (self.rear_geometry is not None and not fresh):
            target, reason, detail = 0., "backing_wait", "waiting for the rear camera"
        else:
            order = np.argsort(path[:, 0])
            for o in self.rear_obstacles or []:
                y = float(np.interp(o["x"], path[order, 0], path[order, 1]))
                margin = s.person_margin if o["person"] else s.vehicle_margin
                if o["y_left"] - self.half_width - margin <= y <= o["y_right"] + self.half_width + margin and o["x"] >= end[0] - 3.:
                    allowed = math.sqrt(2 * s.brake_decel * max(0., o["behind_m"] - s.rear_gap))
                    if allowed < target:
                        target, reason = allowed, "backing_out" if allowed > .1 else "backing_wait"
                        detail = f"{o['class'].replace('_', ' ')} {o['behind_m']:.1f} m behind"
            target = min(target, math.sqrt(2 * 1. * max(0., remaining - .1)))
        if not self.reversing:
            if speed > .15:
                target, reason, detail = 0., "shifting_gear", "stopping before reversing"
            else:
                self.reversing = True
                self.speed_integral = self.accel_command = 0.
        # Steer for a point about 2 m back along the path (the same circle geometry as forward).
        distance = np.hypot(path[:, 0], path[:, 1])
        nearest = int(np.argmin(distance))
        aim = end
        for j in range(nearest, -1, -1):
            if distance[j] >= 2. and path[j, 0] < 0:
                aim = path[j]
                break
        curvature = 2 * float(aim[1]) / max(float(aim[0] ** 2 + aim[1] ** 2), 1e-3)
        steer = math.atan(self.wheelbase * curvature) / self.max_steer
        self.steer += float(np.clip(steer - self.steer, -.07, .07))
        self.steer = float(np.clip(self.steer, -.7, .7))
        if not self.reversing:
            throttle, brake = self._pedals(speed, 0., reason)
        else:
            throttle, brake = self._pedals(speed, target, reason)
            throttle = min(throttle, .5)
        return self._output(throttle, brake, target, reason, detail, float(aim[1]), float(aim[0]))

    # ---- turning round (a U-turn on the route) ------------------------------------------
    def _remember_seen(self, obstacles, pose, sim_time):
        items = []
        for o in obstacles:
            x0, x1 = o["x"], o["x"] + o.get("length", 1.)
            box = np.array([[x0, o["y_left"]], [x1, o["y_left"]], [x1, o["y_right"]], [x0, o["y_right"]]], float)
            items.append({"person": o["person"], "class": o["class"], "corners": to_odometry(box, pose)})
        self.seen = (sim_time, items)
        centres = [it["corners"].mean(axis=0) for it in items if not it["person"]]
        self.seen_log = [e for e in self.seen_log if sim_time - e[0] <= 3.] + [(sim_time, centres)]
        self._track(sim_time, "front", [(it["class"], it["person"], it["corners"]) for it in items])

    def _track(self, sim_time, view, items):
        self.track_log = [e for e in self.track_log if sim_time - e[0] <= 3.] + [(sim_time, view, items)]

    def _moving(self, points, sim_time):
        """A sighting (odometry points) moved 0.5 m or more since the nearest one (within 8 m) of 0.6-1.6 s ago,
        seen by any camera; one with nothing to match is taken as moving. A vehicle filling a side camera's picture
        shows only the part in view, so one crawling past at walking pace looks still."""
        centre = points.mean(axis=0)
        old = [float(np.hypot(*(p.mean(axis=0) - centre))) for t, _, items in self.track_log if .6 <= sim_time - t <= 1.6
               for _, _, p in items]
        near = min(old, default=99.)
        return near > 8. or near >= .5

    def _still(self, centre, sim_time, log=None):
        """A vehicle at `centre` (odometry) was within 0.6 m of there 1.5 s or more ago (standing, not coming)."""
        old = [e for e in (self.seen_log if log is None else log) if 1.5 <= sim_time - e[0] <= 3.]
        return bool(old) and any(min((float(np.hypot(*(c - centre))) for c in cs), default=9.) < .6 for _, cs in old)

    def _street_frame(self, pose=None):
        """The car's pose in the U-turn's street frame: along the original travel, to its right, heading."""
        x0, y0, yaw0 = self.uturn["frame"]
        px, py, psi = pose or self.uturn.get("axle") or self.pose
        c, sn = math.cos(yaw0), math.sin(yaw0)
        return (px - x0) * c + (py - y0) * sn, -(px - x0) * sn + (py - y0) * c, psi - yaw0

    def _corner_sides(self, side, theta):
        """Distance to the right (street frame) of each corner of the car's body."""
        xs = (self.front, self.front, self.front - 2 * self.half_length, self.front - 2 * self.half_length)
        ys = (-self.half_width, self.half_width, self.half_width, -self.half_width)
        return [side + a * math.sin(theta) + b * math.cos(theta) for a, b in zip(xs, ys)]

    def _move_room(self, side, theta, direction, steer):
        """The car can move on a little (`direction` +1 forward, -1 back, at this steering) with every corner
        kept uturn_edge_m inside the road, or moving away from the edge it is near."""
        s, u = self.settings, self.uturn
        reach = .25 + self.speed ** 2 / (2 * s.brake_decel) + .1 * self.speed
        k = math.tan(steer * self.max_steer) / self.wheelbase
        # A car turns about its rear axle (taken as half a wheelbase behind the centre): the tail barely swings out.
        theta2 = theta + direction * reach * k
        rear = side - self.wheelbase / 2 * math.sin(theta) + direction * reach * math.sin((theta + theta2) / 2)
        side2 = rear + self.wheelbase / 2 * math.sin(theta2)
        lo, hi = -u["left"] + s.uturn_edge_m, u["right"] - s.uturn_edge_m
        for now, then in zip(self._corner_sides(side, theta), self._corner_sides(side2, theta2)):
            if (then > hi and then > now) or (then < lo and then < now):
                return False
        return True

    def _on_turning_road(self, points):
        """An obstacle (`points` in the car's own frame) comes into the part of the road the turn uses. The car's
        body stays uturn_edge_m inside the road's edges, so something that does not reach past that strip (a stall,
        a parked car or people on the pavement; camera estimates err towards the car) cannot be hit. 18 Nov run:
        a food stall on the pavement behind ("truck 1.0 m behind") held the backing move for 80 s, and people
        standing on the pavement held the gap for a minute."""
        u = self.uturn
        x0, y0, yaw0 = u["frame"]
        odo = to_odometry(np.asarray(points, float), u.get("axle") or self.pose)
        side = -(odo[:, 0] - x0) * math.sin(yaw0) + (odo[:, 1] - y0) * math.cos(yaw0)
        strip = self.settings.uturn_edge_m
        return bool(side.max() > -u["left"] + strip and side.min() < u["right"] - strip)

    def _rear_points(self, o):
        """A rear obstacle's footprint (car frame) for the turn's checks. Vehicles and stalls stand along the
        street, so with the car across it they are seen side-on: a bumper's overhang beyond the box's bottom
        counts only as far as the car faces along the street (18 Nov: a stall read as a truck was put 1.2 m
        nearer, onto the road). A box cut off at the picture's bottom may reach right up to the bumper."""
        length = LENGTH_M.get(o["class"], 4.)
        if "ground_x" not in o:
            x = o["x"]
        elif o["cut_bottom"]:
            x = self.front - 2 * self.half_length
        else:
            x = o["ground_x"] + OVERHANG_M.get(o["class"], 0.) * abs(math.cos(self._street_frame()[2]))
        return [(x, o["y_left"]), (x, o["y_right"]), (x - length, o["y_right"]), (x - length, o["y_left"])]

    def _read_behind(self, o):
        """A rear box read properly (its distance from the ground under it) not far behind, where something could
        come up to the tail from: not a stall on the pavement while turning round."""
        s = self.settings
        if not (s.uturn_rear_seen_m <= o["behind_m"] <= 6. and o["y_left"] <= self.half_width + 3.
                and o["y_right"] >= -self.half_width - 3.):
            return False
        return self.uturn is None or self._on_turning_road(self._rear_points(o))

    def _rear_believed(self, o):
        """A rear box counts for the turn's checks. Nothing gets within uturn_rear_seen_m of the tail unseen: the car
        stops 2.5 m short of what it sees behind, and anything else coming up is read farther out first. A box that
        near counts only if something was read close behind (or the camera started) and the car has not driven
        uturn_rear_reveal_m forward since, which would show a real one farther out. 18 Nov CARLA rerun: boxes over
        most of the rear picture ("bus 0.0 m behind", "autorickshaw 0.7 m behind", nothing there) held the backing
        moves for about 20 s."""
        s = self.settings
        if o["behind_m"] >= s.uturn_rear_seen_m:
            return True
        if self.rear_seen_at is None:
            return False
        x0, y0, yaw0 = self.rear_seen_at
        x, y, yaw = self.pose
        tail = self.front - 2 * self.half_length
        moved = (x + tail * math.cos(yaw) - x0 - tail * math.cos(yaw0)) * math.cos(yaw0) \
            + (y + tail * math.sin(yaw) - y0 - tail * math.sin(yaw0)) * math.sin(yaw0)
        return moved < s.uturn_rear_reveal_m

    def _seen_near(self, direction, sim_time):
        """Something the cameras see in the way of the next metre or two of this move, or None."""
        s = self.settings
        if direction > 0:
            if self.seen is None or sim_time - self.seen[0] > .6:
                return "waiting for the camera"
            for item in self.seen[1]:
                v = to_vehicle(item["corners"], self.pose)
                if not self._on_turning_road(v):
                    continue
                gap = s.person_margin if item["person"] else s.vehicle_margin
                if v[:, 0].min() <= self.front + 2.5 and v[:, 0].max() >= -self.half_length \
                        and v[:, 1].min() <= self.half_width + gap + .5 and v[:, 1].max() >= -self.half_width - gap - .5:
                    return f"{item['class'].replace('_', ' ')} close ahead"
            return None
        if self.rear_geometry is None:
            return None
        if self.rear_obstacles is None or self.rear_time is None or sim_time - self.rear_time > .6:
            return "waiting for the rear camera"
        for o in self.rear_obstacles:
            if o["behind_m"] <= 2.5 and o["y_left"] <= self.half_width + .6 and o["y_right"] >= -self.half_width - .6 \
                    and self._on_turning_road(self._rear_points(o)) and self._rear_believed(o):
                return f"{o['class'].replace('_', ' ')} {o['behind_m']:.1f} m behind"
        return None

    def _gap_to_turn(self, sim_time):
        """Why it is not yet safe to start turning round across the road, or None."""
        s, u = self.settings, self.uturn
        if self.seen is None or sim_time - self.seen[0] > .6:
            return "waiting for the camera"
        far = u["right"] + 2.
        for item in self.seen[1]:
            v = to_vehicle(item["corners"], self.pose)
            x = float(v[:, 0].min())
            if not self._on_turning_road(v):
                continue   # on the pavement: a stall, a parked car, people standing there
            if item["person"] and x - self.front <= s.uturn_clear_people_m:
                return f"{item['class'].replace('_', ' ')} {max(0., x - self.front):.0f} m ahead"
            if not item["person"] and x - self.front <= s.uturn_clear_ahead_m:
                if x - self.front > 8. and self._still(item["corners"].mean(axis=0), sim_time):
                    continue   # standing well ahead (parked, or queuing the other way): not coming
                return f"{item['class'].replace('_', ' ')} {max(0., x - self.front):.0f} m ahead"
        if self.rear_geometry is not None:
            if self.rear_obstacles is None or self.rear_time is None or sim_time - self.rear_time > .6:
                return "waiting for the rear camera"
            for o in self.rear_obstacles:
                # Overtaking it on the right (where it is about to turn) or right behind it on that side.
                if not o["person"] and o["behind_m"] <= 20. and o["y_right"] > self.half_width + .2 and o["y_left"] < far:
                    centre = to_odometry(np.array([[o["x"] - LENGTH_M.get(o["class"], 4.) / 2, (o["y_left"] + o["y_right"]) / 2]]),
                                         self.pose)[0]
                    if not self._on_turning_road(self._rear_points(o)) or self._still(centre, sim_time, self.rear_log) \
                            or not self._rear_believed(o):
                        continue   # off the road, standing (parked, queuing behind it), or not there: not overtaking
                    return f"{o['class'].replace('_', ' ')} {o['behind_m']:.0f} m behind on its right"
        return None

    def _turn_around(self, speed, sim_time):
        """Turn round across a two-way street: wait for a gap, then forward on full right lock and back on full
        left lock between the road's edges (from the sat-nav's map) until it faces the other way, then line up
        with the far lane and hand back to the route. Stops for anything the cameras see in the way."""
        s, u = self.settings, self.uturn
        dyaw = math.remainder(self.pose[2] - u["last_yaw"], 2 * math.pi)
        u["theta"] += dyaw
        u["last_yaw"] = self.pose[2]
        u["travelled"] += float(math.hypot(u["axle"][0] - u["last_xy"][0], u["axle"][1] - u["last_xy"][1]))
        u["last_xy"] = u["axle"][:2]
        along, side, _ = self._street_frame()
        theta = u["theta"]
        left_to_turn = math.pi - theta
        available = self.last_ok_time is not None and sim_time - self.last_ok_time <= s.unavailable_after_s
        phase, lock = u["phase"], s.uturn_steer
        target, reason, detail = s.uturn_speed, "uturn_forward", f"turned {math.degrees(theta):.0f} degrees"
        want_reverse = phase == "reverse"
        steer_goal = lock
        if sim_time - u["started"] > s.uturn_give_up_s or u["moves"] > 12:
            self._finish_uturn(sim_time, "uturn_given_up", f"gave up after {u['moves']} moves, turned {math.degrees(theta):.0f} degrees",
                               follow_route=False)
            return self._output(0., .7, 0., "uturn_wait", "gave up turning round", 0., 0.)
        if phase == "wait":
            target, reason = 0., "uturn_wait"
            why = self._gap_to_turn(sim_time) if available else "perception unavailable"
            if why is None:
                u["clear_since"] = u["clear_since"] if u["clear_since"] is not None else sim_time
                if sim_time - u["clear_since"] >= s.uturn_clear_s and abs(self.steer - lock) < .1 and speed < .1:
                    u.update(phase="forward", moves=1, move_from=u["travelled"], blocked_since=None)
                    self._event(sim_time, "uturn_move", "forward across the road")
                    phase = "forward"
                detail = "clear: turning the wheel"
            else:
                u["clear_since"] = None
                detail = f"waiting for a gap: {why}"
        if phase == "forward":
            if left_to_turn < math.radians(40.):
                # Nearly round: steer for the far lane's centre line (pure pursuit, 3 m on along it).
                dx, dy = -3., u["across"] - side
                gv = (dx * math.cos(theta) + dy * math.sin(theta), -dx * math.sin(theta) + dy * math.cos(theta))
                d = max(math.hypot(*gv), .5)
                steer_goal = float(np.clip(math.atan(2 * self.wheelbase * gv[1] / d ** 2) / self.max_steer, -lock, lock))
            if abs(left_to_turn) < math.radians(10.) and (abs(side - u["across"]) < .7 or u["travelled"] - u["move_from"] > 8.):
                self._finish_uturn(sim_time, "uturn_done", f"turned round in {u['moves']} moves")
                return self._output(0., 0., s.uturn_speed, "uturn_forward", "turned round", 0., 3.)
            wedged = self._wedged(speed, sim_time)   # a kerb short of the map's edge: as at the edge
            if not self._move_room(side, theta, 1, self.steer) or wedged:
                target, detail = 0., "against something it cannot see" if wedged else "at the road's edge"
                if speed < .15:
                    u.update(phase="reverse", moves=u["moves"] + 1, move_from=u["travelled"], blocked_since=None)
                    self._event(sim_time, "uturn_move", f"backing up{' (stuck going forward)' if wedged else ''}, "
                                                        f"turned {math.degrees(theta):.0f} degrees")
                    want_reverse, steer_goal = True, -lock
            else:
                why = self._seen_near(1, sim_time) if available else "perception unavailable"
                if why is not None:
                    target, reason, detail = 0., "uturn_wait", why
                    if left_to_turn > math.radians(40.) and self._blocked_long(sim_time, speed) \
                            and self._move_room(side, theta, -1, -lock):
                        u.update(phase="reverse", moves=u["moves"] + 1, move_from=u["travelled"], blocked_since=None)
                        self._event(sim_time, "uturn_move", f"backing up (blocked ahead), turned {math.degrees(theta):.0f} degrees")
                        want_reverse, steer_goal, reason = True, -lock, "uturn_reverse"
                else:
                    u["blocked_since"] = None
        elif phase == "reverse":
            reason, steer_goal, detail = "uturn_reverse", -lock, f"backing up, turned {math.degrees(theta):.0f} degrees"
            backed = u["travelled"] - u["move_from"]
            enough = (left_to_turn < math.radians(30.) and backed > .3) or backed > s.uturn_reverse_m
            wedged = self._wedged(speed, sim_time)
            if enough or wedged or not self._move_room(side, theta, -1, self.steer):
                target, detail = 0., "room to go forward" if enough else "against something it cannot see" if wedged                     else "at the road's edge"
                if speed < .15:
                    u.update(phase="forward", moves=u["moves"] + 1, move_from=u["travelled"], blocked_since=None)
                    self._event(sim_time, "uturn_move", f"forward{' (stuck backing up)' if wedged and not enough else ''}, "
                                                        f"turned {math.degrees(theta):.0f} degrees")
                    want_reverse, steer_goal, reason = False, lock, "uturn_forward"
            else:
                why = self._seen_near(-1, sim_time) if available else "perception unavailable"
                if why is not None:
                    target, reason, detail = 0., "uturn_wait", why
                    if self._blocked_long(sim_time, speed) and self._move_room(side, theta, 1, lock):
                        u.update(phase="forward", moves=u["moves"] + 1, move_from=u["travelled"], blocked_since=None)
                        self._event(sim_time, "uturn_move", f"forward (blocked behind), turned {math.degrees(theta):.0f} degrees")
                        want_reverse, steer_goal, reason = False, lock, "uturn_forward"
                else:
                    u["blocked_since"] = None
        if phase in ("forward", "reverse") and target > 0 and available:
            coming = self._coming(sim_time)
            if coming is not None:
                # The car is across the street: stop where it is and let it stop (or go by) rather than move on
                # into its way.
                target, reason, detail = 0., "uturn_wait", coming
        if not available:
            target, reason = 0., "uturn_wait"
        # Change gear only when stopped; turn the wheel before moving off.
        if want_reverse != self.reversing:
            if speed > .15:
                target, reason, detail = 0., "shifting_gear", "stopping to change direction"
            else:
                self.reversing = want_reverse
                self.speed_integral = self.accel_command = 0.
        if abs(self.steer - steer_goal) > .25 and speed < .3:
            target = 0.
        self.steer += float(np.clip(steer_goal - self.steer, -.07, .07))
        self.steer = float(np.clip(self.steer, -lock, lock))
        throttle, brake = self._pedals(speed, target, reason)
        if self.reversing:
            throttle = min(throttle, .5)
        return self._output(throttle, brake, target, reason, detail, 0., 0.)

    def _coming(self, sim_time):
        """Something on the road coming towards the car along the street while it is across it, or None. That
        traffic is beside the car, where only the side cameras see it (things ahead and behind are the moves' own
        checks; the front camera's distances also shift as the car turns, so a parked car looked like one coming).
        Each side sighting of the last 0.6 s is matched to the nearest sighting by any camera of 0.6-1.6 s ago
        (within 8 m); it is coming if it got nearer the car's present position at uturn_coming_speed or more (the
        car's own motion is not counted)."""
        s = self.settings
        here = np.array(self.pose[:2], float)
        latest = {}
        for t, view, items in self.track_log:
            if sim_time - t <= .6 and view in self.side_geometry:
                latest[view] = (t, items)
        old = [(t, person, points.mean(axis=0)) for t, _, items in self.track_log if .6 <= sim_time - t <= 1.6
               for _, person, points in items]
        for view, (t, items) in latest.items():
            for name, person, points in items:
                near = float(np.min(np.hypot(points[:, 0] - here[0], points[:, 1] - here[1])))
                if near > s.uturn_coming_m:
                    continue
                centre = points.mean(axis=0)
                match = min(((float(np.hypot(*(c - centre))), ot, c) for ot, p, c in old if p == person), default=None,
                            key=lambda m: m[0])
                if match is None or match[0] > 8.:
                    continue
                closing = (float(np.hypot(*(match[2] - here))) - float(np.hypot(*(centre - here)))) / max(t - match[1], .3)
                if closing < s.uturn_coming_speed:
                    continue
                v = to_vehicle(points, self.pose)
                if not self._on_turning_road(v):
                    continue   # on the pavement
                # Its direction from the car (18 Nov rerun: a motorcycle 5 m forward and 17 m right read "ahead").
                bearing = math.degrees(math.atan2(*v.mean(axis=0)[::-1]))
                where = "ahead" if abs(bearing) <= 45. else "behind" if abs(bearing) >= 135. \
                    else "on the right" if bearing > 0 else "on the left"
                return f"{name.replace('_', ' ')} coming {where}, {max(0., near - self.half_length):.0f} m"
        return None

    def _blocked_long(self, sim_time, speed):
        """This move has been held by something in its way for 2 s, stopped, after covering 0.3 m or more: end
        it there and go the other way, as at the road's edge (18 Nov: a stall behind, which the rear camera could
        not place, held the backing move with the car half round and room to go forward)."""
        u = self.uturn
        u["blocked_since"] = u.get("blocked_since") or sim_time
        return sim_time - u["blocked_since"] >= 2. and speed < .15 and u["travelled"] - u["move_from"] >= .3

    def _finish_uturn(self, sim_time, kind, detail, follow_route=True):
        self._event(sim_time, kind, detail)
        self.uturn, self.uturn_done_at = None, sim_time
        self.uturn_failed = not follow_route   # the sat-nav reads this and re-plans without turning round
        if follow_route:
            # Follow the route line until settled on the far lane's kerb (as after a junction).
            self.junction = {"path": None, "fallback": False, "planned_at": -1e9, "blocked_since": None}

    def _pedals(self, speed, target, reason):
        """The rule driver's speed controller (india_behavior.Driver.control) for holding
        speed, plus a direct brake whenever the car is faster than its target: a camera
        obstacle can appear late, and that controller eases off too slowly to stop in time."""
        error = target - speed
        self.smoothed_accel = .8 * self.smoothed_accel + .2 * (speed - self.previous_speed) / DT
        if target < self.previous_target - .3 or error < -.5:
            self.speed_integral = min(0., self.speed_integral)
        self.speed_integral = float(np.clip(self.speed_integral + error * DT, -3, 3))
        requested = float(np.clip(.85 * error + .35 * self.speed_integral - .4 * self.smoothed_accel, -3.5, 2.1))
        self.previous_speed, self.previous_target = speed, target
        emergency = reason.startswith("emergency")
        self.accel_command += float(np.clip(requested - self.accel_command, -.15 if not emergency else -8, .12))
        if target < .08 and speed < .2:
            return 0., .7
        if emergency:
            return 0., 1.
        if error < -.3:
            self.accel_command = min(self.accel_command, 0.)
            return 0., float(np.clip(.1 + .16 * -error, 0, .75))
        if self.accel_command >= 0:
            return float(np.clip(.16 + .32 * self.accel_command, 0, .85)), 0.
        return 0., float(np.clip(-self.accel_command * .06, 0, .25))
