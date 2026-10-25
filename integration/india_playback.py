"""Wall-clock playback pacing; never alters the simulation's physics step."""
from collections import deque
import math

SPEEDS = (1.0, 1.5, 2.0, 2.5, 3.0)


class Playback:
    def __init__(self, speed=1.0, *, dt=.05):
        if speed not in SPEEDS or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Use a supported playback speed and a positive fixed step")
        self.speed, self.dt = float(speed), dt
        self.actual = None
        self.samples = []
        self._window = deque()
        self._last_sample = None
        self._paused = False

    def select(self, speed):
        if speed not in SPEEDS:
            raise ValueError("Playback speed must be 1, 1.5, 2, 2.5 or 3")
        changed = self.speed != speed
        self.speed = float(speed)
        if changed:
            self._reset()
        return changed

    def adjust(self, direction):
        index = max(0, min(len(SPEEDS) - 1, SPEEDS.index(self.speed) + direction))
        return self.select(SPEEDS[index])

    def _reset(self):
        self._window.clear()
        self._last_sample = None
        self.actual = None

    def observe(self, wall_s, sim_s, paused):
        if paused or self._paused != paused:
            self._reset()
        self._paused = paused
        if paused:
            return None
        if self._window and (wall_s < self._window[-1][0] or sim_s < self._window[-1][1]):
            raise ValueError("Playback measurements must be monotonic")
        self._window.append((wall_s, sim_s))
        while len(self._window) > 2 and wall_s - self._window[1][0] >= 1.0:
            self._window.popleft()
        first_wall, first_sim = self._window[0]
        elapsed = wall_s - first_wall
        if elapsed >= .5:
            self.actual = max(0., (sim_s - first_sim) / elapsed)
        if self.actual is not None and (self._last_sample is None or wall_s - self._last_sample >= 1):
            record = {"sim_s": sim_s, "requested_speed": self.speed,
                      "measured_speed": self.actual, "window_wall_s": elapsed}
            self.samples.append(record)
            self._last_sample = wall_s
            return record
        return None

    def fps_limit(self, paused=False):
        return 20 if paused else round(self.speed / self.dt)
