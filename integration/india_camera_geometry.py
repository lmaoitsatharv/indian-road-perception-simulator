"""Fixed front-camera geometry for camera-only driving (flat ground, CARLA axes).

Vehicle frame: origin at the car's actor origin on the ground, x forward, y right,
z up (CARLA's convention, so a positive lateral value and a positive steer both
mean "to the right"). Image: u to the right, v down. The camera looks straight
ahead (pitch 0, yaw 0 by default; yaw 180 is a rear-facing camera) with a
horizontal field of view, like a CARLA RGB sensor. Pure NumPy; no simulator access.
"""
from __future__ import annotations

import math

import numpy as np

MASK_SIZE = 512  # the road model's square input/output


class CameraGeometry:
    def __init__(self, width=960, height=540, fov_deg=95., mount=(1.35, .25, 1.45), pitch_deg=0., yaw_deg=0.):
        self.width, self.height = int(width), int(height)
        self.fov_deg = float(fov_deg)
        self.f = self.width / (2 * math.tan(math.radians(self.fov_deg) / 2))
        self.cx, self.cy = self.width / 2, self.height / 2
        self.mount = np.asarray(mount, dtype=float)
        self.pitch = math.radians(pitch_deg)
        self.yaw = math.radians(yaw_deg)   # positive turns the camera to the right (CARLA)

    def to_camera(self, x, y, z=0.):
        """Vehicle-frame points -> (forward, right, up) in the camera frame."""
        dx = np.asarray(x, float) - self.mount[0]
        dy = np.asarray(y, float) - self.mount[1]
        dz = np.asarray(z, float) - self.mount[2]
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = dx * cy + dy * sy, -dx * sy + dy * cy
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        return dx * c + dz * s, dy, -dx * s + dz * c

    def vehicle_from_camera(self, forward, right):
        """A ground-level offset seen from the camera (forward, right) -> vehicle-frame (x, y)."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        forward, right = np.asarray(forward, float), np.asarray(right, float)
        return self.mount[0] + forward * cy - right * sy, self.mount[1] + forward * sy + right * cy

    def project(self, x, y, z=0.):
        """Vehicle-frame points -> image pixels (u, v) and a validity flag."""
        forward, right, up = self.to_camera(x, y, z)
        safe = np.where(forward > .05, forward, np.nan)
        u = self.cx + self.f * right / safe
        v = self.cy - self.f * up / safe
        valid = (forward > .05) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return np.nan_to_num(u, nan=-1.), np.nan_to_num(v, nan=-1.), valid

    def ground_from_pixel(self, u, v):
        """Image pixels -> flat-ground vehicle-frame (x, y); NaN above the horizon."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        # Ray in camera frame: forward 1, right (u-cx)/f, up -(v-cy)/f.
        right, up = (u - self.cx) / self.f, -(v - self.cy) / self.f
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        # Rotate camera ray into the vehicle frame (inverse of to_camera's pitch, then yaw).
        ray_f, ray_r, ray_z = c - up * s, right, s + up * c
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ray_z < -1e-6, -self.mount[2] / ray_z, np.nan)
        return self.vehicle_from_camera(t * ray_f, t * ray_r)

    def to_mask(self, u, v):
        """Camera pixels -> integer (row, column) in the 512x512 road-model output."""
        col = np.clip((np.asarray(u) * MASK_SIZE / self.width).astype(int), 0, MASK_SIZE - 1)
        row = np.clip((np.asarray(v) * MASK_SIZE / self.height).astype(int), 0, MASK_SIZE - 1)
        return row, col

    def nearest_visible_ground(self):
        """Forward distance (vehicle frame) of the lowest image row on the ground."""
        x, _ = self.ground_from_pixel(self.cx, self.height - .5)
        return float(x)


class GroundGrid:
    """A bird's-eye grid of ground cells with a fixed lookup into the road mask."""

    def __init__(self, geometry: CameraGeometry, x_range=(2.5, 25.), y_range=(-10., 10.), cell=.1):
        self.geometry, self.cell = geometry, float(cell)
        self.xs = np.round(np.arange(x_range[0], x_range[1] + cell / 2, cell), 4)
        self.ys = np.round(np.arange(y_range[0], y_range[1] + cell / 2, cell), 4)
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        u, v, self.valid = geometry.project(gx, gy)
        self.mask_row, self.mask_col = geometry.to_mask(u, v)
        # The camera pixel under each ground cell (for reading the picture itself).
        self.pixel_u = np.clip(u.astype(int), 0, geometry.width - 1)
        self.pixel_v = np.clip(v.astype(int), 0, geometry.height - 1)

    def row_of(self, x):
        return int(np.clip(round((x - self.xs[0]) / self.cell), 0, len(self.xs) - 1))

    def col_of(self, y):
        return int(np.clip(round((y - self.ys[0]) / self.cell), 0, len(self.ys) - 1))

    def brightness(self, rgb):
        """Luma (0-255) of the camera picture under each ground cell."""
        picture = np.asarray(rgb)
        if picture.shape[:2] != (self.geometry.height, self.geometry.width):
            raise ValueError("Expected the camera's own picture")
        return picture[self.pixel_v, self.pixel_u].astype(np.float32) @ np.array([.299, .587, .114], np.float32)

    def road(self, mask):
        """Boolean road grid from the road model's 512x512 binary mask (0/1)."""
        values = np.asarray(mask)
        if values.shape != (MASK_SIZE, MASK_SIZE):
            raise ValueError("Expected a 512x512 road-model mask")
        return (values[self.mask_row, self.mask_col] == 1) & self.valid
