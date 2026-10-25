"""Capture paired front RGB/semantic images; no prediction or vehicle control here."""
import hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import queue
import time

import carla
import numpy as np
import pygame

from carla_starter import write_json
from india_camera import configure_rgb
from india_evidence_io import write_rgb_png, write_depth_npz


class RoadObserver:
    def __init__(self, world, ego, owned, sensors, folder, surface_style="legacy", capture_hz=5, camera_profile=None, live_sink=None):
        if capture_hz not in (1, 5):
            raise ValueError("Use an explicit 1 or 5 Hz road capture rate")
        self.folder = folder
        self.live_sink = live_sink
        folder.mkdir(exist_ok=False)
        self.inbox = queue.Queue(maxsize=128)
        self.pending = {"rgb": {}, "labels": {}, "depth": {}}
        if surface_style == "asphalt":
            self.pending["instance"] = {}
        self.snapshots, self.records = {}, []
        self.writer = ThreadPoolExecutor(max_workers=2, thread_name_prefix="road-evidence")
        self.writes, self.write_ms, self.backpressure_s = deque(), [], 0.
        self.dropped = {"queue_overflow": 0, "expired_images": 0}
        self.log = (folder / "frames.jsonl").open("w", encoding="utf-8")
        # A fixed forward camera at the windscreen, independent of the viewer camera.
        mount = carla.Transform(carla.Location(x=1.4, y=.30, z=1.35))
        profile = camera_profile or {"height_m": 1.35, "pitch_deg": 0., "yaw_deg": 0., "fov_deg": 95.}
        mount.location.z = profile["height_m"]
        mount.rotation.pitch, mount.rotation.yaw = profile["pitch_deg"], profile["yaw_deg"]
        self.config = {"schema_version": 2, "carla_version": "0.9.16", "size_wh": [960, 540],
            "fov_horizontal_degrees": 95, "sensor_tick_s": 1/capture_hz, "physics_dt_s": .05,
            "mount_ego_xyz_m": [1.4, .30, 1.35], "mount_rotation_degrees": [0, 0, 0],
            "lens_k": 0, "lens_kcube": 0, "model_control": False,
            "reference": {"road_ids": [1, 24], "ignore_ids": [0],
                "other_ids": "2..28 excluding 24 are non-road; out-of-schema IDs reject scoring",
                "occlusions": "Visible surfaces only; people and vehicles are non-road",
                "unlabeled": "Initially unknown; depth and the saved map resolve nearby surfaces, remaining unknown pixels are excluded with coverage reported",
                "road_markings": "RoadLine 24 and underlying road 1 count as road; debug paint has no independent tag",
                "source": "https://carla.readthedocs.io/en/0.9.16/ref_sensors/#semantic-segmentation-camera",
                "method": "Native semantic tags plus depth-checked OpenDRIVE driving-lane footprint for unlabeled procedural surfaces",
                "geometry_range_m": 80, "surface_height_tolerance_m": .08, "boundary_uncertainty_m": .2,
                "unknown_beyond_geometry_range": "Ignored; known native classes remain labeled",
                "grid": {"origin_xy_m": [-154, -154], "cell_m": .1, "shape_hw": [3081, 3081],
                         "file": "driving_footprint.npz", "waypoint_step_m": .5, "tile_half_length_m": .40}}}
        self.config.update(fov_horizontal_degrees=profile["fov_deg"],
            mount_ego_xyz_m=[1.4,.30,profile["height_m"]],
            mount_rotation_degrees=[profile["pitch_deg"],profile["yaw_deg"],0], camera_profile=profile)
        self.config["dataset_edge_disagreement_policy"] = (
            "At most 0.1% semantic/instance disagreement; preserve both native streams and ignore every disagreeing pixel"
            if camera_profile else None)
        self.config["weather_probe_only"] = bool(profile.get("weather_probe_only",False))
        weather = world.get_weather()
        self.config["actual_weather"] = {key:float(getattr(weather,key)) for key in (
            "cloudiness","sun_altitude_angle","sun_azimuth_angle","precipitation","wetness",
            "scattering_intensity","mie_scattering_scale","rayleigh_scattering_scale","fog_density","fog_distance","fog_falloff")}
        focal = 960 / (2 * math.tan(math.radians(profile["fov_deg"] / 2)))
        self.config["intrinsics_source"] = [[focal, 0, 480], [0, focal, 270], [0, 0, 1]]
        self.config["surface_style"] = surface_style
        if surface_style == "asphalt":
            surface = json.loads((folder.parent.parent / "surface_style.json").read_text(encoding="utf-8"))
            if not surface["actor_ids"] or max(surface["actor_ids"]) > 65535:
                raise ValueError("Appearance actor IDs do not fit the native instance-camera encoding")
            self.config["schema_version"] = 3
            self.config["reference"]["appearance_actor_ids"] = surface["actor_ids"]
            self.config["reference"]["appearance_native_class"] = 3
            self.config["reference"]["other_ids"] = "Known classes remain non-road; only exact owned appearance-instance pixels are resolved with depth and map"
            self.config["reference"]["appearance_method"] = "Only owned asphalt instances are resolved with depth and map; other buildings retain their native non-road label"
            self.config["instance_encoding"] = "Raw RGB: R=semantic class; G=actor ID low byte; B=actor ID high byte"
            paving_path = folder.parent.parent / "streetscape.json"
            if paving_path.exists():
                paving = json.loads(paving_path.read_text(encoding="utf-8"))
                if set(paving["actor_ids"]) & set(surface["actor_ids"]):
                    raise ValueError("Footpath and road appearance instances must be disjoint")
                self.config["streetscape"] = {"version": paving["version"],
                    "nonroad_actor_ids": paving["actor_ids"], "native_class": paving["native_class"],
                    "top_height_m": paving["top_height_m"],
                    "reference": "Known non-road appearances; excluded from the owned asphalt set"}
        grid = self.config["reference"]["grid"]
        surface = pygame.Surface(tuple(reversed(grid["shape_hw"])))
        surface.fill((0, 0, 0))
        points = []
        for wp in world.get_map().generate_waypoints(.5):
            if wp.lane_type != carla.LaneType.Driving:
                continue
            location, forward = wp.transform.location, wp.transform.get_forward_vector()
            half, side = .4, wp.lane_width / 2
            polygon = [[location.x + a*half*forward.x - b*side*forward.y,
                        location.y + a*half*forward.y + b*side*forward.x] for a, b in ((-1,-1),(1,-1),(1,1),(-1,1))]
            pixels = [[round((x+154)/.1), round((y+154)/.1)] for x,y in polygon]
            pygame.draw.polygon(surface, (1, 1, 1), pixels)
            points.append({"road_id": wp.road_id, "lane_id": wp.lane_id, "s": wp.s,
                           "x": location.x, "y": location.y, "lane_width": wp.lane_width})
        footprint = pygame.surfarray.array3d(surface)[:, :, 0].T.copy()
        np.savez_compressed(folder / grid["file"], driving=footprint)
        write_json(folder / "footprint_waypoints.json", {"samples": points})
        self.config["reference"]["grid"]["sha256"] = hashlib.sha256((folder / grid["file"]).read_bytes()).hexdigest()
        cameras = [("rgb", "sensor.camera.rgb"), ("labels", "sensor.camera.semantic_segmentation"),
                   ("depth", "sensor.camera.depth")]
        if surface_style == "asphalt":
            cameras.append(("instance", "sensor.camera.instance_segmentation"))
        for kind, blueprint in cameras:
            bp = world.get_blueprint_library().find(blueprint)
            for key, value in {"image_size_x": "960", "image_size_y": "540", "fov": str(profile["fov_deg"]),
                               "sensor_tick": str(1/capture_hz), "lens_k": "0", "lens_kcube": "0"}.items():
                bp.set_attribute(key, value)
            if kind == "rgb":
                self.config["rgb_postprocessing"] = configure_rgb(bp)
            sensor = world.spawn_actor(bp, mount, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
            owned.append(sensor)
            sensors.append(sensor)
            sensor.listen(lambda image, tag=kind: self.receive(tag, image))
        self.config["evidence_writer"] = {"mode": "two background encoders, commit in capture order", "max_pending_pairs": 4,
            "overflow": "Block the producer; never silently drop paired evidence",
            "encoding": "Lossless RGB8 PNG filter None / DEFLATE1; exact float32 NPY in DEFLATE1 NPZ"}
        write_json(folder / "capture_config.json", self.config)

    def receive(self, kind, image):
        if kind == "rgb" and self.live_sink is not None:
            self.live_sink.receive_rgb(image)
        try:
            self.inbox.put_nowait((kind, image))
        except queue.Full:
            self.dropped["queue_overflow"] += 1

    def tick(self, snapshot, first_time, decision):
        self.snapshots[snapshot.frame] = {"timestamp_s": snapshot.timestamp.elapsed_seconds,
            "sim_s": snapshot.timestamp.elapsed_seconds - first_time, "decision": decision}
        self.collect()
        cutoff = snapshot.frame - 100
        for key in list(self.snapshots):
            if key < cutoff:
                del self.snapshots[key]
        for kind in self.pending:
            for key in list(self.pending[kind]):
                if key < cutoff:
                    del self.pending[kind][key]
                    self.dropped["expired_images"] += 1

    def collect(self):
        while True:
            try:
                kind, image = self.inbox.get_nowait()
            except queue.Empty:
                break
            self.pending[kind][image.frame] = image
        paired = set.intersection(*(set(value) for value in self.pending.values())) & set(self.snapshots)
        for frame in sorted(paired):
            images = {kind: value.pop(frame) for kind, value in self.pending.items()}
            while self.writes and self.writes[0].done():
                self.commit(self.writes.popleft())
            if len(self.writes) >= 4:
                started = time.perf_counter()
                self.commit(self.writes.popleft())
                self.backpressure_s += time.perf_counter()-started
            self.writes.append(self.writer.submit(self.write_pair, frame, images, dict(self.snapshots[frame])))

    def commit(self, future):
        row, elapsed_ms = future.result()
        self.records.append(row)
        self.log.write(json.dumps(row, allow_nan=False)+"\n")
        self.log.flush()
        self.write_ms.append(elapsed_ms)

    def write_pair(self, frame, images, state):
        started = time.perf_counter()
        rgb, semantic, depth_image = (images[kind] for kind in ("rgb", "labels", "depth"))
        if (abs(rgb.timestamp - semantic.timestamp) > 1e-6
                or abs(rgb.timestamp - state["timestamp_s"]) > 1e-6
                or abs(rgb.timestamp - depth_image.timestamp) > 1e-6
                or not np.allclose(rgb.transform.get_matrix(), semantic.transform.get_matrix(), atol=1e-5, rtol=0)
                or not np.allclose(rgb.transform.get_matrix(), depth_image.transform.get_matrix(), atol=1e-5, rtol=0)):
            raise ValueError("RGB and semantic cameras differ in frame time or pose")
        if any(abs(rgb.timestamp-image.timestamp) > 1e-6 or
               not np.allclose(rgb.transform.get_matrix(), image.transform.get_matrix(), atol=1e-5, rtol=0)
               for image in images.values()):
            raise ValueError("An observation camera differs in frame time or pose")
        tags = np.frombuffer(semantic.raw_data, np.uint8).reshape(540, 960, 4)[:, :, 2].copy()
        if tags.max() > 28:
            raise ValueError("Unexpected CARLA semantic IDs; verify the reference mapping")
        name = f"frame_{frame:07d}"
        rgb_path, labels_path = self.folder / f"{name}_rgb.png", self.folder / f"{name}_labels.png"
        if rgb_path.exists() or labels_path.exists():
            raise FileExistsError("Refusing to overwrite a captured camera pair")
        rgb_pixels = np.frombuffer(rgb.raw_data, np.uint8).reshape(540, 960, 4)[:, :, 2::-1]
        sky = tags == 11
        sky_mean = float(rgb_pixels[sky].mean()) if sky.any() else None
        sky_median = float(np.median(rgb_pixels[sky].mean(axis=1))) if sky.any() else None
        if (self.config["dataset_edge_disagreement_policy"] and not self.config["weather_probe_only"] and self.config["actual_weather"]["sun_altitude_angle"] > 20
                and int(sky.sum()) > 100 and sky_median < 10):
            write_rgb_png(self.folder / f"rejected_{name}_rgb.png",rgb_pixels)
            write_json(self.folder / f"rejected_{name}.json",{"sky_rgb_median":sky_median,"sky_rgb_mean":sky_mean,"weather":self.config["actual_weather"]})
            raise ValueError("Invalid black daytime sky; reject this dataset capture before training")
        write_rgb_png(rgb_path, rgb_pixels)
        packed = np.frombuffer(depth_image.raw_data, np.uint8).reshape(540, 960, 4).astype(np.uint32)
        depth = ((packed[:, :, 2] + 256*packed[:, :, 1] + 65536*packed[:, :, 0]) * (1000 / 16777215)).astype(np.float32)
        depth_path = self.folder / f"{name}_depth.npz"
        write_depth_npz(depth_path, depth)
        gray = np.repeat(tags[:, :, None], 3, axis=2)
        write_rgb_png(labels_path, gray)
        row = {"frame": frame, **state, "rgb_file": rgb_path.name, "labels_file": labels_path.name,
            "sky_pixels":int(sky.sum()), "sky_rgb_mean":sky_mean, "sky_rgb_median":sky_median,
            "rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "depth_file": depth_path.name, "depth_sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
            "depth_timestamp_s": depth_image.timestamp,
            "camera_to_world": rgb.transform.get_matrix(), "rgb_timestamp_s": rgb.timestamp,
            "labels_timestamp_s": semantic.timestamp,
            "semantic_counts": np.bincount(tags.ravel(), minlength=29).tolist()}
        if "instance" in images:
            instance = images["instance"]
            instance_path = self.folder / f"{name}_instance.png"
            raw = np.frombuffer(instance.raw_data, np.uint8).reshape(540, 960, 4)
            write_rgb_png(instance_path, raw[:, :, 2::-1])
            ids = raw[:, :, 1].astype(np.uint16) + 256*raw[:, :, 0].astype(np.uint16)
            appearance = np.isin(ids, self.config["reference"]["appearance_actor_ids"])
            disagreement = int(np.sum(raw[:, :, 2] != tags))
            if disagreement and (not self.config["dataset_edge_disagreement_policy"] or disagreement > tags.size*.001):
                raise ValueError("Instance camera semantic channel differs from paired raw labels beyond the declared dataset edge limit")
            row.update(instance_file=instance_path.name, instance_timestamp_s=instance.timestamp,
                       instance_sha256=hashlib.sha256(instance_path.read_bytes()).hexdigest(),
                       appearance_pixels=int(appearance.sum()), semantic_instance_disagreement_pixels=disagreement)
        return row, 1000*(time.perf_counter()-started)

    def finish(self):
        try:
            self.collect()
            while self.writes:
                self.commit(self.writes.popleft())
            values = [row["frame"] for row in self.records]
            write_json(self.folder / "capture_summary.json", {"paired_frames": len(values),
                "first_frame": min(values) if values else None, "last_frame": max(values) if values else None,
                "dropped": self.dropped, "unpaired_at_end": {k: len(v) for k, v in self.pending.items()},
                "write_pair_ms_median": float(np.median(self.write_ms)) if self.write_ms else None,
                "write_pair_ms_p95": float(np.percentile(self.write_ms, 95)) if self.write_ms else None,
                "producer_backpressure_s": self.backpressure_s,
                "reference_status": "Native tags, depth and map footprint captured; inspect geometric reference before interpreting scores",
                "model_control": False})
        finally:
            self.writer.shutdown(wait=True, cancel_futures=False)
            self.log.close()
