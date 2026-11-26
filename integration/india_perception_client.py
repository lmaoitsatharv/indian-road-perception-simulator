"""Loopback transport from the CARLA viewer (Windows) to the two WSL GPU workers.

Each camera frame goes to the road worker and the detector worker at the same
time; result(frame) waits for both answers. A worker that is down or late makes
that frame "unavailable" (the camera driver then brakes to a stop); the link
reconnects at most once a second. Message framing is simulation_live_protocol.

While the car is about to pass, passing, backing up or turning round, one more
picture goes to the detector each frame (a second request on the same link): the
rear camera's or a side camera's, in turn. A missing answer leaves the front result
usable without that view (the driver then will not start a pass or back up).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue
import socket
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_live_protocol import receive_packet, send_packet
from india_camera_driver import Perception

PROTOCOL = "native_carla_camera_driving_v1"
SHAPE = (540, 960, 3)
MASK = 512 * 512


class WorkerLink(threading.Thread):
    def __init__(self, name, port, session, token, deliver, log):
        super().__init__(daemon=True, name="camera-driving-" + name)
        self.role, self.port, self.session, self.token = name, int(port), session, token
        self.deliver, self.log = deliver, log
        self.jobs = queue.Queue(maxsize=6)
        self.closed = threading.Event()
        self.connection = None
        self.last_failure = -1e9
        self.counters = {"sent": 0, "answered": 0, "failed": 0, "connects": 0}

    def submit(self, job):
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            self.counters["failed"] += 1
            self.deliver(job.get("key", self.role), job["frame"], {"ok": False, "error": "request queue full"})

    def run(self):
        while not self.closed.is_set():
            try:
                job = self.jobs.get(timeout=.1)
            except queue.Empty:
                continue
            if job is None:
                break
            frame, view = job["frame"], job.get("key", self.role)
            try:
                if self.connection is None:
                    if time.monotonic() - self.last_failure < 1.:
                        raise ConnectionError("worker offline; next reconnect in under 1 s")
                    self.connection = socket.create_connection(("127.0.0.1", self.port), timeout=2.)
                    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.connection.settimeout(3.)
                    self.counters["connects"] += 1
                header = {**job["header"], "protocol": PROTOCOL, "session": self.session, "token": self.token,
                          "role": self.role, "control_allowed": True, "sent_ns": time.perf_counter_ns()}
                send_packet(self.connection, header, job["payload"])
                self.counters["sent"] += 1
                response, data = receive_packet(self.connection)
                received_ns = time.perf_counter_ns()
                for key in ("session", "frame", "source_rgb_sha256"):
                    if response.get(key) != header[key]:
                        raise ValueError("worker answer identity differs: " + key)
                if response.get("protocol") != PROTOCOL or response.get("role") != self.role:
                    raise ValueError("unexpected worker answer")
                result = {"ok": True, "header": response, "roundtrip_ms": (received_ns - header["sent_ns"]) / 1e6}
                if self.role == "road":
                    if len(data) != 2 * MASK:
                        raise ValueError("road answer has the wrong size")
                    result["mask"] = np.frombuffer(data[:MASK], np.uint8).reshape(512, 512).copy()
                    result["probability"] = np.frombuffer(data[MASK:], np.uint8).reshape(512, 512).copy()
                    if result["mask"].max(initial=0) > 1:
                        raise ValueError("road mask is not binary")
                else:
                    count = int(response.get("boxes", -1))
                    if count < 0 or len(data) != count * 24:
                        raise ValueError("detector answer has the wrong size")
                    result["boxes"] = np.frombuffer(data, np.float32).reshape(count, 6).copy()
                self.counters["answered"] += 1
                self.deliver(view, frame, result)
            except (OSError, ConnectionError, ValueError) as error:
                self.counters["failed"] += 1
                self.last_failure = time.monotonic()
                if self.connection is not None:
                    try:
                        self.connection.close()
                    except OSError:
                        pass
                    self.connection = None
                self.log({"worker": self.role, "view": view, "frame": frame, "error": f"{type(error).__name__}: {error}"})
                self.deliver(view, frame, {"ok": False, "error": f"{view}: {type(error).__name__}: {error}"})

    def close(self):
        self.closed.set()
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class PerceptionClient:
    def __init__(self, session, error_log_path=None):
        self.session = session
        self.condition = threading.Condition()
        self.answers = {}
        self.expected = {}
        self.errors = []
        self._error_file = None if error_log_path is None else Path(error_log_path).open("w", encoding="utf-8", buffering=1)
        self.links = {role: WorkerLink(role, session[role]["port"], session["session"], session["token"], self._deliver, self._log)
                      for role in ("road", "detector")}
        for link in self.links.values():
            link.start()
        self.latency_ms = []
        self.rear = {"sent": 0, "answered": 0, "missing": 0}
        self.sides = {view: {"sent": 0, "answered": 0, "missing": 0} for view in ("left", "right")}

    def _log(self, row):
        self.errors.append(row)
        if self._error_file:
            self._error_file.write(json.dumps(row) + "\n")

    def _deliver(self, role, frame, answer):
        with self.condition:
            self.answers.setdefault(frame, {})[role] = answer
            self.condition.notify_all()

    @staticmethod
    def _request(frame, rgb, simulation_time):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.shape != SHAPE:
            raise ValueError("camera frame has an unexpected shape")
        payload = rgb.tobytes()
        return {"frame": int(frame), "simulation_timestamp_s": float(simulation_time), "source_shape": list(SHAPE),
                "source_rgb_sha256": hashlib.sha256(payload).hexdigest()}, payload

    def submit(self, frame, rgb, simulation_time, rear=None, extra=None):
        """extra: (view, picture), one more camera's picture to run the detector on ("rear", "left" or "right");
        rear=picture is the same as extra=("rear", picture)."""
        if rear is not None:
            extra = ("rear", rear)
        header, payload = self._request(frame, rgb, simulation_time)
        with self.condition:
            for old in [f for f in self.answers if f < frame - 20]:
                del self.answers[old]
            for old in [f for f in self.expected if f < frame - 20]:
                del self.expected[old]
            self.expected[int(frame)] = None if extra is None else extra[0]
        for link in self.links.values():
            link.submit({"frame": int(frame), "header": header, "payload": payload})
        if extra is not None:
            view, picture = extra
            extra_header, extra_payload = self._request(frame, picture, simulation_time)
            (self.rear if view == "rear" else self.sides[view])["sent"] += 1
            self.links["detector"].submit({"frame": int(frame), "key": view, "header": {**extra_header, "view": view},
                                           "payload": extra_payload})

    def result(self, frame, timeout=1.):
        started = time.perf_counter()
        deadline = time.monotonic() + timeout
        with self.condition:
            view = self.expected.pop(frame, None)
            wanted = 2 if view is None else 3
            while len(self.answers.get(frame, {})) < wanted:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self.condition.wait(left)
            answers = self.answers.pop(frame, {})
        waited = (time.perf_counter() - started) * 1000
        road, detector = answers.get("road"), answers.get("detector")
        if not road or not detector or not road["ok"] or not detector["ok"]:
            problems = [a.get("error", "") for a in (road, detector) if a and not a["ok"]]
            if not road or not detector:
                problems.append("no answer within %.1f s" % timeout)
            return Perception(frame, False, error="; ".join(p for p in problems if p), timings={"wait_ms": waited})
        timings = {"wait_ms": waited, "road_ms": road["header"].get("inference_ms"), "detector_ms": detector["header"].get("inference_ms"),
                   "road_roundtrip_ms": road["roundtrip_ms"], "detector_roundtrip_ms": detector["roundtrip_ms"]}
        self.latency_ms.append(max(road["roundtrip_ms"], detector["roundtrip_ms"]))
        rear_boxes = side_boxes = None
        if view is not None:
            answer = answers.get(view)
            counts = self.rear if view == "rear" else self.sides[view]
            if answer and answer["ok"]:
                if view == "rear":
                    rear_boxes = answer["boxes"]
                else:
                    side_boxes = {view: answer["boxes"]}
                counts["answered"] += 1
                timings.update({f"{view}_ms": answer["header"].get("inference_ms"), f"{view}_roundtrip_ms": answer["roundtrip_ms"]})
            else:
                counts["missing"] += 1
                timings[f"{view}_error"] = (answer or {}).get("error", "no %s answer within %.1f s" % (view, timeout))
        return Perception(frame, True, mask=road["mask"], boxes=detector["boxes"], timings=timings,
                          probability=road["probability"], rear_boxes=rear_boxes, side_boxes=side_boxes)

    def summary(self):
        lat = np.array(self.latency_ms) if self.latency_ms else np.zeros(1)
        return {"links": {role: link.counters for role, link in self.links.items()}, "errors": len(self.errors), "rear": dict(self.rear),
                "sides": {view: dict(counts) for view, counts in self.sides.items()},
                "roundtrip_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())}}

    def close(self):
        for link in self.links.values():
            link.close()
        for link in self.links.values():
            link.join(timeout=3.)
        if self._error_file:
            self._error_file.close()
