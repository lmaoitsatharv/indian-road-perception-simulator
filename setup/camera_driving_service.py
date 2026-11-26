"""Shared loopback service loop for the two camera-driving GPU workers (WSL).

One authenticated client (the CARLA viewer) at a time. Every request is checked
for protocol, session, token, role, image size and pixel hash before the model
runs. The worker stops on its stop marker or deadline and records whether its
model weights stayed unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_live_protocol import receive_packet, send_packet
from setup.prepare_simulation_dataset import save

PROTOCOL = "native_carla_camera_driving_v1"
SHAPE = [540, 960, 3]


def serve(session_file, role, load, infer, fingerprint):
    """load() -> model; infer(model, rgb) -> (extra_header, payload_bytes); fingerprint(model) -> weights digest."""
    import numpy as np
    session = json.loads(Path(session_file).read_text())
    folder = ROOT / session[role]["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    status = {"status": "loading", "pid": os.getpid(), "role": role, "session": session["session"], "requests": 0,
              "rejected": 0, "connections": 0, "model_control": True, "training": False}
    save(folder / "worker_status.json", status)
    model = None
    before = None
    log = (folder / "worker_frames.jsonl").open("w", buffering=1)
    try:
        model, runtime = load(folder)
        before = fingerprint(model)
        status.update(status="ready", runtime=runtime)
        save(folder / "worker_status.json", status)
        deadline = time.monotonic() + session["service_max_seconds"]
        stop = ROOT / session["stop_file"]
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", session[role]["port"]))
            server.listen(1)
            server.settimeout(.2)
            save(folder / "worker_ready.json", {"session": session["session"], "port": session[role]["port"], "pid": os.getpid()})
            while time.monotonic() < deadline and not stop.exists():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                status["connections"] += 1
                with connection:
                    connection.settimeout(1.)
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    while time.monotonic() < deadline and not stop.exists():
                        try:
                            header, payload = receive_packet(connection)
                        except socket.timeout:
                            continue
                        except (ConnectionError, OSError, ValueError):
                            break
                        if (header.get("protocol") != PROTOCOL or header.get("session") != session["session"]
                                or header.get("token") != session["token"] or header.get("role") != role
                                or header.get("control_allowed") is not True or header.get("source_shape") != SHAPE
                                or len(payload) != SHAPE[0] * SHAPE[1] * SHAPE[2]
                                or hashlib.sha256(payload).hexdigest() != header.get("source_rgb_sha256")):
                            status["rejected"] += 1
                            break
                        started = time.perf_counter()
                        rgb = np.frombuffer(payload, np.uint8).reshape(SHAPE)
                        extra, data = infer(model, rgb)
                        status["requests"] += 1
                        reply = {key: header[key] for key in ("session", "frame", "simulation_timestamp_s", "source_rgb_sha256")}
                        reply.update(protocol=PROTOCOL, role=role, request_index=status["requests"],
                                     worker_total_ms=(time.perf_counter() - started) * 1000, **extra)
                        log.write(json.dumps(reply) + "\n")
                        try:
                            send_packet(connection, reply, data)
                        except OSError:
                            break
                        if status["requests"] % 50 == 0:
                            save(folder / "worker_status.json", status)
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        (folder / "worker_error.txt").write_text(traceback.format_exc())
        raise
    finally:
        log.close()
        if model is not None and before is not None:
            unchanged = fingerprint(model) == before
            status["model_weights_unchanged"] = unchanged
            if not unchanged:
                status["status"] = "failed"
        save(folder / "worker_status.json", status)
