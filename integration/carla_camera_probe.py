"""Small hardware diagnosis: RGB camera without Traffic Manager or pedestrians."""
import argparse
import json
from pathlib import Path
import queue
import time
import traceback

import carla

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--port", type=int, default=2050)
args, _ = parser.parse_known_args()
args.output.mkdir(exist_ok=False)
image_queue = queue.Queue()
client = carla.Client("127.0.0.1", args.port)
client.set_timeout(10)
world = client.get_world()
original = world.get_settings()
camera = None
result = {"kind": "RGB probe without traffic manager", "images": [], "errors": []}
try:
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    spawn = world.get_map().get_spawn_points()[0]
    spawn.location.z += 2
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "640")
    bp.set_attribute("image_size_y", "360")
    bp.set_attribute("enable_postprocess_effects", "false")
    camera = world.spawn_actor(bp, spawn)
    camera.listen(image_queue.put)
    print("Probe camera listening", flush=True)
    for tick in range(12):
        print(f"Tick {tick} starting", flush=True)
        frame = world.tick(5)
        print(f"Tick {tick} returned {frame}", flush=True)
        time.sleep(0.15)
        while not image_queue.empty():
            image = image_queue.get_nowait()
            result["images"].append({"image_frame": image.frame, "received_after_tick": frame})
            image.save_to_disk(str(args.output / f"probe_{image.frame}.png"))
        (args.output / "probe_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["passed"] = bool(result["images"])
except Exception:
    result["errors"].append(traceback.format_exc())
    result["passed"] = False
finally:
    (args.output / "probe_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if camera is not None:
        camera.stop()
        camera.destroy()
    world.apply_settings(original)
print(json.dumps(result, indent=2), flush=True)
raise SystemExit(0 if result["passed"] else 1)
