"""Capture even early viewer errors instead of losing them in a hidden console."""
import argparse
from pathlib import Path
import runpy
import sys
import traceback

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--camera-probe", action="store_true")
parser.add_argument("--india", action="store_true")
parser.add_argument("--india-world")
args, _ = parser.parse_known_args()
log_dir = root / "setup/carla/logs"
log_dir.mkdir(exist_ok=True)
with (log_dir / f"{args.output.name}.log").open("w", encoding="utf-8", buffering=1) as log:
    sys.stdout = sys.stderr = log
    try:
        if args.india_world == "combined":
            sys.path.insert(0, str(root / "integration"))
            runpy.run_path(str(root / "integration/india_combined_scene.py"), run_name="__main__")
        elif args.india_world:
            sys.path.insert(0, str(root / "integration"))
            runpy.run_path(str(root / "integration/india_world_scene.py"), run_name="__main__")
        elif args.india:
            sys.path.insert(0, str(root / "integration"))
            runpy.run_path(str(root / "integration/carla_india_scene.py"), run_name="__main__")
        elif args.camera_probe:
            runpy.run_path(str(root / "integration/carla_camera_probe.py"), run_name="__main__")
        else:
            runpy.run_path(str(root / "integration/carla_starter.py"), run_name="__main__")
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
