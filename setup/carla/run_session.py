"""Bounded CARLA launcher. Its Windows Job Objects own only the processes it starts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "setup"))
# Reuse only this class, not the historical controller, deadline, queue or lease.
from windows_job import CarlaJob


def atomic(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def owned_native_crash(pids, started):
    roots = {Path(os.environ.get("LOCALAPPDATA", "")) / "CarlaUE4/Saved/Crashes",
             Path.home() / "AppData/Local/CarlaUE4/Saved/Crashes"}
    directories = [d for root in roots if root.is_dir() for d in root.iterdir()]
    for directory in directories:
        context = directory / "CrashContext.runtime-xml"
        try:
            if not context.is_file() or context.stat().st_mtime < started:
                continue
            properties = ET.parse(context).getroot().find("RuntimeProperties")
            pid = int(properties.findtext("ProcessId", "0"))
            if pid in pids:
                return {"process_id": pid, "type": properties.findtext("CrashType"),
                        "error": properties.findtext("ErrorMessage"), "context": str(context),
                        "ownership": "Process was recorded inside this session's server Job Object"}
        except (OSError, ValueError, ET.ParseError, AttributeError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--camera-probe", action="store_true")
    parser.add_argument("--renderer", choices=["vulkan", "dx11", "dx12"], default="vulkan")
    parser.add_argument("--quality", choices=["Low", "Epic"], default="Epic")
    parser.add_argument("--india", action="store_true")
    parser.add_argument("--india-world", choices=("market_street","residential_colony","divided_main_road","rural_road","irregular_junction","all","asset_probe","combined"),
                        help="Opt-in new Indian road worlds; independent of the original neighbourhood")
    parser.add_argument("--india-probe", action="store_true")
    parser.add_argument("--case", default="mixed")
    parser.add_argument("--view", choices=["follow", "driver", "junction"], default="follow")
    parser.add_argument("--sim-seconds", type=float, default=60)
    parser.add_argument("--speed", type=float, choices=(1., 1.5, 2., 2.5, 3.), default=1.)
    parser.add_argument("--speed-check", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--replay-check", action="store_true")
    parser.add_argument("--observe-road", action="store_true")
    parser.add_argument("--dataset-plan", type=Path, help="Predeclared Indian simulation capture episodes")
    parser.add_argument("--live-road-config",type=Path,help="Opt-in native-CARLA live-observer session")
    parser.add_argument("--road-camera-hz", type=int, choices=(1,5), default=5)
    parser.add_argument("--surface-style", choices=("legacy", "asphalt"), default="asphalt")
    parser.add_argument("--weather", choices=("clear", "overcast", "wet", "dusk"), default="clear")
    args = parser.parse_args()
    if args.india_world:
        if args.india or args.observe_road or args.dataset_plan or args.live_road_config or args.suite or args.speed_check:
            parser.error("New worlds use their own native preview; choose --india-world without old-scene/model flags")
        if args.speed != 1:
            parser.error("New worlds currently use 1x playback")
    if args.live_road_config and (not args.india or not args.observe_road or args.road_camera_hz!=5):
        parser.error("Live observation requires --india --observe-road --road-camera-hz 5")
    if args.dataset_plan and (not args.india or not args.smoke or not args.observe_road):
        parser.error("Dataset capture requires --india --smoke --observe-road")
    if not 0 < args.seconds <= 1800:
        parser.error("Duration must be between 1 and 1800 seconds")
    if (args.speed != 1 or args.speed_check) and not args.india:
        parser.error("Playback speed controls are available in the Indian neighbourhood")
    if args.observe_road and not args.india:
        parser.error("Road observation requires --india")
    if args.weather != "clear" and not (args.india or args.india_world):
        parser.error("Weather presets require --india")
    if args.speed_check and (not args.smoke or args.speed != 1 or args.sim_seconds < 36):
        parser.error("Speed control check needs --india --smoke --speed 1 and at least 36 simulated seconds")
    if (ROOT / "setup/carla/STOP_DEMO.json").exists():
        raise RuntimeError("STOP_DEMO.json exists; inspect/remove it intentionally before a new session.")
    installation_path = ROOT / "setup/carla/installation.json"
    if not installation_path.exists():
        raise RuntimeError("CARLA package not installed. Run setup/carla/install_package.py after download.")
    installation = json.loads(installation_path.read_text(encoding="utf-8"))
    executable = Path(installation["executable"])
    if not executable.is_file() or not executable.resolve().is_relative_to(ROOT / "tools/carla"):
        raise RuntimeError("Invalid CARLA installation path")
    lock = ROOT / "setup/carla/session.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("A dedicated session may already be running; inspect setup/carla/session.lock.")
    os.write(fd, str(os.getpid()).encode("ascii"))
    os.close(fd)
    server = viewer = None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    prefix = "carla_indian_worlds" if args.india_world else "carla_india" if args.india else "carla"
    (ROOT / "runs").mkdir(exist_ok=True)
    output = ROOT / "runs" / f"{prefix}_{'smoke' if args.smoke else 'live'}_{stamp}_seed{args.seed}"
    state = {"controller_pid": os.getpid(), "port": 2050, "tm_port": 8050,
             "started_utc": datetime.now(timezone.utc).isoformat(), "output": str(output),
             "ownership": "Separate kill-on-close Job Objects; no existing process attached",
             "status": "starting"}
    record = ROOT / "setup/carla/server_owner.json"
    started_epoch, owned_pids = time.time(), set()
    try:
        # Never attach to an existing simulator or take over its world.
        for port in (2050, 2051, 2052, 8050):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        if re.fullmatch(r"Town[0-9]+(?:HD)?(?:_Opt)?", args.map) is None:
            raise ValueError("Only installed stock Town maps are accepted by this starter")
        map_file = executable.parent / f"CarlaUE4/Content/Carla/Maps/{args.map}.umap"
        if not map_file.is_file():
            raise RuntimeError(f"Map asset missing: {map_file}")
        # This packaged Shipping build ignored URL/command-line map overrides.
        # Set only this installation's startup-map keys, preserving the original.
        engine_ini = executable.parent / "CarlaUE4/Config/DefaultEngine.ini"
        engine_backup = ROOT / "setup/carla/DefaultEngine.original.ini"
        if not engine_backup.exists():
            shutil.copy2(engine_ini, engine_backup)
        old_ini = engine_ini.read_text(encoding="utf-8")
        map_asset = f"/Game/Carla/Maps/{args.map}.{args.map}"
        new_ini, changes = re.subn(r"(?m)^(EditorStartupMap|GameDefaultMap|ServerDefaultMap|TransitionMap)=.*$",
                                  lambda match: match.group(1) + "=" + map_asset, old_ini)
        if changes != 4:
            raise RuntimeError(f"Expected four map settings, found {changes}; config left unchanged")
        if new_ini != old_ini:
            ini_temp = engine_ini.with_suffix(".ini.tmp")
            ini_temp.write_text(new_ini, encoding="utf-8")
            os.replace(ini_temp, engine_ini)
        state["startup_map_config"] = str(engine_ini)
        argv = [str(executable), f"-{args.renderer}", f"-quality-level={args.quality}", "-nosound",
                "-carla-rpc-port=2050", "-ResX=960", "-ResY=540", "-windowed"]
        state["server_argv"] = argv
        server = CarlaJob(argv, executable.parent)
        owned_pids.update(server.pids())
        state["server_pid"] = server.pid
        atomic(record, state)
        print("Starting CARLA. First launch can take several minutes...", flush=True)
        import carla
        ready_deadline = time.monotonic() + 300
        ready = False
        while time.monotonic() < ready_deadline:
            owned_pids.update(server.pids())
            crash = owned_native_crash(owned_pids, started_epoch)
            if crash:
                state["native_crash"] = crash
                raise RuntimeError(f"Owned CARLA startup crash: {crash['error']}")
            if server.poll() is not None:
                raise RuntimeError(f"CARLA engine exited during startup (code {server.poll()}); inspect its latest crash metadata.")
            if (ROOT / "setup/carla/STOP_DEMO.json").exists():
                raise RuntimeError("Stopped while server was starting")
            try:
                client = carla.Client("127.0.0.1", 2050)
                client.set_timeout(2)
                version = client.get_server_version()
                if version != "0.9.16":
                    raise RuntimeError(f"Wrong server version: {version}")
                # Version RPC can answer while the world is still loading or about to crash.
                ready_world = client.get_world()
                ready_map = ready_world.get_map().name
                if ready_map.rsplit("/", 1)[-1] != args.map:
                    raise RuntimeError(f"Wrong startup map: {ready_map}; expected {args.map}")
                ready = True
                break
            except RuntimeError as error:
                if "Wrong server" in str(error) or "Wrong startup map" in str(error):
                    raise
                time.sleep(1)
        if not ready:
            raise TimeoutError("CARLA did not become ready in five minutes; inspect its Saved/Logs directory.")
        argv = [sys.executable, "-u", str(ROOT / "setup/carla/viewer_entry.py"),
                "--port", "2050", "--tm-port", "8050", "--seed", str(args.seed),
                "--map", args.map, "--seconds", str(args.seconds), "--output", str(output),
                "--renderer", args.renderer, "--quality", args.quality]
        if args.smoke:
            argv.append("--smoke")
        if args.camera_probe:
            argv.append("--camera-probe")
        if args.india_world:
            argv.extend(["--india-world", args.india_world,"--sim-seconds",str(args.sim_seconds),"--weather",args.weather])
        if args.india:
            argv.append("--india")
            argv.extend(["--case", args.case, "--sim-seconds", str(args.sim_seconds), "--view", args.view])
            argv.extend(["--speed", str(args.speed)])
            if args.speed_check:
                argv.append("--speed-check")
            if args.suite:
                argv.append("--suite")
            if args.replay_check:
                argv.append("--replay-check")
            if args.observe_road:
                argv.append("--observe-road")
                argv.extend(["--road-camera-hz", str(args.road_camera_hz)])
            argv.extend(["--surface-style",args.surface_style])
            if args.dataset_plan:
                argv.extend(["--dataset-plan", str(args.dataset_plan.resolve())])
            if args.live_road_config:
                argv.extend(["--live-road-config",str(args.live_road_config.resolve())])
            argv.extend(["--weather", args.weather])
        if args.india_probe:
            argv.append("--india-probe")
        viewer = CarlaJob(argv, ROOT)
        state.update(status="running", viewer_pid=viewer.pid, server_version=version,
                     ready_map=ready_map, viewer_argv=argv,
                     viewer_log=str(ROOT / "setup/carla/logs" / f"{output.name}.log"))
        atomic(record, state)
        print("Live camera window open: TAB changes camera, SPACE pauses, ESC closes.", flush=True)
        end = time.monotonic() + args.seconds + 90
        stop_requested_at = None
        next_crash_check = 0
        while viewer.poll() is None and time.monotonic() < end:
            if time.monotonic() >= next_crash_check:
                owned_pids.update(server.pids())
                crash = owned_native_crash(owned_pids, started_epoch)
                if crash:
                    state["native_crash"] = crash
                    raise RuntimeError(f"Owned CARLA native crash: {crash['error']}")
                next_crash_check = time.monotonic() + 2
            if server.poll() is not None:
                state["engine_exit_code"] = server.poll()
                state["interruption_reason"] = "engine_exited"
                break
            if (ROOT / "setup/carla/STOP_DEMO.json").exists():
                if stop_requested_at is None:
                    stop_requested_at = time.monotonic()
                if time.monotonic() - stop_requested_at > 20:
                    break
            time.sleep(0.5)
        code = viewer.poll()
        state.update(viewer_exit_code=code, status="completed" if code == 0 else "failed")
        print(f"Session {state['status']}. Evidence: {output}", flush=True)
        return 0 if code == 0 else 1
    except Exception:
        state.update(status="failed", error=traceback.format_exc())
        print(state["error"], flush=True)
        return 1
    finally:
        if viewer is not None:
            viewer.close()
        if server is not None:
            server.close()
        state.update(ended_utc=datetime.now(timezone.utc).isoformat(), owned_job_handles_closed=True)
        state["owned_server_process_ids"] = sorted(owned_pids)
        atomic(record, state)
        output.mkdir(exist_ok=True)
        atomic(output / "launcher_result.json", state)
        lock.unlink(missing_ok=True)
        # A forced viewer termination may leave only our lock behind. Our job has closed now.
        viewer_lock = ROOT / "setup/carla/viewer.lock"
        if viewer is not None and viewer_lock.exists():
            try:
                lock_data = json.loads(viewer_lock.read_text(encoding="utf-8"))
                if isinstance(lock_data, dict) and lock_data.get("output") == str(output):
                    viewer_lock.unlink()
            except (ValueError, OSError):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
