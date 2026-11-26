#!/usr/bin/env bash
# Run town drives one after another (rule driver, then learned speed, per seed), never two at once.
#   bash rl/queue_town_pairs.sh <after-folder|-> <tag> "<destinations>" <seed> [<seed> ...]
# <after-folder>: a runs/camera_driving_* folder whose drive must have finished first ("-": none).
# MODES="rl" (environment) runs only the learned drives.
# Each drive: 900 simulated seconds, clear weather, --check. Folders: runs/camera_driving_<tag>_s<seed>_{rule,rl}.
set -u
cd "$(dirname "$0")/.."
after="$1"; tag="$2"; places="$3"; shift 3
owners() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"CommandLine like '%run_camera_driving.py%' and Name like 'python%'\").Count" | tr -d '\r'; }
busy() { [ -e setup/carla/session.lock ] || [ "$(owners)" != "0" ]; }
finished() { [ -f "$1/status.json" ] && ! grep -q '"status": "\(driving\|starting_workers\)"' "$1/status.json"; }
if [ "$after" != "-" ]; then
  until finished "$after"; do sleep 20; done
fi
for seed in "$@"; do
  for mode in ${MODES:-rule rl}; do
    while busy; do sleep 20; done
    sleep 30
    while busy; do sleep 20; done
    out="runs/camera_driving_${tag}_s${seed}_${mode}"
    extra=()
    [ "$mode" = rl ] && extra=(--rl-speed)
    echo "$(date +%H:%M) start $out"
    .venv/Scripts/python.exe setup/run_camera_driving.py --sim-seconds 900 --seed "$seed" --check "${extra[@]}" \
      "--destinations=$places" --output "$out" > "$out.launch.log" 2>&1
    echo "$(date +%H:%M) end $out exit $?"
  done
done
