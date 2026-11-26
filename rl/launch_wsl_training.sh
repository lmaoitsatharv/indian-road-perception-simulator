#!/usr/bin/env bash
# Start a PPO training run of the speed governor in WSL, in the background.
#   bash rl/launch_wsl_training.sh <run folder under runs/> [train_speed_ppo.py options...]
set -euo pipefail
cd "$(dirname "$0")/.."
out="runs/$1"; shift
mkdir -p "$out"
RL_THREADS="${RL_THREADS:-4}" setsid nohup "$HOME/.venvs/idd-detect/bin/python" -W ignore rl/train_speed_ppo.py --output "$out" "$@" > "$out/train.log" 2>&1 < /dev/null &
echo "started $out (pid $!)"
