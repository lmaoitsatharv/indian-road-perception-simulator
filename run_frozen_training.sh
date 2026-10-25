#!/usr/bin/env bash
# Run an explicit frozen-encoder phase independently of a notebook.
set -uo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
training_python="${PROJECT_WSL_PYTHON:-$HOME/.venvs/idd-tf220/bin/python}"
if [[ "$training_python" != /* || ! -x "$training_python" ]]; then
  printf '%s\n' 'Set PROJECT_WSL_PYTHON to an existing absolute Linux Python interpreter.' >&2
  exit 2
fi
if (( $# > 1 )); then
  printf '%s\n' 'Usage: bash run_frozen_training.sh [new-run-directory]' >&2
  exit 2
fi
training_run="${1:-runs/frozen_01}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
printf '%s\n' "Frozen-encoder training started at $(date --iso-8601=seconds)"
"$training_python" -u train_idd.py --run-dir "$training_run" --epochs 20 --batch-size 4
result=$?
printf '%s\n' "Training process exited with code $result at $(date --iso-8601=seconds)"
mkdir -p setup
printf '%s\n' "$result" > setup/frozen_01_exit_code.txt
exit "$result"
