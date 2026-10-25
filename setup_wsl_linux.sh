#!/usr/bin/env bash
set -euo pipefail

# Run inside Ubuntu 24.04, as your normal Linux user.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$HOME/.venvs/idd-tf220"
if [[ "$(id -u)" -eq 0 ]]; then
  echo 'Run as your normal Ubuntu user; sudo will ask for your Linux password.' >&2
  exit 1
fi
if ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  echo 'This setup is intended for WSL2 Ubuntu.' >&2
  exit 1
fi
if [[ ! -e /dev/dxg ]]; then
  echo 'WSL GPU access is missing. Check WSL2, BIOS virtualization and the Windows NVIDIA driver.' >&2
  exit 1
fi
if [[ "${1:-}" == '--skip-system-packages' ]]; then
  # For an administrator-assisted setup after these packages were installed.
  dpkg-query -W python3-venv python3-pip libgomp1 > /dev/null
else
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-pip libgomp1
fi
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Ubuntu 24.04 / Python 3.12 for this environment"'
python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements-wsl.txt"
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m ipykernel install --user --name idd-wsl-gpu --display-name 'IDD WSL RTX 4060'
mkdir -p "$PROJECT_DIR/setup"
"$ENV_DIR/bin/python" -m pip freeze > "$PROJECT_DIR/setup/wsl_packages.txt"
cd "$PROJECT_DIR"
"$ENV_DIR/bin/python" "$PROJECT_DIR/check_wsl_gpu.py"
echo 'GPU setup verified. Full training has not started.'
echo "Notebook command: $ENV_DIR/bin/python -m jupyterlab --no-browser --ip=127.0.0.1"
