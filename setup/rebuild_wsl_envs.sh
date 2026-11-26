#!/usr/bin/env bash
# Rebuild both WSL model environments used by camera driving (after a laptop reset).
# Run inside the WSL distribution (Ubuntu-24.04 by default) as the Linux user that runs the models; sudo asks for
# that user's Linux password once.
#   road model:  ~/.venvs/idd-tf220  (TensorFlow 2.20, requirements-wsl.txt)
#   detector:    ~/.venvs/idd-detect (PyTorch 2.9.1 cu128 + Ultralytics, requirements-detection-wsl.txt)
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== 1/2 road model environment (idd-tf220)"
if [[ -x "$HOME/.venvs/idd-tf220/bin/python" ]]; then
  echo "already there, skipping"
else
  bash "$PROJECT_DIR/setup_wsl_linux.sh"
fi

echo "== 2/2 detector environment (idd-detect)"
DET="$HOME/.venvs/idd-detect"
if [[ ! -x "$DET/bin/python" ]]; then
  python3 -m venv "$DET"
fi
"$DET/bin/python" -m pip install --upgrade pip
"$DET/bin/python" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
"$DET/bin/python" -m pip install -r "$PROJECT_DIR/requirements-detection-wsl.txt"
"$DET/bin/python" -c "import torch, ultralytics; assert torch.cuda.is_available(), 'no CUDA in idd-detect'; print('idd-detect OK:', torch.__version__, torch.cuda.get_device_name(0), 'ultralytics', ultralytics.__version__)"
"$HOME/.venvs/idd-tf220/bin/python" -c "import tensorflow as tf; g = tf.config.list_physical_devices('GPU'); assert g, 'no GPU in idd-tf220'; print('idd-tf220 OK:', tf.__version__, g)"

echo "Both environments ready. Start the simulation from Windows with Start_Camera_Driving.cmd."
