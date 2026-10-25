# WSL GPU environment

Road segmentation uses a Linux TensorFlow environment separate from the native Windows CARLA environment. Object detection uses another environment so its PyTorch dependencies do not modify TensorFlow. Start with [getting started](docs/GETTING_STARTED.md) for required local data and trusted model artifacts.

## Reference environment

Recorded TensorFlow experiments used Ubuntu 24.04 under WSL2, Python 3.12, TensorFlow 2.20.0 and Keras 3.15.1 on an RTX 4060 Laptop GPU with about 8 GiB memory. GPU matrix computation and a real batch-4 U-Net forward/backward check passed. A disposable verification update was discarded; full road training subsequently completed 18 epochs and selected epoch 14.

These are reference results, not a claim that WSL, drivers or dependencies are already installed on another computer. Package pins are in `requirements-wsl.txt` and `requirements-detection-wsl.txt`.

## Windows prerequisites

Hardware virtualization, WSL2 and a compatible Windows GPU driver are required. Use the machine manufacturer's instructions for virtualization settings. Installation from elevated PowerShell can begin with:

```powershell
wsl --install --distribution Ubuntu-24.04
wsl --update
wsl --list --verbose
```

Follow restart instructions and initialize the Linux account. The distribution must run as WSL version 2. Windows and Linux environments are separate; do not activate a Windows virtual environment from Linux.

## TensorFlow setup and verification

Open Ubuntu in the checked-out repository directory and run:

```bash
bash setup_wsl_linux.sh
```

The setup creates `~/.venvs/idd-tf220`, installs pinned TensorFlow/Keras dependencies, registers a notebook kernel and checks GPU computation. Its project-batch verification also requires local road data and encoder artifacts. A source-only checkout does not contain those inputs. Inspect `setup/wsl_gpu_verification.json`; environment installation alone is not proof that a project training batch succeeded.

WSL GPU access uses the Windows NVIDIA driver. This workflow does not install a Linux display driver into WSL. The TensorFlow CUDA package extra supplies the framework libraries for the recorded configuration.

## Notebooks

From a verified Ubuntu environment and repository directory:

```bash
source ~/.venvs/idd-tf220/bin/activate
python -m jupyterlab --no-browser --ip=127.0.0.1 --port=8890
```

Keep the access token private. Select the verified Linux kernel. Clean notebooks do not include saved execution output, and their training switch defaults to false. Starting a notebook server does not start a full training run.

## Detection environment

Create a separate environment such as `~/.venvs/idd-detect`. Install matching GPU PyTorch/torchvision builds followed by `requirements-detection-wsl.txt`; that file records package versions and the official PyTorch index used. Do not install these into `idd-tf220` merely to run detection.

Detector readiness verifies actual runtime, trusted initialization, class order and prepared-data provenance. See [detector preparation](docs/DETECTION_PREPARATION.md) and [training](docs/DETECTION_TRAINING.md). GPU discovery or a forward-only probe does not establish training memory or accuracy.

## Data and filesystem considerations

Windows drives appear under `/mnt/` in WSL. The road loader reconstructs pair paths from the current split directory and IDs rather than historical absolute path columns. Retain the established manifest, normalization, mask values and split assignments when transferring data. The real road split is 5,507/687/689; its final test evaluation is already complete.

Cross-filesystem PNG loading can limit throughput. If an additional Linux-storage copy is needed, verify counts/hashes and preserve membership and source files. Faster storage must not silently resplit data or introduce different labels.

## Troubleshooting

Before full training inspect the verification result, WSL version and `/usr/lib/wsl/lib/nvidia-smi` where available. Preserve GPU-memory failures and diagnose them in a fresh process; do not change masks or preprocessing to conceal them. Framework/version changes require repeating runtime checks.

Official references: [WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install), [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and [TensorFlow installation](https://www.tensorflow.org/install/pip).
