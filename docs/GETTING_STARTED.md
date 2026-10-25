# Getting started

The simplest entry point is the rule-driven CARLA town. Model inference and training use separate runtimes and external artifacts.

## Requirements

Use Windows and PowerShell, official 64-bit Python 3.12 with the py launcher, Git and a compatible CARLA 0.9.16 graphics setup. Allow at least 30 GB free for the compressed engine and extracted installation, plus space for recordings/data/runs.

GPU model work was measured through WSL2 Linux. Recorded framework versions are compatibility information, not proof that every hardware/driver combination was tested. The MATLAB planning references are optional and are not required to run the native town.

## Obtain the source

    git clone https://github.com/lmaoitsatharv/indian-road-perception-simulator.git
    cd indian-road-perception-simulator

Run subsequent commands from that repository root.

## Install and run CARLA

    .\setup\project\bootstrap_carla.ps1 -DownloadCarla
    .\Start_Combined_Indian_Traffic.ps1 -SimulationSeconds 120 -Quality Low

The bootstrap creates .venv-carla, installs [the simulator Python dependencies](../setup/carla/requirements.txt), downloads the official Windows engine archive and runs the checked extraction utility. Engine files go under tools/carla/0.9.16/ and machine-specific installation records are created locally under setup/carla/.

The pinned environment uses CARLA 0.9.16, pygame 2.6.1 and NumPy 1.26.4. The Python carla package alone is not the engine executable. The installer refuses to overwrite an existing unverified installation. Its archive hash is a local integrity record, not a publisher-supplied checksum.

The combined town contains all five districts together. Select an appearance condition with:

    .\Start_Combined_Indian_Traffic.ps1 -SimulationSeconds 120 -Seed 8101 -Weather overcast

Selectable weather is not evidence of robustness in that condition. The current driver uses preset routes and rules, not camera predictions or selected destinations.

| Combined-town control | Action |
| --- | --- |
| Tab | Switch front/follow camera |
| Space | Pause/resume |
| S | Save a screenshot |
| Esc | Close the owned session |

Each run writes a new local runs/carla_indian_worlds_* folder. Inspect its recorded outcome instead of treating an opened window as a passed scenario. Close through the viewer so owned actors, sensors and processes are released.

The smaller neighbourhood provides separate traffic scenarios:

    .\Start_Indian_Traffic.ps1 -Seed 7 -Scenario mixed

Its additional camera/replay/playback controls are documented in [simulation controls](../INDIAN_SIMULATION.md). Neither rule-driven scene requires trained models or datasets.

## Separate Python runtimes

| Runtime | Dependency specification | Purpose |
| --- | --- | --- |
| Windows .venv-carla | [setup/carla/requirements.txt](../setup/carla/requirements.txt) | Native simulator client/viewer |
| Windows TensorFlow | [requirements-inference.txt](../requirements-inference.txt) | CPU road inference and utilities |
| WSL TensorFlow | [requirements-wsl.txt](../requirements-wsl.txt) | GPU segmentation training/inference |
| Separate WSL detector | [requirements-detection-wsl.txt](../requirements-detection-wsl.txt) | Pinned PyTorch/Ultralytics workflow |

Do not combine all specifications in the CARLA environment: framework and NumPy requirements differ. Package installation does not supply checkpoints, datasets or readiness reports.

For Windows CPU inference:

    py -3.12 -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt

For GPU segmentation, use WSL2 with Ubuntu 24.04/Python 3.12 and working NVIDIA GPU access. The [WSL guide](../WSL_SETUP.md) and setup_wsl_linux.sh describe setup and GPU verification. The verification also needs the relevant external prepared data.

The isolated detector specification records PyTorch 2.9.1, torchvision 0.24.1 and Ultralytics 8.3.225. Inside a dedicated WSL Python environment, its pinned installation is:

    python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements-detection-wsl.txt

Verify runtime/driver compatibility and actual GPU availability before training. This does not create the prepared detector dataset or provenance.

## External models and provenance

The [model manifest](../models/perception_models.json) records tasks, relative checkpoint paths, hashes and result scopes. The [real-road lock](../models/segmentation_final_lock.json) also fixes threshold, preprocessing and split identity.

Obtain trusted checkpoints and required metadata through a distribution channel that permits those files. Place them at the documented relative paths and compare SHA-256 values. Substituting a different checkpoint does not preserve the recorded result.

The real-road inference guard also requires dataset/splits/pair_splits.csv with its locked hash and split membership. Full training images are not needed merely to predict a supplied photograph, but the manifest remains part of the current verification contract. It is absent from a source-only clone.

After those prerequisites are supplied, explicit Windows CPU prediction is:

    & .\.venv\Scripts\python.exe predict_road.py --image media/input.png --output-dir runs/prediction_new --resize --device cpu

Replace media/input.png with an image you may use. The resize switch explicitly transforms a non-512x512 source to the trained shape, and the saved transform matters for geometry. Choose a new output directory; existing results are refused. Missing required weights/manifest is expected to block inference rather than justify disabling checks.

The simulation-road model has a separate manifest/scope. Its live launcher additionally requires the checkpoint, confirmation/provenance records, native CARLA setup, Windows utility environment and configured WSL GPU runtime. It is not a turnkey source-only startup path. See [live observation](LIVE_SIMULATION_OBSERVATION.md) and [the code map](CODE_MAP.md). Live predictions do not enter vehicle controls.

## Local runtime paths

WSL launchers use `PROJECT_WSL_DISTRIBUTION` (default `Ubuntu-24.04`) and `PROJECT_WSL_PYTHON` when set. Without an explicit interpreter, they resolve the selected Linux user's home and use its `.venvs/idd-tf220/bin/python`. The PowerShell inference/evaluation wrappers also accept `-WslDistribution` and `-WslPython`. Supply an absolute Linux interpreter path when the environment is installed elsewhere.

Optional MATLAB tools accept `--matlab`, then `MATLAB_EXECUTABLE` or a MATLAB executable on `PATH`. Legacy plotting helpers can use `PROJECT_PLOT_PYTHON`; that runtime needs the packages in `requirements-training-review.txt`. These settings select local tools and do not install packages or bypass model/data verification.

The Jupyter wrapper also accepts `-Port`. Inside Linux, `run_frozen_training.sh` accepts an optional fresh run directory and reads `PROJECT_WSL_PYTHON` or the current user's default environment path. Windows-to-WSL conversion assumes the standard `/mnt/<drive>` automount; configure a compatible layout if that mount convention differs.

The dated overnight queue/controller utilities are retained as historical experiment code. They require their original session/lease/provenance contracts and are not general startup commands. Use the explicit trainers and fresh run directories for a new experiment.

## Training data and run preparation

No training dataset is included. Obtain source data under its terms, prepare/audit the required views and preserve the documented split/label contracts. Data and weights stay outside source commits.

The detector entry point defaults to readiness inspection. Training requires its explicit start switch and a fresh output directory. Recorded recipes refer to particular verified parents and provenance; editing paths does not regenerate those proofs.

Real-road and synthetic-road trainers use separate contracts. Synthetic invalid pixels require masked loss/metrics and must not become ordinary background. Preserve failed runs and compare candidates with matched data, resolution and precision.

## Where to go next

Read [results](RESULTS.md), [the code map](CODE_MAP.md), [roadmap](NEXT_STEPS.md) and [documentation index](INDEX.md). Recorded metrics are not reruns on your machine. Only selected evidence is committed; complete data/run/model artifacts remain external. Keep RPC/preview services on trusted local interfaces as described in [security guidance](../SECURITY.md).
