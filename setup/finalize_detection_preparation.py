"""Record the completed preparation work while keeping real-data training gated."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import file_sha,load_policy,save_json
from setup.verify_detection_runtime import protection


def load(path):
    return json.loads(Path(path).read_text())


def main():
    run=ROOT/"runs/detection_readiness_20261023_01"
    runtime=load(run/"runtime_isolated/runtime_verification.json")
    data=load(run/"data_checks_release/verification.json")
    workflow=load(run/"workflow_checks/verification.json")
    integration=load(run/"framework_integration/verification.json")
    assert all(r["passed"] and not r["training_started"] for r in (runtime,data,workflow,integration))
    assert protection()==runtime["protected_sha256"]
    preflight=subprocess.run([sys.executable,str(ROOT/"train_detection.py"),"--output",str(run/"training_readiness.json")],
        capture_output=True,text=True,cwd=ROOT)
    assert preflight.returncode==2,preflight.stdout+preflight.stderr
    ready=load(run/"training_readiness.json")
    assert not ready["ready_for_explicit_training_start"] and ready["checks"]["gpu_forward_runtime_checked"]
    assert ready["checks"]["initial_weights_verified"]
    download=load(ROOT/"dataset/idd117k_raw/download_progress.json")
    result={"status":"preparation_complete_waiting_for_verified_real_dataset",
        "completed_utc":datetime.now(timezone.utc).isoformat(),"training_started":False,
        "detector":"YOLO11s proposed candidate, official COCO initial weights; not IDD-fine-tuned",
        "planned_names":load_policy()["names"],"data_checks":data,"workflow_checks":workflow,
        "framework_integration":integration,"runtime":runtime,"training_readiness":ready,
        "download_snapshot":{k:download[k] for k in ("status","updated_utc","percent","completed_bytes","total_bytes","recent_MB_s")},
        "original_road_models_and_driver_unchanged":True,"model_control":False,"rl_started":False,
        "remaining_work":["Complete byte verification, combined gzip integrity and full archive inventory",
            "Review native coordinate conventions and all source labels across both releases",
            "Index the older original detection portion without inventing drive IDs; merge with 95K source manifest",
            "Prepare and audit full provider splits, duplicates, drive isolation, actual boxes and class coverage",
            "Measure initial backward-pass memory and throughput when training is explicitly started"],
        "report":"docs/DETECTION_PREPARATION.md"}
    save_json(run/"preparation_results.json",result)
    save_json(ROOT/"docs/results/detection_preparation.json",result)
    names=["detection_policy.json","detection_training_recipe.json","requirements-detection-wsl.txt",
        "idd_detection_data.py","train_detection.py","setup/prepare_idd_detection.py","setup/index_idd95k_detection.py",
        "setup/inspect_idd_archive.py","setup/verify_detection_data.py","setup/verify_detection_workflow.py",
        "setup/verify_detection_runtime.py","setup/verify_detection_integration.py","setup/finalize_detection_preparation.py"]
    source=run/"source";source.mkdir(exist_ok=False)
    for name in names:
        destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    save_json(run/"source_hashes.json",{name:file_sha(ROOT/name) for name in names})
    report=f"""# Object-detector preparation — 23 October 2026

The independent preparation work is complete while IDD-117K continues downloading.
**No IDD detector has been trained.** The real-road and simulation-road models,
their inference defaults and the rule driver remain unchanged by hash comparison.
Perception remains outside vehicle control; no RL, tracking or distance estimator
was started.

## What is ready

- A separate WSL environment: `$HOME/.venvs/idd-detect/bin/python`.
  PyTorch {runtime['torch']}, torchvision {runtime['torchvision']}, Ultralytics
  {runtime['ultralytics']}; dependency checking found no conflicts. Package locks
  and isolated runtime settings are retained in the readiness run. The existing
  `idd-tf220` environment was not used for installations.
- Official **YOLO11s COCO-pretrained weights**, with recorded origin and SHA-256
  `{runtime['checkpoint_sha256']}`. These are an initial candidate, not an
  IDD-trained or approved real-world detector.
- A converter for the observed IDD JSON lists and separate legacy VOC XML. It
  requires each source's coordinate convention and actual image dimensions,
  converts to normalized detector boxes and records clipping and duplicate boxes.
- A source-index helper for the **observed 95K folder layout**, preserving native
  train/val/test and numeric drive folders. It checks the published 95K counts.
  The older original release still needs its own inspected layout/identity adapter.
- A dataset exporter that retains labelled negative images, blocks missing or
  unknown labels, and checks drive-level and exact-image leakage. Official test
  images are metadata-only; they are never opened or used as training negatives.
- A guarded training entry point. Its default action is **readiness checking**;
  actual training requires an explicit start argument, a fresh output folder,
  verified archives, a full audited dataset and the expected provider counts.

## What the partial archive actually showed

The already downloaded, committed first 64 MiB were rehashed before inspection.
The prefix begins with `IDD117K_Detection/IDD_95kDetection/val/Labels_json/` and
then `val/leftImg8bit/`. This was a prefix/schema inspection, not complete gzip
integrity verification or a full archive inventory.

Twelve validation annotation files contained **655 annotated objects**. They were
read only to establish the JSON schema and observed class names; no real image
pixels were decoded and no model accuracy was measured. An annotation is a list
of objects such as `name` plus `bbox` with `xmin`, `ymin`, `xmax`, `ymax`. It does
not contain image dimensions, so assuming every frame is 1920x1080 would be wrong.

Coordinate conventions remain explicit: continuous edge coordinates and one-based
inclusive VOC coordinates have different conversions. Their use must be reviewed
against the complete source release before the source manifest is finalised.

## Planned classes

The proposed 12 outputs are person, rider, car, motorcycle, bicycle, autorickshaw,
bus, truck, animal, traffic light, traffic sign and other vehicle. The last class
merges the source's vehicle-fallback/caravan/trailer/train labels instead of
silently turning those vehicles into background. Class names and IDs are fixed
in `detection_policy.json`. Full-release label inventory and class coverage are
still pending; this is a project mapping, not the official IDD benchmark taxonomy.

Explicit non-target labels such as ego vehicle and road remain background.
Unknown classes and difficult/ignore/crowd flags currently quarantine the whole
image and block readiness, rather than silently dropping positive objects.
If such flags occur in the full release, implement and validate an appropriate
ignore policy before training; do not discard difficult examples without review.

## Verification completed

The data and workflow checks passed: non-square coordinate round trips, VOC
origin conversion, clipping, overlapping rider/bike boxes, true negative images,
missing/invalid/unknown annotations, drive conflicts, byte-identical and
re-encoded pixel-identical images across splits, source indexing and training
refusal for fixture/incomplete data. The framework itself loaded the exported
fixture labels with 12 boxes and retained its separate empty-label negative.
It accepted the generated YAML/class order and produced 640x640 letterboxed
tensors. Its 12-class head loaded compatible pretrained features and completed
a finite-output GPU forward pass.

On the RTX 4060 Laptop GPU, the COCO model's direct FP32 forward check used
batch 4 at 640x640: median **{runtime['batch_forward_median_ms']:.2f} ms per batch**,
maximum **{runtime['batch_forward_max_ms']:.2f} ms**, peak allocated memory
**{runtime['peak_allocated_bytes']/1024**2:.1f} MiB**. This is a synthetic-input
forward check, not camera latency, real-image accuracy, or proof that training's
backward pass fits the same memory. No optimizer/backward steps were executed.

An initial runtime test fell back to a temporary settings folder. The final
test explicitly created and verified its own settings directory; both records
are preserved. The official checkpoint file remained unchanged in all checks.

## Proposed first training run, not started

The saved recipe proposes pretrained YOLO11s, 640-pixel letterboxing, batch 4,
seed 42, AdamW at 1e-4, up to 20 epochs, patience 5 and AMP. Checkpoints use the
pinned framework's validation fitness. These are starting settings, not a proven
optimum or a training-time estimate. Measure the first real training batches
and backward memory before committing to the longer run.

Run a readiness check from the project folder on Windows:

```powershell
.venv/Scripts/python.exe train_detection.py
```

A nonzero return with a list of waiting conditions is expected while downloading.
After data inspection/import/audit and an explicit training request, the separate
WSL environment can start a fresh run:

```bash
$HOME/.venvs/idd-detect/bin/python train_detection.py --start-training --output runs/detection_train_NEW
```

The command will refuse to start until its data gates pass. It never silently
uses a tiny fixture or an incomplete subset. The current default expects the
combined provider counts of 96,897 train, 20,202 validation and 24,644 test-metadata
images. The 95K index alone is intentionally insufficient for that recipe.

## Remaining after download

1. Finish part verification, check the combined gzip CRC/member inventory, and
   plan extraction against available space without concatenating another 73 GB file.
2. Inspect the older release, source coordinate conventions and full class set;
   create the complete source manifest automatically from provider metadata.
3. Convert/audit actual images and labels, retaining drive/split identity and
   resolving any leakage, unusual flags, missing classes or malformed boxes.
4. Start training only when requested, measure actual learning/throughput and
   evaluate the fixed candidate on appropriate labelled real data. The official
   unlabelled test images cannot supply local accuracy numbers.

Road segmentation and this detector are separate models. Detection produces
object categories and image rectangles; tracking identity and estimating metres
are separate following steps. No live driving use is implied by these checks.

## Evidence and sources

Full evidence: `runs/detection_readiness_20261023_01/`. Portable results:
[detection_preparation.json](results/detection_preparation.json). Current download:
`dataset/idd117k_raw/DOWNLOAD_STATUS.md`.

- [Official IDD catalogue](https://insaan.iiit.ac.in/datasets/)
- [Official IDD annotation/evaluation code](https://github.com/AutoNUE/public-code)
- [YOLO11 model documentation](https://docs.ultralytics.com/models/yolo11/)
- [Official matched PyTorch/torchvision builds](https://pytorch.org/get-started/previous-versions/)

The detector dependency/model follows Ultralytics' AGPL-3.0 / enterprise licensing;
the downloaded dataset has its own portal terms. Origins and versions are retained
with the local runtime and checkpoint metadata.

Numerical coordinate-check illustration (synthetic, no accuracy claim):

![Native-coordinate round trip](<{(run/'data_checks_release/coordinate_check.png').as_posix()}>)
"""
    (ROOT/"docs/DETECTION_PREPARATION.md").write_text(report,encoding="utf-8")
    (run/"PREPARATION_REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps({"status":result["status"],"training_started":False,
        "download_percent":download["percent"],"data_check_groups":data["check_groups"],
        "workflow_check_groups":len(workflow["checks"]),"gpu_runtime_passed":True,
        "framework_integration_passed":True,"road_models_unchanged":True},indent=2))


if __name__=="__main__":
    main()
