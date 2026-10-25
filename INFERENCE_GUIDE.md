# Using the locked road model

## What is ready

The selected morning-best U-Net/MobileNetV2 checkpoint can predict raw images through the existing WSL RTX 4060 environment. The model is loaded once per process. Prediction runs with `training=False`, without an optimizer or updates to weights or BatchNorm statistics.

The [real-video guide](REAL_VIDEO_GUIDE.md) covers the completed three-clip video
demo, source timestamps, per-frame 0/1 exports, measured GPU timings and visual
review. It reuses this exact inference class and fixed checkpoint. The external
clips have no road ground-truth masks; no video accuracy score is reported.

The saved final result remains **91.6549% test road IoU on 689 images**. This tool did not repeat that test or change the checkpoint, threshold, preprocessing or architecture.

## Run one image from Windows

Open PowerShell at the repository root with the required trusted model and data artifacts available:

```powershell
.\run_road_prediction.ps1 `
  -Image '.\dataset\splits\val\images\Image_9.png' `
  -OutputDir 'runs\my_prediction_01'
```

Choose a new output folder each time. Existing output folders are refused. The launcher uses the verified Ubuntu environment and RTX 4060. It does not silently fall back to CPU.

For a new image of another size, explicitly add `-Resize`. This directly resizes the oriented image to 512×512 using bilinear interpolation. It can distort proportions; future camera calibration must account for the recorded source-to-model image transform. Resizing is not a new model training step.

For several files, pass an array to `-Image`. They are read one at a time; the model is reused. Offline capture timestamps remain unknown unless a caller provides them through the Python API/CLI. A list of unrelated photographs is not a time-aligned driving sequence.

## Saved outputs

Each image gets a separate folder:

| File | Meaning |
|---|---|
| `road_mask_01.png` | Actual binary data: uint8 pixels are 0 or 1 |
| `road_probability.npy` | Float32 road probabilities before thresholding |
| `model_input_rgb.png` | Raw 512×512 RGB input used for this prediction |
| `mask_display.png` | White/black preview: 0/255 for display only |
| `road_overlay.png` | Green road prediction over the image |
| `frame.json` | Model hash, threshold, camera-pixel coordinates, input transform and timing |
| `../run_status.json` | Runtime, completed images and checkpoint preservation |

The binary PNG looks almost black in an ordinary image viewer because value 1 is close to black. Read it numerically or use the separate display preview. Do not replace the 0/1 data with the display file.

## Review improvements — 13 October 2026

Inference now checks the retained selection, final protocol, checkpoint hash and
split manifest before loading the model, and rechecks the checkpoint after loading
and after prediction. An image replaced after preflight stops the run. The bytes
decoded are also the bytes hashed.

Exports reject unsafe folder names, display-only 255 masks, non-finite
probabilities/timing, and masks inconsistent with the fixed `>=0.5` threshold.
`frame.json` is published last with an atomic UTF-8 write; a frame without that
file is incomplete. Interrupted/failed runs retain their status and evidence.
One supplied capture timestamp can describe only one image per CLI invocation.

The review verified the contracts using synthetic inputs and an actual locked CPU
model pass. It did not rerun test accuracy or measure new GPU latency. See
`runs/review_20261013/model_pipeline_verification.json`.

## Avoid normalizing twice

`idd_inference.LockedRoadSegmenter` offers two explicit entry points:

```python
from idd_inference import LockedRoadSegmenter, read_rgb

segmenter = LockedRoadSegmenter(device="gpu", gpu_memory_mib=3072)
rgb, metadata = read_rgb("my_512_image.png")
prediction = segmenter.predict_rgb(rgb)  # raw uint8 RGB: normalizes once

# For a tensor already prepared by the existing preprocessing pipeline:
# prediction = segmenter.predict_normalized(normalized_float32_rgb)
# This second entry point performs no rescaling.
```

Raw RGB must be uint8 with shape `(512,512,3)`. Already normalized input must be finite float32 of that shape in [-1,1]. Shape and range checks help catch incorrect use; callers must still know which preprocessing has already happened.

The transform is `raw / 127.5 - 1`. Black becomes -1 and white becomes +1. Masks are never passed through this transform. The output is thresholded at the locked **0.5**, using `>=`.

The GPU memory cap is 3,072 MiB by default. It is a pool limit, not a claim that the model continuously uses 3 GiB. Run the class in a fresh Python process so it can set this limit before the GPU context starts.

## What was actually verified

- Six fixed validation examples: IDs 9, 48, 103, 606, 4160 and 6514.
- Raw and already normalized entry points produced exactly identical probabilities.
- NumPy normalization exactly matched the existing TensorFlow MobileNetV2 preprocessing.
- All six masks reproduced the earlier saved masks exactly when using the original batch size of 16.
- Batch-one inference changed **43 of 1,572,864 pixels** across those examples, about **0.00273%**, from mixed-precision computation order near the threshold. Maximum probability difference was 0.006104. The threshold and weights were not changed to hide the differences.
- Saved mask/probability values round-tripped exactly; output tensors were on GPU:0.
- In-memory weights/BatchNorm, all three model files, the split manifest and final test metrics stayed unchanged.
- The PowerShell launcher ran end to end on validation Image 9, and MATLAB read its binary mask exactly.

Evidence: `runs/inference_verification_20261003_retry01/verification.json`, `runs/inference_diagnosis_20261003/diagnosis.json` and `runs/inference_cli_example_20261003/`. The initial overly strict comparison and its diagnosis remain recorded; no new test image contents were read.

## Using the output in MATLAB

```matlab
addpath(fullfile(pwd, 'integration', 'matlab'));
frame = read_camera_prediction( ...
  fullfile(pwd, 'runs', 'inference_cli_example_20261003', '001_Image_9'));
imshow(frame.road_mask);  % logical display; the original PNG stays uint8 0/1
```

This returns a logical camera mask. Its metric planning map is empty and `planning_use_valid` is false because camera calibration is absent. A road-surface probability also does not identify every obstacle or prove a location is safe to drive through.

For runtime loading semantics, see the [official TensorFlow model-loading API](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model). Local TensorFlow 2.20.0 execution is the verification evidence for this project.
