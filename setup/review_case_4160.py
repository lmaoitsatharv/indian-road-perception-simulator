"""Inspect saved validation predictions without running training or model inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
review = root / "runs/frozen_01/validation_review"
image = np.asarray(Image.open(root / "dataset/splits/val/images/Image_4160.png").convert("RGB"))
target = np.asarray(Image.open(root / "dataset/splits/val/masks/Mask_4160.png"))
prediction = np.asarray(Image.open(review / "predicted_masks/Prediction_4160.png"))
assert image.shape == (512, 512, 3)
assert target.shape == prediction.shape == (512, 512)
assert set(np.unique(target)) == set(np.unique(prediction)) == {0, 1}
tp = int(np.count_nonzero((target == 1) & (prediction == 1)))
fn = int(np.count_nonzero((target == 1) & (prediction == 0)))
fp = int(np.count_nonzero((target == 0) & (prediction == 1)))
iou = tp / (tp + fn + fp)
missed = fn / (tp + fn)
stats = {"id": 4160, "split": "validation", "threshold": 0.5,
         "road_iou": iou, "true_positive_pixels": tp,
         "false_negative_pixels": fn, "false_positive_pixels": fp,
         "fraction_of_provided_road_missed": missed,
         "inputs_modified": False, "model_inference_performed": False}
(review / "case_4160_metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
figure, axes = plt.subplots(1, 4, figsize=(16, 4.8), constrained_layout=True)
axes[0].imshow(image)
axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
axes[3].imshow(image)
overlay = np.zeros((512, 512, 4), dtype=float)
overlay[(target == 1) & (prediction == 0)] = [1, 0.15, 0.1, 0.55]
overlay[(target == 0) & (prediction == 1)] = [0, 0.65, 1, 0.65]
axes[3].imshow(overlay)
for axis, title in zip(axes, ["Original image", "Provided road mask", "Predicted road mask", "Errors relative to provided mask"]):
    axis.set_title(title, fontsize=11)
    axis.axis("off")
figure.suptitle(f"Validation ID 4160 | road IoU {iou:.3f} | {missed:.1%} of provided road missed", fontsize=14)
figure.legend(handles=[Patch(color="#ff261a", label="Missed road"),
                       Patch(color="#00a6ff", label="Extra predicted road")],
              loc="lower center", ncol=2, frameon=False)
figure.savefig(review / "case_4160_errors.png", dpi=150)
plt.close(figure)
(review / "case_4160_review.md").write_text(f"""# Validation case 4160

Compared existing 512 x 512 image, provided mask, and saved prediction at threshold 0.5.
Road IoU: **{iou:.5f}**. **{missed:.1%}** of the provided road pixels were missed.
True positives: {tp:,}; missed road pixels: {fn:,}; extra predicted road pixels: {fp:,}.

The model predicts only road/non-road. The missing foreground is classified as
non-road; there is no separate object or shadow class. Shadows and paving texture
are plausible contributing factors, but this single image does not isolate them.
This is a validation failure, even though global validation IoU is 92.03%.

## Proposed controlled improvement

Review existing shadow/textured-road training examples and the original road-class
definition. Then compare a separate robustness trial with the current baseline:
training-only brightness/contrast and realistic shadow variation, with moderate
settings checked visually. Photometric changes affect images only, before the
existing single [-1, 1] normalization; corresponding masks remain 0/1 unchanged.
Any later encoder fine-tuning is a separate decision/experiment.

Assess both unchanged full-validation metrics and a declared difficult-case subset.
Do not optimize only for this image or modify/delete labels because the model misses them.
The reserved test set remains untouched. No new training, inference, label changes,
or preprocessing changes were performed to create this review.

Figure: `case_4160_errors.png` (red: missed road; blue: extra predicted road).
""", encoding="utf-8")
print(json.dumps(stats, indent=2))
