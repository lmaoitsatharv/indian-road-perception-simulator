# Understand the project from zero

**A practical guide for Atharv and Sarthak - Milestone 2, 26 October 2026**

This guide explains the implementation that exists in this repository. It separates completed work, experiments, limitations and planned work. Read it alongside the source rather than memorizing sentences without understanding them. The [progress report](MILESTONE_2_PROGRESS.md) is the short assessment-facing version. The [file guide](FILE_GUIDE.md) covers every published file and the [ownership plan](TEAM_OWNERSHIP.md) divides preparation and remaining responsibility.

## 1. What exactly are we building?

The project has two foundations: computer vision that understands road images, and an Indian-style traffic simulation in which driving behavior can be tested. The eventual goal is to connect them to a car that can follow a chosen route and react to obstacles.

Today, three current model roles are implemented, with several retained checkpoints and experimental variants:

1. A **real-road segmentation model** marks road pixels in real photographs or video.
2. A **simulation-road segmentation model** does the same task for the particular synthetic scenes we generate.
3. A **real-image object detector** produces labelled boxes around people, vehicles, animals and traffic objects.

The simulator also has a **rule-based driver**. It already moves vehicles using steering, throttle and braking. That driver receives exact actor positions from the simulator. It currently does not receive the models' predicted road mask or object boxes as driving inputs. The live road-model preview draws predictions for observation.

**A speakable answer:** "We have built and evaluated road segmentation and object detection, and we have a working Indian-style traffic simulator. The simulator currently drives with a rule-based baseline. Our next stage is connecting the perception outputs to driving decisions and destination navigation."

Do not describe this stage as a completed camera-driven autonomous car, a trained RL agent or a reconstruction of a real Indian city. Those statements would describe different work from what is implemented.

## 2. The essential vocabulary

An **image** is a grid of pixels. An RGB pixel has red, green and blue values, usually integers from 0 to 255. A 512x512 RGB image contains 512 rows, 512 columns and three channels.

A **label** is the supplied answer used during learning. For segmentation the label is a mask: each pixel is 1 for the supplied road class or 0 for background. For detection the labels are rectangles and category numbers. The label is not the model's prediction.

A **model** is a network of mathematical operations with adjustable numbers called **weights** and **biases**. During training, data and labels help the optimizer adjust those numbers. During inference, fixed weights produce predictions for a new input.

A **tensor** is an array of numbers. A batch of four road images has shape `[4, 512, 512, 3]` in the TensorFlow pipeline. Its target batch has shape `[4, 512, 512, 1]`. PyTorch's detector internally uses channel-first image tensors, typically `[batch, 3, height, width]`. The framework handles that convention; the two input contracts should not be mixed.

A **parameter** is a learned number. A **hyperparameter** is a setting chosen for the experiment, such as learning rate, batch size or input resolution. Model weights are parameters; 0.00003 as a learning rate is a hyperparameter.

A **framework** supplies the automatic differentiation, layers and training machinery. We use TensorFlow/Keras for road segmentation and PyTorch/Ultralytics for detection. CARLA is the simulator, not the neural-network training framework. WSL2 is the Linux environment on Windows, not a model.

## 3. Segmentation, detection, tracking, planning and control

| Term | Question it answers | Example in this project |
| --- | --- | --- |
| Segmentation | Which pixels belong to road? | A 512x512 probability map and binary road mask |
| Detection | What objects appear, and where are their boxes? | A person box with class and confidence |
| Tracking | Is this the same object over successive frames? | A persistent ID for one pedestrian; integration remains future work |
| Distance/motion estimation | How far away is it, and is it approaching? | A future range/closing-motion estimate checked against simulation reference |
| Route planning | Which connected roads should the car use to reach a goal? | Destination selection and routing remain future combined-town work |
| Behavior decision | Should we stop, slow, proceed or consider passing? | Current hand-coded rules; bounded RL comparison is planned |
| Control | What steering, throttle and brake commands execute that decision? | Current CARLA vehicle controller |

A detector box does not provide an object's identity over time or distance in metres. A road mask is not an obstacle-free path: a car may occupy a road region, and the segmentation label describes visible road surfaces rather than a driving permission. A correct route also needs a controller to follow it.

**Teacher: "Does the model drive the car?"** Answer: "Not yet. The model currently produces perception outputs. The simulator car uses rules and simulator positions. We still need the observation-to-decision integration and its validation."

## 4. The three datasets: keep their stories separate

| Dataset | Target | Current split | Important qualification |
| --- | --- | --- | --- |
| Resized real-road image/mask copy | One binary road label per pixel | 5,507 train / 687 validation / 689 test | Original sequence IDs and exact original semantic remapping are unavailable |
| IDD-117K detection collection, clean v3 view | Twelve classes of bounding boxes | 56,729 train / 17,675 validation | 24,644 official test identities are unlabelled metadata, not a locally scored test |
| Generated simulation road dataset | Road mask plus valid-pixel mask | 4,000 train / 500 validation / 500 test | Whole episodes assigned to one split; synthetic layouts remain related |

The road data and detection data are not interchangeable just because both relate to IDD. Their counts, labels, cleaning, normalization and evaluation protocols differ.

**Training data** is used for gradient updates. **Validation data** is used to choose a model or experiment setting without updating its weights on those images. **Test data** is held back for a final evaluation after the model/protocol are fixed. Repeatedly choosing experiments from a test score turns that set into development information.

For the real road model, the chosen checkpoint and threshold were locked before evaluating all 689 test images. That test is now inspected. For the detector, the current result is development validation, not an independent labelled test. For the simulation model, a separate fixed-model test and later fresh-scene confirmation were completed; those scenes are also now inspected.

**Data leakage** means evaluation data or closely related information unfairly influences training or selection. Exact duplicates across train and validation are one source. Adjacent frames from the same drive are another. Hashing can find exact matches; it cannot prove that every similar frame or correlated scene has been removed.

## 5. Real-road preprocessing, step by step

The original flat copy had 6,993 numbered photographs and 6,993 corresponding masks. The pairing key is the number: `Image_9.png` goes with `Mask_9.png`. A directory listing alone is unsafe because file order can change. `prepare_idd_splits.py` inventories numeric IDs, rejects missing/duplicate IDs, computes hashes and records assignments in a CSV manifest.

A **manifest** is an explicit list of the examples and their identities. `dataset/splits/pair_splits.csv` is the authoritative current split list. The loader reconstructs paths from the current folder and pair IDs, so old absolute path strings in the CSV do not control loading.

The initial audit found **zero exact duplicate input images**. There was one repeated-mask group consisting of 110 entirely zero masks. An identical target mask is not the same thing as an identical input image. For example, two different photographs could legitimately both have no visible target class.

Those 110 pairs were excluded from the split copies at the user's request after sampled images showed road surfaces. Original files and removal records were preserved. Their original label definition was not fully established, so we must not claim all 110 were proven incorrectly labelled. This filtering changed coverage. The retained total is 6,883 = 5,507 + 687 + 689.

`idd_data.py` then performs these operations when a batch is requested:

1. Read the manifest, reject repeated IDs and verify the files match it.
2. Decode image PNG as three-channel RGB and mask PNG as one channel.
3. Require the saved 512x512 shapes. This loader validates the existing resized data; it does not silently resize arbitrary new source images.
4. Cast values to float32. Keep target labels exactly 0 or 1.
5. For training, shuffle and sometimes flip image and mask horizontally using the same random decision.
6. Apply MobileNetV2 preprocessing to RGB once.
7. Batch four examples, preserve the last smaller batch and prefetch a bounded amount.

The normalization is `normalized = pixel / 127.5 - 1`. Thus 0 becomes -1, 127.5 becomes 0 and 255 becomes 1. This scale matches the pretrained encoder's expected input.

**Never divide these masks by 255.** The actual labels already are 0/1. Dividing 1 by 255 produces approximately 0.00392, which damages the target. A separate black/white display mask may use 0/255 for visibility; that is not the training target.

An **augmentation** creates a variation during training. If a photograph flips but its mask does not, the road answer no longer aligns with the image. Validation/test receive no augmentation. Later selected real-model experiments also changed brightness/contrast and applied smooth synthetic shadows to RGB; labels stayed aligned and unchanged in meaning.

**Why not normalize twice?** The loader already produces [-1,1]. Applying the formula again would turn these numbers into an unintended range near -1. `idd_model.py` deliberately contains no second input rescaling.

**Important split limitation:** flat filenames do not reveal original drives. Exact-image grouping helps against direct duplication, but we cannot claim drive-independent splitting for this small segmentation copy.

Code to open: [idd_data.py](../idd_data.py), [prepare_idd_splits.py](../prepare_idd_splits.py), [check_preprocessing.py](../check_preprocessing.py). The initial split script is historical; do not rerun it over the cleaned manifest.

## 6. Detection import and duplicate removal, step by step

The larger detector collection arrived in five archive parts. The importer reads them as one stream, verifies the parts and gzip/tar inventory, and extracts safely without creating another huge concatenated archive. A SHA-256 is a fingerprint of file content. Matching hashes help verify a copy is unchanged; they do not prove that the source annotation was correct or that a local hash is a publisher-supplied checksum.

There are two inspected annotation layouts: newer JSON and older XML. The source index uses actual provider lists and recording identities instead of guessing membership from convenient filenames. Provider counts are 96,897 train, 20,202 validation and 24,644 unlabelled test identities. Extra source files are not assigned invented split membership.

The filtered view preserves original provider split assignments:

1. Exclude labelled records sharing a recording with the reserved provider test.
2. Exclude training records sharing a recording with validation.
3. Exclude unresolved mixed uploads conservatively.
4. Decode labelled candidates and verify actual native image dimensions, annotation shape and class policy.
5. Exclude invalid whole image/annotation pairs and exact training duplicates of validation.
6. Preserve valid annotated images with no target objects as negative examples.

A **negative example** is a genuinely annotated image with no target object. An image without an annotation is not proof that no object exists. The official unlabelled test images never become negatives and are not decoded for local model scoring.

The earlier v2 view had 67,159 training entries and 17,675 validation images. It already excluded 56 invalid pairs and 6,767 training entries matching validation by original bytes or decoded RGB pixels. Both hashes matter: two JPEG encodings can have different file bytes but identical decoded pixels.

The training-only v3 cleaning then groups rows by their decoded pixel hash:

- If all copies have the same saved label hash, retain one deterministic canonical example and exclude extra repeated entries.
- If identical pixels have different saved label hashes, exclude the entire ambiguous group instead of silently choosing or merging annotations.
- Validation membership and labels remain unchanged.

**The arithmetic:** 67,159 - 5,045 ambiguous entries - 5,385 extra consistent duplicates = **56,729 unique retained training images**. The 5,045 entries belonged to 2,516 ambiguous pixel groups. Exclusions were written to provenance records; raw provider images/annotations were not deleted or automatically relabelled.

Different saved label hashes can reflect true annotation disagreement or simply differing serialization/order. The conservative exclusion does not prove every discarded annotation was wrong. Exact duplicate removal also does not eliminate every visually similar image.

Prepared image views use **hard links** where possible. A hard link is another filename for the same file content on the same filesystem. It saves storage; it is not image augmentation or compression. Editing linked content would affect all names, so the source/prepared images are treated as immutable. Labels and audit records are separate.

Code to open: [archive importer](../setup/import_idd117k_archive.py), [provider index](../setup/index_idd117k_detection.py), [recording curation](../setup/curate_idd_detection_manifest.py), [quality filtering](../setup/finalize_idd_detection_quality.py), [v3 cleaning](../setup/prepare_clean_detection_view.py).

## 7. How annotations become detector labels

The project's categories are person, rider, car, motorcycle, bicycle, autorickshaw, bus, truck, animal, traffic light, traffic sign and other vehicle. `detection_policy.json` fixes their order. Class 0 is person; class 2 is car. A trained checkpoint and its labels must agree on that order.

The source policy maps declared labels into these classes. Some fallback vehicle categories map to other vehicle. Known non-target categories remain background. Unknown or unsupported annotations block readiness instead of quietly becoming background. This is a project-specific detection protocol, not the official full IDD benchmark.

Source boxes are interpreted using the inspected coordinate contract and actual image dimensions. A valid box must have positive width/height and finite coordinates. Boxes crossing the visible image edge can be clipped and recorded; wholly invalid pairs are excluded. Repeated boxes are removed and counted. XML dimensions must agree with the decoded image.

YOLO labels store one object per line:

```text
class_id center_x center_y width height
```

The four coordinates are normalized by native image width/height. For a 200x100 image with box edges `(10,20)` to `(110,70)`:

```text
center_x = ((10 + 110) / 2) / 200 = 0.30
center_y = ((20 + 70) / 2) / 100 = 0.45
width    = (110 - 10) / 200      = 0.50
height   = (70 - 20) / 100       = 0.50
car row  = 2 0.30 0.45 0.50 0.50
```

The saved labels were checked independently, including normalized ranges, class IDs, counts and comparisons with sampled original annotations. Conversion correctness is different from annotation truth: a source object can still be mislabelled even if its coordinates are converted exactly.

At runtime, the detector uses **letterboxing**: resize while preserving aspect ratio, then pad to the required input shape. This is different from stretching every source image to a square. The detector's own pipeline handles its image scaling, normally to [0,1]; do not feed it the road model's MobileNetV2 [-1,1] preprocessing.

Code to open: [idd_detection_data.py](../idd_detection_data.py), [detection_policy.json](../detection_policy.json), [independent label audit](../setup/audit_prepared_idd_detection.py).

## 8. How the real road model works

The architecture is a U-Net-style decoder connected to a pretrained MobileNetV2 encoder. **Transfer learning** means reusing features learned previously rather than learning all visual filters from random numbers. The ImageNet weights initialize the encoder. They are not a ready-made Indian-road model.

The original classification head is removed. The encoder transforms a 512x512x3 input into progressively smaller feature maps. Early maps retain spatial detail; later maps describe more abstract patterns. The selected outputs are at 256x256, 128x128, 64x64, 32x32 and 16x16, with channel counts 96, 144, 192, 576 and 320 respectively.

A **convolution** applies learned filters across local regions to produce feature responses. A **channel** in a feature map is one learned response pattern, not necessarily an RGB color. A **ReLU** sets negative activations to zero and introduces nonlinearity. A network without suitable nonlinearities could not represent the same rich input/output relationships.

The decoder upsamples the 16x16 feature map. Four upsampling blocks use 256, 128, 64 and 32 filters. Each combines its upsampled features with a matching encoder feature through a **skip connection**. This helps recover edges and location detail that downsampling reduced. A skip connection transfers features, not answer masks.

A final upsample and refinement restore 512x512. A 1x1 convolution with sigmoid gives one probability per pixel. A **sigmoid** maps a real number into 0..1. A pixel at 0.82 is a high predicted road probability under this model, not a guaranteed safe-driving confidence.

The fixed inference threshold is 0.5: probability at least 0.5 becomes road, otherwise background. Probabilities are used during training; thresholding is used for final binary masks and overlap evaluation. Thresholding before the training loss would obstruct useful gradients.

The initial network had 6,202,337 parameters, with 4,357,473 trainable during the frozen-encoder stage. "Frozen" means those encoder parameters are excluded from optimizer updates. Their features still participate in the forward calculation.

**Batch normalization** uses learned scale/offset and running statistics. In this project the encoder remains in inference mode to preserve pretrained statistics. Later fine-tuning can update selected convolution weights while keeping BatchNorm fixed. This is why merely saying "we unfroze the model" is incomplete.

Code to open: [idd_model.py](../idd_model.py), especially `build_unet`, `BCEDiceLoss`, `RoadIoU`, `RoadDice` and `compile_model`.

## 9. What the road training loss means

A **loss** gives the optimizer a differentiable numerical measure of error. The implemented road loss adds binary cross-entropy and soft Dice loss, with equal weights, averaged per image.

For one target `y` and probability `p`:

```text
BCE = -[y * ln(p) + (1-y) * ln(1-p)]
soft Dice = (2 * sum(y*p) + epsilon) / (sum(y) + sum(p) + epsilon)
road loss = mean per-image BCE + mean per-image (1 - soft Dice)
epsilon = 0.000001
```

If a true road pixel has `p=0.8`, its BCE is about 0.223. If it has `p=0.2`, BCE is about 1.609. Confidently predicting the wrong answer is penalized more strongly. Soft Dice encourages overlap between predicted road probabilities and target road regions. The small epsilon avoids division problems in empty cases.

This is a training loss, not the reported thresholded test Dice. Soft Dice uses probabilities. Reported binary Dice uses true/false positive/negative counts after thresholding. Lower loss often accompanies learning, but it does not guarantee every held-out metric improves.

**Teacher: "Why use two terms?"** Answer: "Cross-entropy supervises the individual pixel probabilities, while soft Dice emphasizes overlap of the road region. Their combination trains both local predictions and region agreement. We still select and evaluate the model using declared validation metrics."

## 10. How weights actually change

Think of a weight as an adjustable setting in the network. We do not manually open a checkpoint and replace thousands of values. The training algorithm calculates how those values should change.

One learning cycle is:

1. Load a batch of training inputs and targets.
2. Run the **forward pass** to obtain probabilities or object predictions.
3. Calculate loss by comparing predictions with supplied targets.
4. Run **backpropagation** to calculate gradients of loss with respect to trainable weights.
5. The **optimizer** uses the gradients, learning rate and its own state to update weights.
6. Repeat over the training schedule; evaluate validation without applying gradient updates.

A **gradient** describes how the loss would change for a small weight change. A simple illustrative gradient-descent update is:

```text
new_weight = old_weight - learning_rate * gradient
old_weight = 0.50, learning_rate = 0.10, gradient = -0.20
new_weight = 0.50 - 0.10*(-0.20) = 0.52
```

Adam/AdamW use additional running estimates of gradients and squared gradients. The simple formula explains the direction, not the exact update used by those optimizers. AdamW also applies decoupled weight decay to relevant parameter groups. Weight decay discourages uncontrolled parameter growth; it is not data deduplication.

**Learning rate** controls update size. Too large can destabilize training; too small can make progress slow. Lower is not automatically better. We use smaller updates when adapting an already trained model, then compare results instead of assuming a setting works.

A **batch** is the small group processed together. An **epoch** usually means one pass through the training set. In our weighted detector sampler, an epoch instead makes a fixed number of draws with replacement: some images repeat and some are not sampled that epoch. It is incorrect to claim every training image was definitely seen once in each weighted epoch.

**Gradient accumulation** combines gradients from several small batches before an optimizer step. Therefore a completed batch count is not automatically the optimizer-step count. `train_detection.py` records actual optimizer steps. The final three-epoch run recorded 2,659 optimizer updates.

**Validation does not train the network.** We use inference/evaluation behavior and no optimizer updates on validation data. It still influences model selection, so it is development evidence rather than a final untouched test.

## 11. The actual real-road training history

The initial phase used the 5,507 training pairs and 687 validation pairs, 512x512 input, batch 4, Adam learning rate 0.001 and gradient clipping at norm 1. The encoder was frozen; the new decoder learned. The maximum was 20 epochs, with early stopping and learning-rate reduction. It completed 18 epochs; epoch 14 was the original best, with validation road IoU 92.0345%.

**Early stopping** ends a run after the chosen validation signal fails to improve enough for its configured patience. A **learning-rate scheduler** changes the learning rate during a run. A **checkpoint** saves a model state so it can be loaded again. The best checkpoint and last checkpoint serve different purposes.

Separate later experiments changed augmentation, trainable encoder blocks, learning rate and BatchNorm treatment. The selected morning-best trial used shadow augmentation, trainable encoder blocks 13-16 except BatchNorm, fixed decoder BatchNorm, batch 16 and learning rate **0.000003**. It initialized from the original trained checkpoint. Epoch 1 of that separate trial was selected; that does not mean the entire model learned from scratch in one epoch.

The selected model's validation road IoU was 92.1258%. Its checkpoint, preprocessing, threshold and protocol were locked before testing. On all 689 reserved real test images, it scored **91.6549% road IoU, 95.6457% Dice, 95.19% precision and 96.11% recall**. The original and boundary variants were not separately tried on that test set.

A boundary-focused experiment improved some edge measures but did not become the default real model. Later simulator adaptations also failed strict real retention checks. They were preserved as experiments rather than silently replacing the real model. Better synthetic performance alone is not enough to approve real-road replacement.

**Teacher: "How do you know weights changed?"** Answer: "The training checks verified finite gradients and actual optimizer updates. In the frozen stage, decoder weights changed while the pretrained encoder remained unchanged. We saved checkpoints and reloaded them to verify predictions and metrics. Later fine-tuning used explicitly selected trainable blocks."

Code to open: [train_idd.py](../train_idd.py), [idd_experiments.py](../idd_experiments.py), [model configuration](../models/model_config.json), [locked protocol](../models/segmentation_final_lock.json).

## 12. The detector and its actual training history

YOLO11s is a separate object-detection architecture. It estimates object locations and classes; it does not produce the road mask. The `s` denotes the small model variant. The starting checkpoint was COCO-pretrained. Pretraining provides a useful initialization; Indian-road labels are still needed to adapt it to our class contract.

The pinned detector framework reports **box loss**, **classification loss** and **distribution focal loss (DFL)**. Box-related losses train localization; classification trains categories; DFL helps represent box-coordinate distributions. This is not the road model's BCE-plus-Dice objective. The project's trainer delegates the detector network/loss machinery to the pinned framework and adds readiness, sampling, recovery and reporting logic.

| Stage | Initialization and data | Settings | Result scope |
| --- | --- | --- | --- |
| First completed detector | COCO YOLO11s; v2 data, 67,159 train entries | 640 input, batch 4, AdamW, 20 epochs | 47.26% mAP50 / 31.24% mAP50-95, full project validation |
| Resolution check | Same frozen trained weights | 960 input, no new training | 55.87% / 37.17%; this gain comes from inference resolution |
| Clean-view refinement | Verified trained detector; v3 unique training view | 960 input, 30 epochs, initial LR 0.0003, rare-image sampling | Final saved model 58.054% / 38.638% at 960 |
| Latest short adaptation | Verified epoch-30 parent; same v3 view | 1280, FP32, batch 2, workers 0, LR 0.00003, 3 epochs | Matched final 62.245% / 41.525% at 1280 |

These are stages with their own initialization and optimizer settings. Do not present them as one uninterrupted exact optimizer resume, or compare different resolutions as if the entire gain came from learning.

The first 20-epoch detector printed its completed training/final validation, then received SIGKILL and exited with code -9. The cause was not established. Independently reopened finite checkpoints and GPU inference checks supported the saved result, but that process did not exit cleanly. Later parent and three-epoch pilot completion records are separate. Preserving the failed exit is part of reporting the experiment accurately.

The final pilot uses AdamW, cosine learning-rate decay, no new warmup, mosaic off, scale augmentation 0.3, horizontal flip probability 0.5, seed 42 and a checkpoint every epoch. It was one changed diagnostic attempt after an initial 1280 attempt failed with an unknown CUDA error. FP32/batch-2/zero-workers settings reduced concurrency factors; they did not establish the original error's root cause.

The rare-image sampler boosts primary training images containing bicycle by up to 2, animal by up to 4 and traffic light by up to 2. It uses the largest applicable boost for an image, not a multiplication of all boosts. Pixel-group correction avoids duplicated identities receiving disproportionate base weight. The v3 view is already unique, but the correction is retained. Validation sampling is not weighted. Repeating 120 unique animal scenes is not equivalent to collecting 480 different scenes.

The three-epoch pilot is compared to its own parent at the same 1280 resolution, GPU FP32 and all 17,675 validation images / 319,438 objects:

| Metric | Parent | Final | Change in percentage points |
| --- | --- | --- | --- |
| mAP50 | 61.688% | 62.245% | +0.557 |
| mAP50-95 | 41.125% | 41.525% | +0.400 |
| Precision | 74.767% | 74.229% | -0.539 |
| Recall | 55.362% | 55.784% | +0.423 |

The learning gain is modest. Final animal AP50 is only 11.487%; other vehicle is 33.985%. Bicycle, autorickshaw and traffic-light AP50 decline slightly. A separate +25% learning-rate screen did not justify adopting the higher rate. A failed or inferior experiment still supplies useful evidence.

The latest selected detector is `runs/detection_train_20261025_02/weights/best.pt`. The parent best was preserved. Checkpoint binaries remain outside Git; the metadata and result summaries are published.

Code to open: [train_detection.py](../train_detection.py), [detection_refinement.py](../detection_refinement.py), [clean recipe](../detection_refinement_clean_recipe.json), [latest recorded recipe](configs/detection_pilot_1280.json), [matched result](DETECTION_RESOLUTION_PILOT_RESULT.md).

## 13. How training is started, saved and recovered

Opening a notebook, importing a Python module or displaying a recipe does not prove that training happened. A forward pass alone does not change weights. A discarded smoke step verifies an operation but is not the final trained model.

The real-road entry point is `train_idd.py`. In a prepared environment with the required dataset and encoder cache, a new frozen-stage run is explicitly started with a fresh output path:

```bash
python train_idd.py --run-dir runs/road_example_NEW --epochs 20 --batch-size 4
```

The detector entry point performs readiness checks by default. It requires an explicit start switch and fresh output directory to train:

```bash
# Readiness only; requires the local prepared data and provenance.
python train_detection.py --recipe detection_refinement_clean_recipe.json

# Example of an explicit new run; do not run just to rehearse the viva.
python train_detection.py --recipe detection_refinement_clean_recipe.json --output runs/detector_example_NEW --start-training
```

These commands show how the entry points work. A source-only clone lacks data, weights and local verification artifacts, so it is not immediately ready for training. Historical recipes also name their historical parent/provenance; changing a path does not create valid readiness evidence.

Before the detector trains, readiness checks verify the dataset manifest, image/label audit, class names, source provenance, starting checkpoint hash and framework/runtime proof. Training is executed in a separate WSL environment with GPU support. The launcher keeps an ownership record and bounded logs so a stopped terminal is not confused with a stopped worker.

**Files created by a run:**

- `recipe.json` / `run_config.json`: settings actually used.
- `status.json` / `run_status.json`: whether it is preparing, training, completed or failed.
- `results.csv` / `history.csv`: numerical metrics by epoch.
- `batch_progress.json`: recent batch/loss/optimizer-step progress.
- `best.pt` or `best.keras`: validation-selected model.
- `last.pt` / `last.keras`: latest saved state; not necessarily the best predictor.
- `last_resume_fp32.pt`: detector raw model, EMA, optimizer moments and scaler state preserved at full precision for recovery.
- Source snapshots and SHA-256 values: what code/model/data manifest the experiment used.

**EMA** is an exponential moving average of model parameters, used by the framework for evaluation. It smooths the parameter trajectory; it is not a second dataset.

**Exact optimizer continuation** needs more than model weights: it needs optimizer state, epoch/schedule, EMA and scaler where applicable. A stripped inference checkpoint may not contain these. Random sampling and augmentation may also not replay identically after a restart, so do not claim bit-for-bit continuation merely because the epoch was restored.

An earlier recovery exposed low-precision AdamW moment problems. The custom saver now preserves a separate FP32 state. An optimizer reset was used only after a documented unstable restoration and a discarded finite-loss check. Resetting the optimizer preserves learned model weights but loses its running moment history; it is different from exact continuation.

**FP32** uses 32-bit floating-point numbers. **Mixed precision / AMP** uses lower precision where suitable to save time/memory, with safeguards such as loss scaling. Lower precision is not automatically inaccurate or automatically the cause of a CUDA failure. We retain actual error records and verify changed attempts.

## 14. Metrics you should calculate and explain

For road pixels, **TP** is correctly predicted road, **FP** is background predicted as road, and **FN** is road missed by the model.

```text
IoU       = TP / (TP + FP + FN)
Dice      = 2*TP / (2*TP + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2*Precision*Recall / (Precision + Recall)
```

Example: TP=80, FP=10, FN=20. IoU=80/110=72.73%, Dice=160/190=84.21%, precision=80/90=88.89%, recall=80/100=80%. Dice and IoU differ because their denominators differ. They do not contradict each other.

The implemented road metrics sum TP/FP/FN across the evaluated split, then compute the ratio. This is global road-class overlap, not a mean of per-image IoUs or a mean across road/background classes. If the entire evaluated set has an empty union, the implemented empty/empty overlap is 1.

**Boundary IoU** measures agreement around explicitly defined boundary regions, here with declared pixel widths. It catches edge changes that a large road-area overlap can hide. A candidate that improves overall road IoU can still worsen boundary quality.

For detection, **box IoU** is intersection area divided by union area of predicted and labelled boxes. At a chosen overlap threshold, a class-correct matched prediction can count as a true positive. Duplicate predictions of one labelled object are not all true positives.

A **confidence threshold** controls which predictions are retained. Lowering it often recovers more objects while adding false detections. An **IoU matching threshold** specifies the box-overlap requirement; it is a different setting. **NMS** suppresses strongly overlapping candidate boxes during postprocessing; it is different again from dataset deduplication.

**AP** summarizes a class's precision-recall curve across confidence levels under the evaluation matching rules. It is not simply precision at confidence 0.25. **mAP50** averages the class AP values using a 0.50 box-overlap threshold. **mAP50-95** averages over stricter thresholds from 0.50 to 0.95 in steps of 0.05, then classes. Stricter localization makes it lower in this result.

Thus **62.245% mAP50 does not mean exactly 62.245% of all objects are detected**, and it does not mean 62.245% of the project is complete. The aggregate recall in the matched final evaluation is 55.784%, under that evaluator's selected operating point. Different class/object-size results matter.

**Teacher: "Why is simulator IoU almost 100% but detector mAP about 62%?"** Answer: "They measure different tasks, datasets and reference protocols. The synthetic road segmentation problem is constrained and has ignored uncertain pixels. Detection covers many object types and small/occluded objects in real scenes. Those percentages cannot be directly ranked."

## 15. How the simulation was created

CARLA supplies the 3D engine, vehicle/pedestrian actors, physics, sensors and an API for controlling them. We wrote first-party map, scenery, traffic, capture and audit code around it. We did not implement a 3D physics engine from scratch.

**OpenDRIVE** describes road geometry, lanes and junction links. It is not a neural-network format. The connected-town specification defines seven junction nodes and nine streets. Each street has endpoints, width, sidewalk/shoulder choices, optional curvature, lane arrangement and a cruise target. Junction connectors join legal lane ends.

`india_combined_map.py` builds the road network and validates its connectivity. `india_combined_scenery.py` places the district-specific scenery. `india_combined_traffic.py` creates persistent actor routes and applies commands. `india_combined_scene.py` owns the native world, viewer, sensors and episode loop.

The five districts are market street, residential colony, divided main road, rural road and irregular junction. They exist simultaneously in one world. The main loop is about 1.64 km; the non-junction streets total about 2.6 km. There are 36 directed connectors. This is an original illustrative layout, not a downloaded real-city map or a scene generated by the detector.

The main road has two lanes each way and a median; narrower streets have their own widths/footpaths/shoulders. Left-hand lane direction and joins are checked against native waypoints. OpenDRIVE and CARLA coordinate conventions need careful handling; the implementation uses native waypoint transforms when driving/capturing rather than assuming a drawing coordinate equals the runtime pose.

The scene uses installed generic cars, bikes, cycles, vans, bus and scenery meshes. Project-authored signs and shop textures provide Hindi/English details. Original autorickshaw OBJ artwork exists, but it is not a registered, rigged, cooked and physics-verified native vehicle. "We made source artwork" is the accurate claim; "the custom autorickshaw is driving in CARLA" is not.

A selected 650-second run kept one world and the same 25 vehicles, while ego travelled about 2.21 km through all five districts. Twenty-one pedestrians were present. That run had zero recorded vehicle contacts/overlaps/centre departures/direction errors under its checks, but its tightest sampled gap was 0.273 m. One finite scenario does not establish broad robustness.

**Teacher: "Is the car animated along prerecorded coordinates?"** Answer: "The native scene applies steering, throttle and braking to CARLA vehicles. Physics advances them. Waypoints are references for the controller, not pose teleports. Saved traces and continuity checks support that distinction."

## 16. The simulation loop and current vehicle controller

A typical native scene does this:

1. Create the owned CARLA world and actors, configure synchronous simulation and sensors.
2. Advance at a fixed physics time step of 0.05 seconds, equivalent to 20 steps per simulated second.
3. Read current actor positions, headings, dimensions and speeds from the simulator snapshot.
4. Project each vehicle onto its reference route and find a point ahead.
5. Apply route/cruise limits, following gaps, junction priority and pedestrian/emergency rules.
6. Convert target motion into steering, throttle and brake commands.
7. Save positions, commands, decision reasons and synchronized camera evidence.
8. Refresh the viewer, handle pause/camera/screenshot/exit controls and clean up owned actors/processes.

The driver uses a look-ahead point to steer along the route. In the combined town, turning is rate-limited/smoothed. Speed feedback compares desired and observed speed to choose bounded throttle/brake. Native CARLA physics determines the resulting movement. A nonzero throttle command is not itself proof of a particular acceleration; the trace is what measures motion.

A **junction reservation** organizes conflicting approaches. A **body-gap check** considers vehicle dimensions, not just distance between center points. The unusually wide installed bus is restricted to the divided corridor because allowing it into narrow roads caused a retained development failure. We changed the route eligibility rather than pretending its physical body was smaller.

The **rule driver** is useful as a baseline: it gives a working comparison before replacing observations or decisions with learned ones. It also uses ideal simulator knowledge, so it must be labelled ground-truth-driven. Its performance is not evidence that the camera detector can supply equally reliable observations.

The viewer already supports camera changes, pause, screenshot and exit. Those are interface controls. Manual WASD driving, a manual/automatic takeover mode and destination selection are separate features if implemented later.

Launcher chain: `Start_Combined_Indian_Traffic.cmd` -> `Start_Combined_Indian_Traffic.ps1` -> `setup/carla/run_session.py` -> viewer/combined scene -> map/scenery/traffic -> controller -> CARLA commands. The session runner uses Windows Job Objects so cleanup targets its own process tree rather than unrelated programs.

## 17. How overtaking works, and what still remains

The earlier small-neighbourhood `india_behavior.py` has an explicit passing state machine. A **state machine** chooses behavior according to the current state and conditions, rather than making unrelated decisions every frame.

Conceptually the sequence is following -> checking whether passing is allowed -> passing -> waiting for a return gap if necessary -> returning left -> completed. It finds the slower leader, estimates the time/clearance needed, checks approaching traffic and pedestrians, and checks lateral body separation. When the vehicle is ready to merge back, it checks the current return gap; starting a pass does not guarantee that the future return gap will remain clear.

If a condition blocks the pass, the event log records the reason. A clear road alone is insufficient when the car would not have enough space to merge back. Passing should also be restricted around junctions or unsuitable road sections. The controller's current logic is hand-coded, not learned from detector boxes.

Selected tests cover a clear pass, oncoming blockage, pedestrian interaction, return-gap blockage and junction conflicts. A passing demo is successful only if the trace contains the intended behavior, not merely because the simulator window opened. The blocked-oncoming case deliberately does not claim route completion during its finite interval.

The newer combined town has its own loop-based traffic implementation. The earlier passing proof does not automatically prove destination navigation or refined overtaking in that town. That is a remaining integration/evaluation item.

Code to open: [india_behavior.py](../integration/india_behavior.py), especially `_overtake` and `Driver.control`; [combined traffic](../integration/india_combined_traffic.py); [scenario report](results/carla/SCENARIO_REPORT.md).

## 18. How simulation labels and the separate road model were built

To train a synthetic road model, we collected matched RGB, semantic, instance and depth streams with camera pose/calibration and OpenDRIVE geometry. **RGB** is the visible image. **Semantic IDs** identify categories. **Instance IDs** distinguish particular objects. **Depth** supplies distance along camera rays under CARLA's convention. They must share frame/time and camera setup before labels can be aligned reliably.

Some custom asphalt meshes lack the native Road semantic tag. Treating those pixels as background would create wrong labels. The reference code uses the exact owned surface instances, synchronized depth and the driving-lane footprint to resolve suitable visible road pixels. Known people, vehicles, buildings and footpaths remain non-road. Occluded road is not labelled visible road.

Uncertain regions are ignored: examples include a conservative 0.2 m boundary neighborhood and geometry beyond 80 m. A valid-pixel mask records where the target can be trusted. The saved reference uses 0=non-road, 1=road and 255=ignore. The binary training mask uses 0/1; its invalid zeros are placeholders, not negative labels.

During dataset preparation, RGB is resized bilinearly to 512x512. Road/validity labels use identical nearest-neighbor coordinates so categories are not blended. At training time RGB receives the same single MobileNetV2 normalization. Target/validity stay binary. Both BCE and Dice explicitly exclude invalid pixels, including their denominators. Validity is target metadata and is never supplied as a visual feature to the network.

Thirty-two 80-second episodes cover four generated topologies with traffic/weather/camera variation. Exact repeated RGB pixels and very low-change adjacent candidate views are excluded before selecting 5,000 pairs. Low change is measured on a 48x27 RGB thumbnail; a mean absolute difference below 0.75 on the 0..255 scale rejects that candidate. The original candidate evidence is retained.

Whole episodes are assigned to one split: 4,000 train, 500 validation, 500 test. This reduces temporal leakage. Related layouts/assets still occur, so this is not city-independent or real-world evaluation. Held-out scene families are not exhaustively covered by four test episodes.

The separate simulator U-Net starts with a fresh decoder and ImageNet encoder, not the real-road checkpoint. It uses frozen-encoder Adam training at 0.001, batch 4 and masked BCE/Dice. Training ended after 11 epochs; epoch 10 was selected. Fixed-model test IoU was 99.9288%, Dice 99.9644% and valid coverage 98.2174% on 500 synthetic images. A later 480-frame fresh-scene confirmation scored 99.7478% road IoU. These are different evaluations, both synthetic.

Code to open: [simulation_data.py](../simulation_data.py), [simulation_model.py](../simulation_model.py), [train_simulation.py](../train_simulation.py), [reference builder](../carla_road_reference.py), [dataset builder](../setup/build_simulation_dataset.py).

## 19. Live camera predictions, latency and stale results

The live road preview runs the simulation road model on the native front camera at 5 Hz. A separate GPU service returns a result associated with the exact source frame/session and pixel hash. The overlay is drawn on that source image, not on a newer unrelated camera frame.

One pending frame replaces an older pending frame instead of growing an unlimited queue. A bounded latest-frame design avoids showing progressively older predictions when the model cannot keep up. It does not guarantee that the system always meets a real-time deadline.

**Latency** is delay. **Frame age** is how long ago the source frame originated when a result is received or displayed. GPU inference is only one part: acquisition, transport, resizing, inference, postprocessing and display all contribute. Timestamps from unrelated Windows/Linux clocks cannot be subtracted directly; this implementation uses a common Windows monotonic time base for the cross-process ages.

The measured normal observation run received 100/100 results with zero reconnects. Median GPU inference was 17.31 ms; median source-step-to-result age upper bound was 60.36 ms and first presentation age upper bound was 120.11 ms. These numbers describe the road observer in its measured scene. They are not a combined detector-plus-segmenter driving-loop benchmark.

Stale/offline results are hidden at completed redraws. The declared age bound is 500 ms in wall and simulation time, with a presentation reserve. A pause does not freeze wall-clock age. Controlled tests injected delay/disconnection/pause and checked handling. Two displayed intervals exceeded the target until the next redraw, by up to 14.87 ms. Therefore per-redraw checks passed, but strict continuous on-screen deadline compliance did not.

The matched rule-driver comparison recorded identical actor/control traces with and without the observer in that finite test. That supports the observation-only separation for the test; it is not universal simulator determinism.

Code to open: [simulation_live_protocol.py](../simulation_live_protocol.py), [india_live_preview.py](../integration/india_live_preview.py), [worker](../setup/live_simulation_worker.py), [live model scope](../idd_live_simulation_inference.py).

## 20. Why there are separate real and simulation models

**Domain shift** means the training/input distribution changes. Synthetic lighting, materials, camera response, label definitions and geometry can differ from real road photographs. A model that excels on one can perform worse on the other.

Earlier adaptation experiments improved synthetic development but failed strict real-validation road/boundary and reviewed-video gates. We preserved the original locked real model and used explicit simulator-only model choices. No appearance/color heuristic guesses which model should run. Source/protocol metadata selects the allowed scope.

The newer simulator-only model's high synthetic score does not replace the real model's 689-image test result. The detector is another distinct task and model. A manifest records checkpoint hashes and scope; it is not a magic conversion between tasks.

**Teacher: "Why not use one highest-scoring model everywhere?"** Answer: "The scores come from different distributions and tasks. A synthetic-road gain did not preserve the real-road results, so we kept real inference locked and isolated simulator experiments. A model needs validation for the input domain and task where it will be used."

## 21. Planning references, destination navigation and later RL

The repository also contains older MATLAB planning/controller references. One static Hybrid A* example plans a vehicle-feasible route on a declared metric map. Its tracking reference uses Stanley steering, bounded PI speed feedback/feedforward and a kinematic bicycle model. These are separate reference experiments, not the current connected CARLA town's live destination planner.

**A*** searches a graph/grid using travel cost plus an estimate of remaining cost. **Hybrid A*** includes vehicle pose/heading and motion constraints, making it more appropriate for a car than a point moving in four grid directions. An **occupancy map** marks free/occupied space in metric coordinates. An uncalibrated image road mask cannot simply be treated as such a map.

For destination selection, the combined town already provides nodes, roads and lane connectors. Remaining work includes selecting legal goal locations, routing from the current lane, generating a continuous drivable reference, following it and stopping at the goal. A preset loop is not user-selected destination navigation. Drawing a goal marker is not evidence the car can reach it.

**Reinforcement learning** learns a policy through interactions and rewards. The agreed future scope is deliberately bounded: STOP / SLOW / PROCEED, selecting target speed above route control. Route planning, collision checks, an independent emergency override and the low-level controller remain separate.

No RL policy is currently trained. Do not invent a DQN architecture, reward formula, training-episode count or reward curve. Future design must state observations, actions, rewards, termination rules and what uses simulator ground truth versus camera estimates. Compare it with the existing rule baseline on the same held-out scenarios, including failures and comfort/progress, rather than showing only a successful run.

## 22. Finding your way around the files

Start with these groups; use the [complete file guide](FILE_GUIDE.md) for individual helpers.

| Group | Read first | What it contributes |
| --- | --- | --- |
| Road data | `idd_data.py`, `check_preprocessing.py` | Correct image/mask batches and verified normalization |
| Road network/training | `idd_model.py`, `train_idd.py`, `idd_experiments.py` | Architecture, loss, metrics and controlled training |
| Real inference | `idd_inference.py`, `predict_road.py`, `predict_road_video.py` | Fixed-model image/video prediction without training |
| Detector data | `idd_detection_data.py`, `detection_policy.json`, `setup/prepare_clean_detection_view.py` | Box contracts, conversion, class mapping and cleaning |
| Detector training | `train_detection.py`, `detection_refinement.py`, recipes | Readiness, framework training, weighted sampling and FP32 recovery |
| Synthetic learning | `simulation_data.py`, `simulation_model.py`, `train_simulation.py` | Validity-aware labels, masked objective and separate training |
| Connected town | `integration/india_combined_*.py` | Map, scene, scenery and persistent traffic |
| Earlier driving | `integration/india_behavior.py`, `integration/carla_india_scene.py` | Rule driving, passing cases and native scene loop |
| Live observation | `simulation_live_protocol.py`, `integration/india_live_preview.py`, worker | Bounded transport, freshness and matching-frame display |
| Evaluation | `evaluate_*.py`, `setup/verify_*.py`, `docs/results/` | Scores, independent checks and curated saved evidence |
| Launch/setup | `Start_*.cmd`, `Start_*.ps1`, `setup/carla/`, requirements | Environment, session ownership and reproducible startup |
| MATLAB | `integration/matlab/` | Earlier static planning/tracking and replay references |

**Python `.py`:** executable source or reusable module. **Notebook `.ipynb`:** explanatory cells; source-clean versions have no saved execution output. **JSON:** structured configuration or result data. **YAML:** framework settings/dataset definition. **CSV:** rows such as per-epoch metrics or split membership. **PNG:** image or small texture. **`.keras`/`.pt`:** binary model state, supplied separately. **`.xodr`:** road network. **`.obj`/`.mtl`:** source mesh/material artwork. **`.ps1`/`.cmd`/`.sh`:** launch/setup scripts for different shells.

`dataset/`, `runs/`, `tools/`, environments, source footage and checkpoints are intentionally excluded from Git. They are not useless: they are large local data, software, results or recovery state. `archive/` preserves unrelated or historical material. GitHub is the source/review handoff, not a copy of the entire laptop.

If asked about a helper filename, state its input, operation, output and whether it changes data/weights/controls. For example: "`setup/prepare_clean_detection_view.py` reads the already audited v2 pairs, groups training pixel hashes, excludes ambiguous/repeated entries, writes a fresh v3 view and provenance, and leaves validation and raw sources unchanged. It does not train."

## 23. How to show evidence without overclaiming

The first sentence for any number should identify task, split, model and conditions. Say "62.245% detector mAP50 on 17,675 project validation images at 1280 FP32," not simply "our accuracy is 62%."

A screenshot proves what was visible in that capture; it does not prove all cases work. A source function proves a behavior is implemented, but saved traces and independent audits are needed to support a claim it happened successfully. A healthy log or forward pass is not proof of optimizer learning. A runtime that exited after saving complete metrics still needs its failed exit disclosed.

The four selected detector-box examples were generated from a frozen epoch-1 checkpoint of the pilot, not the final epoch-3 model. They illustrate duplicate boxes, small misses and class confusion; they are not an aggregate score or a random independent test sample.

Current weak points include animal detection, small/occluded people, broad other-vehicle labels, source noise, correlated synthetic scenes, tight traffic clearance, transient jerk, GPU/runtime interruptions and the live redraw timing tail. Naming a measured limitation helps explain the next experiment; it does not invalidate every working component.

**A useful experiment report has:** hypothesis, unchanged baseline, one deliberate change, matched data/settings, full versus sample scope, class/condition breakdown, saved failures and selection decision. Raising resolution can improve results without any weight change; a modest learning-rate increase can fail; a higher average can hide a class regression.

## 24. Viva questions: short answers to practise aloud

### Project and architecture

**1. What is your project in one sentence?** We are developing Indian-road perception alongside a traffic simulator, with destination driving and perception-based decisions as the next integration stage.

**2. What is complete at this milestone?** Data pipelines, evaluated road models, a trained 12-class detector, native traffic simulation and an observation-only road preview.

**3. What remains?** Better weak-class detection, live tracking/range, perception-to-driving decisions, selected destinations, combined-town overtaking refinement and broader tests; RL is later.

**4. Is this one neural network?** No. Road segmentation and object detection use different models and labels, and vehicle control is currently rule-based.

**5. Why two road models?** Synthetic and real distributions differ. We kept the real deployment locked while training/evaluating a separate synthetic model.

**6. What does the detector output?** Bounding boxes, class IDs and confidence values, not road masks, metre distances, persistent tracks or steering commands.

### Data and preprocessing

**7. How did you pair photographs and masks?** Numeric pair IDs and a manifest, with missing/repeated ID checks; not arbitrary folder order.

**8. How many road examples are retained?** 5,507 train, 687 validation and 689 test, totaling 6,883 pairs.

**9. Did you remove 110 duplicate photographs?** No. Their masks were all zero. The audit found no exact duplicate road input images. Those 110 pairs were excluded under the recorded filtering decision.

**10. Can a duplicate mask be valid?** Yes. Different images can share the same target pattern; image duplication and target duplication are different concepts.

**11. What does SHA-256 do?** It fingerprints content so exact changes/matches can be checked. It is not a correctness score or encryption of the dataset.

**12. Why hash both JPEG bytes and RGB pixels?** Re-encoded files can differ in bytes while decoding to the same pixels. Pixel hashing catches that exact decoded match.

**13. How did the detector training set become 56,729 images?** From 67,159 v2 entries, exclude 5,045 ambiguous entries and 5,385 extra consistent duplicates; preserve validation.

**14. What if identical pixels have different labels?** The conservative clean-view rule excludes the whole ambiguous training group and records the reason, rather than silently inventing a merged answer.

**15. Did you delete the original dataset?** No. Prepared views and exclusion records preserve provenance; raw sources remain local.

**16. Are all near-duplicates gone?** That is not established. Exact byte/pixel duplicates were checked; similar adjacent frames are a broader issue.

**17. Why preserve provider splits?** It avoids moving examples opportunistically and allows recording/identity exclusions to be audited while retaining source membership.

**18. Why not use the official detector test as negatives?** Missing labels do not mean no objects exist. Its 24,644 identities are metadata-only for this workflow.

**19. Why float32 inputs?** Neural-network computations use floating point; conversion also permits normalization while retaining mask labels correctly.

**20. What normalization is used?** Road RGB uses `x/127.5-1` once. Detector preprocessing is the separate framework's contract; the same formula is not shared indiscriminately.

**21. Why nearest-neighbor labels?** Interpolating categorical labels with bilinear averaging can create invalid fractional classes or blend ignore boundaries.

**22. What is a hard link?** Another filesystem name for the same content, used to avoid redundant prepared image copies; it is not a new data sample.

### Training and models

**23. What is transfer learning?** Reuse pretrained visual features, then adapt suitable parts to our task using labelled data.

**24. What is frozen?** A layer whose parameters do not receive optimizer updates; it still computes features.

**25. Why MobileNetV2?** It supplies relatively compact pretrained features suitable for a resource-limited encoder. We do not claim it was proven globally optimal.

**26. Why U-Net?** Its upsampling decoder and skip connections restore spatial detail needed for per-pixel road output.

**27. Why sigmoid?** There is one binary road/non-road probability per pixel, so a one-channel 0..1 output is appropriate.

**28. What is the road loss?** Mean per-image BCE plus mean per-image one-minus-soft-Dice, with equal weights.

**29. How did you change weights?** Forward prediction, loss, backpropagation and optimizer updates; we did not manually edit checkpoint numbers.

**30. What is a learning rate?** The setting controlling the scale of weight updates. Its effect is measured with validation rather than assumed.

**31. What is an epoch?** Usually a dataset pass; our weighted detector epochs are a fixed number of sampled draws with replacement.

**32. Is every batch an optimizer update?** Not necessarily. Gradient accumulation can combine several batches; we record actual updates separately.

**33. Does validation update weights?** No. It measures the current model and guides selection without training on those images.

**34. How was the best model chosen?** By the declared validation criterion, then saved and reloaded; the final real test was evaluated only after its protocol was fixed.

**35. Is the latest road model trained from random weights?** Its encoder starts with ImageNet pretraining; the initial decoder is new, and selected fine-tuning begins from an already trained model.

**36. Why freeze BatchNorm in fine-tuning?** To avoid unintentionally changing running statistics while adjusting selected visual filters, especially with small batches.

**37. Why YOLO11s?** It is the chosen compact pretrained detector candidate for this hardware. We measured resolution/training behavior rather than claiming every alternative was benchmarked.

**38. Why higher resolution?** It makes small objects occupy more pixels, which improved the measured average/person results, but it increases computation and can regress classes.

**39. Did the whole 960-to-1280 gain come from training?** No. A frozen-weight resolution comparison isolates inference gains; the three-epoch pilot's separate matched gain was only 0.557 mAP50 points.

**40. Why rare-class sampling?** To expose training more often to underrepresented image categories while keeping validation unchanged. Repetition does not create new unique scenes.

**41. What are best and last checkpoints?** Best is selected using validation; last is the latest saved state. They need not be identical or equally useful for recovery.

**42. What must a resume checkpoint include?** Model/EMA, optimizer state, completed epoch/schedule and scaler where relevant. We also verify hashes and finite tensors.

**43. Why FP32 recovery?** Earlier saved low-precision optimizer moments caused unstable restoration, so a separate full-precision state preserves more reliable continuation information.

**44. Why not keep increasing the learning rate?** A paired +25% screen did not show a convincing win. Faster/larger updates do not guarantee better generalization.

### Evaluation

**45. What is IoU?** Correctly overlapping road area divided by the union of predicted and true road areas.

**46. Why not only pixel accuracy?** Large background regions can make accuracy look good while road overlap is poor.

**47. What is recall?** The fraction of labelled positives recovered at the evaluated operating point; low person recall means missed people remain important.

**48. What is mAP50?** Mean class average precision using 0.50 box IoU matching, summarized over confidence thresholds.

**49. Why is mAP50-95 lower?** It also requires tighter localization at stricter overlap thresholds.

**50. What is your latest detector result?** 62.245% mAP50 and 41.525% mAP50-95 on 17,675 project validation images, 1280 input, matched GPU FP32 evaluation.

**51. Is that an official IDD benchmark score?** No. We use a documented project class mapping and filtered validation membership.

**52. What is your real-road test result?** 91.6549% road IoU and 95.6457% Dice on all 689 fixed-protocol test images.

**53. Does 99.93% synthetic IoU prove real-road quality?** No. It describes a constrained synthetic dataset, validity policy and fixed model, not transfer to real driving.

**54. What regressed in the latest pilot?** Bicycle, autorickshaw and traffic-light AP50 declined slightly; the average improvement was not universal.

**55. Are the four box examples a benchmark?** No. They are deliberately selected illustrations from an earlier frozen epoch-1 pilot checkpoint.

### Simulation and integration

**56. What is CARLA doing?** Providing the world, actors, physics and sensors. Our source constructs maps, traffic rules, scenery, capture and checking around it.

**57. How is the map made?** Procedural OpenDRIVE road/lane geometry and junction connectors, with scenery placed separately.

**58. Is this a real Indian town?** It is an original Indian-style illustrative layout with generic installed assets and project-authored signs.

**59. What moves the car?** A rule driver sends throttle/brake/steering to native CARLA physics; it uses route waypoints and simulator actor states.

**60. Can I select a destination today?** Not in the completed combined-town interface. The current routes are preset; goal selection/routing is planned.

**61. How does passing work?** The earlier rule state machine checks a slower leader, oncoming/pedestrian constraints, lateral clearance and room to return before completing a pass.

**62. Has combined-town overtaking been proved by the old tests?** No. The town has different route/control code and needs its own integration and scenario checks.

**63. Where do synthetic labels come from?** Synchronized native sensor IDs/depth and map geometry, not predictions from the model being trained.

**64. Why ignore some pixels?** Unknown edges or geometry should not become false target labels. Both loss and metrics exclude invalid positions.

**65. What is valid coverage?** The fraction of total pixels included in the declared evaluation; synthetic test coverage was 98.2174%.

**66. What is a stale prediction?** A result whose source frame is too old to be used/displayed under the age policy, even if the network computed it correctly.

**67. Why not report only model milliseconds?** A car receives delayed sensor results after capture, transport and processing; complete age matters as well as inference cost.

**68. What is the role of the old MATLAB files?** Separate static/dynamic planning and replay references; they are not secretly driving the current connected CARLA scene.

**69. Have you trained reinforcement learning?** No. The future policy is bounded STOP/SLOW/PROCEED and will be compared with the rule baseline.

**70. What does zero collisions prove?** That the recorded finite checks found no collisions under their conditions. It does not prove all seeds, conditions or real roads are safe.

### Repository and teamwork

**71. Why are datasets/weights missing from GitHub?** They are large and have separate distribution/access requirements. Source, metadata and curated evidence are published; trusted weights/data are supplied separately.

**72. What does a requirements file do?** Records Python packages/version constraints so a compatible environment can be prepared. It does not install the CARLA engine or supply trained weights by itself.

**73. Why clean notebook outputs?** To avoid large/personal/stale embedded output in Git. Actual experiment results remain in their reports and local run evidence.

**74. What did each member do?** Explain the actual activities each person completed. The ownership plan allocates data/perception to Atharv and simulation/planning/control to Sarthak for review, validation and remaining work; it is not fabricated historical authorship.

**75. Why so many verify/report scripts?** The main code performs operations; independent checks and reports make results inspectable and preserve failures. They are supporting tools, not 200 separate models.

## 25. A two-day revision route

First, both understand chapters 1-4 and the completed/planned distinction. Atharv then concentrates on chapters 5-14; Sarthak concentrates on chapters 15-21. Both read chapter 22 and the short questions. Exchange explanations and open the source for any answer that feels like a memorized phrase.

Practise five calculations: RGB normalization, YOLO box normalization, IoU/Dice/precision/recall, one illustrative weight update and percentage-point improvement. Practise three traces: an image into a model, a model training batch into a checkpoint, and a simulator snapshot into a vehicle command.

For a teacher's unfamiliar question, identify the module and its input/output first. If something has not been measured or implemented, say so and connect it to the remaining plan. Accurate understanding of a bounded system is more defensible than inventing a feature or a number.

## Sources inside the project

This guide is grounded in the source/configurations linked above and these result records: [real segmentation](results/segmentation.json), [detector data](results/idd_detection_import.json), [latest matched detector](results/detection_resolution_pilot.json), [simulation dataset](results/simulation_dataset.json), [simulation test](results/simulation_test.json), [fresh confirmation](results/simulation_confirmation.json), [town](results/combined_indian_town.json) and [live observation](results/live_simulation_observation.json). Historical full galleries and checkpoint binaries remain local as explained in [the documentation index](INDEX.md).
