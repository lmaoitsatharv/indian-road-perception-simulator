# Object-detector runtime and data preparation

This records preparation completed on 23 October 2026. It established runtime, conversion and readiness contracts before IDD fine-tuning; it did not measure a trained detector at that stage. Training subsequently completed, including the clean-view refinement and three-epoch 1280 pilot. See [current results](RESULTS.md), [training history](DETECTION_TRAINING.md) and [matched final result](DETECTION_RESOLUTION_PILOT_RESULT.md).

## Runtime and initialization

The isolated `idd-detect` WSL environment used PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128 and Ultralytics 8.3.225. Dependency checking found no conflicts. This environment is separate from TensorFlow's `idd-tf220`; installation did not modify the road-model runtime. See [getting started](GETTING_STARTED.md) for setup and required local artifacts.

Initial YOLO11s weights were official COCO pretraining, SHA-256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`. They were not an IDD-trained accuracy result or an approved driving model. Origin, versions and scope are retained in local provenance metadata.

## Source inspection and conversion

A rehashed 64 MiB archive prefix exposed the newer JSON layout under `IDD117K_Detection/IDD_95kDetection/val/`. Twelve annotations contained 655 objects. They were read to establish schema and class names; no image pixels were decoded for that check. Prefix inspection did not establish complete archive integrity. Full streaming archive/import checks followed separately.

Newer JSON stores lists of named objects with `xmin/ymin/xmax/ymax`; older VOC-like XML is parsed separately. Actual image dimensions and explicit coordinate contracts are required. Complete source review selected literal zero-origin `xyxy_edges` for both actual sources rather than assuming one-based VOC geometry.

`idd_detection_data.py` converts to normalized detector boxes, records clipping/duplicate boxes, retains valid annotated negatives and rejects missing or unsupported labels. Source indexing preserves provider split and recording identities. Official unlabelled test identities remain metadata and never become negative training examples.

## Class contract

The fixed twelve categories are person, rider, car, motorcycle, bicycle, autorickshaw, bus, truck, animal, traffic light, traffic sign and other vehicle. Vehicle fallback/caravan/trailer/train map to other vehicle. Explicit non-target labels remain background. Unknown classes or unsupported difficult/ignore/crowd flags reject the image and block readiness rather than silently dropping positives. The class order and policy are in `detection_policy.json`.

This is a project mapping, not the complete official IDD benchmark taxonomy. Complete import verified all twelve classes in retained labelled splits. The initial v2 view had 67,159 training / 17,675 validation entries; subsequent training-only v3 cleaning retained 56,729 unique training images with validation unchanged. See [import](IDD_DETECTION_IMPORT.md) and [clean-view analysis](DETECTION_IMPROVEMENT.md).

## Completed preparation checks

Checks covered non-square coordinate round trips, VOC origin conversion, clipping, overlapping rider/bike boxes, genuine negatives, missing/invalid/unknown annotations, recording conflicts, byte-identical and re-encoded pixel-identical leakage, indexing and refusal to train from incomplete/fixture data.

The pinned framework loaded the synthetic fixture's twelve boxes and separate negative image, accepted YAML/class order, and produced 640x640 letterboxed tensors. Its twelve-class head loaded compatible pretrained features and completed a finite-output GPU forward pass.

The reference RTX 4060 Laptop FP32 COCO forward probe used batch 4 at 640x640: median 24.52 ms per batch, maximum 30.94 ms and peak allocated memory 317.3 MiB. This was a synthetic-input forward-only check, not real-image accuracy, camera latency or backward-memory proof. No optimizer update occurred in preparation. Initial and corrected runtime-settings checks remain separate preserved records.

## Guarded training entry point

Default execution checks readiness; full training requires an explicit switch and fresh output:

```bash
python train_detection.py
python train_detection.py --start-training --output runs/detection_train_NEW
```

Run these in the appropriate prepared environment. A source-only checkout lacks datasets, trusted weights and local verification artifacts, so the gates will not pass merely because source exists. They check combined provider counts of 96,897 train / 20,202 validation / 24,644 test metadata and the recipe's expected prepared counts. A tiny fixture or 95K-only index is insufficient.

The historical first recipe used COCO YOLO11s, 640 input, batch 4, seed 42, AdamW 0.0001, maximum 20 epochs, patience 5 and AMP. These became the first completed run's settings; later refinement/pilot settings are separate. Forward success is not a substitute for measured real backward/optimizer execution.

## Scope and evidence

The detector predicts image rectangles/classes, not persistent tracks, metre distances or vehicle commands. Both road models and the rule driver retain separate scopes. Portable preparation evidence is [detection_preparation.json](results/detection_preparation.json). Full fixtures, coordinate illustration, package records and runtime proof are local under `runs/detection_readiness_20261023_01/`, outside the public checkout.

References: [IDD catalogue](https://insaan.iiit.ac.in/datasets/), [annotation code](https://github.com/AutoNUE/public-code), [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) and [PyTorch builds](https://pytorch.org/get-started/previous-versions/). Dependencies and datasets retain their respective licensing/access terms.
