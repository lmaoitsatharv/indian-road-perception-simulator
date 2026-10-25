# IDD detection import and preprocessing — 23 October 2026

This report records the 23 October v2 import/preparation stage. Training and later v3 cleaning subsequently completed; see [training](DETECTION_TRAINING.md), [clean-view analysis](DETECTION_IMPROVEMENT.md) and [current results](RESULTS.md).

**Preprocessing and readiness checks are complete.** The combined archive,
extraction, native-image audit, independent saved-label audit and final file
checks passed. The preparation stage performed zero epochs and zero optimizer steps; subsequent training is reported separately. Both road-segmentation models remain separate.

## Stored data

- Original five downloaded parts: `dataset/idd117k_raw/`.
- Unmodified extracted source: `dataset/idd117k_extracted/IDD117K_Detection/`.
- First image/annotation audit, including rejected pairs: `dataset/idd117k_detection_v1/`.
- Final quality-filtered training view: `dataset/idd117k_detection_v2/`.
- Import evidence: `runs/idd_detection_import_20261023_01/`.

The first two locations preserve the provider data. Prepared images use hard
links within the same drive; labels and audit records are separate files. An
initial Windows sandbox restriction caused copy fallback; a bounded recovery
verifies both source and copy SHA-256 before replacing only import-created
copies with equivalent hard links. It does not delete or modify raw images.
That recovery completed: 91,601 audited images use links, reclaiming
46,915,324,190 duplicate bytes. All 84,834 final training/validation images are
hard links, with no extra full-image copies in the final view.

## Archive and provider membership

All five part hashes were independently checked while reading the combined
gzip stream. Gzip CRC/length checks and the complete tar inventory passed.
Extraction rechecked those hashes and exactly matched the inventory:
258,980 regular files, 73,262,082,977 unpacked bytes, 141,814 JPEGs,
75,305 JSON annotations, 41,857 XML annotations and four text files.

The source readme and actual provider lists agree on 96,897 training images,
20,202 validation images and 24,644 unlabelled test images. The older release
also contains unlisted extras; their exact identities are recorded in the source
index and they are not assigned invented split membership or negative labels.
Unlabelled test images are extracted for storage/integrity but never decoded
for preprocessing, training, previews or local accuracy evaluation.

## Recording exclusions

The 95K release's numeric recording groups are disjoint across its provider
splits. The older release has 76 overlapping recording groups when simultaneous
camera views, dated chunks and reviewed date-only aliases are grouped together.
An unresolved mixed upload folder is kept as one conservative group and excluded
from labelled preparation.

The original manifest and split lists remain unchanged. A filtered manifest
excludes labelled records sharing a recording with reserved test; training
records sharing a recording with validation; and unresolved mixed uploads.
No retained row moves to a different provider split.

| View | Train | Validation | Test metadata |
| --- | ---: | ---: | ---: |
| Original provider lists | 96,897 | 20,202 | 24,644 |
| After recording exclusions, before image/box checks | 73,968 | 17,689 | 24,644 |
| Final prepared dataset | 67,159 | 17,675 | 24,644 |

There are 25,442 recording/identity exclusions: 12,622 training records sharing
validation recordings, 12,158 labelled records sharing test recordings and 662
records in the unresolved upload batch. Every excluded identity and reason is
saved in `source_manifest_curated_exclusions.jsonl`.

The complete native-image audit then decoded all 91,657 labelled candidates.
It excluded 56 invalid whole pairs (55 degenerate/reversed boxes and one
foreground box wholly outside its image), plus 6,767 training entries whose
encoded bytes or decoded pixels matched validation. These quality exclusions
are recorded separately in `source_manifest_quality_exclusions.jsonl`. No exact
pixel duplicate remains across the final labelled splits.

## Native images and labels

The 12 project classes are person, rider, car, motorcycle, bicycle,
autorickshaw, bus, truck, animal, traffic light, traffic sign and other vehicle.
Vehicle-fallback/caravan/trailer/train annotations map to other vehicle.
Declared non-target labels such as ego vehicle and pole remain background.
This class mapping and filtered validation set are a project protocol, not the
official IDD detection benchmark protocol.

Source JSON and XML coordinates are retained literally as zero-origin box
edges. Actual image dimensions are read; XML dimensions must match. This avoids
assuming that VOC-like XML uses one-based coordinates or that every frame is
1920×1080. Boundary clipping and duplicate-box removal are recorded. Native
image data is not stretched or recompressed. Training uses aspect-preserving
letterboxing to 640 pixels at runtime.

Every selected labelled image is decoded, checked for RGB/orientation and
hashed both as original bytes and decoded pixels. Missing labels never become
negative examples. Invalid image/annotation pairs are quarantined whole; exact
training duplicates of validation are excluded. Valid annotated empty images
remain negatives. Unknown source classes or unexpected failures require review.

The final view reuses the completed native-image audit, revalidates annotation
hashes and saved labels, and records every quality exclusion. An independent
pass reopens every final label and compares source-versus-saved boxes on a
deterministic image sample. Original annotations may contain human noise;
these checks do not certify every class label or every possible near-duplicate.

The final view retains 425 clipped boxes and records removal of 45 duplicate
boxes. It preserves 151 training and 53 validation images with valid annotations
but no target objects. All 12 target classes occur in both labelled splits.
Native sizes span 1280×600, 1280×720, 1280×964, 1920×1080 and 2048×1024.

Within training, 7,914 repeated-pixel entries remain across provider source
identities: 59,245 distinct pixel images in 67,159 entries. Validation has 17,675
distinct pixel images. There are 2,516 within-training pixel groups with multiple
saved label serializations, which can reflect annotation differences or ordering.
These within-split repetitions are retained and recorded; no claim is made that
every image entry is unique or that the source annotations are noise-free.

## Independent verification

The independent pass reopened **all 84,834 saved label files**, verified their
hashes, normalized coordinates, object counts, class IDs and negative-image
counts. It separately reopened 166 source image/annotation pairs, rehashed and
decoded their images and compared original boxes with saved-label coordinates.
Maximum sampled coordinate error was 0.0000009601 pixels, below the 0.001-pixel
check threshold. This is conversion verification, not model accuracy.

Twelve source-versus-saved annotation previews are retained in
the comparison gallery: `runs/idd_detection_import_20261023_01/actual_data_audit/index.html` (local artifact, supplied separately)..
Two previews, covering both source releases, were also visually inspected;
converted boxes match the source positions. Source class-label noise remains a
limitation, including an autorickshaw labelled as car in one source sample.

All 42 converter/archive/index/curation/quality check groups passed. The actual
final YAML, class order and image-folder counts passed separately in WSL with
PyTorch 2.9.1+cu128 and Ultralytics 8.3.225; 87 selected file sizes/timestamps
also matched across Windows and WSL. CUDA was available on the RTX 4060 Laptop
GPU. No actual training batch or backward-memory check was performed.

The exhaustive Windows readiness pass rechecked every final prepared image's
size/timestamp and every label's SHA-256, together with manifest, YAML, policy,
archive and source-index provenance. All readiness gates passed. The final
17 protected fingerprints match the pre-import values, covering road checkpoints,
real inference, original split metadata and the rule driver. Eighteen final
source/configuration files are snapshotted beside the evidence.

Machine-readable results: [idd_detection_import.json](results/idd_detection_import.json).
Full evidence: `runs/idd_detection_import_20261023_01/results.json`,
`actual_data_audit/verification.json`, `training_readiness_windows.json`,
`framework_check/verification.json` and `protected_after.json` within that run.
The prepared dataset also contains its own `README.md`.

## Training follows preprocessing

Preprocessing itself performs no optimizer updates. The separate 20-epoch YOLO11s run later completed and was followed by clean-view refinement. The default training entry point still checks readiness; a fresh experiment requires an explicit start action and new output directory. Road model checkpoints, inference defaults and the rule driver
are protected independently by hashes.

Source catalogue: [IIIT Hyderabad IDD datasets](https://insaan.iiit.ac.in/datasets/).
Source annotation definitions: [AutoNUE public code](https://github.com/AutoNUE/public-code).
