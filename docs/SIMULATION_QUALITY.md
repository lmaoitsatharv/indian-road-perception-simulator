# Asphalt and simulation quality pass — 20 October 2026

This page retains the completed V4.1 measurements. The later V4.2 footpath,
reference-audit, recording and separate model work is documented in
[ROAD_BOUNDARY_FIX.md](ROAD_BOUNDARY_FIX.md). The seven-case results below should
not be read as a new seven-case run of V4.2.

## Visible changes

The default Indian neighbourhood has a dark, grainy asphalt surface, lighter
pavements and worn white centre/outer-edge markings. The original OpenDRIVE
carriageway and its vehicle collision surface remain in place. Purely decorative
surface pieces have collisions disabled and sit at z=0.015 m, below the raised
footpaths. Vehicles and pedestrians keep their collisions enabled.

Native brake lights, passing/return indicators and junction turn indicators now
reflect the existing driver state. Changes are recorded in `light_states.json`
with both requested and returned native light flags. Not every installed vehicle
mesh necessarily has every lamp. The signals do not change its steering or speed.
Readback is checked after a world tick because CARLA can return the preceding
frame's state immediately after a command.

Clear, overcast, wet and dusk presets are selectable. These are appearance
conditions, not a calibrated tyre/friction experiment. Wet/dusk scenes enable
vehicle low beams. The viewer displays the selected weather.

The standard launcher defaults to Low graphics for the laptop. Epic remains
selectable. The selected 1x–3x playback speed and measured achieved speed remain
separate; screenshot writing, camera capture and hardware can reduce throughput.

```powershell
.\Start_Indian_Traffic.ps1 -Seed 7 -Speed 3
.\Start_Indian_Traffic.ps1 -Seed 7 -Weather dusk -Quality Epic
.\Start_Indian_Traffic.ps1 -Weather wet -Camera driver
.\Start_Indian_Traffic.ps1 -RoadSurface legacy
```

The final command restores the earlier road appearance; it does not restore an
older driving-policy version. The original observation baseline and its source
snapshots remain preserved locally.

## Pedestrian interaction and incident evidence

The preceding mixed seed-7/42 audits found minimum sampled body gaps of 0.174 m
and 0.235 m between a crossing pedestrian and a footpath walker. Zero overlap did
not imply comfortable spacing. Before departing the kerb, a crossing person now
checks the nearby walker's predicted motion through the first two metres of the
crossing. An occupied entry gap delays departure. Someone already crossing keeps
walking, and vehicles retain their yielding rules.
Crossing starts/ends also sit farther from the longitudinal walking tracks, so
a completed crossing person does not stand directly beside the walking stream.

The check uses a 1.2 m centre-clearance setting and a short constant-speed
prediction. It is a conservative illustrative rule, not human-behaviour
calibration. Individual wait/accept events are retained. The viewer also records
body gaps below 0.5 m with 0.7 m release hysteresis. These are diagnostic settings,
not universal safety thresholds; overlaps and native collisions remain separate.

The original vehicle control, passing gates, headway and junction reservation
rules remain unchanged. The policy comparison tests explicitly isolate the added
pedestrian-entry check and still compare the preceding vehicle rules and 912
passing scenarios. Native runs must additionally check actual progress, clearances,
crossing completion, braking/jerk, controls and cleanup.

The sampled body-clearance calculation now uses scalar arithmetic instead of
allocating small NumPy arrays repeatedly. All 1,505 generated geometry cases
matched the preceding kernel and independent audit. Its measured kernel time was
about 15x lower in this laptop microbenchmark. That is not a claim of 15x faster
whole-world playback. Read [the exact measurements](results/geometry_kernel_benchmark.json).

## Honest road-model comparison

The first asphalt pilot retained the preceding V3 traffic policy. It revealed
heavy synthetic bloom in the recorded model camera, while the viewer had bloom
disabled. Its pooled road IoU was **43.89%**; that failed result and every frame
are preserved in [the glare-build record](results/asphalt_glare_comparison.json).
The model and viewer now share zero motion blur, zero bloom and zero lens-flare
settings. The values actually accepted by the camera are saved in capture config.

The final model-camera runs use the corrected V4.1 scene, the same locked morning
model, normalization and 0.5 threshold. They include the declared pedestrian and
lighting changes. No training or selection uses these new frames. Read
[the portable appearance results](results/asphalt_comparison.json).

| Recorded scene | Road IoU | Evaluated pixel coverage |
| --- | ---: | ---: |
| Original light-grey scene | 78.72% | 97.56% |
| First asphalt capture with synthetic glare | 43.89% | 98.16% |
| Final asphalt scene and matched camera settings | 79.06% | 98.16% |

The final comparison contains all 1,100 paired frames. Its **+0.33 percentage
point** difference from the historical scene is small. It does not show that the
trained model improved. The independent audit recounted every saved mask, checked
1,760 depth-projected road points with zero map mismatches, and checked 3,621,078
known-building/actor pixels for reference occlusion. The three checkpoints and
loaded model weights kept their fingerprints. The completed real-data test stays
unchanged at 91.6549% road IoU.

The reusable appearance asset is natively tagged Building because of its asset
folder. Treating its pixels as ordinary buildings would corrupt evaluation.
Asphalt captures therefore add a fourth, aligned instance-ID camera. Only pixels
whose IDs match the saved owned surface actors enter the depth/map resolution;
unrelated buildings retain their negative label. The raw native tags, encoded
instance image, depth, timestamps, camera poses and hashes remain saved unchanged.
The analysis breakdown uses a separate `owned_asphalt_appearance_instance`
category, which is not a new native CARLA class.

The same 80 m range, 0.08 m height tolerance, 0.1 m driving-lane raster and 0.2 m
uncertain-edge exclusion remain. Known vehicles/people remain non-road. The
independent audit checks instance identities, building/actor occlusion, saved-mask
counts and projected points against OpenDRIVE. Older captures still use their
original reference protocol. The early asphalt pilot without instance evidence
is explicitly rejected for scoring.

This changes texture, markings and a thin appearance surface; extra capture
overhead, post-processing and pedestrian changes can also affect comparisons. It is an
appearance experiment on correlated synthetic development scenes, not an isolated
colour experiment or an improvement to measured real-road accuracy. The completed
689-image real-data test and all three model files remain separate and unchanged.

## Repeatable native checks

After stopping any active dedicated session, run the bounded suite in a new
output directory:

```powershell
.\.venv-carla\Scripts\python.exe setup/carla_india/recheck_quality.py --output runs/NEW_QUALITY_CHECK
```

Each scenario uses a fresh owned CARLA session. The suite includes mixed seeds
7/42/19, clear passing, a direct pedestrian yield, oncoming blockage and a guarded
return, including overcast/wet/dusk conditions. Saved traces are independently
audited. Failures remain preserved; a launch/runtime failure stops the suite.
Read [the portable quality results](results/simulation_quality.json) for the
actual measured outcomes and remaining limits. Full local captures, traces,
source snapshots and failed material probes stay under `runs/`.
The first V4 check is retained separately: immediate light readbacks produced
false failures, and the crossing-exit spacing needed another change. It was
stopped before the remaining scenario launch; final checks use a new directory.

The seven final checks passed with no recorded native collisions or sampled body
overlaps. The three mixed seeds each retained one bus/parked-car alert at about
0.483 m. The clear-pass and guarded-return cases completed a pass; the wet
oncoming case remained behind the cycle and did not reach its goal in 60 seconds.
All declared crossings, delayed light readbacks and native viewer controls passed.

| Case / seed | Weather | Completed passes | Ego goal | Closest sampled body gap | Close alerts |
| --- | --- | ---: | --- | ---: | ---: |
| mixed / 7 | clear | 1 | Yes | 0.483 m | 1 |
| mixed / 42 | clear | 1 | Yes | 0.483 m | 1 |
| mixed / 19 | clear | 1 | Yes | 0.483 m | 1 |
| clear pass / 7 | clear | 1 | Yes | 0.844 m | 0 |
| pedestrian / 7 | overcast | 0 | Yes | No pair in the 4 m search band | 0 |
| oncoming / 7 | wet | 0 | No, within 60 s | 2.496 m | 0 |
| guarded return / 7 | dusk | 1 | Yes | 0.815 m | 0 |

Ego maximum braking over 0.5 s was 4.66 m/s² in the mixed cases, with jerk p95
4.18 m/s³; these were not comfort improvements over the preceding build. Full
actor braking/jerk measurements and incidents are retained in the results.

At selected 3x, the ordinary viewer test loops achieved about 2.65–2.76x. The
two runs recording all paired model cameras achieved about 0.61x. These are
whole-loop measurements on this laptop, excluding startup and including test
pauses/writes. The recorded evaluation does not establish real-time camera latency.
Only the clear mixed scenes are used for the frozen-model comparison; wet/dusk
traffic checks do not establish segmentation robustness in those conditions.

This remains one illustrative left-hand neighbourhood with generic installed
assets. Authentic autorickshaws, richer traffic density, irregular road geometry
and calibration against Indian footage remain future work. Driving still uses
simulator ground truth and hand-coded rules. Road predictions do not control the
car, and no RL policy is trained. The later RL scope remains STOP/SLOW/PROCEED
over the route controller with independent safety checks.
