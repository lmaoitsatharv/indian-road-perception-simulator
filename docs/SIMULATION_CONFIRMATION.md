# Fresh simulation confirmation — 23 October 2026

The fixed epoch-10 simulation model was checked on **eight new 60-second Indian-style left-hand episodes**, covering four changed map layouts and all six traffic-case families. All 480 frames at 1 Hz were scored; none were removed based on accuracy. No checkpoint, threshold or model weights changed.

| Model | Road IoU | Dice | Valid-pixel coverage |
|---|---:|---:|---:|
| New simulation-only model | 99.7478% | 99.8738% | 98.0143% |
| Locked real-road baseline on the same frames | 79.9195% | 88.8392% | 98.0143% |

New-model minus baseline road IoU: **+19.8283 percentage points**.

These measurements cover new seeds, installed building assets, asphalt appearances and junction spacing within the same illustrative road families. They do not establish real-road or unseen-city performance.

## Scene changes fixed before capture

Junction half-spacing changed from 12 m to 10 or 11 m; the four canonical OpenDRIVE map hashes differ from training. Approach lengths, 4.5 m lanes and LHT lane rules remain fixed. New low-rise/apartment meshes (House12, Apartment01 and Apartment02) replace the original Apartment03/04 facades. Road texture colour/grain, sun direction, camera heights (1.30/1.45 m), pitch/yaw and FOV (92/98 degrees) also vary.

Preflight checked 30,600 footpath samples with no driving-lane intrusion and verified original default-map hashes remain unchanged. Native capture metadata confirms the new meshes actually spawned. All cameras were synchronized, no evidence frames dropped, and the 360 daytime frames with substantial visible sky passed the brightness check.

## Per-episode comparison

| Layout | Case | Weather | Seed | Baseline IoU | New model IoU |
|---|---|---|---:|---:|---:|
| crossroad | mixed | clear | 4101 | 85.9964% | 99.3910% |
| crossroad | oncoming | wet | 4109 | 80.9639% | 99.3173% |
| t_north | pedestrian | overcast | 4201 | 77.0488% | 99.9007% |
| t_north | clear_overtake | dusk | 4209 | 76.9643% | 99.8423% |
| t_south | junction | clear | 4301 | 82.8316% | 99.8998% |
| t_south | return_gap | overcast | 4309 | 79.6271% | 99.6872% |
| straight | mixed | wet | 4401 | 81.0773% | 99.9560% |
| straight | pedestrian | dusk | 4409 | 75.2106% | 99.9964% |

Per-frame comparison: 480 improved, 0 worsened, 0 equal. Counts use exact integer-ratio comparisons. Weather, appearance and geometry changes are bundled; this check does not isolate their individual effects.

## Independent checks and remaining limitations

Both sets of 480 saved masks were independently recounted against their 0/1/255 references. Depth-projected sampled road points were checked against each native OpenDRIVE map; known actors and raised footpaths remained non-road in the reference. The two models used identical capture hashes and identical reference arrays. Uncertain pixels are excluded, with coverage above. Small declared semantic/instance edge mismatches (at most 0.1%) are excluded exactly, matching the dataset policy; historical captures without that declaration still reject any mismatch.

All eight native health checks passed with zero collisions, body overlaps, logged road departures or wrong-way events. Three close-clearance alerts remain, with a minimum sampled body gap of 0.204 m. Two pedestrian runs (seeds 4201/4409) did not exercise the historical vehicle-yield criterion: pedestrians waited for a gap. Those behaviour criteria remain marked unexercised, not passed. Some mixed runs did not finish the full route within 60 seconds. This is a perception comparison with independent rule-based traffic, not a driving-safety evaluation.

Worst new-model frame: crossroad, seed 4109, 47.1s, road IoU **93.0735%**. Inspect its saved comparison and the worst frame from every episode; pooled overlap can hide local errors. Stationary intervals remain included and adjacent frames are correlated.

## Evidence and next step

- Gallery of all 480 matched comparisons: `runs/simulation_confirmation_20261023_01/index.html`.
- Counts, per-frame changes and per-layout breakdown: `results.json` and `comparison_frames.json` in that folder.
- Fixed simulator-only route: `simulation_model.json`; selection and capture protocol: `selection.json` / `plan.json`.
- Native and protocol verification: `capture_verification.json`, `preflight.json`, `protocol_verification.json`.
- Independent pixel/depth checks: `fresh_model/independent_verification.json` and `locked_baseline/independent_verification.json`.
- Portable results: [results/simulation_confirmation.json](results/simulation_confirmation.json).

Real-image/video inference, real checkpoints, model catalog/lock, original splits and the trained simulation checkpoint remain unchanged. No reserved real-test images were opened; no training or RL occurred. The opt-in manifest is for recorded CARLA evaluation only, not shared real deployment.

The subsequent bounded live-observation implementation is complete. Read
[LIVE_SIMULATION_OBSERVATION.md](LIVE_SIMULATION_OBSERVATION.md) for measured
source-to-result age, frame alignment, stale/offline suppression, recovery and
the retained display expiry limitation. Vehicle/pedestrian detection/tracking
and any controller use remain separate future work. This fresh set is now
inspected and cannot serve as untouched confirmation after tuning.
