# Indian traffic prototype

This is a MATLAB simulation/replay reference, separate from the native CARLA [connected town](../docs/COMBINED_INDIAN_TOWN.md).

## Recorded revision - 13 October 2026

The public simulator now delegates to `run_indian_traffic_trial` with reviewed
mixed-traffic settings. It uses full-body route following, logged prospective
NPC motion holds, persistent pedestrian detours around stopped clusters,
parked-vehicle clearance and smoother following. Three 300-second reviewed seeds
(42, 7, 2026) had zero independently sampled ego/other overlaps and zero forced
NPC holds. Seed 7's longest stop fell from 219 to 0.7 seconds, while seed 2026's
earlier car/two-wheeler incident did not recur in the reviewed run.

These are joint development changes, not isolated experiments or general safety
validation. Emergency acceleration changes remain abrupt. Mixed tests disabled
overtaking; three separate controlled cases checked clear-road passing and
oncoming/pedestrian rejection. Details and exact configurations are in
`../runs/review_20261013/REVIEW_REPORT.md`.

## Historical first prototype — 4 October 2026

This is a synthetic development world with left-side traffic, cars, auto-rickshaws,
two-wheelers, three parked vehicles, and ten pedestrians. Pedestrians walk on
footpaths, start crossings at seeded random times on straight road sections,
wait for occupied spaces, and can walk around stopped vehicles. The distributions
are illustrative; they have not been fitted to Indian traffic recordings.

## Open the saved simulation

```matlab
addpath(fullfile(pwd, 'integration', 'matlab'));
replay_indian_traffic_world( ...
    fullfile(pwd, 'runs', 'indian_traffic_20261004_seed42_02'));
```

Press Play; select 1x, 4x or 16x. Reset and the time slider work. Repeat replays
the same recorded traffic, rather than generating a new random experiment.

To generate a separate run with another seed:

```matlab
launch_indian_traffic_world(43,300);
```

Each run uses a fresh output directory. The ego car physically follows a fixed
loop through Stanley steering and bounded speed feedback, with rule-based
STOP/SLOW/PROCEED and a separate short-range braking override. Other vehicles use
prescribed lane positions and simple following rules. This particular world does
not run Hybrid A* or use the camera segmentation model. Existing Hybrid A* and
segmentation artifacts remain separate and unchanged. RL is pending.

## Saved evidence and known faults

- `runs/indian_traffic_20261004_seed42_01`: preserved failed development run.
  At 224.7 seconds a crossing pedestrian entered the body of the already stopped
  ego car. Several other overlaps were also recorded.
- `runs/indian_traffic_20261004_seed42_02`: pedestrian obstacle awareness and
  narrower two-wheeler offset variation added. Completed 300 seconds, one loop,
  45 crossing starts / 41 crossing completions, zero sampled ego collisions and
  no sampled ego road departure. Ego maximum speed was 3 m/s (10.8 km/h).
- Other actors still produced **246 overlap samples at 0.5-second inspection
  intervals**. These are samples, not 246 distinct collision events. This remains
  a fault in the traffic prototype and must be resolved before RL training or
  a general collision-free simulation claim.
- The ego selected STOP for 169.2 seconds of this run. Progress and conservative
  decision behaviour need improvement after the actor-interaction faults.
- Collision checks sample the ego body every 0.02 seconds; they are not a
  continuous-time safety guarantee. This is one seed, not a held-out evaluation.
- `world.mat`, `replay.json`, `ego_trace.csv`, `events.json`, `world_config.json`,
  `result.json` and source snapshots preserve the actual simulation evidence.

## Replay repair and verification

The first viewer failed because a nested-function variable named `start` conflicted
with `start(clockTimer)`. It was renamed `trailStart`. Road polygons now use MATLAB
`subtract` rather than the numeric minus operator. The status panel identifies
ego collisions separately from other-actor overlap samples.

`setup/verify_indian_replay.m` exercised actual MATLAB callbacks. The saved
`runs/indian_traffic_20261004_seed42_02/replay_verification.json` confirms Play/Pause,
16x playback, seeking to 240 seconds, Reset and deletion of the viewer timer on
close. The corrected viewer was also opened and visually inspected in MATLAB.
No simulation rerun, training or test-set inference was needed for this repair.

MATLAB reports a polygon simplification warning during visual road construction;
the road and footpaths rendered successfully. The simulation's analytic road
geometry and saved physics are independent of these display polygons.

Next: audit and correct remaining NPC/pedestrian interactions, then verify
several seeds and reserved scenarios before introducing a future bounded RL policy.
