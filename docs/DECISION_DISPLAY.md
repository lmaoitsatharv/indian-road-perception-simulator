# Visible driving decisions

The camera window now shows two extra rows:

- **Current action and motion reason:** following traffic, yielding to a pedestrian,
  waiting at the junction, braking, passing or returning left. Current and target
  speeds are displayed in km/h.
- **Passing explanation:** the actual blocking condition and, when applicable, the
  available and required distance. The explanation is refreshed every physics step.

These rows explain the existing rules. They do not change the driving policy,
target speeds, safety margins or 0.05-second physics step.

## Seed 7 example

At 28.1 simulated seconds in the mixed scenario:

```text
FOLLOW | Keeping distance from traffic ahead | speed 8.5 / target 8.5 km/h
Pass blocked: junction too close | available 47.4 m / required 71.0 m
```

The decision log also retains simultaneous blockers: an approaching motorcycle's
gap was 78.9 m while the policy required 124.9 m. The screen shows the final primary
reason selected by the existing policy; the log preserves all evaluated blockers.
Required distances describe this prototype's passing calculation, not a validated
real-road standard.

## Capture and logs

**S** saves the displayed camera image, the entire window including its explanations,
and a JSON sidecar containing that displayed decision. It works while paused.
**Tab**, **Space**, **R**, **+ / -**, **1 / 2 / 3** and **Esc** retain their existing roles.

Each episode saves `ego_decisions.jsonl`, with one record per ego physics step.
It records the controller reason, pass state, blockers, exact distances, speeds and
display text. `source_frame` identifies the snapshot used to compute the command;
`world_frame` identifies the resulting simulator tick. The camera-frame index links
to that same resulting frame and its display text. The viewer uses the cached
decision associated with the displayed camera frame rather than a later snapshot.

Smoke checks save representative HUD images and sidecars automatically. Full run
records remain local; the [portable audit](results/decision_display.json) contains
the selected measured results and tested source hashes.

## Verification on 19 October 2026 UTC

Five focused tests passed, including a comparison of 912 policy cases with the
previous immutable commit. After removing only the explanation writes, the passing
method's syntax tree matches that commit exactly; the stepping and actuator methods
also match. The existing four playback tests passed.

A native 110-second mixed seed-7 run at selected 3x verified:

| Check | Result |
| --- | ---: |
| Ego decisions linked to physics steps | 2,201 |
| Camera frames linked to the corresponding decision | 2,175 |
| Junction-distance records independently checked | 294 |
| HUD captures with matching sidecars | 9 |
| Goal reached | 81.95 simulated seconds |
| Completed overtakes | 1 |
| Native collisions / sampled body overlaps | 0 / 0 |
| Actor steps compared against the previous run | 41,819 |
| Maximum position / command differences | 0 m / 0 |

The checked session closed its owned jobs. Full evidence is retained locally in
`runs/carla_india_smoke_20261019T182836_913871Z_seed7/`.

An earlier completed check also passed. Reopening the live viewer then exposed a
startup timeout: actor settling had consumed the ten-second camera-delivery timer
before the camera loop began. The timer now starts when the camera loop starts.
The failed zero-frame run is preserved in
`runs/carla_india_live_20261019T182717_266696Z_seed7/`; it closed its owned jobs.
The measured results above are from the subsequent completed recheck of the fix.

These are selected regression checks, not a claim that all traffic is safe. The
explanations come from hand-coded rules and simulator ground truth. The locked
segmentation model is unchanged and does not control the car; no RL policy was
trained. A later [live road observer](LIVE_SIMULATION_OBSERVATION.md) adds observation-only predictions; object tracking and perception-driven control remain separate integration work.
