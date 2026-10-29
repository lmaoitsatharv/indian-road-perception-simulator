# Roadmap

The current foundation includes evaluated real/synthetic road segmenters, a 12-class detector, a connected rule-driven CARLA town and observation-only live road prediction. The stages below are planned integration, not completed driving behavior. See [results](RESULTS.md) for current evidence.

## 1. Improve and validate object perception

Review missed animals, small/occluded people, bicycles and broad other-vehicle labels. Separate annotation errors, class errors, localization errors, duplicates and genuine misses. Use audited training-only examples for changes while preserving validation.

Work in progress (validation only, nothing deployed): [detector error review](DETECTION_ERROR_REVIEW_20261028.md), [animal data plan](ANIMAL_DATA_PLAN_20261029.md) and [filtered animal view and re-score](ANIMAL_FILTER_RESCORE_PLAN_20261029.md).

Run short hypothesis-driven experiments, preserve the parent and compare every class under matched resolution/precision. The current 1280 result is 62.245% mAP50 / 41.525% mAP50-95 on project validation, not an independent test or evidence of adequate recall for control. Check the real-trained detector on the intended CARLA camera distribution before relying on its observations.

## 2. Add live detection, tracking and range

Start both model types from the same raw RGB frame while preserving separate preprocessing. Map their outputs into a checked common coordinate system using saved resize/letterbox transforms, calibration and frame/session/time identities.

Implement persistent tracks and range/closing-motion observations. Depth or radar may be declared additional simulator sensors with their own synchronization/association checks. Keep evaluation ground truth distinct from inputs claimed for a perception-driven controller.

Road masks do not establish legal lanes or obstacle-free corridors. Traffic-light/sign categories do not determine light state or sign meaning. Those interpretations need additional modules or explicitly declared simulator inputs.

## 3. Connect bounded decisions to control

Establish a rule baseline consuming the declared perception observations and proposing STOP / SLOW / PROCEED with target speed. Check route geometry, vehicle footprint, obstacles, stopping distance and stale/uncertain inputs. Retain an independent braking override and explicit failure behavior.

Use fixed verified model checkpoints during driving inference. Updating tracks, route progress and commands is not neural training. Save evidence that model-derived observations affect decisions; an overlay alone does not establish control integration.

## 4. Select destinations and follow routes

Expose legal stopping destinations on the connected map, route from the current lane, generate a continuous drivable reference, follow it and stop in a declared goal region at suitable terminal speed. Test several start/goal pairs and unsuccessful/unreachable requests.

Choose useful progress among feasible legal routes. Route length, expected delay, traffic and turning constraints can inform efficiency; safety/observability remain constraints. A preset loop or a goal marker is not destination navigation.

## 5. Refine combined-town overtaking

The selected passing evidence belongs to the smaller neighbourhood. Independently implement/check combined-town passing with clear-pass, oncoming-blocked, pedestrian-conflict, lateral-clearance and return-gap cases.

Record states, blockers, clearance and merge completion. Waiting can be the correct result. Preserve failed cases rather than treating one successful scene as general performance.

## 6. Measure the complete pipeline

Run CARLA, both perception branches and downstream logic together. Measure camera-to-command age, important misses, false braking, dropped/stale frames, identity errors, contacts/close passes, route completion, goal stopping, waiting and comfort. Separate component timings do not establish complete-loop latency.

Compare against the same-seed ground-truth rule baseline using development and held-out scenarios. Broaden density, crossing timing, geometry and weather. Preserve selection boundaries and report uncertainty/failures.

## 7. Evaluate a bounded learned policy

After observation/control contracts and the baseline are established, compare an RL policy limited to STOP / SLOW / PROCEED or corresponding target-speed choice. Specify observations, actions, reward, termination and which inputs are camera estimates versus simulator truth.

Planning, collision checks, emergency braking and low-level control remain separate. No RL policy is currently trained. Future claims require matched held-out comparisons covering progress, failures and comfort.
