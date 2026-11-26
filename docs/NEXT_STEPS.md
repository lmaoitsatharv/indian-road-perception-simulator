# Roadmap

The current foundation includes evaluated real/synthetic road segmenters, a 12-class detector, a connected CARLA town with rule-driven traffic, and a car driven by the two models alone that picks destinations, passes, takes junctions and turns round, with an optional learned speed policy. Stages 3, 4, 5 and 7 below are now done in simulation and stage 6 in part; what remains is listed under each. See [results](RESULTS.md) and [camera driving](CAMERA_DRIVING.md) for the evidence.

## 1. Improve and validate object perception

Review missed animals, small/occluded people, bicycles and broad other-vehicle labels. Separate annotation errors, class errors, localization errors, duplicates and genuine misses. Use audited training-only examples for changes while preserving validation.

Work in progress (validation only, nothing deployed): [detector error review](DETECTION_ERROR_REVIEW_20261028.md), [animal data plan](ANIMAL_DATA_PLAN_20261029.md) and [filtered animal view and re-score](ANIMAL_FILTER_RESCORE_PLAN_20261029.md).

Run short hypothesis-driven experiments, preserve the parent and compare every class under matched resolution/precision. The current 1280 result is 62.245% mAP50 / 41.525% mAP50-95 on project validation, not an independent test or evidence of adequate recall for control. Check the real-trained detector on the intended CARLA camera distribution before relying on its observations.

## 2. Add live detection, tracking and range

Start both model types from the same raw RGB frame while preserving separate preprocessing. Map their outputs into a checked common coordinate system using saved resize/letterbox transforms, calibration and frame/session/time identities.

Implement persistent tracks and range/closing-motion observations. Depth or radar may be declared additional simulator sensors with their own synchronization/association checks. Keep evaluation ground truth distinct from inputs claimed for a perception-driven controller.

Road masks do not establish legal lanes or obstacle-free corridors. Traffic-light/sign categories do not determine light state or sign meaning. Those interpretations need additional modules or explicitly declared simulator inputs.

## 3. Connect bounded decisions to control

**Done in simulation (6-7 Nov).** The camera driver turns the road mask and detector boxes into PROCEED / SLOW / STOP with a target speed and a reason, checks the path and the car's footprint against obstacles, stops on stale perception and keeps an emergency brake. Every decision is saved with the frame it came from. Remaining: depth or range sensing instead of flat-ground estimates.

Establish a rule baseline consuming the declared perception observations and proposing STOP / SLOW / PROCEED with target speed. Check route geometry, vehicle footprint, obstacles, stopping distance and stale/uncertain inputs. Retain an independent braking override and explicit failure behavior.

Use fixed verified model checkpoints during driving inference. Updating tracks, route progress and commands is not neural training. Save evidence that model-derived observations affect decisions; an overlay alone does not establish control integration.

## 4. Select destinations and follow routes

**Done in simulation (8-9 Nov; either side of a street and U-turns, 18 Nov).** Destinations are picked on the map; the sat-nav routes on the lane graph and the car stops 0.2-0.5 m from the pin. Remaining: shorter waits for a turn at busy junctions, and junction paths that do not cut the inside of sharp turns.

Expose legal stopping destinations on the connected map, route from the current lane, generate a continuous drivable reference, follow it and stop in a declared goal region at suitable terminal speed. Test several start/goal pairs and unsuccessful/unreachable requests.

Choose useful progress among feasible legal routes. Route length, expected delay, traffic and turning constraints can inform efficiency; safety/observability remain constraints. A preset loop or a goal marker is not destination navigation.

## 5. Refine combined-town overtaking

**Done in simulation (7 Nov; side cameras, 23 Nov).** The car passes a bus stopped at its stop after checking the rear camera, waits for traffic coming up behind or alongside, and stops and backs out if something comes the other way. Remaining: more seeds and passing in wet weather.

The selected passing evidence belongs to the smaller neighbourhood. Independently implement/check combined-town passing with clear-pass, oncoming-blocked, pedestrian-conflict, lateral-clearance and return-gap cases.

Record states, blockers, clearance and merge completion. Waiting can be the correct result. Preserve failed cases rather than treating one successful scene as general performance.

## 6. Measure the complete pipeline

Run CARLA, both perception branches and downstream logic together. Measure camera-to-command age, important misses, false braking, dropped/stale frames, identity errors, contacts/close passes, route completion, goal stopping, waiting and comfort. Separate component timings do not establish complete-loop latency.

Compare against the same-seed ground-truth rule baseline using development and held-out scenarios. Broaden density, crossing timing, geometry and weather. Preserve selection boundaries and report uncertainty/failures.

## 7. Evaluate a bounded learned policy

**Done in simulation (12 Nov).** A PPO policy chooses a fraction of the shield's speed; in CARLA it never exceeded a posted limit and braked hard less than the rule driver, though most of its gain comes from the shield. See [learned speed results](RL_SPEED_RESULTS.md). Remaining: rewards that learn something the shield's rules do not already give.

After observation/control contracts and the baseline are established, compare an RL policy limited to STOP / SLOW / PROCEED or corresponding target-speed choice. Specify observations, actions, reward, termination and which inputs are camera estimates versus simulator truth.

Planning, collision checks, emergency braking and low-level control remain separate from the policy. Future claims require matched held-out comparisons covering progress, failures and comfort.
