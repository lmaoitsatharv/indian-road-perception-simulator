# Learned speed for the camera car (12 November)

**Goal (from the user):** use reinforcement learning so the camera car reaches its destination faster, without
breaking traffic rules and while keeping everyone safe.

## What the learning decides, and what it never decides

The camera driver already works out every reason to go slower: a vehicle or person on its path, the junction where
it must wait for its turn, a curve, the destination, a wide or unclear road, a road end, a pass in progress. Each is a
speed limit with a reason. Until now the car took the lowest of them, with a fixed 6 m/s cruise on top.

With the learned speed (`DriverSettings.rl_speed = 1`, launcher `-RLSpeed`):

| Part | What it is | Learned? |
| --- | --- | --- |
| **Shield** | The lowest of every safety constraint the rule driver has (unchanged: same gaps, same emergency stops, same junction waits, same pedestrian rules), plus the **posted speed limit** from the sat-nav's map, slowing at 1.5 m/s² in time for a lower limit ahead | No |
| **Policy** | A small network that picks a fraction of the shield's speed (0, 0.3, 0.5, 0.65, 0.78, 0.88, 0.95 or 1) ten times a second | Yes |

The policy can only slow the car below the shield, never speed it up, so it cannot break a rule or a safety
constraint the shield enforces. Two rule-driver choices are left out of the shield because they are about pace, not
safety:

- the fixed 6 m/s cruise, replaced by the posted limit (faster on the main road and rural roads, **slower** on the
  market, colony and irregular-junction streets, where 6 m/s is over the limit);
- the 3 m/s crawl over the last 30 m to a destination (the stopping curve to the pin stays).

Two additions to the shield, for the learned car only:

- **Holding the limit:** the rule driver's speed controller (`CameraDriver._pedals`) settles under its target:
  the car needs about 0.61 throttle to hold any speed from 3 m/s up (fitted on three town drives), which leaves the
  controller 0.42 m/s short at every speed (0.49 m/s under a 6 m/s target in the town runs). While the posted limit
  is what holds the car back, the controller's set-point is raised by 80% of that shortfall
  (`india_rl_speed.hold_offset`, 0.34 m/s), fading out linearly from 0.25 m/s under the limit to 0.25 m/s over it
  (`hold_weight`), so the car drives just under the limit rather than 0.5 m/s under it. The target in the records
  stays the limit. If the car is ever 0.3 m/s over a limit for 1 s, the raise is paused for 30 s.
- **Easing into junctions:** from 60 m out the shield lowers smoothly (2 m/s²) to the 3.5 m/s junction speed by the
  start of the 20 m junction zone, instead of the step from 8 m/s to 2.5-3.5 m/s that made the car brake hard there.
  Once the turn is granted and the road is not flagged wide or unclear, it eases to the junction speed by the
  junction's entry instead, rather than crawling the last 20 m of its own street.
- **Below the gearbox's hunting band:** on streets posted 5.6-7.3 m/s the learned car aims for 5.6 m/s, because the
  simulated car's automatic gearbox hunts between gears when asked to hold 6-6.8 m/s (see the rulings). On the
  8 m/s main road and everywhere posted lower it aims for the limit itself.

The steering, the junction paths, passing, backing out and the stuck-car recovery are unchanged.

## Traffic rules the car keeps

The town has no signs or signals, so "the rules" are written down here:

1. **Speed limits:** each street's district speed (the `cruise` of `india_combined_map.STREETS`, the speed every
   other driver keeps): market 4.4 m/s, main road 8, rural roads 6-6.8, colony 4.8-5, irregular junction street 5.2.
   Inside a junction, the lower of the two streets it joins (the camera car's own junction speeds of 3.5 m/s on a
   planned path and 2.5 m/s on the route line still apply).
2. **Right of way at junctions:** never enter a junction before the town's turn-taking gives the car its turn.
3. **Pedestrians:** stop for a person on or walking towards the path; slow to 2.5 m/s beside one close to it.
4. **Gaps:** 4 m to a vehicle (6 m to a bus or truck) at standstill, and a 2.5 m/s² stopping curve to it.
5. **Keep left** (steering, unchanged).

The rule driver as it is breaks rule 1: on 9 November (pair 18/19, run 19) it spent 30 s more than 0.5 m/s over the
limit, up to 1.1 m/s over, 4.8% of its moving time.

## How it is trained

Training in CARLA itself is too slow (a 900 s town drive takes about 18 minutes plus 3 minutes to load) and the
models need the GPU. The policy is trained instead in **a fast stand-in for the town** (`rl/speed_env.py`, numpy,
512 drives at once, about 40,000 decisions a second):

- **Roads:** 510 real routes of the town from the car's own planner (`rl/export_town_routes.py`): posted limits,
  curvature and junctions every metre. The 110 place-to-place routes are kept for evaluation only.
- **Traffic:** vehicles ahead keep their district speed, slow to 2.7 m/s near junctions, wait their turn, sometimes
  turn off, and buses stop at the roadside (the car passes after waiting 3 s, as the camera driver does).
- **People:** on the footpaths by district density; a few jaywalkers who cross the way the town's do (they wait
  until nothing is within 12 m and nothing is coming within 12 m + 2.8 s, then cross at 1.15-1.35 m/s).
- **Junctions:** within 35 m the car waits for its turn (none 30% of the time, otherwise a log-normal wait), and not
  before vehicles queued ahead of it have gone.
- **Perception as the camera driver has it:** a 10 Hz camera 0.1 s late, missed detections, distance errors that
  grow with distance, people's sideways speed from a short track, phantom vehicles and phantom road ends.
- **The driver's own rules** for what is on its path, the same speed constraints, and its own speed controller
  (`CameraDriver._pedals`) on a pedal model fitted to the camera car in the 9 November town run (throttle 0.6 holds
  speed, 0.8 gives +1.5 m/s², full brake -5.2 m/s², 0.1 s lag).

Collisions and near misses are judged on true positions. The reward is progress (0.1 per metre), minus 0.02 per
0.1 s, plus 4 for arriving, minus 40 for a collision, minus penalties for near misses, braking harder than 2.5 m/s²,
emergency stops, jerk, going over the limit and entering a junction out of turn.

PPO (`rl/train_speed_ppo.py`, PyTorch in WSL): 512 drives x 128 decisions per update, 2 x 128 tanh layers. Every 10
updates the greedy policy drives the 110 held-out place-to-place routes (330 drives, seeds never trained on) next to
the rule driver; the kept policy is the fastest one with no more collisions, near misses, rule breaks or timeouts
than the rule driver. It is exported for numpy (`models/rl_speed/policy.npz`) so the camera driver needs no PyTorch.

## What the car is allowed to know

Only what the camera driver already knows (its speed, its own constraints, its tracks of people) and the sat-nav's
map (limits, the next junction, the destination). `setup/verify_camera_driving.py`'s isolation check covers
`india_rl_speed.py` too.

## Checks

- `setup/verify_rl_speed.py`: limit profiles, the policy file, the learned speed never above the shield for any
  action, cap and reason, and an emergency keeps its full brake.
- `setup/verify_rl_speed.py --suites`: every camera-driving and destination check again with the learned speed on.
- Town drives in pairs (rule driver / learned speed), same seed and destinations, scored by
  `rl/analyse_town_run.py`: leg times, collisions, resets, time over the limit, hard braking.

Results: [RL_SPEED_RESULTS.md](RL_SPEED_RESULTS.md).
