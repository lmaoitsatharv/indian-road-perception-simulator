# Learned speed: results (12 November)

Design: [RL_SPEED_DESIGN_20261112.md](RL_SPEED_DESIGN_20261112.md). Decisions taken while building it:
[RL_SPEED_RULINGS_20261112.md](RL_SPEED_RULINGS_20261112.md).

## In short

<!-- filled in from the town drives below -->

## In the stand-in town (330 held-out drives)

The 110 place-to-place routes, three drives each from 500 m before the destination, seeds never used in training,
junction waits as in the town (median 18 s when there is one). The same 330 drives for every driver (common random
numbers). "Over the limit" counts any amount over 0.05 m/s; the rule driver is the camera driver as it was.

| Driver | Time per drive | Over the limit | Largest excess | Hard braking (< -3 m/s²) | Emergency braking | Collisions / near misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule driver (6 m/s cruise) | 171.3 s | 24.4 s | +1.06 m/s | 3.85 s | 2.06 s | 0 / 0 |
| Rule driver keeping the posted limits | 173.2 s | 0 | -0.33 m/s | 3.59 s | 2.22 s | 0 / 0 |
| Shield only (always its full speed) | 169.9 s | 0 | -0.33 m/s | 3.25 s | 2.19 s | 0 / 0 |
| Shield + holding the limit | 166.5 s | 0 | -0.04 m/s | 3.40 s | 2.08 s | 0 / 0 |
| Shield + holding the limit + easing to the junction entry | 164.7 s | 0 | -0.04 m/s | 3.64 s | 2.11 s | 0 / 0 |
| **Learned policy** (run 04: shield + holding the limit) | 166.4 s | 0 | -0.04 m/s | 3.40 s | 2.08 s | 0 / 0 |
| **Learned policy** (run 05: shield + holding the limit + easing to the junction entry) | 164.4 s | 0 | -0.04 m/s | 3.63 s | 2.06 s | 0 / 0 |

A second held-out set (660 drives, starting 300-900 m before the destination, other seeds) gives the same picture:

| Driver | Time per drive | Over the limit | Hard braking | Emergency braking | Collisions / near misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule driver | 200.9 s | 30.5 s | 4.59 s | 2.48 s | 0 / 0 |
| Rule driver keeping the posted limits | 205.3 s | 0 | 4.19 s | 2.58 s | 0 / 0 |
| Learned policy (run 04) with holding the limit + junction-entry easing | **195.0 s (-3.0%)** | 0 | 4.23 s | 2.54 s | 0 / 0 |

What this shows:

- Keeping to the posted limits costs the rule driver 1.1% (it is over them 14% of the time in these drives).
- The learned car is **2.6% faster than the rule driver and 3.7% faster than the rule driver keeping the limits**,
  never over a limit, with less hard braking.
- **Almost all of that comes from the shield, not from the learning.** The policy drives at its shield's speed in
  97-98% of its decisions, and the best checkpoint is within 0.3 s of "always the shield's speed". The time goes to
  things a speed choice cannot change under the same rules: waiting for turns, following slower vehicles, the
  junction speeds, and the limits themselves.
- Under stress (up to 50% of people jaywalking, three times as many missed detections) neither the shield nor the
  policy had a collision or more than 0.3 s of near misses in 330 drives: the town's jaywalkers only cross when the
  gap is long, so caution never paid off for the policy.

## In the CARLA town: first version (seed 8101)

Two pairs of 900 s drives, the rule driver and the learned speed, same seed, same destinations, clear weather,
`-Check`, scored by `rl/analyse_town_run.py` on the simulator's ground truth. The learned car used policy run 04
and the **first limit hold** (see below). Owner folders: `runs/camera_driving_20261112_{rule01,rl01,rule02,rl02}`.

| | List 1 rule | List 1 learned | List 2 rule | List 2 learned |
| --- | ---: | ---: | ---: | ---: |
| Destinations reached | 3 of 3 | 3 of 3 | 4 of 4 | 4 of 4 |
| Last arrival | 680.2 s | 717.6 s | 845.4 s | **733.4 s** |
| Stopped waiting for a junction turn | 16.4 s | 54.4 s | 58.3 s | 54.2 s |
| Legs without those waits | 653.7 s | 653.2 s | 771.8 s | **664.1 s** |
| Collisions / resets | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| More than 0.5 m/s over a limit | 139.6 s | **0 s** | 30.1 s | **0 s** |
| Over a limit at all / largest excess | 156.1 s / 1.11 m/s | 6.6 s / 0.38 m/s | 96.7 s / 1.11 m/s | 6.1 s / 0.39 m/s |
| Hard braking (< -3 m/s²) | 14.3 s | 32.0 s | 16.3 s | 34.3 s |
| Emergency braking | 8.2 s | 6.3 s | 17.5 s | 6.4 s |
| Off road / wrong side | 0 / 0 s | 0 / 0 s | 0 / 0 s | 0.1 / 2.1 s |

Lists: 1 = Residential square, Outskirts bend, Irregular junction; 2 = Village road, Outskirts bend, Colony street,
Market junction (the 8-9 November test lists).

- **List 2: the learned car arrived 112 s (13%) earlier** and spent 108 s less moving. List 1 tied once junction
  waits are taken out; with them the learned car arrived 37 s later, because it reached Irregular junction while a
  crossing car and bicycle held it (34 s) and the rule driver, there at another moment, did not wait at all. When a
  car arrives at a junction decides most of its wait, so single pairs are noisy; more seeds are queued.
- **Speed limits:** the rule driver cruises at 5.5 m/s on every street, 1.1 m/s over the market street's 4.4 m/s
  and 2.5 m/s under the main road's 8. The learned car held 4.3 on the market street and 7.6 on the main road.
- **What went wrong: hard braking doubled.** The first limit hold raised the set-point too far at higher speeds
  (its pedal model came from the rule car only, never above 5.5 m/s) and switched off the moment the car crossed
  the limit. The set-point then fell 0.6 m/s at once, the speed controller's integral reset, the throttle dipped to
  zero and the car's automatic gearbox shifted: 18 drops of about 2 m/s at -5 to -6 m/s² with the throttle still
  on, mostly on the 6-6.8 m/s rural roads (open-road medians there were only 4.0-5.7 m/s). The rule car never
  reaches those speeds. Fixed for the next drives: a 0.34 m/s raise measured on these drives, faded out smoothly
  near the limit, and a 30 s pause if the car is ever 0.3 m/s over for 1 s; the stand-in's pedal model was refitted
  on the same three drives so training sees this car's real response.
- The policy itself went below its shield in 7% of list-2 frames, almost all while stopped (arrived, waiting for a
  turn), where it makes no difference.

## In the CARLA town: second version (seed 42, list 1)

Policy run 07 (trained on the refitted pedal model), the measured 0.34 m/s hold faded near the limit, and easing
to the junction entry. Owner folders `runs/camera_driving_20261112_list1_s42_{rule,rl}`.

| | Rule | Learned |
| --- | ---: | ---: |
| Destinations reached / last arrival | 3 of 3, 684.9 s | 3 of 3, 688.6 s |
| Stopped waiting for a turn / legs without it | 16.2 s / 658.6 s | 27.8 s / **650.5 s** |
| Collisions / resets | 0 / 0 | 0 / 0 |
| More than 0.5 m/s over a limit / at all | 135.5 s / 154.9 s | **0 s** / 6.3 s (at most 0.38 m/s) |
| Hard braking / emergency braking | 14.2 s / 10.5 s | 28.6 s / **3.8 s** |
| Wrong side | 2.6 s | **11.8 s** |

Two problems, both found in the frame records and fixed for the next drive:

- **Hard braking was still double, and the hold was not the cause.** On the 6.0 m/s rural street the speed swung
  between 3.9 and 6.35 m/s every 2.5 s with the throttle on: the simulated car's automatic gearbox shifts up at about
  5.9 m/s and loses ~2 m/s while shifting, then shifts back down. The rule car cruises at 5.5 m/s and never gets
  there. The learned car now aims for 5.6 m/s on streets posted 5.6-7.3 m/s.
- **11.8 s on the wrong side**, all in one wait: easing to the junction entry let the car reach the mouth of
  Irregular junction at 5.3 m/s, where a wide road area and the turn wait made it swing 1.2 m over the centre line
  and stop there. Junction-entry easing is off again (the car slows to junction speed by the start of the 20 m zone,
  as the rule driver does).

In the stand-in, the final shield (gear band, no junction-entry easing) gives 168.8 s per drive against the rule
driver's 170.5 s (-1.0%) and the rule driver keeping the limits 172.5 s (-2.1%), never over a limit, with 17% less
hard braking than the rule driver (3.86 s against 4.63 s).

## In the CARLA town: final version (seed 7, list 1)

Policy run 07, the faded 0.34 m/s hold, the gear band, no junction-entry easing. Owner folders
`runs/camera_driving_20261112_list1_s7_{rule,rl}`.

| | Rule | Learned |
| --- | ---: | ---: |
| Destinations reached / last arrival | 3 of 3, 697.0 s | 3 of 3, 700.4 s |
| Stopped waiting for a turn / legs without it | 28.8 s / 658.1 s | 3.7 s / 686.5 s |
| Collisions / resets | 0 / 0 | 0 / 0 |
| Over a limit at all / largest excess | 152.2 s / +1.11 m/s | **0 s / -0.16 m/s** |
| Hard braking / emergency braking | 16.9 s / 6.6 s | **14.7 s** / 8.8 s |
| Off road / wrong side | 2.2 / 0 s | 0.2 / 1.9 s |
| Open-road speed by posted limit (median) | 5.5 m/s everywhere | 4.4 → 4.22, 4.8 → 4.63, 5.0 → 4.83, 6.0-6.8 → 5.4, 8.0 → 7.83 |

- **Hard braking is now below the rule driver's**, and the car was never over a limit at all.
- **List 1 is mostly market and colony streets (4.4-5.0 m/s limits)**, where the rule driver's 5.5 m/s is over the
  limit; keeping to them cost the learned car about 4% of its moving time here, and it finished 3 s later overall
  with less waiting at junctions. List 2 has more of the 8 m/s main road (see the first-version pair: 13% faster).

## Final version: three list-1 drives against the rule driver

The rule driver's code is unchanged by this work (the learned speed is off by default), so the learned car's final
version was also driven on the seeds that already had rule baselines. Owner folders
`runs/camera_driving_20261112v3_list1_s{8101,42}_rl` and `runs/camera_driving_20261112_list1_s7_rl`.

| Seed | Last arrival rule / learned | Over a limit at all, rule / learned | Hard braking rule / learned | Emergency braking rule / learned | Collisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8101 | 680.2 / 690.5 s | 156.1 s / **0 s** | 14.3 / **10.2 s** | 8.2 / 7.5 s | 0 / 0 |
| 42 | 684.9 / 693.1 s | 154.9 s / **0 s** | 14.2 / **10.6 s** | 10.5 / 7.9 s | 0 / 0 |
| 7 | 697.0 / 700.4 s | 152.2 s / **0 s** | 16.9 / **14.7 s** | 6.6 / 8.8 s | 0 / 0 |

All six drives reached all three destinations with no reset. The learned car was never over a limit (closest 0.16 m/s
under) and braked hard about a quarter less; it arrived 3-10 s (0.5-1.5%) later on this list, which is mostly
4.4-5.0 m/s market and colony streets where the rule driver's 5.5 m/s is over the limit.

**List 2, seed 8101, final version** (`runs/camera_driving_20261112v3_list2_s8101_rl`): 4 of 4 by 842.7 s against
the rule driver's 845.4 s; never over a limit (closest 0.16 m/s under) against 96.7 s over; hard braking 16.1 s
against 16.3 s; emergency braking 14.7 s against 17.5 s; no collision, reset or wrong side.

The first version had finished this list in 733.4 s. Leg by leg, most of that difference was luck: on legs 3-4 the
rule driver and the final version each spent about 108 s and 36 s queuing behind a slow vehicle that the first
version, arriving at another moment, never met. The rest was real: easing to the junction entry saved about 69 s of
junction crawling over the four legs. It is back on, but only once the turn is granted and the road is not flagged
wide or unclear (the s42 failure needed both).

<!-- granted-entry drives below -->
