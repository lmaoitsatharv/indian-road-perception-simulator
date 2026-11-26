# Learned speed: decisions made while building it (12 November)

The user asked for RL that gets the camera car to its destination faster without breaking traffic rules and keeping
everyone safe, then went to sleep. These are the decisions taken on their behalf, each with what it costs if wrong.
Design: [RL_SPEED_DESIGN_20261112.md](RL_SPEED_DESIGN_20261112.md). Results: [RL_SPEED_RESULTS.md](RL_SPEED_RESULTS.md).

## Scope

- **What the RL decides: the target speed only**, as a fraction of a fixed shield's speed (the scope already
  recorded in NEXT_STEPS.md: "STOP / SLOW / PROCEED, selecting target speed above the existing route controller.
  Keep independent emergency and collision checks"). Steering, junction paths, passing and backing are unchanged.
  Cost if wrong: decisions with more time at stake (overtaking slow traffic on the divided road, lane choice,
  route choice) are not learned.
- **The shield keeps every safety constraint of the rule driver unchanged** (gaps, emergency stops, pedestrian
  rules, junction waits, junction speeds, wide/unclear road speeds) and adds the posted limit. Only the fixed 6 m/s
  cruise and the 3 m/s crawl over the last 30 m are left out. Cost if wrong: none for safety (the policy can only
  slow the car); arriving uses the 2.5 m/s² stopping curve to the pin instead of a crawl.
- **Branch `rl-speed-governor` in the main checkout, nothing committed** (the user did not ask for commits; town
  runs need the gitignored models and runs/). Cost if wrong: none.

## Traffic rules

- **Posted limit = each street's district speed** (`india_combined_map.STREETS[*].cruise`, which every other driver
  keeps); inside a junction the lower of its two streets'. The town has no signs; this was the only consistent
  reading. Cost if wrong: the learned car is slower than it needs to be on the market (4.4), colony (4.8-5) and
  irregular-junction (5.2) streets; `DriverSettings.rl_speed_limits = 0` makes the 6 m/s cruise the limit again.
- **The town's 2.6 m/s junction-connector cruise is not a limit**; the car's own junction speeds (3.5 / 2.5 m/s)
  stay. Cost if wrong: the car turns about 1 m/s faster than the other drivers on a planned path, as the rule
  driver already does.
- **Over the limit means more than 0.5 m/s over** for the "speeding" count, and any amount (>0.05 m/s) for the
  strict count; both are reported. Cost if wrong: none (both are in the results).

## Shield details

- **Slowing for a lower limit ahead at 1.5 m/s²**, reaching it as the car's front gets there. Cost if wrong: a
  few metres of slower driving before each lower limit.
- **Holding the limit (learned car only):** while the posted limit itself is what holds the car back, the speed
  controller's set-point is raised by 80% of its steady shortfall (`india_rl_speed.hold_offset`); the recorded
  target stays the limit. **Revised after the first town drive (rl01):** the first version used a pedal model fitted
  only on the rule car (which never went above 5.5 m/s), so it raised the set-point by up to 0.77 m/s at 8 m/s, and
  it switched the raise off the moment the car crossed the limit. In CARLA the car then overshot a 6.5 m/s limit, the
  set-point dropped 0.6 m/s at once, the controller's integral reset, the throttle dipped to zero and the car's
  automatic gearbox shifted: 18 drops of about 2 m/s at -5 to -6 m/s² on the rural roads, and twice the rule
  driver's hard braking. Now the raise is 0.34 m/s at every speed from 3 m/s up (refitted on three town drives up to
  8 m/s: the car needs 0.61 throttle to hold any such speed) and fades out linearly between 0.25 m/s under and
  0.25 m/s over the limit, with a 30 s pause if the car is 0.3 m/s over for 1 s. The stand-in's pedal model was
  refitted the same way. Cost if wrong: a few tenths of a m/s below the limit, or a small overshoot that the pause
  catches.
- **Easing down before a junction (learned car only):** from 60 m out, at most `sqrt(3.5² + 2·2·(d − 20))` where d is
  the distance to the junction, so the car reaches the 3.5 m/s junction speed as the 20 m junction zone starts
  instead of braking from 8 to 2.5 m/s at once. In the stand-in this cut hard braking 11% (3.64 → 3.27 s per drive)
  and was not slower. Cost if wrong: about 0.3 s per junction.
- **Junction-entry easing turned off again (after town drive s42_rl).** Reaching the junction speed at the junction's
  entry instead of by the start of the 20 m zone saved 1.1% in the stand-in, but in the town the faster approach met
  a "wide road area" at the mouth of Irregular junction at 5.3 m/s; the car swung right while braking for its turn
  and waited 11.8 s with 1.2 m over the centre line. `DriverSettings.rl_junction_entry` stays, off by default. Cost if
  wrong: about 2 s per junction.
- **...and back on, only for a granted turn on clear road (after the list-2 final drive).** Comparing the list-2
  drives leg by leg, junction-entry easing had saved about 69 s of junction crawling over four legs (8% of the
  drive); the s42 failure needed a turn that was not yet granted and a wide junction mouth. The car now eases to the
  junction entry only when the sat-nav reports no wait (`hold_at_m` is None) and the road is neither wide nor
  unclear; otherwise it slows by the start of the 20 m zone as before. Cost if wrong: a fast approach to a junction
  whose turn is granted but whose mouth the road model reads as normal could still swing wide.
- **The gearbox's hunting band (after s42_rl):** the camera car's automatic gearbox shifts up at about 5.9 m/s and
  loses ~2 m/s doing it, so asked to hold 6-6.8 m/s it hunts between 3.9 and 6.4 m/s every 2.5 s (hard braking twice
  the rule driver's). The rule car cruises at 5.5 m/s and never meets it. On streets posted 5.6-7.3 m/s the learned
  car now aims for 5.6 m/s (`india_rl_speed.cruise_limit`, settling near 5.45 m/s with the hold), easing into it the
  same way as into a lower limit; on the 8 m/s main road it passes the shift once and holds 7.8 m/s. This is about
  this simulated car's drivetrain, not a traffic rule. The stand-in does not model the hunting itself. Cost if
  wrong: the rural roads are driven at the rule driver's speed, not their limit.
- **The user's decision (12 Nov, morning): no gearbox.** The camera car now has one fixed ratio
  (`india_combined_scene.single_gear`: the lowest original gear that still reaches 10 m/s at the engine's maximum
  rpm, instant changes), for the rule driver and the learned car alike; `--keep-gearbox` restores the stock box.
  The gear band is off (`india_rl_speed.GEAR_BAND = None`), so the learned car aims for the posted limit on the
  rural roads again. Every drive before this used the stock gearbox, so new rule baselines are needed, and the
  limit hold's 0.34 m/s was measured on the geared car: it is re-measured on the first single-gear rule drive
  before the learned car drives.
- **Within 10% of the shield the shield's reason is still shown** (e.g. "person ahead"), below that
  "learned_slow". Cost if wrong: none (the fraction is in every decision record as `rl`).

## Training

- **Trained in a stand-in for the town, not CARLA** (`rl/speed_env.py`): a 900 s town drive takes ~21 minutes; the
  stand-in runs ~40,000 decisions a second. Its pedestrian density was cut after the first version made people
  bind the speed 33% of the time (town runs: 1.8%); jaywalkers follow the town's own gap rule. Cost if wrong: the
  policy learns from a town that is easier or harder than CARLA's; the shield bounds what that can do.
- **Kept policy = the fastest evaluated checkpoint with no more collisions, near misses, rule breaks, time over the
  limit or timeouts than the rule driver** on 330 held-out drives (the 110 place-to-place routes, seeds never
  trained on). Cost if wrong: a slightly slower checkpoint might generalise better.
- **Common random numbers:** every random draw in the stand-in is a hash of (episode seed, time step or item), so
  the rule driver and the learned car meet the same town. Cost if wrong: none.

## Checks

- **Two existing checks behave differently with the learned speed on and were not changed:** they ask for the
  rule driver's exact behaviour (its reason label behind a car 12 m ahead, and stopping 1-2.5 m short of a
  junction; the first policy stopped 2.7 m short). The latest policy passes all 42 destination checks and all but one
  camera-driving check with the learned speed on; `setup/verify_rl_speed.py` checks the shield itself. Cost if
  wrong: none (the rule driver's checks still run unchanged with the default settings).

- **A policy trained with ten times the penalty on hard and emergency braking (run 06) drove the same as the others**
  (164.8 s, 3.64 s of hard braking against 3.63 s): in the stand-in the hard braking comes from things the car only
  sees late (people, phantom vehicles and road ends), which no speed choice anticipates. Not used.
- **The gearbox drop is modelled in the stand-in** (a throttle that dips to zero above 2.5 m/s brings 0.5 s at
  -5 m/s²), so training is penalised for set-point jumps. Cost if wrong: none for the shield.
- **2.1 s on the wrong side in the list-2 learned drive** happened at 1.4-2.8 m/s while steering out of the Outskirts
  bend junction; the rule car passed the same spot at 2.3 m/s with the same 0.9 m lateral offset. Treated as the
  junction path's tracking, not the learned speed. Cost if wrong: a speed-related steering problem left unfixed.

## Looked at and not done

- **Learned route choice** from the town's junction waits (`rl/junction_times.py`): with the 19 earlier runs'
  waits, 24 of the 110 place-to-place routes would change (all round-the-block loops), saving about 12 s of ~500 s
  each. After the bend-sharing fix (dbb01f4) the list-1 rule drive waited only 18.5 s for turns in 900 s, so this
  is now worth even less. Not built.
- **Overtaking slow vehicles on the divided main road** (two lanes each way) would save the most time behind
  slower traffic but is a new, safety-critical manoeuvre; left for the user to decide.
