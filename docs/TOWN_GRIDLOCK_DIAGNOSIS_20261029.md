# Connected-town gridlock: diagnosis and fix (29 October 2026)

**What happened.** (Fix and test results: see the end of this note.) In the completed 900 s run of 27 October (`runs/carla_indian_worlds_live_20261027T032526_345363Z_seed8101`), all 25
vehicles are stopped continuously from 786.85 s to the end; the ego car last moved at about 662 s. The run still reports `passed`, because the
pass check (`integration/india_combined_scene.py`) looks at contacts and road departures, not progress. The 650 s run in the docs ends just
before this happens. **This section is the original read-only diagnosis.** (Area: simulation/planning.)

## Evidence from the final frame (`setup/analyse_town_gridlock.py`)

| Queue (road, lane) | Vehicles | Front vehicle | Front vehicle is waiting for |
| --- | ---: | --- | --- |
| road 2, lane 2 | 8 | other_vehicle_5 | a junction token |
| road 4, lane 1 | 6 | car_8 | a junction token |
| road 5, lane -1 | 6 | bicycle_13 | a junction token |
| road 9, lane 1 | 3 | bicycle_20 | a junction token |
| road 3, lane -1 | 1 | car_14 | a junction token |
| road 2, lane 3 | 1 | bus_24 | its bus stop |

- Every follower is held by the car directly ahead (gaps all about 4.1 m), and **every queue ends at a vehicle waiting for a junction
  token**. There is no vehicle-only loop: the lock runs *through the junction tokens*.
- No vehicle is moving, and the token holders are not among the five front vehicles (a holder never reports `junction_wait`). So each
  junction token must be held by a vehicle that **cannot move**: a vehicle further back in some queue.

## Likely cause in `integration/india_combined_traffic.py` (lines ~283–314)

1. **Tokens go to vehicles that are not at the front.** A junction token is granted to the earliest-arriving vehicle anywhere within
   35 m of the junction (`-5 < distance < 35`, FIFO on `waiting_since`). Queued cars are only about 8 m apart, so the 2nd–4th cars in a
   queue are inside that zone too. If one of them has an earlier `waiting_since` (for example a car that was in the zone before another
   vehicle turned in ahead of it from a nearby junction), it gets the token, but it cannot move because the car in front is waiting for
   that same token. That is a lock with no way out.
2. **No "don't block the junction" check.** A holder is never asked whether its exit road has space. In a town made of loops, full exit
   roads plus single tokens can form a circular wait across junctions.
3. **No recovery.** A token is released only when its holder passes through, so a stuck holder keeps it forever.

Point 1 is an inference from the code and the final frame; reservation state is not written to the trace, so the exact holders are not proven.

## Proposed fix (as first written; now implemented, see the end)

1. Grant a token only to the **front vehicle of each approach**: no same-direction body between it and the junction.
2. Grant it only if the **exit has room** (no body within about one car length plus a margin past the junction on the holder's route).
3. **Watchdog:** if a holder has not moved 0.5 m in 10 s while still before the junction, release the token and give it to another approach.
4. **Log reservations** (holder per node) in the trace, so a future lock can be proven, not inferred.
5. **Fail the run on gridlock:** add a progress check to the pass gate, for example "fail if every vehicle is stopped for more than 60 s"
   and "ego must keep making progress".

**How to check the fix:** rerun the same seed 8101 for 900 s, then two or three other seeds, and confirm that vehicles keep moving after
786 s, with zero contacts, zero departures and no new close-gap problems. Needs CARLA and the GPU (after the detector run is finished).

## Fix (29 October 2026, implemented; verification partly done)

Changed `integration/india_combined_traffic.py` (junction tokens) and `integration/india_combined_scene.py` (pass gate). Simulated
traffic only; no model, map or camera change.

1. **Front vehicle only.** A token goes only to a vehicle with no same-direction body between it and the junction entry.
2. **Room past the junction.** It goes only if no stopped vehicle sits in the holder's lane just past the junction (about one body
   length plus 4 m).
3. **Watchdog.** A holder still before the junction that has not moved 0.5 m in 10 s gives the token back and rejoins the queue.
4. **Queue time bug (found while testing).** After a vehicle released a token, the old code re-entered it in that junction's queue
   with the release time, because it was still within 7 m of the junction. On its next lap it reached the junction with a time from
   minutes earlier and jumped ahead of vehicles already waiting. This is the most likely way a queued follower ended up holding a
   token in the original lock. A vehicle now joins a queue only while approaching (5 m inside to 35 m before the junction).
5. **Logging.** Each trace row has `holds_junction`; `result.json` has `junction_token_grants` and `junction_watchdog_releases`.
6. **Progress gate.** A run now fails if every vehicle is stopped for 60 s or more, or the ego car gains less than 1 m in 120 s
   (`longest_all_vehicles_stopped_s`, `longest_ego_stall_s`).

### Measured results (simulation, seed 8101; summaries from `setup/analyse_town_progress.py`)

| Run | Code | Simulated | Contacts / overlaps / departures | Vehicles moving after 786 s | Longest single-vehicle wait | Ego distance |
| --- | --- | ---: | --- | ---: | --- | ---: |
| 27 Oct (before) | original | 900 s | 0 / 0 / 0 | **0 of 25** (all stopped from 787 s) | 456 s (bicycle_13, junction) | 2,237 m |
| 29 Oct | rules 1–3, 5–6 | 900 s | 0 / 0 / 0 | **23 of 25** | 185 s (a van starved at junction C) | 2,882 m |
| 29 Oct | all rules | 465 s (run interrupted) | 0 / 0 / 0 | not reached | — | 1,469 m |
| 29 Oct | all rules | 566 s (30-min wall-clock limit) | 0 / 0 / 0 | not reached | 75 s (car_8, junction) | 1,802 m |

- The bus is excluded from "moving": it parks at its service stop by design. The minimum sampled gap stayed 0.273 m in every run,
  the same as before, so the tight-clearance limitation is unchanged.
- The 185 s wait in the second row is what exposed bug 4. In the last row, with every rule, no vehicle waited longer than 75 s
  within the first 566 s. The earlier run had already waited 63 s by that point, so this is not yet proof.
- **Seeds do not vary the traffic.** Seeds 7101, 9101 and 6101 gave byte-identical results to 8101 (the seed only picks car
  colours), so they repeat one scenario rather than testing new ones. Varying spawn positions would need a code change.
- **Still to do:** a full 900 s run with every rule, queued to start automatically after other GPU work finishes.
  The last run was about 3× slower than earlier in the day for an unknown reason; at that speed 900 s cannot finish inside the
  30-minute wall-clock limit.
