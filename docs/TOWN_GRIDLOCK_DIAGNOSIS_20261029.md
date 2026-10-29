# Connected-town gridlock: diagnosis (29 October 2026, read-only, nothing changed)

**What happened.** In the completed 900 s run of 27 October (`runs/carla_indian_worlds_live_20261027T032526_345363Z_seed8101`), all 25
vehicles are stopped continuously from 786.85 s to the end; the ego car last moved at about 662 s. The run still reports `passed`, because the
pass check (`integration/india_combined_scene.py`) looks at contacts and road departures, not progress. The 650 s run in the docs ends just
before this happens. **Not fixed; no simulation code was changed.** (Area: simulation/planning.)

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

## Proposed fix (for review; not implemented)

1. Grant a token only to the **front vehicle of each approach**: no same-direction body between it and the junction.
2. Grant it only if the **exit has room** (no body within about one car length plus a margin past the junction on the holder's route).
3. **Watchdog:** if a holder has not moved 0.5 m in 10 s while still before the junction, release the token and give it to another approach.
4. **Log reservations** (holder per node) in the trace, so a future lock can be proven, not inferred.
5. **Fail the run on gridlock:** add a progress check to the pass gate, for example "fail if every vehicle is stopped for more than 60 s"
   and "ego must keep making progress".

**How to check the fix:** rerun the same seed 8101 for 900 s, then two or three other seeds, and confirm that vehicles keep moving after
786 s, with zero contacts, zero departures and no new close-gap problems. Needs CARLA and the GPU (after the detector run is finished).
