# Playback controls — checked 19 October 2026

The native CARLA Indian neighbourhood supports **1×, 1.5×, 2×, 2.5× and 3×**.
Use **+ / -** to step through the speeds, or **1 / 2 / 3** directly. Replay (**R**)
keeps the selected speed. The default launch stays at 1×.

```powershell
.\Start_Indian_Traffic.ps1 -Seed 7 -Speed 3
```

The HUD reports the selected speed and recent measured simulation seconds per real
second. Measurement resets on a speed change or pause; paused time is excluded.
Startup, rendering, screenshots and hardware load can reduce the actual rate.
The controller, vehicle targets, collision checks and fixed 0.05-second physics
step are unchanged. This follows CARLA's [fixed-step synchronous approach](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/).

## Actual verification

- Four local pacing/measurement checks passed.
- Native Pygame key events exercised every speed, plus/minus, direct 1/2/3,
  pause/resume, camera changes, screenshots and replay.
- The first episode replayed at 12.95 simulated seconds. The following episode
  inherited its 2.5× selection and completed 110 simulated seconds.
- Median measured throughput at selected 3× was **3.00×** across 32 recent-window
  samples; individual windows ranged **2.20×–3.10×**. The short-window display and
  frame limiter are approximate, not a strict timing guarantee.
- The final mixed seed-7 episode reached the goal at **81.95 simulated seconds**
  and completed **one overtake**, matching the earlier 1× result. It recorded zero
  native vehicle collision events, sampled body overlaps, departures and wrong-way
  events. Independent saved-trace recounts passed.
- All recorded ego steps remained 0.05 seconds (maximum numerical discrepancy
  below one nanosecond). Camera timestamps remained aligned. Accelerated camera
  delivery is not guaranteed to supply one image for every physics step; full
  actor traces and checks still run on every simulation step.
- The bounded test controller closed its own Job Objects. An earlier
  launch was stopped through that controller after it could not reach its native
  server; its separate failed-start record was retained.

Machine-readable checks and tested source hashes are in
[playback_controls.json](results/playback_controls.json). Full local evidence and
the rendered HUD capture are under the run path recorded there. This is a finite
scenario test, not broad long-session GPU reliability or real-road safety proof.

## Next project work

Start with [visible decision reasons and the perception bridge](NEXT_STEPS.md).
This feature changes playback pacing only; the scene remains a rule-controlled
prototype using simulator positions, with no road-model driving or trained RL policy.
