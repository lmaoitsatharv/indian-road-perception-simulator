# Historical MATLAB traffic simulation and replay

In the MATLAB Command Window:

```matlab
addpath(fullfile(pwd, 'integration', 'matlab'));
launch_indian_traffic_3d
```

The camera menu offers **Follow car**, **Driver view** (right-hand driver's position), and **Overview**. Play/Pause, Reset, the timeline and 1x/4x/16x affect playback. Repeat replays the same recorded experiment.

To see the controlled overtaking demonstration:

```matlab
launch_indian_traffic_3d('overtaking')
```

The default is now the reviewed five-minute v8 seed-42 mixed-traffic replay.
Its selection is stored separately in `runs/review_20261013/selected_replay.json`.
The three reviewed mixed seeds had zero independently sampled body overlaps and
no NPC motion holds. Rare emergency jerk remains high. Mixed trials disabled
overtaking; the separate controlled overtaking demonstration tests that behavior.
To open the selected mixed scene explicitly:

```matlab
launch_indian_traffic_3d('pedestrians')
```

To compare with the preserved v6 seed-42 baseline:

```matlab
launch_indian_traffic_3d('baseline')
```

The teal ego car uses actual simulated steering and acceleration. Yellow vehicles are auto-rickshaws, purple vehicles are two-wheelers, and blue vehicles are other cars. Pedestrians walk on footpaths and may cross the road. Buildings and trees are visual decoration; physical road and actor footprints are explicitly checked by the simulator.

This is an illustrative synthetic environment and replay of saved simulation states. It is not a calibrated Indian traffic model, a camera-fed autonomous car, or an RL-trained policy. STOP/SLOW/PROCEED and overtaking are currently rules. The intended RL behaviour layer remains a separate future stage.

## Evidence and improvements

See `runs/traffic_overnight_20261004/MORNING_REPORT.md` and `COMPARISON.md` in that folder. Each trial has its config, copied source code, initial placement check, result, event list, full actor states and ego trace. Independent audits identify overlapping bodies and their timestamps, plus speed, waiting and braking metrics for all actors. V6 also logs each driving decision's reason and actor ID. Failed runs remain available.

The 13 October review verified autoplay, pause, reset, seeking, repeat, all three
cameras, playback speed and cleanup in MATLAB R2026a. Paused viewers stop their
own timer; closing or deleting one viewer removes only that timer. Damaged or
non-finite recordings are rejected before a figure is created. Verification is
saved in `runs/review_20261013/viewer_verify_03/verification.json`.

The reviewed public `run_indian_traffic_world` entry point delegates to the same
v8 implementation and settings used by the mixed regressions. It no longer
contains a second, older simulator. Generating another seed remains a new
experiment; replaying a recording does not regenerate traffic.
