# Atharv and Sarthak: balanced module ownership and viva preparation

This is an allocation of responsibility for understanding, review, validation and remaining implementation. It is not a reconstructed authorship log or a claim that either member independently wrote every existing line. Record completed personal contributions only after the named member has actually done and can demonstrate them. Do not invent commits, tests or experiments.

This keeps the established division: Atharv leads perception/data; Sarthak leads simulation, planning and control. Both study the whole system. Large amounts of code, training time and scenery do not by themselves measure one member's contribution.

## Balanced responsibility matrix

| Work area | Atharv leads | Sarthak leads | Shared check |
| --- | --- | --- | --- |
| Requirements | Camera-perception problem and dataset suitability | Driving/navigation requirements and scenario coverage | Explain current versus planned pipeline |
| Data | Real-road pairing, masks, splits; IDD detector conversion and deduplication | Synthetic capture, synchronization, geometry and validity labels | Explain why train/validation/test must stay separate |
| Models | U-Net/MobileNetV2, YOLO11s, losses, augmentation, training and checkpoints | Model input/output requirements for simulator integration and timing | Interpret each result without mixing metrics |
| Simulation | Review camera appearance and scene-label compatibility | Connected maps, actors, scenery, native physics and interface | Trace a camera frame and a vehicle command |
| Driving | Define observations needed from perception | Rule baseline, route following, overtaking, junction handling and independent braking | Explain why camera observation is not control |
| Evaluation | Road IoU/Dice, detector AP and class weaknesses | Completion, collisions, clearances, waiting, jerk and frame age | Review one failure and one success together |
| Remaining development | Better weak-class examples; live detector/tracker observations | Destination routing; perception-to-speed bridge; overtaking refinement | Matched scenario validation and final report |
| Milestone material | Data/model figures and explanations | Simulation/control architecture and evidence | Edit report, cross-question and rehearse together |

## Atharv's preparation tasks

1. Open `idd_data.py` and explain `read_pairs`, `decode_pair`, `augment_pair` and `normalize_pair` in everyday words.
2. Calculate the normalization of 0, 127.5 and 255; explain why a mask value of 1 must not become 1/255.
3. Trace an image through the U-Net: input, encoder, skip connections, decoder and probability head.
4. Calculate one BCE/Dice/IoU example and one normalized bounding box by hand.
5. Explain v2-to-v3 detector cleaning: 67,159 - 5,045 - 5,385 = 56,729. Explain what happens when identical pixels have different annotations.
6. Read the latest paired detector results; identify the parent, final checkpoint, resolution and three regressing classes.
7. Explain a batch, epoch, gradient, optimizer step, learning rate, transfer learning, frozen layer and checkpoint.
8. Give Sarthak a five-minute explanation of the perception pipeline without reading the guide.

## Sarthak's preparation tasks

1. Open `india_combined_map.py` and explain nodes, streets, lane direction and junction connectors.
2. Trace the combined launcher through the owned session runner, scene, traffic and low-level control functions.
3. Explain how waypoint routes become throttle, brake and steering, and how native physics advances the car.
4. Open the earlier `india_behavior.py` overtaking logic; explain leader, oncoming gap, pedestrian blocker, passing and returning.
5. Explain RGB/semantic/instance/depth sensors and why their frame IDs and poses must match.
6. Explain road/valid targets, the ignored edge region and why validity is not a model input.
7. Describe live latest-frame buffering, stale/offline states, and the difference between GPU inference time and full frame age.
8. Explain the older MATLAB Hybrid A*/Stanley reference as a separate reference, not the active CARLA planner.
9. Give Atharv a five-minute explanation of the simulator and remaining navigation integration.

## Both members must answer

- What exact problem do the two model types solve?
- Which three datasets exist, and what are their splits?
- Which weights actually changed, and how do we know learning happened?
- Which results are validation, real test, synthetic test or selected scenario results?
- Does the camera model control the car today? What still needs connecting?
- Does a road mask mean the road is obstacle-free? Does a large box give distance in metres?
- What failed, what was preserved, and why was a higher score not automatically deployed?
- Which features remain: tracking/range, perception-driven decisions, selectable destinations, combined-town passing and later RL?

## Suggested presentation handoff

Atharv introduces the problem and data/model pipeline. Sarthak explains how the simulator is built and how the current driver works. Atharv presents model results and limitations; Sarthak presents the driving evidence and remaining integration. Both answer the other person's basic questions. Adjust speaking time to your comfort, and share the technical preparation equally even if one person does more of the presenting.

Use role descriptions such as "I am responsible for reviewing and explaining the data/perception module" or "I am responsible for simulation and planning integration." Only change these into past-tense personal implementation claims when they match work actually performed.

## Two-day study sequence

**First session:** both read chapters 1-5 of the beginner guide; Atharv continues data/model chapters while Sarthak continues simulation/control chapters. Each completes the calculation/code-trace exercises above.

**Second session:** exchange explanations. The listener asks why each design choice exists, what a failure would look like and which saved evidence supports the answer. Re-open the actual source rather than memorizing an answer disconnected from it.

**Final session:** rehearse a short progress walkthrough and the viva questions. Keep the latest results table visible. Say "planned" when a feature is planned and "not measured yet" when evidence is absent.
