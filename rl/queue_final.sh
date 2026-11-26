#!/usr/bin/env bash
# Final-version learned drives against the existing rule baselines (12 Nov night): after the seed-7 pair.
cd "$(dirname "$0")/.."
MODES=rl bash rl/queue_town_pairs.sh runs/camera_driving_20261112_list1_s7_rl 20261112v3_list1 "Residential square;Outskirts bend;Irregular junction" 8101 42
MODES=rl bash rl/queue_town_pairs.sh - 20261112v3_list2 "Village road;Outskirts bend;Colony street;Market junction" 8101
