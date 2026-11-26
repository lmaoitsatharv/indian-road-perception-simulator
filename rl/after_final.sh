#!/usr/bin/env bash
# After the list-2 final re-drive: run every offline check with the machine idle, then list-2 pairs for seeds 42 and 7.
cd "$(dirname "$0")/.."
until grep -q "end runs/camera_driving_20261112v3_list2_s8101_rl" runs/queue_20261112_final.log; do sleep 20; done
sleep 60
.venv-carla/Scripts/python.exe setup/verify_rl_speed.py --suites > runs/verify_rl_speed_20261112_final.log 2>&1
echo "verify exit $?" >> runs/verify_rl_speed_20261112_final.log
bash rl/queue_town_pairs.sh - 20261112v3_list2 "Village road;Outskirts bend;Colony street;Market junction" 42 7
