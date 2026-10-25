"""Repeat the same no-model baseline to quantify native-run variation."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from setup.run_live_simulation import native_run
from setup.prepare_simulation_dataset import save


if __name__=="__main__":
    output=ROOT / "runs/simulation_live_repeat_baseline_20261023_01"
    output.mkdir(exist_ok=False)
    previous=ROOT / "runs/simulation_live_20261023_02"
    record=native_run(output,previous / "scene_plan.json",4101,40)
    save(output / "status.json",{"status":"completed","model_control":False,"live_inference":False,
                                 "native_capture":record["launcher"]["output"]})
