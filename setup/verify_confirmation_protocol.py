"""Regression checks for the fresh-model manifest and reference edge policy."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from carla_road_reference import checked_appearance, EDGE_DISAGREEMENT_POLICY
from idd_simulation_inference import load_simulation_model_spec
from setup.prepare_simulation_dataset import protected,save


def main():
    output=ROOT / "runs/simulation_confirmation_20261023_01"
    plan=json.loads((output / "plan.json").read_text())
    assert protected()==plan["protected_sha256"]
    manifest=output / "simulation_model.json"
    spec,files=load_simulation_model_spec(manifest)
    assert files["checkpoint"] == ROOT / "runs/simulation_frozen_20261022_01/best.keras"
    rejected=[]
    with tempfile.TemporaryDirectory(dir=output,prefix="protocol-check-") as directory:
        file=Path(directory) / "manifest.json"
        for field,value in (("scope","real_camera"),("training_allowed",True),
                            ("shared_real_deployment_approved",True),("threshold",.4),
                            ("checkpoint_sha256","0"*64)):
            file.write_text(json.dumps({**spec,field:value}))
            try:
                load_simulation_model_spec(file)
            except ValueError:
                rejected.append(field)
            else:
                raise AssertionError(field)
    tags=np.full((40,50),3,np.uint8)
    instance=np.zeros((40,50,3),np.uint8)
    instance[...,0]=3
    instance[...,1]=7
    config={"reference":{"appearance_actor_ids":[7]}}
    row={"appearance_pixels":2000,"semantic_instance_disagreement_pixels":0}
    appearance,ignored=checked_appearance(tags,instance,config,row)
    assert appearance.all() and not ignored.any()
    instance[0,0,0]=12
    row["semantic_instance_disagreement_pixels"]=1
    try:
        checked_appearance(tags,instance,config,row)
    except ValueError:
        rejected.append("undeclared_edge_mismatch")
    else:
        raise AssertionError("Strict captures accepted mismatching pixels")
    config["dataset_edge_disagreement_policy"]=EDGE_DISAGREEMENT_POLICY
    appearance,ignored=checked_appearance(tags,instance,config,row)
    assert ignored.sum()==1 and not appearance[0,0] and appearance.sum()==1999
    instance[0,:3,0]=12
    row["semantic_instance_disagreement_pixels"]=3
    try:
        checked_appearance(tags,instance,config,row)
    except ValueError:
        rejected.append("excessive_edge_mismatch")
    else:
        raise AssertionError("Excessive disagreement accepted")
    for name in ("predict_road.py","predict_road_video.py"):
        source=(ROOT / name).read_text()
        assert "SimulationRoadSegmenter" not in source and "LockedRoadSegmenter(" in source
    assert protected()==plan["protected_sha256"]
    result={"passed":True,"rejected_changes":rejected,"exact_mismatching_pixel_exclusion":True,
            "real_inference_default_unchanged":True,"tensorflow_loaded":"tensorflow" in sys.modules}
    assert not result["tensorflow_loaded"]
    save(output / "protocol_verification.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    main()
