"""Explicit native-CARLA observation scope, separate from recorded and real use."""
import json
from pathlib import Path

from idd_inference import ROOT,SHAPE,LockedRoadSegmenter,file_sha256,load_final_lock


def load_live_spec(path):
    path=Path(path).resolve(strict=True)
    runs=(ROOT / "runs").resolve()
    if not path.is_relative_to(runs) or path.stat().st_size>100_000:
        raise ValueError("Live model manifest must be a small local runs/ artifact")
    spec=json.loads(path.read_text())
    expected={"schema":"simulator_live_observer_v1","scope":"native_carla_observation_only",
        "training_allowed":False,"control_allowed":False,"shared_real_deployment_approved":False,
        "input_shape":list(SHAPE),"threshold":.5,"normalization":"RGB /127.5 -1 exactly once"}
    for key,value in expected.items():
        if type(spec.get(key)) is not type(value) or spec[key]!=value:
            raise ValueError("Live observer protocol differs: "+key)
    files={}
    for key in ("checkpoint","confirmation"):
        relative=Path(spec[key])
        target=(ROOT / relative).resolve(strict=True)
        if relative.is_absolute() or not target.is_relative_to(runs) or file_sha256(target)!=spec[key+"_sha256"]:
            raise ValueError("Live observer file/hash differs: "+key)
        files[key]=target
    confirmed=json.loads(files["confirmation"].read_text())
    if confirmed["status"]!="completed" or confirmed["checkpoint_sha256"]!=spec["checkpoint_sha256"]:
        raise ValueError("Live checkpoint lacks its completed recorded confirmation")
    baseline,_=load_final_lock()
    if spec["base_checkpoint_sha256"]!=baseline["checkpoint_sha256"]:
        raise ValueError("Protected real-model anchor differs")
    return spec,files


class LiveSimulationRoadSegmenter(LockedRoadSegmenter):
    def __init__(self,manifest):
        spec,files=load_live_spec(manifest)
        super().__init__(device="gpu",gpu_memory_mib=1024)
        with self.tf.device(self.device):
            self.model=self.tf.keras.models.load_model(files["checkpoint"],compile=False,safe_mode=True)
        if tuple(self.model.input_shape[1:])!=SHAPE or tuple(self.model.output_shape[1:])!=(512,512,1):
            raise ValueError("Live model input/output shape differs")
        self.model.trainable=False
        self.checkpoint=files["checkpoint"]
        self.lock={**self.lock,"checkpoint":spec["checkpoint"],"checkpoint_sha256":spec["checkpoint_sha256"],"scope":spec["scope"]}
        self.runtime.update(checkpoint_sha256=spec["checkpoint_sha256"],scope=spec["scope"],
                            training_allowed=False,control_allowed=False,
                            model_manifest_sha256=file_sha256(manifest))
