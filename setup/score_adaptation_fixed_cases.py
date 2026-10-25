"""Two predeclared, known development examples; never model selection or training."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"setup"))
from adapt_road_boundaries import count_pixels,scores,protected
from build_adaptation_gallery import picture

CASES=[("before_paving", "runs/carla_asphalt_observation_20261020_02",979,
        "runs/carla_india_smoke_20261020T005852_322596Z_seed7/episode_01_mixed_seed7/road_camera/frame_0000979_rgb.png"),
       ("after_paving", "runs/carla_paving_observation_20261020_01",947,
        "runs/carla_india_smoke_20261020T172949_077967Z_seed7/episode_01_mixed_seed7/road_camera/frame_0000947_rgb.png")]


def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
    args=parser.parse_args();out=args.output.resolve(strict=True)
    selection=load(out/"selection.json")
    if not selection["selected_epoch"] or load(out/"run_status.json")["phase"]!="completed":
        raise ValueError("Use an already selected, completed candidate")
    target=out/"fixed_cases";target.mkdir(exist_ok=False)
    model_path=out/"candidate.keras"
    if sha(model_path)!=selection["sha256"]:raise ValueError("Selected checkpoint changed")
    import tensorflow as tf
    import idd_model
    devices=tf.config.list_physical_devices("GPU")
    if len(devices)!=1:raise RuntimeError("Expected the verified GPU")
    tf.config.set_logical_device_configuration(devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
    tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism()
    model=tf.keras.models.load_model(model_path,compile=False)
    @tf.function(input_signature=[tf.TensorSpec((1,512,512,3),tf.float32)],jit_compile=False)
    def infer(images):return model(images,training=False)
    infer(tf.zeros((1,512,512,3),tf.float32))
    entries=[]
    for label,folder,frame,rgb_path in CASES:
        folder=ROOT/folder;source=ROOT/rgb_path
        rows=load(folder/"01_mixed_seed7/frames.json")
        row=next(row for row in rows if row["frame"]==frame)
        if sha(source)!=row["source"]["rgb_sha256"]:raise ValueError("Source RGB differs")
        rgb=np.array(Image.open(source).convert("RGB").resize((512,512),Image.Resampling.BILINEAR),dtype=np.uint8)
        truth=np.array(Image.open(folder/row["reference"]).convert("L"),dtype=np.uint8)
        base=np.array(Image.open(folder/row["mask"]).convert("L"),dtype=np.uint8)
        tensor=infer(rgb[None].astype(np.float32)/np.float32(127.5)-np.float32(1.))
        assert tensor.device.upper().endswith("GPU:0")
        probability=np.asarray(tensor,dtype=np.float32)[0,...,0]
        assert np.isfinite(probability).all() and 0<=probability.min()<=probability.max()<=1
        mask=(probability>=.5).astype(np.uint8)
        base_metrics,candidate_metrics=scores(count_pixels(truth,base)),scores(count_pixels(truth,mask))
        assert abs(base_metrics["road_iou"]-row["metrics"]["road_iou"])<1e-12
        image_path,mask_path=target/(label+"_rgb.png"),target/(label+"_candidate_01.png")
        Image.fromarray(rgb).save(image_path);Image.fromarray(mask).save(mask_path)
        record={"name":label,"sim_s":row["sim_s"],"image":str(image_path),"mask":str(folder/row["reference"])}
        png=target/(label+"_comparison.png")
        picture(record,folder/row["mask"],mask_path,base_metrics["road_iou"],candidate_metrics["road_iou"],png)
        entries.append({"name":label,"frame":frame,"sim_s":row["sim_s"],"baseline":base_metrics,
            "candidate":candidate_metrics,"delta_pp":100*(candidate_metrics["road_iou"]-base_metrics["road_iou"]),
            "rgb_source_sha256":sha(source),"reference_unchanged":True,
            "comparison":str(png.relative_to(out)),"mask":str(mask_path.relative_to(out)),"inference_device":tensor.device})
    report={"candidate_sha256":selection["sha256"],"selected_epoch":selection["selected_epoch"],
            "known_development_examples":True,"withheld_examples":False,"selection_changed":False,
            "training_performed":False,"threshold":.5,"normalization":"RGB/127.5-1 once", "cases":entries,
            "protected_files_unchanged":protected()==load(out/"manifest.json")["protected_files"]}
    assert report["protected_files_unchanged"] and sha(model_path)==selection["sha256"]
    (out/"fixed_case_comparison.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"cases":[{"name":e["name"],"baseline_iou":e["baseline"]["road_iou"],"candidate_iou":e["candidate"]["road_iou"],"delta_pp":e["delta_pp"]} for e in entries],"protected_files_unchanged":True}))


if __name__=="__main__":main()
