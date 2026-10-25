"""Check exported labels with the real framework and build a 12-class head.

Uses synthetic fixtures and forward passes only, never real-image accuracy,
backpropagation or an optimizer. Existing model/checkpoint files are unchanged.
"""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import file_sha,load_policy,save_json


def main():
    output=ROOT/"runs/detection_readiness_20261023_01/framework_integration"
    output.mkdir(exist_ok=False)
    config=output/"ultralytics_settings"; config.mkdir()
    mpl=output/"matplotlib"; mpl.mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(config),MPLCONFIGDIR=str(mpl),MPLBACKEND="Agg",YOLO_AUTOINSTALL="False")
    import torch
    from ultralytics import YOLO,settings
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import DetectionModel
    settings.update({k:False for k in ("sync","wandb","comet","clearml","mlflow","neptune","raytune","tensorboard","dvc","hub") if k in settings})
    torch.set_num_threads(4)
    torch.manual_seed(42)
    fixture=ROOT/"runs/detection_readiness_20261023_01/data_checks_release/export"
    before={str(p.relative_to(fixture)):file_sha(p) for p in fixture.rglob("*") if p.is_file()}
    data=check_det_dataset(str(fixture/"data.yaml"),autodownload=False)
    assert list(data["names"].values())==load_policy()["names"]
    dataset=YOLODataset(img_path=data["train"],imgsz=640,batch_size=4,data=data,
                        task="detect",augment=False,cache=False)
    assert len(dataset)==2 and sorted(len(label["cls"]) for label in dataset.labels)==[0,12]
    items=[dataset[i] for i in range(2)]
    assert all(tuple(item["img"].shape)==(3,640,640) for item in items)
    assert sorted(len(item["cls"]) for item in items)==[0,12]
    for relative,digest in before.items():
        assert file_sha(fixture/relative)==digest
    weight=ROOT/"models/cache/detection/yolo11s.pt"
    digest=file_sha(weight)
    base=YOLO(str(weight))
    custom=DetectionModel(cfg=base.model.yaml,ch=3,nc=12,verbose=False)
    custom.load(base.model,verbose=False)
    custom.to("cuda:0").eval().requires_grad_(False)
    with torch.inference_mode():
        prediction=custom(torch.zeros((2,3,640,640),device="cuda:0"))
    raw=prediction[0] if isinstance(prediction,tuple) else prediction
    assert raw.shape[:2]==(2,16) and torch.isfinite(raw).all()
    assert all(p.grad is None for p in custom.parameters()) and file_sha(weight)==digest
    result={"passed":True,"training_started":False,"optimizer_steps":0,"real_images_evaluated":0,
        "json_yaml_accepted_by_pinned_framework":True,"class_order_matches_policy":True,
        "training_fixture_images":2,"negative_examples_preserved":1,"fixture_box_count":12,
        "letterbox_tensor_shape":[3,640,640],"custom_head_classes":12,
        "custom_head_output_shape":list(raw.shape),"pretrained_checkpoint_unchanged":True,
        "training_memory_checked":False,"scope":"synthetic_fixture_and_forward_only_integration"}
    save_json(output/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
