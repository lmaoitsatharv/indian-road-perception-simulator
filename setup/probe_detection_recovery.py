"""Discarded training-only smoke check for safe checkpoint recovery; no accuracy evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(output):
    output.mkdir(exist_ok=False)
    (output/"settings").mkdir();(output/"matplotlib").mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(output/"settings"),MPLCONFIGDIR=str(output/"matplotlib"),
        YOLO_AUTOINSTALL="False",WANDB_DISABLED="true",MPLBACKEND="Agg",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.data.utils import get_hash
    import ultralytics.data.dataset as dataset_module
    # The probe may build a subset index in memory, but must never publish it
    # over the full training dataset's cache.
    dataset_module.save_dataset_cache_file=lambda prefix,path,x,version:x.update(version=version)
    from detection_refinement import parallel_dataset_hash,save_full_precision_resume
    torch.set_num_threads(4); torch.manual_seed(42)
    dataset=ROOT/"dataset/idd117k_detection_v3"
    rows=[json.loads(line) for line in (dataset/"pairs.jsonl").open() if '"train"' in line]
    selected=[rows[i] for i in range(0,len(rows),max(1,len(rows)//16))][:16]
    assert len(selected)==16 and all(row["split"]=="train" for row in selected)
    paths=[str(dataset/row["prepared_image"]) for row in selected]
    hash_paths=paths+[str(dataset/row["prepared_label"]) for row in selected]+[str(output/"missing_fixture.jpg")]
    assert parallel_dataset_hash(hash_paths)==get_hash(hash_paths)
    (output/"train_images.txt").write_text("\n".join(paths)+"\n")
    recipe=json.loads((ROOT/"detection_refinement_clean_recipe.json").read_text())
    allowed={k:recipe[k] for k in ("imgsz","batch","lr0","momentum","weight_decay","box","cls","dfl","mosaic","scale") if k in recipe}
    cfg=get_cfg(overrides=allowed)
    names=json.loads((ROOT/"detection_policy.json").read_text())["names"]
    data=YOLODataset(img_path=str(output/"train_images.txt"),imgsz=960,batch_size=4,augment=True,
        hyp=cfg,rect=False,cache=False,data={"names":dict(enumerate(names)),"nc":len(names)},task="detect")
    net=YOLO(str(ROOT/"runs/detection_train_20261024_04/weights/last.pt"),task="detect").model.float().cuda().train()
    net.requires_grad_(True)
    net.criterion=None
    for name,param in net.named_parameters():
        if ".dfl." in name:
            param.requires_grad_(False)
    net.args=cfg
    holder=object.__new__(DetectionTrainer);holder.args=cfg
    optimizer=holder.build_optimizer(net,name="AdamW",lr=recipe["lr0"],momentum=cfg.momentum,decay=cfg.weight_decay,iterations=30000)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    losses=[];steps=0
    optimizer.zero_grad(set_to_none=True)
    for i in range(48):
        batch=data.collate_fn([data[(i*4+j)%len(data)] for j in range(4)])
        batch={key:(value.cuda(non_blocking=True) if isinstance(value,torch.Tensor) else value) for key,value in batch.items()}
        batch["img"]=batch["img"].float()/255
        with torch.autocast("cuda",enabled=True):
            loss,items=net(batch)
            loss=loss.sum()
        values=items.detach().float().cpu().tolist()
        assert all(torch.isfinite(torch.tensor(values))), values
        assert values[1]<20,values
        losses.append(values)
        scaler.scale(loss).backward()
        if (i+1)%16==0:
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(net.parameters(),10)
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);steps+=1
        if i%8==0:
            print(json.dumps({"batch":i+1,"losses":values,"optimizer_steps":steps}),flush=True)
    assert steps==3 and all(torch.isfinite(p).all().item() for p in net.parameters())
    (output/"weights").mkdir()
    stub=types.SimpleNamespace(last=ROOT/"runs/detection_train_20261024_04/weights/last.pt",
        model=net,optimizer=optimizer,scaler=scaler,ema=types.SimpleNamespace(ema=net))
    save_full_precision_resume(stub,output)
    complete=torch.load(output/"weights/last_resume_fp32.pt",map_location="cpu",weights_only=False)
    assert complete["project_full_precision_resume"] and complete["model"] is not None
    assert all(value.dtype==torch.float32 for state in complete["optimizer"]["state"].values()
        for key,value in state.items() if key!="step" and isinstance(value,torch.Tensor) and value.is_floating_point())
    report={"passed":True,"scope":"discarded optimizer smoke check; training split only; no validation/test evaluation",
        "training_images":len(data),"batches":len(losses),"optimizer_steps":steps,"batch":4,"imgsz":960,
        "classification_loss_max":max(v[1] for v in losses),"first_losses":losses[0],"last_losses":losses[-1],
        "all_tensors_finite":True,"parallel_cache_hash_matches_framework":True,
        "fp32_checkpoint_roundtrip_verified":True,"candidate_weights_saved":False,
        "discarded_probe_checkpoint":"weights/last_resume_fp32.pt"}
    (output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output.resolve())
