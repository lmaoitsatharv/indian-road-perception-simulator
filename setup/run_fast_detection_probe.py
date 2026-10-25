"""CPU-only paired validation screening; does not update weights or deploy settings."""
import argparse
from datetime import datetime,timezone
import json
import gc
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(output):
    os.nice(10)
    for folder in ("settings","matplotlib"):(output/folder).mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="",YOLO_CONFIG_DIR=str(output/"settings"),MPLCONFIGDIR=str(output/"matplotlib"),
        MPLBACKEND="Agg",YOLO_AUTOINSTALL="False",WANDB_DISABLED="true")
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    from ultralytics import YOLO,settings
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils import ops
    from idd_detection_data import file_sha,save_json
    from setup.verify_detection_runtime import protection
    settings.update({key:False for key in ("sync","wandb","comet","clearml","mlflow","neptune","raytune","tensorboard","dvc","hub") if key in settings})
    selection=json.loads((output/"selection.json").read_text())
    names=json.loads((ROOT/"detection_policy.json").read_text())["names"]
    before=protection()
    checkpoint_hashes={key:file_sha(output/(key+"_snapshot.pt")) for key in ("starting","latest")}
    class ProbeValidator(DetectionValidator):
        def init_metrics(self,model):
            super().init_metrics(model);ProbeValidator.latest=self
            self.size_targets=np.zeros((12,3),dtype=int);self.size_found=np.zeros((12,3),dtype=int)
        def update_metrics(self,predictions,batch):
            super().update_metrics(predictions,batch)
            for index,prediction in enumerate(predictions):
                truth=self._prepare_batch(index,batch)
                cls=truth["cls"].cpu().numpy().astype(int)
                boxes=ops.scale_boxes(truth["imgsz"],truth["bboxes"].clone(),truth["ori_shape"],ratio_pad=truth["ratio_pad"])
                wh=(boxes[:,2:]-boxes[:,:2]).cpu().numpy()
                bins=np.digitize(np.round(np.sqrt(wh.prod(axis=1))*640/max(truth["ori_shape"]),2),[32,96])
                np.add.at(self.size_targets,(cls,bins),1)
                kept=torch.nonzero(prediction["conf"]>=.25).reshape(-1)
                ious=box_iou(truth["bboxes"],prediction["bboxes"][kept]).cpu().numpy()
                predicted_cls=prediction["cls"][kept].cpu().numpy().astype(int)
                ious*=cls[:,None]==predicted_cls[None,:]
                matches=np.array(np.nonzero(ious>=.5)).T
                if len(matches)>1:
                    matches=matches[ious[matches[:,0],matches[:,1]].argsort()[::-1]]
                    matches=matches[np.unique(matches[:,1],return_index=True)[1]]
                    matches=matches[np.unique(matches[:,0],return_index=True)[1]]
                found=matches[:,0] if len(matches) else np.zeros(0,dtype=int)
                np.add.at(self.size_found,(cls[found],bins[found]),1)
    variants=[("latest_960","latest",960),("latest_1280","latest",1280),("starting_960","starting",960)]
    results={}
    for label,checkpoint,size in variants:
        started=time.monotonic();print("Starting CPU probe: "+label,flush=True)
        model=YOLO(str(output/(checkpoint+"_snapshot.pt")),task="detect")
        raw_epoch=model.ckpt.get("epoch")
        # Exported/stripped checkpoints use -1; they do not represent epoch zero.
        model_epoch=None if raw_epoch is None or int(raw_epoch)<0 else int(raw_epoch)+1
        metrics=model.val(data=str(output/"dataset/data.yaml"),validator=ProbeValidator,device="cpu",imgsz=size,
            batch=1,workers=0,half=False,conf=.001,iou=.7,max_det=300,plots=False,save_json=False,
            project=str(output),name=label,verbose=False)
        validator=ProbeValidator.latest
        assert validator.seen==selection["images"]
        assert validator.size_targets.sum(axis=1).tolist()==[selection["objects"][name] for name in names]
        classes={}
        for position,class_id in enumerate(metrics.box.ap_class_index):
            eligible=np.flatnonzero(metrics.box.p_curve[position]>=.75)
            threshold=int(eligible[np.argmax(metrics.box.r_curve[position][eligible])]) if len(eligible) else None
            classes[names[int(class_id)]]={"ap50":float(metrics.box.ap50[position]),"ap50_95":float(metrics.box.ap[position]),
                "precision":float(metrics.box.p[position]),"recall":float(metrics.box.r[position]),
                "at_precision_0_75":None if threshold is None else {"confidence":float(metrics.box.px[threshold]),
                    "precision":float(metrics.box.p_curve[position][threshold]),"recall":float(metrics.box.r_curve[position][threshold])}}
        record={"checkpoint_epoch_completed":model_epoch,"checkpoint_epoch_metadata":raw_epoch,
            "images":validator.seen,"resolution":size,
            "metrics":{str(key):float(value) for key,value in metrics.results_dict.items()},"per_class":classes,
            "size_bin_definition":"sqrt(native box area) scaled to longest image side 640: <32,32-96,>=96; not physical distance",
            "size_targets":validator.size_targets.tolist(),"size_matched_at_conf_0_25_iou_0_5":validator.size_found.tolist(),
            "cpu_speed_ms_per_image":metrics.speed,"elapsed_seconds":round(time.monotonic()-started,2)}
        save_json(output/(label+".json"),record);results[label]=record
        save_json(output/"progress.json",{"completed_variants":list(results),"training_started":False})
        print(json.dumps({"completed":label,"mAP50":record["metrics"]["metrics/mAP50(B)"],"seconds":record["elapsed_seconds"]}),flush=True)
        ProbeValidator.latest=None
        del model,metrics,validator
        gc.collect()
    report={"passed":protection()==before,"checked_utc":datetime.now(timezone.utc).isoformat(),"selection":selection,
        "results":results,"protected_files_unchanged":protection()==before,
        "checkpoint_copies_unchanged":all(file_sha(output/(key+"_snapshot.pt"))==value for key,value in checkpoint_hashes.items()),
        "cpu_only":True,"optimizer_steps":0,"training_started":False,"official_test_used":False,"control_allowed":False,
        "scope":"paired stratified development screening; not the full 17675-image score or an independent test"}
    assert report["passed"] and report["checkpoint_copies_unchanged"]
    save_json(output/"comparison.json",report)
    print(json.dumps({"passed":True,"complete":list(results)},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output.resolve())
