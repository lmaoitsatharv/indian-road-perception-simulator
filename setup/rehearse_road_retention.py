"""Separate decoder adaptation with real TRAIN teacher rehearsal and strict gates.

Preparation and metric helpers are CPU-only. TensorFlow is imported exclusively
by train, after an explicit parent GPU grant. Reserved real test paths are never
constructed. No deployment lock, retained model or previous trial is modified.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"setup"))
from adapt_road_boundaries import INITIAL, INITIAL_SHA, sha, atomic, load, protected, real_records, count_pixels, scores, weight_hash

SEED=20261021


def path_value(value):
    """Accept parent-generated Windows paths in the established WSL runtime."""
    value=str(value)
    if len(value)>2 and value[1]==":" and value[0].isalpha() and ROOT.as_posix().startswith("/mnt/"):
        return Path("/mnt/"+value[0].lower()+"/"+value[3:].replace("\\","/"))
    path=Path(value)
    return path if path.is_absolute() else ROOT/path


def verified_record(row):
    row=dict(row)
    for key in ("image","mask"):
        path=path_value(row[key]).resolve(strict=True)
        expected=row.get(key+"_sha256")
        actual=sha(path)
        if expected and actual!=expected:
            raise ValueError("Changed immutable input: "+str(path))
        row[key],row[key+"_sha256"]=str(path),actual
    return row


def strict_ratios(candidate,baseline):
    """Exact integer comparisons; no rounding allowance or epsilon tolerance."""
    def at_least(a,b,c,d):
        return b>0 and d>0 and a*d>=c*b
    a,b=candidate,baseline
    road=at_least(a["tp"],a["tp"]+a["fp"]+a["fn"],b["tp"],b["tp"]+b["fp"]+b["fn"])
    edge=at_least(a["boundary_intersection_5"],a["boundary_union_5"],b["boundary_intersection_5"],b["boundary_union_5"])
    return {"road_iou_not_lower":road,"boundary5_iou_not_lower":edge,"passed":road and edge}


def prepare(args):
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Use a new runs/ directory")
    output.mkdir(parents=True,exist_ok=False)
    prior=load(args.reuse_data/"manifest.json")
    current_protected=protected()
    if current_protected!=prior["protected_files"] or sha(INITIAL)!=INITIAL_SHA:
        raise ValueError("Changed original model/manifest")
    real=real_records()
    selected=sorted(np.random.default_rng(SEED).choice(len(real["train"]),size=args.real_train_images,replace=False))
    real_train=[verified_record(real["train"][index]) for index in selected]
    synthetic=[verified_record(row) for row in prior["train"] if row["domain"]=="synthetic"]
    development=[verified_record(row) for row in prior["development"]]
    guard=load(args.guard_manifest)
    records=guard["records"] if isinstance(guard,dict) else guard
    guards={}
    for item in records:
        row=verified_record(item)
        group=row["group"]
        row.update(domain="real_guard",name=str(row.get("name",row.get("id"))))
        guards.setdefault(group,[]).append(row)
    if sorted(map(len,guards.values()))!=[6,12]:
        raise ValueError("Expected separately declared 12 scoring and 6 diagnostic real-video frames")
    used={row["image_sha256"] for row in real_train+synthetic}
    if any(row["image_sha256"] in used for rows in guards.values() for row in rows):
        raise ValueError("A real-video guard image overlaps training inputs")
    full_val=[verified_record(row) for row in real["val"]]
    manifest={"real_train":real_train,"synthetic_train":synthetic,"real_validation":full_val,
              "development":development,"real_video_guards":guards,"protected_files":current_protected,
              "guard_manifest_sha256":sha(args.guard_manifest),"reused_data":str(args.reuse_data.resolve()),
              "reserved_test_images_read":False,"fresh_future_check_opened":False}
    atomic(output/"manifest.json",manifest)
    config={"seed":SEED,"initial_checkpoint":str(INITIAL),"initial_sha256":INITIAL_SHA,
            "epochs":args.epochs,"batch_size":4,"learning_rate":args.learning_rate,
            "real_train_images":len(real_train),"synthetic_train_images":len(synthetic),"real_val_images":687,
            "real_video_groups":{name:len(rows) for name,rows in guards.items()},
            "real_video_guard_manifest_sha256":sha(args.guard_manifest),
            "teacher":"Original locked morning-best, probabilities/logits generated from real TRAIN only",
            "teacher_logit_clip_probability":1e-4,"teacher_cache_dtype":"float32, lossless NPZ",
            "distillation_weight":args.distillation_weight,"distillation_boundary_alpha":4.,
            "real_supervised_boundary_alpha":1.,"synthetic_supervised_boundary_alpha":2.,
            "real_supervised_negative_weight":1.,"synthetic_supervised_negative_weight":1.25,
            "encoder_trainable":False,"batchnorm_trainable":False,
            "real_augmentation":"None: match exact canonical real images used by the teacher",
            "synthetic_augmentation":"Deterministic shared horizontal flip plus gentle brightness/contrast",
            "normalization":"RGB/127.5-1 exactly once","threshold":.5,"threshold_comparator":">=",
            "selection":"Strict real687, real-video scoring and diagnostic IoU AND boundary5IoU >= original (exact count ratios); synthetic DEV gain >0.5pp; maximize DEV among passing epochs",
            "previous_experimental_dev_iou":.8386963869162799,"synthetic_dev_minimum_gain":.005,
            "real_default_must_remain_original":True,"deployed":False,
            "prepared_utc":datetime.now(timezone.utc).isoformat()}
    source=output/"source";source.mkdir()
    for path in (Path(__file__),ROOT/"setup/adapt_road_boundaries.py",ROOT/"setup/boundary_metrics.py",ROOT/"carla_road_reference.py",ROOT/"idd_model.py"):
        shutil.copy2(path,source/path.name)
    config["source_sha256"]={path.name:sha(path) for path in source.iterdir()}
    atomic(output/"config.json",config)
    atomic(output/"progress.json",{"phase":"prepared_waiting_for_gpu","teacher_targets_generated":False})
    print(json.dumps({"output":str(output),"real_train":len(real_train),"synthetic_train":len(synthetic),"guards":config["real_video_groups"]}))


def decode(record):
    rgb=np.array(Image.open(record["image"]).convert("RGB"),dtype=np.uint8)
    target=np.array(Image.open(record["mask"]).convert("L"),dtype=np.uint8)
    if rgb.shape!=(512,512,3) or target.shape!=(512,512) or not set(np.unique(target))<={0,1,255}:
        raise ValueError("Invalid canonical image/mask")
    if record["domain"]=="real" and not set(np.unique(target))<={0,1}:
        raise ValueError("Changed binary real-data mask")
    return rgb.astype(np.float32)/np.float32(127.5)-np.float32(1.),target


def train(args):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
    import tensorflow as tf
    import idd_model
    output=args.output.resolve(strict=True)
    if (output/"run_status.json").exists():raise FileExistsError("Preserve completed and failed trials")
    config,manifest=load(output/"config.json"),load(output/"manifest.json")
    if protected()!=manifest["protected_files"] or sha(INITIAL)!=INITIAL_SHA:
        raise ValueError("A protected input changed")
    devices=tf.config.list_physical_devices("GPU")
    if len(devices)!=1:raise RuntimeError("Use the verified WSL GPU runtime")
    tf.config.set_logical_device_configuration(devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=5120)])
    tf.keras.utils.set_random_seed(SEED);tf.config.experimental.enable_op_determinism()
    started=time.monotonic();deadline=started+args.max_wall_seconds
    state={"phase":"running","reserved_test_images_read":False,"fresh_future_check_opened":False}
    atomic(output/"run_status.json",state)
    def bound():
        if (output/"STOP.json").exists() or time.monotonic()>deadline:raise RuntimeError("STOP marker or wall-time bound")
    def progress(phase,**fields):atomic(output/"progress.json",{"phase":phase,"elapsed_s":time.monotonic()-started,**fields})
    def evaluate(model,records,label):
        folder=output/"evaluations"/label;folder.mkdir(parents=True,exist_ok=False)
        (folder/"masks_01").mkdir();rows=[];total=None
        @tf.function(input_signature=[tf.TensorSpec((None,512,512,3),tf.float32)],jit_compile=False)
        def infer(images):return model(images,training=False)
        for start in range(0,len(records),4):
            bound();batch=records[start:start+4];decoded=[decode(row) for row in batch]
            tensor=infer(np.stack([item[0] for item in decoded]));prob=np.asarray(tensor,dtype=np.float32)
            assert tensor.device.upper().endswith("GPU:0") and np.isfinite(prob).all()
            for row,(_,truth),probability in zip(batch,decoded,prob,strict=True):
                mask=(probability[...,0]>=.5).astype(np.uint8);counts=count_pixels(truth,mask)
                total=counts if total is None else {key:total[key]+counts[key] for key in total}
                path=folder/"masks_01"/(row["name"]+".png");Image.fromarray(mask).save(path)
                rows.append({"name":row["name"],"metrics":scores(counts),"mask_sha256":sha(path)})
            if start%64==0:progress("evaluating",dataset=label,done=min(start+4,len(records)),images=len(records))
        result={"images":len(records),"metrics":scores(total),"rows":rows,"threshold":.5}
        atomic(folder/"metrics.json",result);return result
    def datasets_for(model,prefix):
        result={"real_validation":evaluate(model,manifest["real_validation"],prefix+"_real_validation"),
                "development":evaluate(model,manifest["development"],prefix+"_development")}
        result["real_video_guards"]={group:evaluate(model,rows,prefix+"_video_"+group)
                                     for group,rows in manifest["real_video_guards"].items()}
        return result
    def distillation_loss(y_true,y_pred):
        target,teacher_z,domain=[tf.cast(y_true[...,index:index+1],tf.float32) for index in range(3)]
        real=domain[:,0,0,0];valid=tf.cast(target<2,tf.float32);truth=tf.where(valid>0,target,0.)
        dilated=tf.nn.max_pool2d(truth,7,1,"SAME");eroded=-tf.nn.max_pool2d(-truth,7,1,"SAME")
        safe=-tf.nn.max_pool2d(-valid,7,1,"SAME");ring=(dilated-eroded)*safe
        alpha=tf.reshape(tf.where(real>0,config["real_supervised_boundary_alpha"],config["synthetic_supervised_boundary_alpha"]),(-1,1,1,1))
        negative=tf.reshape(tf.where(real>0,config["real_supervised_negative_weight"],config["synthetic_supervised_negative_weight"]),(-1,1,1,1))
        weight=valid*(1.+alpha*ring)*(truth+(1.-truth)*negative)
        pred=tf.cast(y_pred,tf.float32)
        bce=tf.keras.backend.binary_crossentropy(truth,pred)
        supervised=tf.math.divide_no_nan(tf.reduce_sum(weight*bce,(1,2,3)),tf.reduce_sum(weight,(1,2,3)))
        intersection=tf.reduce_sum(truth*pred*valid,(1,2,3))
        supervised+=1.-(2.*intersection+1e-6)/(tf.reduce_sum((truth+pred)*valid,(1,2,3))+1e-6)
        clipped=tf.clip_by_value(pred,1e-4,1.-1e-4);student_z=tf.math.log(clipped)-tf.math.log1p(-clipped)
        distill_weight=valid*(1.+config["distillation_boundary_alpha"]*ring)
        distillation=tf.math.divide_no_nan(tf.reduce_sum(distill_weight*tf.square(student_z-teacher_z),(1,2,3)),tf.reduce_sum(distill_weight,(1,2,3)))
        return supervised+real*config["distillation_weight"]*distillation
    try:
        # Ignored pixels contribute neither supervised nor distillation gradient.
        packed=np.zeros((2,16,16,3),np.float32);packed[:,4:12,4:12,0]=1;packed[:,6:9,6:9,0]=255
        packed[0,...,2]=1;packed[...,1]=np.log(np.float32(.4))-np.log1p(np.float32(-.4))
        pred=tf.Variable(np.full((2,16,16,1),.4,np.float32))
        with tf.GradientTape() as tape:value=tf.reduce_mean(distillation_loss(packed,pred))
        gradient=tape.gradient(value,pred).numpy();ignored=packed[...,0:1]==255
        assert (gradient[ignored]==0).all() and np.isfinite(gradient).all()
        changed=pred.numpy();changed[ignored]=.99
        np.testing.assert_allclose(distillation_loss(packed,pred),distillation_loss(packed,changed),atol=1e-6)
        atomic(output/"loss_verification.json",{"ignored_gradient_zero":True,"ignore_changes_do_not_change_loss":True,"finite_gradients":True})
        teacher=tf.keras.models.load_model(INITIAL,compile=False);teacher.trainable=False
        teacher_weights=weight_hash(teacher.weights)
        baseline=datasets_for(teacher,"baseline")
        atomic(output/"baseline.json",baseline)
        cache=output/"teacher_train_logits";cache.mkdir();cache_records=[]
        @tf.function(input_signature=[tf.TensorSpec((None,512,512,3),tf.float32)],jit_compile=False)
        def teach(images):return teacher(images,training=False)
        real_train=manifest["real_train"]
        for start in range(0,len(real_train),4):
            bound();batch=real_train[start:start+4];images=np.stack([decode(row)[0] for row in batch])
            probability=np.asarray(teach(images),dtype=np.float32)[...,0]
            clipped=np.clip(probability,np.float32(1e-4),np.float32(1.-1e-4))
            logits=(np.log(clipped)-np.log1p(-clipped)).astype(np.float32)
            for row,logit in zip(batch,logits,strict=True):
                path=cache/(row["name"]+".npz");np.savez_compressed(path,logits=logit)
                cache_records.append({"name":row["name"],"real_train_id":row["id"],"image_sha256":row["image_sha256"],
                                      "logits":str(path),"sha256":sha(path)})
            if start%64==0:progress("teacher_train_cache",done=min(start+4,len(real_train)),images=len(real_train))
        assert weight_hash(teacher.weights)==teacher_weights
        atomic(output/"teacher_cache_manifest.json",{"source_checkpoint_sha256":INITIAL_SHA,"records":cache_records,"only_real_train":True,"dtype":"float32"})
        del teacher,teach;tf.keras.backend.clear_session(free_memory=True);gc.collect()
        model=tf.keras.models.load_model(INITIAL,compile=False);encoder=model.get_layer("mobilenetv2_encoder");encoder.trainable=False
        bn=[layer for layer in model.layers if isinstance(layer,tf.keras.layers.BatchNormalization)]
        for layer in bn:layer.trainable=False
        encoder_hash=weight_hash(encoder.weights);bn_hash=weight_hash([w for layer in bn for w in layer.weights]);before=weight_hash(model.trainable_weights)
        optimizer=tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"],clipnorm=1.))
        model.compile(optimizer=optimizer,loss=distillation_loss,jit_compile=False)
        teacher_lookup={row["name"]:row for row in cache_records}
        records=real_train+manifest["synthetic_train"]
        def generator():
            for index in np.random.default_rng(SEED+epoch).permutation(len(records)):
                row=records[index];image,target=decode(row)
                packed=np.zeros((512,512,3),np.float32);packed[...,0]=target
                if row["domain"]=="real":
                    with np.load(teacher_lookup[row["name"]]["logits"],allow_pickle=False) as data:packed[...,1]=data["logits"]
                    packed[...,2]=1.
                else:
                    rng=np.random.default_rng(SEED+epoch*10000+int(index))
                    if rng.random()<.5:image,packed=np.fliplr(image).copy(),np.fliplr(packed).copy()
                    contrast=rng.uniform(.95,1.05);brightness=rng.uniform(-.04,.04)
                    mean=image.mean((0,1),keepdims=True);image=np.clip((image-mean)*contrast+mean+brightness,-1,1).astype(np.float32)
                yield image,packed
        dataset=tf.data.Dataset.from_generator(generator,output_signature=(tf.TensorSpec((512,512,3),tf.float32),tf.TensorSpec((512,512,3),tf.float32))).batch(4).prefetch(1)
        options=tf.data.Options();options.threading.private_threadpool_size=2;dataset=dataset.with_options(options)
        class Monitor(tf.keras.callbacks.Callback):
            def on_train_batch_end(self,batch,logs=None):
                bound()
                if not np.isfinite(float(logs["loss"])):raise RuntimeError("Nonfinite rehearsal loss")
                if batch%20==0:progress("training",epoch=epoch,batch=batch,loss=float(logs["loss"]))
        history=[];best_dev=baseline["development"]["metrics"]["road_iou"]+config["synthetic_dev_minimum_gain"];selected=None
        best_sim_dev=best_dev;sim_epoch=None
        for epoch in range(1,config["epochs"]+1):
            bound();fit=model.fit(dataset,epochs=1,steps_per_epoch=(len(records)+3)//4,shuffle=False,callbacks=[Monitor()],verbose=2)
            checkpoint=output/f"epoch_{epoch:02d}.keras";model.save(checkpoint,include_optimizer=False)
            measured=datasets_for(model,f"epoch_{epoch:02d}")
            gates={"real_validation":strict_ratios(measured["real_validation"]["metrics"],baseline["real_validation"]["metrics"])}
            for group in manifest["real_video_guards"]:
                gates["video_"+group]=strict_ratios(measured["real_video_guards"][group]["metrics"],baseline["real_video_guards"][group]["metrics"])
            retained=all(value["passed"] for value in gates.values());dev=measured["development"]["metrics"]["road_iou"]
            qualifies=retained and dev>best_dev
            row={"epoch":epoch,"train_loss":float(fit.history["loss"][-1]),"evaluations":measured,"strict_gates":gates,
                 "real_retention_passed":retained,"selected_at_epoch":qualifies,"checkpoint_sha256":sha(checkpoint)}
            history.append(row);atomic(output/"history.json",history)
            if qualifies:
                shutil.copy2(checkpoint,output/"candidate.keras");selected=epoch;best_dev=dev
            if dev>best_sim_dev:
                shutil.copy2(checkpoint,output/"simulation_candidate.keras");sim_epoch=epoch;best_sim_dev=dev
            print(json.dumps({"epoch":epoch,"real_iou":measured["real_validation"]["metrics"]["road_iou"],
                "real_boundary5":measured["real_validation"]["metrics"]["boundary_iou_5px"],"dev_iou":dev,"strict_gates":gates,"selected":qualifies}),flush=True)
        assert weight_hash(encoder.weights)==encoder_hash and weight_hash([w for layer in bn for w in layer.weights])==bn_hash
        assert weight_hash(model.trainable_weights)!=before
        selection={"selected_epoch":selected,"checkpoint":str(output/"candidate.keras") if selected else None,
                   "sha256":sha(output/"candidate.keras") if selected else None,"frozen_utc":datetime.now(timezone.utc).isoformat(),
                   "real_default_is_original":True,"deployed":False,"fresh_future_check_opened":False}
        atomic(output/"selection.json",selection)
        simulator_selection={"selected_epoch":sim_epoch,
            "checkpoint":str(output/"simulation_candidate.keras") if sim_epoch else None,
            "sha256":sha(output/"simulation_candidate.keras") if sim_epoch else None,
            "criterion":"Maximum DEV IoU with >0.5pp gain over original; no fresh-heldout selection",
            "approved_as_shared_model":bool(sim_epoch and history[sim_epoch-1]["real_retention_passed"]),
            "real_default_must_remain_original":True,"deployed":False,
            "frozen_utc":datetime.now(timezone.utc).isoformat(),"fresh_future_check_opened":False}
        atomic(output/"simulation_selection.json",simulator_selection)
        comparison={"baseline":baseline,"epochs":history,"selection":selection,"encoder_unchanged":True,
                    "simulator_selection":simulator_selection,
                    "batchnorm_unchanged":True,"teacher_train_only":True,"teacher_unchanged":True,
                    "decoder_updated":True,"optimizer_updates":int(optimizer.iterations.numpy()),
                    "protected_files_unchanged":protected()==manifest["protected_files"],"reserved_test_images_read":False,
                    "fresh_future_check_opened":False,"previous_experimental_dev_iou":config["previous_experimental_dev_iou"]}
        atomic(output/"comparison.json",comparison)
        if not comparison["protected_files_unchanged"]:raise RuntimeError("Protected input changed")
        if selected:
            # Reload with the identical four-image inference batch to validate
            # serialization independently of legitimate batch-shape roundoff.
            del model,encoder,bn,optimizer,dataset,fit
            tf.keras.backend.clear_session(free_memory=True);gc.collect()
            reloaded=tf.keras.models.load_model(output/"candidate.keras",compile=False)
            @tf.function(input_signature=[tf.TensorSpec((None,512,512,3),tf.float32)],jit_compile=False)
            def canary(images):return reloaded(images,training=False)
            records=manifest["real_validation"][:4];pred=(np.asarray(canary(np.stack([decode(row)[0] for row in records])))[...,0]>=.5).astype(np.uint8)
            for row,mask in zip(records,pred,strict=True):
                saved=np.array(Image.open(output/"evaluations"/f"epoch_{selected:02d}_real_validation"/"masks_01"/(row["name"]+".png")))
                np.testing.assert_array_equal(mask,saved)
            atomic(output/"reload_verification.json",{"same_batch4_masks_match":True,"canaries":4,"checkpoint_sha256":selection["sha256"]})
        state.update(phase="completed",selected_epoch=selected,elapsed_s=time.monotonic()-started)
        progress("completed",selected_epoch=selected)
    except BaseException as error:
        state.update(phase="failed_or_stopped",error=str(error),elapsed_s=time.monotonic()-started)
        raise
    finally:
        state["protected_files_unchanged"]=protected()==manifest["protected_files"]
        atomic(output/"run_status.json",state)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("prepare","train"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reuse-data",type=Path,default=ROOT/"runs/road_boundary_adaptation_20261020_03")
    parser.add_argument("--guard-manifest",type=Path)
    parser.add_argument("--real-train-images",type=int,default=1024)
    parser.add_argument("--learning-rate",type=float,default=1e-6)
    parser.add_argument("--distillation-weight",type=float,default=4.)
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--max-wall-seconds",type=int,default=1200)
    args=parser.parse_args()
    if args.mode=="prepare":
        if args.guard_manifest is None:parser.error("Preparation needs the independent real-video guard manifest")
        if not 512<=args.real_train_images<=2048 or not 0<args.learning_rate<=3e-6 or not 1<=args.epochs<=3:
            parser.error("Use bounded rehearsal dimensions")
        if not 0<=args.distillation_weight<=20:parser.error("Distillation weight must be0..20")
        prepare(args)
    else:
        if not 60<=args.max_wall_seconds<=1800:parser.error("Use a60..1800-second bound")
        train(args)


if __name__=="__main__":main()
