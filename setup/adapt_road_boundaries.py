"""Bounded, separate road-model adaptation. The deployed lock is read-only.

prepare uses only NumPy/Pillow and audited saved simulator cameras. train imports
TensorFlow only after the parent has released CARLA's GPU session. Test images
are never constructed as records, opened or evaluated by this tool.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "setup"))
from carla_road_reference import build_reference
from boundary_metrics import boundary, morph

INITIAL = ROOT / "runs/overnight_20261003/trial_13_shadow_fixed_bn_low_lr/best.keras"
INITIAL_SHA = "93a8faf3d43b77f0ce74d5cdbaf94124728d371c3b68b2a3386f9f72fc106725"
MANIFEST_SHA = "512cc054e2291ec5e197b0dca913ba94698cf055062b54511977816872005e90"
SEED = 20261020


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def safe_file(folder, name, expected):
    path = (folder / name).resolve(strict=True)
    if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
        raise ValueError("Capture source fingerprint/path mismatch")
    return path


def protected():
    paths = [ROOT / row["checkpoint"] for row in load(ROOT / "models/retained_models.json")["models"]]
    paths += [ROOT / "models/segmentation_final_lock.json", ROOT / "models/retained_models.json",
              ROOT / "dataset/splits/pair_splits.csv"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def real_records():
    manifest = ROOT / "dataset/splits/pair_splits.csv"
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError("Changed real-data split manifest")
    result = {"train": [], "val": []}
    counts = {"train": 0, "val": 0, "test": 0}
    seen = set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            pair_id, split = int(row["id"]), row["split"]
            if pair_id in seen or split not in counts:
                raise ValueError("Invalid split metadata")
            seen.add(pair_id)
            counts[split] += 1
            if split not in result:
                continue  # Never construct paths into the reserved test split.
            result[split].append({"id": pair_id, "domain": "real", "name": f"real_{pair_id}",
                "image": str(ROOT / f"dataset/splits/{split}/images/Image_{pair_id}.png"),
                "mask": str(ROOT / f"dataset/splits/{split}/masks/Mask_{pair_id}.png")})
    if counts != dict(train=5507, val=687, test=689):
        raise ValueError("Unexpected split counts")
    for rows in result.values():
        rows.sort(key=lambda row: row["id"])
    return result


def prepare_capture(source, output, role, max_frames):
    episode = Path(source).resolve(strict=True)
    camera = episode / "road_camera"
    config, summary = load(camera / "capture_config.json"), load(camera / "capture_summary.json")
    result = load(episode / "result.json")
    launcher = load(episode.parent / "launcher_result.json")
    if not (result["runtime_passed"] and result["cleanup_complete"] and launcher["owned_job_handles_closed"]
            and launcher["status"] == "completed" and config["schema_version"] == 3):
        raise ValueError("Only clean schema-3 simulator captures can enter adaptation")
    if any(summary["dropped"].values()):
        raise ValueError("Incomplete simulator capture")
    grid = camera / config["reference"]["grid"]["file"]
    if sha(grid) != config["reference"]["grid"]["sha256"]:
        raise ValueError("Changed saved lane footprint")
    footprint = np.load(grid, allow_pickle=False)["driving"]
    rows = [json.loads(line) for line in (camera / "frames.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != summary["paired_frames"]:
        raise ValueError("Incomplete paired-camera rows")
    selected = sorted(set(np.linspace(0, len(rows)-1, min(max_frames, len(rows)), dtype=int)))
    folder = output / "data" / role
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    for index in selected:
        row = rows[index]
        if max(abs(row["rgb_timestamp_s"]-row[key]) for key in
               ("labels_timestamp_s", "depth_timestamp_s", "instance_timestamp_s")) > 1e-6:
            raise ValueError("Unaligned cameras")
        rgb_path = safe_file(camera, row["rgb_file"], row["rgb_sha256"])
        labels_path = safe_file(camera, row["labels_file"], row["labels_sha256"])
        depth_path = safe_file(camera, row["depth_file"], row["depth_sha256"])
        instance_path = safe_file(camera, row["instance_file"], row["instance_sha256"])
        tags = np.array(Image.open(labels_path).convert("L"), dtype=np.uint8)
        instance = np.array(Image.open(instance_path).convert("RGB"), dtype=np.uint8)
        if not np.array_equal(instance[..., 0], tags):
            raise ValueError("Instance/semantic cameras disagree")
        ids = instance[..., 1].astype(np.uint16)+256*instance[..., 2].astype(np.uint16)
        appearance = np.isin(ids, config["reference"]["appearance_actor_ids"])
        if int(appearance.sum()) != row["appearance_pixels"]:
            raise ValueError("Owned asphalt pixel count differs")
        depth = np.load(depth_path, allow_pickle=False)["meters"]
        target, diagnostics = build_reference(tags, depth, row["camera_to_world"], config, footprint, appearance)
        target = np.array(Image.fromarray(target).resize((512, 512), Image.Resampling.NEAREST), dtype=np.uint8)
        rgb = Image.open(rgb_path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        name = f"frame_{row['frame']:07d}"
        image_path, mask_path = folder / f"{name}_rgb.png", folder / f"{name}_mask.png"
        rgb.save(image_path)
        Image.fromarray(target).save(mask_path)
        if not set(np.unique(target)) <= {0, 1, 255} or (target != 255).mean() < .85:
            raise ValueError("Unexpected geometry-reference labels/coverage")
        records.append({"name": name, "domain": "synthetic", "image": str(image_path), "mask": str(mask_path),
            "frame": row["frame"], "sim_s": row["sim_s"], "source_rgb_sha256": row["rgb_sha256"],
            "image_sha256": sha(image_path), "mask_sha256": sha(mask_path), "geometry": diagnostics})
    atomic(folder / "manifest.json", {"role": role, "source": str(episode), "capture_config": config,
        "episode_config": load(episode.parent / "config.json"), "records": records,
        "limits": "Same synthetic town. Different seed/weather/case does not establish independent real-world generalization."})
    return records


def prepare(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Candidate evidence belongs inside a new runs/ directory")
    output.mkdir(parents=True, exist_ok=False)
    if sha(INITIAL) != INITIAL_SHA:
        raise ValueError("Initial model fingerprint differs")
    captures = [args.train_capture.resolve(), args.dev_capture.resolve(), args.heldout_capture.resolve()]
    if len(set(captures)) != 3:
        raise ValueError("Training, development and withheld episodes must be distinct")
    original = protected()
    real = real_records()
    chosen = sorted(np.random.default_rng(SEED).choice(len(real["train"]), size=512, replace=False))
    train_real = [real["train"][i] for i in chosen]
    train_syn = prepare_capture(captures[0], output, "train", 110)
    dev_syn = prepare_capture(captures[1], output, "development", 60)
    # No withheld RGB/labels are decoded here. They are prepared only after the
    # selected checkpoint has been frozen and hashed by the training stage.
    manifest = {"train": train_real+train_syn, "real_validation": real["val"], "development": dev_syn,
                "heldout_source": str(captures[2]), "protected_files": original}
    atomic(output / "manifest.json", manifest)
    shutil.copy2(__file__, output / "adapt_road_boundaries.py")
    snapshot=output/"source";snapshot.mkdir()
    for path in (Path(__file__), ROOT/"carla_road_reference.py", ROOT/"idd_model.py", ROOT/"setup/boundary_metrics.py"):
        shutil.copy2(path,snapshot/path.name)
    atomic(output / "config.json", {"seed": SEED, "initial_checkpoint": str(INITIAL),
        "initial_sha256": INITIAL_SHA, "learning_rate": args.learning_rate, "batch_size": 4, "epochs": 2,
        "encoder_trainable": False, "batchnorm_trainable": False,
        "loss": "Masked per-image boundary-weighted BCE plus soft Dice; 255 ignored, radius3 alpha2, negatives1.25",
        "augmentation": "Deterministic horizontal flip and gentle brightness/contrast on train only",
        "normalization": "Raw RGB /127.5 -1 exactly once", "threshold": .5, "comparator": ">=",
        "selection": "Maximize synthetic-development IoU with real IoU drop <=0.2pp and real 5px boundary-IoU drop <=0.5pp; require synthetic gain >=1pp",
        "test_images_read": False, "real_train_images": len(train_real), "synthetic_train_images": len(train_syn),
        "real_val_images": 687, "synthetic_dev_images": len(dev_syn), "candidate_is_deployed": False,
        "heldout_opened_before_selection": False, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256":{path.name:sha(path) for path in snapshot.iterdir()}})
    atomic(output / "progress.json", {"phase": "prepared_waiting_for_gpu", "train_images": len(manifest["train"])})
    print(json.dumps({"prepared": str(output), "train": len(manifest["train"]), "val": 687, "dev": len(dev_syn)}))


def count_pixels(truth, pred):
    valid = truth != 255
    t, p = truth == 1, pred.astype(bool)
    result = {"tp": int((t & p & valid).sum()), "fp": int((~t & p & valid).sum()),
              "fn": int((t & ~p & valid).sum()), "tn": int((~t & ~p & valid).sum()),
              "ignored": int((~valid).sum()), "total": int(valid.size)}
    for radius in (5, 14):
        safe = morph(valid, radius, border=True)
        tb, pb = boundary(t, radius), boundary(p, radius)
        result[f"boundary_intersection_{radius}"] = int((tb & pb & safe).sum())
        result[f"boundary_union_{radius}"] = int(((tb | pb) & safe).sum())
    return result


def scores(counts):
    def ratio(n, d):
        return n/d if d else None
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return {**counts, "road_iou": ratio(tp, tp+fp+fn), "dice": ratio(2*tp, 2*tp+fp+fn),
        "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
        "boundary_iou_5px": ratio(counts["boundary_intersection_5"], counts["boundary_union_5"]),
        "boundary_iou_14px": ratio(counts["boundary_intersection_14"], counts["boundary_union_14"]),
        "coverage": 1-counts["ignored"]/counts["total"]}


def weight_hash(variables):
    value = hashlib.sha256()
    for variable in variables:
        value.update(np.asarray(variable.numpy()).tobytes())
    return value.hexdigest()


def examples(output, records, baseline, candidate, baseline_label, candidate_label, group):
    """Retain improvements, regressions and low-scoring cases, not only wins."""
    folder = output / "examples" / group
    folder.mkdir(parents=True, exist_ok=False)
    base_rows = {row["name"]: row for row in baseline["rows"]}
    cand_rows = {row["name"]: row for row in candidate["rows"]}
    delta = lambda row: cand_rows[row["name"]]["metrics"]["road_iou"]-base_rows[row["name"]]["metrics"]["road_iou"]
    chosen = {}
    for kind, ordered in (("worst", sorted(records,key=lambda row:cand_rows[row["name"]]["metrics"]["road_iou"])),
                          ("regression",sorted(records,key=delta)), ("gain",sorted(records,key=delta,reverse=True))):
        for record in ordered[:4]:
            chosen.setdefault(record["name"],(kind,record))
    entries = []
    for name,(kind,record) in chosen.items():
        rgb = np.array(Image.open(record["image"]).convert("RGB"))
        target = np.array(Image.open(record["mask"]).convert("L"))
        first = np.array(Image.open(output/"evaluations"/baseline_label/"masks_01"/(name+".png")))
        second = np.array(Image.open(output/"evaluations"/candidate_label/"masks_01"/(name+".png")))
        truth = np.zeros_like(rgb);truth[target==1]=255;truth[target==255]=[110,72,130]
        def overlay(mask):
            view=rgb.copy();yes=mask==1;view[yes]=(.6*view[yes]+.4*np.array([30,220,110])).astype(np.uint8)
            return view
        def error(mask):
            view=rgb.copy();view[(target==1)&(mask==0)]=[245,65,65];view[(target==0)&(mask==1)]=[40,135,250]
            return view
        base_iou,candidate_iou = base_rows[name]["metrics"]["road_iou"],cand_rows[name]["metrics"]["road_iou"]
        picture=Image.new("RGB",(1440,626),(17,25,36));draw=ImageDraw.Draw(picture)
        draw.text((10,8),f"{group}: {name} | {kind} | delta {100*(candidate_iou-base_iou):+.2f} pp",fill="white")
        panels=[("Camera",rgb),("Geometry reference (purple ignored)",truth),
                ("Baseline errors: red missed / blue extra",error(first)),
                (f"Baseline IoU {base_iou*100:.2f}%",overlay(first)),
                (f"Candidate IoU {candidate_iou*100:.2f}%",overlay(second)),
                ("Candidate errors: red missed / blue extra",error(second))]
        for index,(title,view) in enumerate(panels):
            x,y=(index%3)*480,55+(index//3)*298
            draw.text((x+8,y-18),title,fill="white")
            picture.paste(Image.fromarray(view).resize((480,270),Image.Resampling.NEAREST),(x,y))
        file=folder/(name+".jpg");picture.save(file,quality=92)
        entries.append({"name":name,"kind":kind,"baseline_iou":base_iou,"candidate_iou":candidate_iou,
                        "delta_pp":100*(candidate_iou-base_iou),"comparison":str(file.relative_to(output))})
    atomic(folder/"manifest.json",entries)
    return entries


def report(output, comparison):
    history=comparison["epochs"]
    chosen=comparison["selection"]["selected_epoch"]
    fields=("road_iou","precision","recall","boundary_iou_5px")
    lines=["# Separate road-boundary adaptation", "", "## Result", "",
           (f"Epoch {chosen} passed the predeclared real-validation retention and synthetic-development gain guards."
            if chosen else "Neither epoch passed all predeclared selection guards. The locked original remains preferred."),
           "The deployed model lock is unchanged. All three retained checkpoints and the real split manifest were preserved.",
           "", "## Full real validation (687 images)", "",
           "| Model | Road IoU | Precision | Recall | Boundary IoU 5px |", "|---|---:|---:|---:|---:|"]
    def table_row(name,metrics):
        return "| "+name+" | "+" | ".join(f"{100*metrics[key]:.3f}%" for key in fields)+" |"
    lines.append(table_row("Locked baseline",comparison["baseline_real"]))
    lines.extend(table_row(f"Epoch {row['epoch']}",row["real_validation"]) for row in history)
    lines += ["", "## Synthetic development (checkpoint selection)", "",
        "| Model | Road IoU | Precision | Recall | Boundary IoU 5px |", "|---|---:|---:|---:|---:|"]
    lines.append(table_row("Locked baseline",comparison["baseline_development"]))
    lines.extend(table_row(f"Epoch {row['epoch']}",row["synthetic_development"]) for row in history)
    lines += ["", "## Withheld synthetic recording", ""]
    if chosen:
        lines += ["Opened after selection freeze.", "",
            "| Model | Road IoU | Precision | Recall | Boundary IoU 5px |", "|---|---:|---:|---:|---:|"]
        lines.append(table_row("Locked baseline",comparison["baseline_heldout"]))
        lines.append(table_row("Selected candidate",comparison["candidate_heldout"]))
    else:
        lines += ["Not opened or evaluated because no candidate passed all selection guards. It remains available for a later independently selected candidate."]
    lines += ["", "## Protocol and limits", "",
        "- Training: 512 real TRAIN images plus an evenly spaced selection from the assigned synthetic training recording.",
        f"- Two decoder-only epochs at Adam LR {load(output/'config.json')['learning_rate']:g}, batch 4; frozen encoder and every BatchNorm layer; single [-1,1] normalization and >=0.5 threshold.",
        "- Geometry-reference labels 0/1/255; 255 contributes no loss and is excluded from evaluation. Boundary metrics also exclude the ignore-neighbourhood halo.",
        "- Candidate selection maximizes synthetic-development IoU, requiring >=1pp gain, real-validation IoU drop <=0.2pp and boundary-IoU5 drop <=0.5pp.",
        "- All 687 validation images were evaluated before training and after each epoch. No reserved 689-image test image was opened or evaluated.",
        "- The new withheld recording is decoded only for a qualifying selected checkpoint after writing selection.json. It shares the same town and route family, so it is a limited simulator check rather than independent real-world evidence.",
        "- Approximate geometry-assisted references are not human ground truth. The reference is never generated from model predictions.",
        "- All epoch checkpoints, per-image masks/counts, loss history, gains, regressions and worst cases remain available.",
        "- This candidate does not control the car and is not deployed by this experiment.", ""]
    (output/"MODEL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(11,3.2))
        epochs=[row["epoch"] for row in history]
        axes[0].plot(epochs,[row["train_loss"] for row in history],"o-");axes[0].set_title("Training loss")
        for index,(label,key) in enumerate((("Real validation IoU","real_validation"),("Synthetic development IoU","synthetic_development")),1):
            axes[index].plot(epochs,[100*row[key]["road_iou"] for row in history],"o-")
            baseline=comparison["baseline_real" if index==1 else "baseline_development"]["road_iou"]*100
            axes[index].axhline(baseline,color="gray",linestyle="--",label="locked baseline")
            axes[index].set_title(label);axes[index].legend()
        for axis in axes:axis.set_xlabel("Epoch");axis.set_xticks(epochs);axis.grid(alpha=.25)
        fig.tight_layout();fig.savefig(output/"curves.png",dpi=160);plt.close(fig)
    except ImportError:
        pass  # Exact history.json/CSV remain the authoritative curve evidence.


def train(args):
    # Parent explicitly grants the GPU slot before invoking this command.
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import idd_model  # Register the retained checkpoint's serializable objects.
    output = args.output.resolve(strict=True)
    config, manifest = load(output / "config.json"), load(output / "manifest.json")
    if (output / "run_status.json").exists():
        raise FileExistsError("Use a new candidate; do not repeat or overwrite an existing trial")
    started, deadline = time.monotonic(), time.monotonic()+args.max_wall_seconds
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) != 1:
        raise RuntimeError("Expected the verified WSL GPU runtime")
    tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=5120)])
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    state = {"phase": "initializing", "test_images_read": False}
    atomic(output / "run_status.json", state)
    def check_bound():
        if (output / "STOP.json").exists() or time.monotonic() > deadline:
            raise RuntimeError("Candidate stopped at its stop marker or wall-time bound")
    def progress(phase, **fields):
        atomic(output / "progress.json", {"phase": phase, "elapsed_s": time.monotonic()-started, **fields})
    def decode(record):
        rgb = np.array(Image.open(record["image"]).convert("RGB"), dtype=np.uint8)
        target = np.array(Image.open(record["mask"]).convert("L"), dtype=np.uint8)
        if rgb.shape != (512,512,3) or target.shape != (512,512) or not set(np.unique(target)) <= {0,1,255}:
            raise ValueError("Invalid adaptation image/mask")
        if record["domain"] == "real" and not set(np.unique(target)) <= {0,1}:
            raise ValueError("The real-data mask changed")
        return rgb.astype(np.float32)/np.float32(127.5)-np.float32(1.), target
    def evaluate(model, records, label, keep_masks=False):
        rows, totals = [], None
        folder = output / "evaluations" / label
        folder.mkdir(parents=True, exist_ok=False)
        if keep_masks:
            (folder / "masks_01").mkdir()
        @tf.function(input_signature=[tf.TensorSpec((None,512,512,3),tf.float32)], jit_compile=False)
        def infer(images):
            return model(images, training=False)
        for start in range(0, len(records), 4):
            check_bound()
            batch = records[start:start+4]
            decoded = [decode(row) for row in batch]
            probabilities = np.asarray(infer(np.stack([v[0] for v in decoded])), dtype=np.float32)
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Nonfinite prediction")
            for rec, (image, target), probability in zip(batch, decoded, probabilities, strict=True):
                mask = (probability[...,0] >= .5).astype(np.uint8)
                counts = count_pixels(target, mask)
                totals = counts if totals is None else {key: totals[key]+counts[key] for key in counts}
                row = {"name": rec["name"], "metrics": scores(counts)}
                if keep_masks:
                    path = folder / "masks_01" / (rec["name"]+".png")
                    Image.fromarray(mask).save(path)
                    row["mask_sha256"] = sha(path)
                rows.append(row)
            if start % 64 == 0:
                progress("evaluating", dataset=label, images_done=min(start+4,len(records)), images_total=len(records))
        result = {"images": len(rows), "metrics": scores(totals), "rows": rows,
                  "threshold": .5, "ignore_halo_for_boundary_metrics": True}
        atomic(folder / "metrics.json", result)
        return result
    def masked_loss(y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
        valid = tf.cast(y_true < 2, tf.float32)
        truth = tf.where(valid > 0, y_true, 0.)
        dilated = tf.nn.max_pool2d(truth, 7, 1, "SAME")
        eroded = -tf.nn.max_pool2d(-truth, 7, 1, "SAME")
        safe = -tf.nn.max_pool2d(-valid, 7, 1, "SAME")
        weight = valid*(1.+2.*(dilated-eroded)*safe)*(1.+.25*(1.-truth))
        bce = tf.keras.backend.binary_crossentropy(truth, y_pred)
        weighted = tf.math.divide_no_nan(tf.reduce_sum(weight*bce, (1,2,3)), tf.reduce_sum(weight, (1,2,3)))
        intersection = tf.reduce_sum(truth*y_pred*valid, (1,2,3))
        dice = (2.*intersection+1e-6)/(tf.reduce_sum((truth+y_pred)*valid, (1,2,3))+1e-6)
        return weighted+1.-dice
    try:
        if protected()!=manifest["protected_files"] or sha(INITIAL)!=INITIAL_SHA:
            raise RuntimeError("A protected input changed after preparation")
        # Ignored pixels must have zero influence on loss and zero gradient.
        sample = np.zeros((1,16,16,1), np.float32)
        sample[:,5:10,5:10] = 1
        sample[:,6:9,6:9] = 255
        p = tf.Variable(np.full(sample.shape,.4,np.float32))
        with tf.GradientTape() as tape:
            value = tf.reduce_mean(masked_loss(sample,p))
        gradient = tape.gradient(value,p).numpy()
        assert (gradient[sample == 255] == 0).all() and np.isfinite(gradient).all()
        changed = p.numpy(); changed[sample == 255] = .99
        np.testing.assert_allclose(masked_loss(sample,p).numpy(),masked_loss(sample,changed).numpy(),atol=1e-6)
        atomic(output / "loss_verification.json", {"ignored_pixel_gradient_zero": True,
            "changing_ignored_predictions_keeps_loss": True, "finite_loss_gradient": True})
        model = tf.keras.models.load_model(INITIAL, compile=False)
        model.get_layer("mobilenetv2_encoder").trainable = False
        bn = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
        for layer in bn:
            layer.trainable = False
        encoder = model.get_layer("mobilenetv2_encoder")
        encoder_before, bn_before = weight_hash(encoder.weights), weight_hash([w for layer in bn for w in layer.weights])
        trainable_before=weight_hash(model.trainable_weights)
        baseline_real = evaluate(model, manifest["real_validation"], "baseline_real", True)
        baseline_dev = evaluate(model, manifest["development"], "baseline_development", True)
        optimizer=tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"], clipnorm=1.))
        model.compile(optimizer=optimizer, loss=masked_loss, jit_compile=False)
        config.update(tensorflow=tf.__version__, keras=tf.keras.__version__, gpu=str(devices[0]),
                      gpu_memory_bound_mib=5120, trainable_parameters=sum(int(np.prod(v.shape)) for v in model.trainable_weights))
        atomic(output / "config.json", config)
        train_records = manifest["train"]
        def generator():
            # Shuffle lightweight indices before decoding, avoiding a multi-GB
            # shuffle buffer of full 512px float images on the laptop.
            for index in np.random.default_rng(SEED+current_epoch).permutation(len(train_records)):
                rec = train_records[index]
                image, target = decode(rec)
                yield image, target[...,None].astype(np.float32)
        dataset = tf.data.Dataset.from_generator(generator, output_signature=(
            tf.TensorSpec((512,512,3),tf.float32),tf.TensorSpec((512,512,1),tf.float32)))
        def augment(index, values):
            image, target = values
            flip = tf.random.stateless_uniform([], seed=[SEED,tf.cast(index,tf.int32)]) < .5
            image,target = tf.cond(flip,lambda:(tf.image.flip_left_right(image),tf.image.flip_left_right(target)),lambda:(image,target))
            contrast = tf.random.stateless_uniform([], seed=[SEED+1,tf.cast(index,tf.int32)],minval=.95,maxval=1.05)
            brightness = tf.random.stateless_uniform([], seed=[SEED+2,tf.cast(index,tf.int32)],minval=-.04,maxval=.04)
            image = tf.clip_by_value(tf.image.adjust_contrast(image,contrast)+brightness,-1.,1.)
            return image,target
        dataset = dataset.enumerate().map(augment,num_parallel_calls=2,deterministic=True).batch(4).prefetch(1)
        options = tf.data.Options(); options.threading.private_threadpool_size=2
        dataset = dataset.with_options(options)
        class Bound(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                check_bound()
                if not math.isfinite(float(logs["loss"])):
                    raise RuntimeError("Nonfinite training loss; preserve this failed trial")
                if batch % 20 == 0:
                    progress("training",epoch=current_epoch,batch=batch,loss=float(logs["loss"]))
        history, selected_epoch, best_dev = [], None, baseline_dev["metrics"]["road_iou"]+.01
        selected = output / "candidate.keras"
        for current_epoch in (1,2):
            check_bound()
            result = model.fit(dataset,epochs=1,callbacks=[Bound()],verbose=2)
            epoch_path = output / f"epoch_{current_epoch:02d}.keras"
            model.save(epoch_path, include_optimizer=False)
            real = evaluate(model,manifest["real_validation"],f"epoch_{current_epoch:02d}_real",True)
            dev = evaluate(model,manifest["development"],f"epoch_{current_epoch:02d}_development",True)
            retention = (real["metrics"]["road_iou"] >= baseline_real["metrics"]["road_iou"]-.002 and
                         real["metrics"]["boundary_iou_5px"] >= baseline_real["metrics"]["boundary_iou_5px"]-.005)
            improves = retention and dev["metrics"]["road_iou"] >= best_dev
            row = {"epoch": current_epoch,"train_loss":float(result.history["loss"][-1]),
                   "real_validation":real["metrics"],"synthetic_development":dev["metrics"],
                   "real_retention_guard_passed":retention,"selected_at_this_epoch":improves,"checkpoint_sha256":sha(epoch_path)}
            history.append(row);atomic(output / "history.json",history)
            if improves:
                shutil.copy2(epoch_path, selected)
                best_dev,selected_epoch=dev["metrics"]["road_iou"],current_epoch
            print(json.dumps({"epoch":current_epoch,"real_iou":real["metrics"]["road_iou"],
                "dev_iou":dev["metrics"]["road_iou"],"retained":retention,"selected":improves}),flush=True)
        assert weight_hash(encoder.weights) == encoder_before
        assert weight_hash([w for layer in bn for w in layer.weights]) == bn_before
        assert weight_hash(model.trainable_weights) != trainable_before
        optimizer_updates=int(optimizer.iterations.numpy())
        # Freeze selection before decoding any withheld capture.
        selection = {"selected_epoch":selected_epoch,"checkpoint":str(selected) if selected_epoch else None,
            "sha256":sha(selected) if selected_epoch else None,"frozen_utc":datetime.now(timezone.utc).isoformat(),
            "deployed":False,"original_remains_locked":True,"heldout_used_for_selection":False}
        atomic(output / "selection.json",selection)
        # Selection is immutable now. Release the training graph and optimizer
        # before loading each inference-only comparator; do not keep three full
        # models resident on the 8GB laptop GPU.
        import gc
        del model,encoder,bn,optimizer,dataset,result
        tf.keras.backend.clear_session(free_memory=True)
        gc.collect()
        withheld,withheld_base,withheld_candidate=[],None,None
        if selected_epoch:
            withheld = prepare_capture(manifest["heldout_source"],output,"heldout_after_selection",110)
            baseline_model = tf.keras.models.load_model(INITIAL,compile=False)
            withheld_base = evaluate(baseline_model,withheld,"baseline_heldout",True)
            del baseline_model
            tf.keras.backend.clear_session(free_memory=True)
            gc.collect()
            selected_model = tf.keras.models.load_model(selected,compile=False)
            @tf.function(input_signature=[tf.TensorSpec((None,512,512,3),tf.float32)],jit_compile=False)
            def reloaded_canary(images):
                return selected_model(images,training=False)
            canaries=manifest["real_validation"][:4]
            masks=(np.asarray(reloaded_canary(np.stack([decode(record)[0] for record in canaries])))[...,0]>=.5).astype(np.uint8)
            for record,predicted in zip(canaries,masks,strict=True):
                saved=np.array(Image.open(output/"evaluations"/f"epoch_{selected_epoch:02d}_real"/"masks_01"/(record["name"]+".png")))
                np.testing.assert_array_equal(predicted,saved)
            atomic(output/"reload_verification.json",{"selected_checkpoint_sha256":sha(selected),
                "real_validation_canaries":4,"saved_masks_match_reloaded_model":True})
            withheld_candidate=evaluate(selected_model,withheld,"candidate_heldout",True)
        comparison={"baseline_real":baseline_real["metrics"],"baseline_development":baseline_dev["metrics"],
            "epochs":history,"selection":selection,"baseline_heldout":withheld_base["metrics"] if withheld_base else None,
            "candidate_heldout":withheld_candidate["metrics"] if withheld_candidate else None,"candidate_heldout_is_selected":selected_epoch is not None,
            "withheld_images":len(withheld),"real_validation_images":687,
            "limits":"Same synthetic town/route family. Geometry-assisted labels are approximate and not human ground truth. No new real independent-test claim.",
            "encoder_unchanged":True,"batchnorm_weights_and_statistics_unchanged":True,
            "decoder_weights_updated":True,"optimizer_updates":optimizer_updates,
            "protected_files_unchanged":protected()==manifest["protected_files"],"test_images_read":False}
        atomic(output / "comparison.json",comparison)
        if not comparison["protected_files_unchanged"]:
            raise RuntimeError("A protected file changed")
        with (output/"history.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream);writer.writerow(["epoch","train_loss","real_iou","real_boundary_iou5","synthetic_dev_iou","retention","selected"])
            for row in history:
                writer.writerow([row["epoch"],row["train_loss"],row["real_validation"]["road_iou"],
                    row["real_validation"]["boundary_iou_5px"],row["synthetic_development"]["road_iou"],
                    row["real_retention_guard_passed"],row["selected_at_this_epoch"]])
        diagnostic_epoch=selected_epoch if selected_epoch else 2
        real_candidate=load(output/"evaluations"/f"epoch_{diagnostic_epoch:02d}_real"/"metrics.json")
        examples(output,manifest["real_validation"],baseline_real,real_candidate,"baseline_real",
                 f"epoch_{diagnostic_epoch:02d}_real","real_validation")
        dev_candidate=load(output/"evaluations"/f"epoch_{diagnostic_epoch:02d}_development"/"metrics.json")
        examples(output,manifest["development"],baseline_dev,dev_candidate,"baseline_development",
                 f"epoch_{diagnostic_epoch:02d}_development","synthetic_development")
        if selected_epoch:
            examples(output,withheld,withheld_base,withheld_candidate,"baseline_heldout",
                     "candidate_heldout","withheld_synthetic")
        report(output,comparison)
        state.update(phase="completed",selected_epoch=selected_epoch,elapsed_s=time.monotonic()-started)
        progress("completed",selected_epoch=selected_epoch)
    except BaseException as error:
        state.update(phase="failed_or_stopped",error=str(error),elapsed_s=time.monotonic()-started)
        raise
    finally:
        state["protected_files_unchanged"]=protected()==manifest["protected_files"]
        atomic(output / "run_status.json",state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","train"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--train-capture",type=Path)
    parser.add_argument("--dev-capture",type=Path)
    parser.add_argument("--heldout-capture",type=Path)
    parser.add_argument("--max-wall-seconds",type=int,default=1200)
    parser.add_argument("--learning-rate",type=float,default=3e-6,help="Preparation only: predeclared decoder Adam learning rate")
    args=parser.parse_args()
    if args.mode=="prepare":
        if not 0 < args.learning_rate <= 1e-5:
            parser.error("Use a bounded decoder learning rate in (0,1e-5]")
        if any(value is None for value in (args.train_capture,args.dev_capture,args.heldout_capture)):
            parser.error("Preparation needs all three distinct assigned capture directories")
        prepare(args)
    else:
        if not 60 <= args.max_wall_seconds <= 1800:
            parser.error("Use a bounded 60..1800-second run")
        train(args)


if __name__=="__main__":
    main()
