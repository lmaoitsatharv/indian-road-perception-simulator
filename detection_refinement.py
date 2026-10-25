"""Single-GPU, training-only rare-image sampling with duplicate-pixel correction."""
from collections import Counter
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import os


def parallel_dataset_hash(paths):
    """Same framework size/path hash, with concurrent Windows-mount stat calls."""
    def size(path):
        try:
            return os.stat(path).st_size
        except OSError:
            return 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        total=sum(pool.map(size,paths))
    result=hashlib.sha256(str(total).encode())
    result.update("".join(paths).encode())
    return result.hexdigest()


def save_full_precision_resume(trainer, output):
    """Publish a complete raw-model/EMA/AdamW recovery state atomically."""
    import torch
    full=torch.load(trainer.last,map_location="cpu",weights_only=False)
    full["optimizer"]=copy.deepcopy(trainer.optimizer.state_dict())
    for state in full["optimizer"]["state"].values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):
                state[key]=value.detach().cpu()
    full["model"]=copy.deepcopy(trainer.model).cpu().float()
    full["ema"]=copy.deepcopy(trainer.ema.ema).cpu().float()
    full["scaler"]=trainer.scaler.state_dict()
    full["project_full_precision_resume"]=True
    temporary=output/"weights/last_resume_fp32.tmp"
    torch.save(full,temporary)
    temporary.replace(output/"weights/last_resume_fp32.pt")


def sampling_weights(labels, image_names, manifest_rows, boosts):
    """Keep all training identities eligible; equalise pixel-group base probability."""
    import numpy as np
    if any(not str(key).isdigit() or not 0 <= int(key) < 12 or
           not np.isfinite(float(value)) or not 1 <= float(value) <= 4 for key, value in boosts.items()):
        raise ValueError("Rare-image probability boosts must name project classes and stay between 1 and 4")
    by_name = {Path(row["prepared_image"]).name: row for row in manifest_rows if row["split"] == "train"}
    groups = Counter(row["pixels_sha256"] for row in by_name.values())
    if len(by_name) != len(labels) or len(labels) != len(image_names):
        raise ValueError("Sampling inputs must cover the complete original training manifest")
    weights = []
    for label, image in zip(labels, image_names):
        row = by_name[Path(image).name]
        cls = set(int(value) for value in label["cls"].reshape(-1))
        boost = max([1.0] + [float(boosts.get(str(value), 1.0)) for value in cls])
        if not 1 <= boost <= 4:
            raise ValueError("Rare-image probability boosts must be bounded between 1 and 4")
        weights.append(boost / groups[row["pixels_sha256"]])
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Every original training identity must retain positive probability")
    return weights


def make_refinement_trainer(config, dataset_path, output, *, reset_optimizer=False):
    import numpy as np
    import torch
    from torch.utils.data import WeightedRandomSampler
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.models.yolo.detect import DetectionTrainer
    import ultralytics.data.dataset as dataset_module
    dataset_module.get_hash=parallel_dataset_hash

    if config.get("type") != "rare_images_with_pixel_group_correction" or config.get("validation_weighted") is not False:
        raise ValueError("Only explicit training-only sampling is supported")

    class RefinementTrainer(DetectionTrainer):
        def _load_checkpoint_state(self, checkpoint):
            restored=dict(checkpoint)
            if reset_optimizer:
                restored["optimizer"]=None
                restored["scaler"]=None
            super()._load_checkpoint_state(restored)
            if checkpoint.get("project_full_precision_resume") and checkpoint.get("model") is not None:
                self.model.load_state_dict(checkpoint["model"].float().state_dict())

        def save_model(self):
            super().save_model()
            # Framework checkpoints quantize AdamW moments to FP16. Preserve a
            # separate complete FP32 state for stable recovery after interruption.
            save_full_precision_resume(self,output)

        def get_dataloader(self, dataset_path_arg, batch_size=16, rank=-1, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path_arg, batch_size, rank, mode)
            if rank != -1:
                raise ValueError("This verified sampling recipe is single-GPU only")
            if config["seed"] != self.args.seed:
                raise ValueError("Sampler and training seeds must match")
            data = self.build_dataset(dataset_path_arg, mode, batch_size)
            if data.rect:
                raise ValueError("Weighted training requires rect=False")
            with (dataset_path / "pairs.jsonl").open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            weights = sampling_weights(data.labels, data.im_files, rows, config["class_image_boosts"])
            generator = torch.Generator().manual_seed(self.args.seed)
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(data), replacement=True, generator=generator)
            total = float(weights.sum())
            expected = {}
            for cls in range(len(self.data["names"])):
                mask = np.array([cls in set(label["cls"].reshape(-1).astype(int)) for label in data.labels])
                expected[self.data["names"][cls]] = {
                    "original_images": int(mask.sum()),
                    "expected_primary_samples_per_epoch": float(weights[mask].sum() / total * len(data))}
            record = {"passed": True, "training_identities": len(data), "draws_per_epoch": len(data),
                "replacement": True, "all_training_identities_have_positive_probability": True,
                "minimum_weight": float(weights.min()), "maximum_weight": float(weights.max()),
                "expected_class_exposure": expected, "configuration": config,
                "validation_sampling_changed": False, "labels_changed": False,
                "scope": "primary training images; auxiliary mosaic choices follow framework behaviour"}
            (output / "training_sampling.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            return InfiniteDataLoader(data, batch_size=min(batch_size, len(data)), shuffle=False, sampler=sampler,
                num_workers=self.args.workers, pin_memory=True, collate_fn=data.collate_fn,
                worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(self.args.seed),
                prefetch_factor=2 if self.args.workers else None, drop_last=False)

    return RefinementTrainer
