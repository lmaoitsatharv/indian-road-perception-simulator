"""Check the actual dataset configuration in the isolated detector runtime."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import load_policy,save_json,source_path
from train_detection import readiness


def main():
    output=ROOT/"runs/idd_detection_import_20261023_01/framework_check"
    output.mkdir(exist_ok=False)
    settings_dir=output/"ultralytics_settings";settings_dir.mkdir()
    mpl=output/"matplotlib";mpl.mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(settings_dir),MPLCONFIGDIR=str(mpl),MPLBACKEND="Agg",YOLO_AUTOINSTALL="False")
    import torch
    import ultralytics
    from ultralytics import settings
    from ultralytics.data.utils import check_det_dataset
    assert Path(settings.file).resolve().is_relative_to(settings_dir)
    settings.update({k:False for k in ("sync","wandb","comet","clearml","mlflow","neptune","raytune","tensorboard","dvc","hub") if k in settings})
    assert torch.__version__=="2.9.1+cu128" and ultralytics.__version__=="8.3.225"
    dataset=ROOT/"dataset/idd117k_detection_v2"
    prepared=json.loads((dataset/"preparation.json").read_text())
    assert prepared["passed"] and torch.cuda.is_available()
    data=check_det_dataset(str(dataset/"data.yaml"),autodownload=False)
    assert list(data["names"].values())==load_policy()["names"] and data["nc"]==12
    counts={s:sum(1 for _ in Path(data[s]).glob("*.jpg")) for s in ("train","val")}
    assert counts=={s:prepared["counts"][s] for s in counts}
    sampled=0
    for line in (dataset/"pairs.jsonl").open(encoding="utf-8"):
        row=json.loads(line)
        if int(row["image_sha256"][:8],16)%1337:
            continue
        info=source_path(dataset,row["prepared_image"]).stat()
        assert (info.st_size,info.st_mtime_ns//1000000)==(row["image_bytes"],row["image_mtime_ms"])
        sampled+=1
    # The exhaustive file/label audit is done on Windows; this separately checks
    # WSL paths, framework parsing and cross-platform metadata on a hash sample.
    checked=readiness(dataset,inspect_files=False)
    assert checked["ready_for_explicit_training_start"],checked
    result={"passed":True,"training_started":False,"optimizer_steps":0,"image_pixels_opened":False,
        "torch_version":torch.__version__,"ultralytics_version":ultralytics.__version__,
        "cuda_available":True,"gpu":torch.cuda.get_device_name(0),"class_count":12,"image_counts":counts,
        "wsl_image_timestamp_and_size_samples":sampled,"actual_yaml_parsed_by_framework":True,
        "class_order_matches_policy":True,"framework_test_split_configured":"test" in data,
        "full_file_audit_location":"training_readiness_windows.json","metadata_readiness":checked}
    save_json(output/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
