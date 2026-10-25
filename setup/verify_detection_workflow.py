"""Check source indexing and that incomplete/fixture data cannot start training."""
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import read_manifest,save_json
from setup.index_idd95k_detection import index_release


def rejected(function,match):
    try:
        function()
    except ValueError as error:
        assert match in str(error),str(error)
        return
    raise AssertionError("Expected rejection")


def main():
    root=ROOT/"runs/detection_readiness_20261023_01"
    output=root/"workflow_checks"
    output.mkdir(exist_ok=False)
    source=output/"sources/IDD_95kDetection"
    for i,split in enumerate(("train","val","test"),1):
        image=source/split/"leftImg8bit"/f"{i:05}"/"00001.jpg"
        image.parent.mkdir(parents=True)
        if split=="test":
            image.write_bytes(b"not-an-image-test-metadata-only")
        else:
            Image.new("RGB",(64,48),(30*i,20,100)).save(image)
            label=source/split/"Labels_json"/f"{i:05}"/"00001.json"
            label.parent.mkdir(parents=True)
            label.write_text("[]")
    manifest=output/"index.csv"
    report=index_release(source,manifest,"xyxy_edges",expected=None)
    assert report["counts"]=={"train":1,"val":1,"test":1} and report["passed"] is False
    records=read_manifest(manifest)
    assert [r["split"] for r in records]==["train","val","test"]
    assert records[2]["annotation"]=="-" and all(r["family"]=="IDD_95kDetection" for r in records)
    annotation=source/"train/Labels_json/00001/00001.json"
    annotation.unlink()
    rejected(lambda:index_release(source,output/"missing.csv","xyxy_edges",expected=None),"missing images or annotations")
    annotation.write_text("[]")
    duplicate=source/"train/leftImg8bit/00001/00001.jpeg"
    Image.new("RGB",(64,48),(1,2,3)).save(duplicate)
    rejected(lambda:index_release(source,output/"ambiguous.csv","xyxy_edges",expected=None),"Ambiguous duplicate image key")
    fixture=root/"data_checks_final/export"
    result_path=output/"fixture_readiness.json"
    process=subprocess.run([sys.executable,str(ROOT/"train_detection.py"),"--dataset",str(fixture),"--output",str(result_path)],
        cwd=ROOT,capture_output=True,text=True)
    assert process.returncode==2,process.stdout+process.stderr
    readiness=json.loads(result_path.read_text())
    assert readiness["ready_for_explicit_training_start"] is False and readiness["training_started"] is False
    assert readiness["checks"]["prepared_real_dataset"] is False
    # This explicit start attempt must be rejected before a run directory exists.
    denied=ROOT/"runs/detection_train_REJECTED_FIXTURE"
    assert not denied.exists()
    attempt=subprocess.run([sys.executable,str(ROOT/"train_detection.py"),"--dataset",str(fixture),
        "--start-training","--output",str(denied)],cwd=ROOT,capture_output=True,text=True)
    assert attempt.returncode!=0 and "Training refused:" in attempt.stderr and not denied.exists()
    save_json(output/"verification.json",{"passed":True,"training_started":False,
        "checks":["observed_95k_layout_preserves_provider_splits_and_drive_ids",
                  "unlabelled_test_pixels_never_opened_or_turned_into_negatives",
                  "missing_annotations_and_ambiguous_image_keys_rejected",
                  "fixture_scope_rejected_by_training_readiness",
                  "explicit_start_on_fixture_rejected_before_creating_run"],
        "incomplete_download_ready":False,"fixture_training_run_created":False})
    print((output/"verification.json").read_text())


if __name__=="__main__":
    main()
