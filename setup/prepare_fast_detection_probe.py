"""Prepare an isolated, fixed validation probe without changing active training data."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import file_sha,save_json
from project_runtime import linux_path


def main(output,count):
    import numpy as np
    if output.parent!=(ROOT/"runs").resolve() or not 96<=count<=512:
        raise ValueError("Use a fresh project run and a bounded diagnostic sample")
    output.mkdir(exist_ok=False)
    dataset=ROOT/"dataset/idd117k_detection_v3"
    cache=np.load(dataset/"labels/val.cache",allow_pickle=True).item()
    preparation=json.loads((dataset/"preparation.json").read_text())
    assert len(cache["labels"])==preparation["counts"]["val"]==17675
    rng=np.random.default_rng(20261025)
    chosen=set()
    for class_id in range(12):
        available=[i for i,row in enumerate(cache["labels"]) if class_id in row["cls"].reshape(-1)]
        chosen.update(int(i) for i in rng.choice(available,size=min(8,len(available)),replace=False))
    remaining=np.array(sorted(set(range(len(cache["labels"]))) - chosen))
    chosen.update(int(i) for i in rng.choice(remaining,size=count-len(chosen),replace=False))
    images=output/"dataset/images/val";labels=output/"dataset/labels/val"
    images.mkdir(parents=True);labels.mkdir(parents=True)
    records=[];objects=np.zeros(12,dtype=int);image_counts=np.zeros(12,dtype=int)
    for index in sorted(chosen):
        row=cache["labels"][index]
        basename=Path(row["im_file"]).name
        source_image=dataset/"images/val"/basename
        source_label=dataset/"labels/val"/(Path(basename).stem+".txt")
        target_image=images/basename;target_label=labels/source_label.name
        shutil.copy2(source_image,target_image);shutil.copy2(source_label,target_label)
        assert file_sha(source_image)==file_sha(target_image) and file_sha(source_label)==file_sha(target_label)
        cls=row["cls"].reshape(-1).astype(int)
        objects+=np.bincount(cls,minlength=12);image_counts[np.unique(cls)]+=1
        records.append({"image":basename,"source_index":index,"image_sha256":file_sha(target_image),
            "label_sha256":file_sha(target_label)})
    linux_root=linux_path(ROOT)
    probe_path=linux_root+"/"+output.relative_to(ROOT).as_posix()+"/dataset"
    config={"path":probe_path,"train":linux_root+"/dataset/idd117k_detection_v3/images/train",
        "val":"images/val","names":dict(enumerate(preparation["names"]))}
    # JSON is a valid YAML subset, avoiding an additional preparation dependency.
    (output/"dataset/data.yaml").write_text(json.dumps(config,indent=2)+"\n")
    run=ROOT/"runs/detection_train_20261024_04"
    with (run/"results.csv").open() as stream:
        history=list(csv.DictReader(stream))
    source=run/"weights/best.pt"
    before=source.stat()
    shutil.copy2(source,output/"latest_snapshot.pt")
    after=source.stat()
    assert (before.st_mtime_ns,before.st_size)==(after.st_mtime_ns,after.st_size)
    assert file_sha(source)==file_sha(output/"latest_snapshot.pt")
    baseline=ROOT/"runs/detection_train_20261024_02/weights/best.pt"
    shutil.copy2(baseline,output/"starting_snapshot.pt")
    assert file_sha(output/"starting_snapshot.pt")=="57f1d54b93c794677d26fd22773284de2752d878477bb9ac04a761cc7f035153"
    save_json(output/"selection.json",{"passed":True,"images":count,"population_images":17675,
        "selection":"fixed seed; minimum eight validation images per class, then uniform fill; development screening only",
        "seed":20261025,"objects":dict(zip(preparation["names"],objects.tolist())),
        "class_images":dict(zip(preparation["names"],image_counts.tolist())),"records":records,
        "latest_snapshot_sha256":file_sha(output/"latest_snapshot.pt"),
        "baseline_snapshot_sha256":file_sha(output/"starting_snapshot.pt"),
        "epoch3_metrics":next(row for row in history if int(row["epoch"])==3),
        "last_completed_metrics_at_snapshot":history[-1],"official_test_used":False,"training_data_changed":False,
        "created_utc":datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"passed":True,"images":count,"objects":int(objects.sum()),
        "latest_epoch_at_copy":history[-1]["epoch"],"output":str(output)},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--count",type=int,default=192)
    args=parser.parse_args();main(args.output.resolve(),args.count)
