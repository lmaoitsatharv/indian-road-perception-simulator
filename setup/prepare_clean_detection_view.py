"""Preserve validation; remove ambiguous/repeated training pixels in a fresh view."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import MANIFEST_FIELDS, file_sha, read_manifest, save_json
from setup.verify_detection_runtime import protection


def select_pairs(rows):
    groups = defaultdict(list)
    validation = []
    for row in rows:
        if row["split"] == "train":
            groups[row["pixels_sha256"]].append(row)
        elif row["split"] == "val":
            validation.append(row)
        else:
            raise ValueError("Only previously audited train/validation pairs may enter this view")
    if set(groups) & {row["pixels_sha256"] for row in validation}:
        raise ValueError("Training pixels overlap validation")
    keep, excluded, ambiguous = set(), [], set()
    for digest, group in groups.items():
        if len({row["label_sha256"] for row in group}) > 1:
            ambiguous.add(digest)
            excluded.extend({**row, "exclusion_reason": "duplicate_pixels_with_differing_annotations"} for row in group)
        else:
            canonical = min(group, key=lambda row: row["prepared_image"])
            keep.add(canonical["prepared_image"])
            excluded.extend({**row, "exclusion_reason": "identical_pixel_and_label_duplicate",
                "canonical_prepared_image":canonical["prepared_image"]} for row in group if row is not canonical)
    selected = [row for row in rows if row["split"] == "val" or row["prepared_image"] in keep]
    return selected, excluded, ambiguous


def main(output):
    output = output.resolve()
    if output.parent != ROOT / "dataset" or output.exists():
        raise ValueError("Use a fresh dataset directory in this project")
    source = ROOT / "dataset/idd117k_detection_v2"
    original = json.loads((source / "preparation.json").read_text())
    assert original["passed"] and file_sha(source / "pairs.jsonl") == original["manifest_sha256"]
    # A small selection fixture checks the substantive exclusion behaviour.
    def row(name, pixel, label, split="train"):
        return {"prepared_image":"images/"+split+"/"+name+".jpg", "pixels_sha256":pixel,
                "label_sha256":label,"split":split}
    fixture = [row("a","same","x"),row("b","same","x"),row("c","bad","x"),row("d","bad","y"),row("v","val","v","val")]
    kept, removed, bad = select_pairs(fixture)
    assert [r["prepared_image"] for r in kept] == ["images/train/a.jpg","images/val/v.jpg"]
    assert len(removed) == 3 and bad == {"bad"}
    before = protection()
    rows = [json.loads(line) for line in (source / "pairs.jsonl").open(encoding="utf-8")]
    selected, excluded, ambiguous = select_pairs(rows)
    expected = Counter(row["split"] for row in selected)
    assert dict(expected) == {"train":56729,"val":17675}
    assert len(ambiguous) == 2516 and len(excluded) == 10430
    output.mkdir()
    for folder in ("images/train","images/val","labels/train","labels/val","provenance"):
        (output / folder).mkdir(parents=True,exist_ok=True)
    for item in original["archive_provenance"].values():
        assert file_sha(source/item["file"]) == item["sha256"]
        shutil.copy2(source/item["file"],output/item["file"])
    pairs = output / "pairs.jsonl"
    objects = {split:Counter() for split in ("train","val")}
    negatives = Counter()
    names = original["names"]
    def reuse(record):
        split=record["split"]
        for key,folder,suffixes in (("prepared_image","images",{".jpg",".jpeg"}),
                                   ("prepared_label","labels",{".txt"})):
            relative=PurePosixPath(record[key])
            if len(relative.parts)!=3 or relative.parts[:2]!=(folder,split) or relative.suffix.lower() not in suffixes:
                raise ValueError("Invalid audited relative path")
        image=source/record["prepared_image"]; label=source/record["prepared_label"]
        info=image.lstat()
        assert stat.S_ISREG(info.st_mode) and info.st_size==record["image_bytes"]
        assert info.st_mtime_ns//1000000==record["image_mtime_ms"] and stat.S_ISREG(label.lstat().st_mode)
        data=label.read_bytes()
        assert hashlib.sha256(data).hexdigest()==record["label_sha256"]
        counts=Counter()
        for line in data.decode("utf-8").splitlines():
            fields=line.split(); assert len(fields)==5
            cls=int(fields[0]); cx,cy,bw,bh=map(float,fields[1:])
            assert 0<=cls<len(names) and all(math.isfinite(v) for v in (cx,cy,bw,bh)) and bw>0 and bh>0
            assert min(cx-bw/2,cy-bh/2)>=-1e-8 and max(cx+bw/2,cy+bh/2)<=1+1e-8
            counts[names[cls]]+=1
        assert sum(counts.values())==record["foreground_objects"]
        target=output/record["prepared_image"]
        os.link(image,target)
        assert os.path.samefile(image,target)
        with (output/record["prepared_label"]).open("xb") as stream:
            stream.write(data)
        return record,counts
    started=last=time.monotonic(); completed=0
    iterator=iter(selected)
    with pairs.open("x",encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=6) as executor:
        while batch:=list(itertools.islice(iterator,96)):
            for record,counts in executor.map(reuse,batch):
                stream.write(json.dumps(record,separators=(",",":"))+"\n")
                objects[record["split"]].update(counts); negatives[record["split"]]+=not counts; completed+=1
            if time.monotonic()-last>=20:
                print(json.dumps({"clean_pairs_created":completed,"total":len(selected)}),flush=True);last=time.monotonic()
    assert objects["val"]==Counter(original["objects"]["val"])
    original_provider=ROOT/"runs/idd_detection_import_20261023_01/source_manifest_quality.csv"
    assert file_sha(original_provider)==original["source_manifest_sha256"]
    provider=read_manifest(original_provider)
    kept_ids={(row["family"],row["source_id"]) for row in selected}
    retained=[row for row in provider if row["split"]=="test" or (row["family"],row["source_id"]) in kept_ids]
    manifest=output/"provenance/source_manifest.csv"
    with manifest.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=MANIFEST_FIELDS);writer.writeheader();writer.writerows(retained)
    counts=dict(expected,test_metadata_only=sum(row["split"]=="test" for row in retained))
    assert counts["test_metadata_only"]==24644
    with (output/"provenance/training_exclusions.jsonl").open("x",encoding="utf-8") as stream:
        for record in excluded:stream.write(json.dumps(record,separators=(",",":"))+"\n")
    shutil.copy2(source/"data.yaml",output/"data.yaml")
    cleaning={"source_dataset":"dataset/idd117k_detection_v2","source_preparation_sha256":file_sha(source/"preparation.json"),
        "source_pairs_sha256":original["manifest_sha256"],"ambiguous_training_pixel_groups_excluded":len(ambiguous),
        "excluded_training_entries":len(excluded),"exclusions_by_reason":dict(Counter(r["exclusion_reason"] for r in excluded)),
        "retained_training_pixel_groups":expected["train"],"training_pixels_unique":True,
        "validation_membership_and_labels_unchanged":True,"labels_rewritten":False,
        "source_annotation_truth_not_assumed":True,"selection_fixture_passed":True}
    report={**original,"source_manifest_sha256":file_sha(manifest),"counts":counts,"curated_manifest_counts":counts,
        "manifest_sha256":file_sha(pairs),"data_yaml_sha256":file_sha(output/"data.yaml"),
        "objects":{split:dict(value) for split,value in objects.items()},"negatives":dict(negatives),
        "link_count":len(selected),"copy_count":0,"training_started":False,"training_cleanup":cleaning,
        "clipped_boxes":sum(r.get("clipped_boxes",0) for r in selected),
        "duplicate_boxes_removed":sum(r.get("duplicate_boxes_removed",0) for r in selected),
        "elapsed_seconds":round(time.monotonic()-started,1)}
    assert all(report["objects"][split].get(name,0)>0 for split in ("train","val") for name in names)
    assert protection()==before
    save_json(output/"preparation.json",report)
    save_json(output/"progress.json",{"status":"completed","passed":True,"training_started":False,"counts":counts})
    save_json(output/"provenance/cleaning_verification.json",{"passed":True,**cleaning,"counts":counts,
        "protected_files_unchanged":True,"official_test_used":False,"source_pairs_sha256":original["manifest_sha256"],
        "final_pairs_sha256":file_sha(pairs),"validation_boxes":sum(objects["val"].values())})
    print(json.dumps({"passed":True,"counts":counts,"cleaning":cleaning,"training_boxes":sum(objects["train"].values())},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output)
