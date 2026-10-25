"""Check the optimized pre-training file guard without loading or training a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import save_json
from train_detection import check_prepared_files


def main(output):
    output=output.resolve()
    if not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Fixtures stay in project runs")
    output.mkdir(parents=True,exist_ok=False)
    dataset=output/"fixture"
    for folder in ("images/train","images/val","labels/train","labels/val"):
        (dataset/folder).mkdir(parents=True)
    rows=[]
    for index,split in enumerate(("train","train","train","val")):
        image=dataset/f"images/{split}/{index}.jpg"
        label=dataset/f"labels/{split}/{index}.txt"
        image.write_bytes(b"fixture image metadata only "+bytes([index]))
        label.write_bytes(b"0 0.5 0.5 0.2 0.2\n" if index else b"")
        info=image.stat()
        rows.append({"split":split,"prepared_image":image.relative_to(dataset).as_posix(),
            "prepared_label":label.relative_to(dataset).as_posix(),"image_bytes":info.st_size,
            "image_mtime_ms":info.st_mtime_ns//1000000,"label_sha256":hashlib.sha256(label.read_bytes()).hexdigest()})
    manifest=dataset/"pairs.jsonl"
    def write(values=rows):
        manifest.write_text("".join(json.dumps(row)+"\n" for row in values),encoding="utf-8")
    def checked(workers=6):
        return check_prepared_files(dataset,{"train":3,"val":1},workers=workers)
    def rejects(values):
        write(values)
        try:
            checked()
        except ValueError:
            return
        raise AssertionError("Malformed manifest path/split was accepted")
    write()
    assert checked()==checked(1)==({"train":3,"val":1},True)
    checks=["serial_and_parallel_results_match_including_empty_valid_label"]
    image=dataset/rows[1]["prepared_image"]
    times=image.stat()
    os.utime(image,ns=(times.st_atime_ns,times.st_mtime_ns+2_000_000))
    assert checked()[1] is False
    os.utime(image,ns=(times.st_atime_ns,times.st_mtime_ns))
    # DrvFS can round explicitly restored times to whole seconds. Freeze the
    # actual restored fixture timestamp rather than weakening the real guard.
    rows[1]["image_mtime_ms"]=image.stat().st_mtime_ns//1000000
    write();assert checked()[1] is True
    checks.append("changed_image_timestamp_rejected")
    old=image.read_bytes();image.write_bytes(old+b"changed")
    assert checked()[1] is False
    image.write_bytes(old);os.utime(image,ns=(times.st_atime_ns,times.st_mtime_ns))
    rows[1]["image_mtime_ms"]=image.stat().st_mtime_ns//1000000
    write();assert checked()[1] is True
    checks.append("changed_image_size_rejected")
    label=dataset/rows[1]["prepared_label"]
    old=label.read_bytes();label.write_bytes(old.replace(b"0.2",b"0.3"))
    assert checked()[1] is False
    label.write_bytes(old)
    checks.append("same_size_changed_label_hash_rejected")
    label.unlink();assert checked()[1] is False;label.write_bytes(old)
    checks.append("missing_label_rejected_not_a_negative")
    for field,value in (("prepared_image","../outside.jpg"),("prepared_label","labels/val/1.txt"),
                        ("prepared_label","labels/train/wrong.txt"),("prepared_label","labels\\train\\1.txt"),
                        ("prepared_image","C:/outside.jpg"),("split","test")):
        changed=[dict(row) for row in rows];changed[1][field]=value;rejects(changed)
    write();checks.append("traversal_wrong_split_identity_windows_path_and_test_rejected")
    assert not check_prepared_files(dataset,{"train":4,"val":1})[1]
    checks.append("incomplete_counts_rejected")
    skipped=[]
    if os.name!="nt":
        outside=output/"outside.txt";outside.write_bytes(old)
        label.unlink();label.symlink_to(outside)
        assert checked()[1] is False
        label.unlink();label.write_bytes(old)
        checks.append("leaf_symlink_rejected_before_reading")
        parent=dataset/"labels/train";preserved=dataset/"labels/train_original"
        assert parent.resolve().is_relative_to(output) and preserved.absolute().is_relative_to(output)
        parent.rename(preserved);parent.symlink_to(preserved,target_is_directory=True)
        try:
            checked()
        except ValueError:
            pass
        else:
            raise AssertionError("Symlinked dataset parent accepted")
        assert parent.is_symlink();parent.unlink();preserved.rename(parent)
        checks.append("symlinked_parent_directory_rejected")
    else:
        skipped.append("POSIX symlink checks run separately in WSL")
    assert checked()==({"train":3,"val":1},True)
    result={"passed":True,"checks":checks,"skipped":skipped,"training_started":False,
        "model_loaded":False,"scope":"metadata_file_guard_fixtures_only"}
    save_json(output/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output)
