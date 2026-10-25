"""Reuse audited per-image framework arrays, checking every retained label again."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import file_sha,save_json
from project_runtime import linux_path


def main():
    source=ROOT/"dataset/idd117k_detection_v2"
    target=ROOT/"dataset/idd117k_detection_v3"
    prepared=json.loads((target/"preparation.json").read_text())
    audit=json.loads((ROOT/"runs/detection_clean_audit_20261024_02/verification.json").read_text())
    assert audit["passed"] and audit["objects"]==prepared["objects"]
    assert file_sha(target/"pairs.jsonl")==prepared["manifest_sha256"]
    rows=[json.loads(line) for line in (target/"pairs.jsonl").open()]
    linux_root=linux_path(target).rstrip("/")+"/"
    proof={"passed":True,"training_started":False,"all_retained_labels_compared_to_cached_arrays":True,
           "normal_framework_hash_checks_preserved":True,"new_images_or_labels_created":False,"splits":{}}
    for split in ("train","val"):
        old_path=source/"labels"/(split+".cache")
        old=np.load(old_path,allow_pickle=True).item()  # Locally generated, previously verified framework cache.
        by_name={Path(label["im_file"]).name:label for label in old["labels"]}
        selected=sorted([row for row in rows if row["split"]==split],key=lambda row:row["prepared_image"])
        def reuse(row):
            path=target/row["prepared_label"]
            data=path.read_bytes()
            assert hashlib.sha256(data).hexdigest()==row["label_sha256"]
            values=np.fromstring(data.decode("utf-8"),sep=" ",dtype=np.float32).reshape(-1,5)
            label=by_name[Path(row["prepared_image"]).name]
            assert label["shape"]==(row["native_height"],row["native_width"])
            assert label["normalized"] is True and label["bbox_format"]=="xywh"
            np.testing.assert_array_equal(values[:,0:1],label["cls"])
            np.testing.assert_array_equal(values[:,1:],label["bboxes"])
            return {**label,"im_file":linux_root+row["prepared_image"]},len(data),row["image_bytes"]
        with ThreadPoolExecutor(max_workers=6) as executor:
            checked=list(executor.map(reuse,selected))
        labels=[item[0] for item in checked]
        images=[label["im_file"] for label in labels]
        label_paths=[linux_root+row["prepared_label"] for row in selected]
        # Pinned 8.3.225 get_hash: sum file sizes, followed by concatenated paths.
        size=sum(label_size+image_size for _,label_size,image_size in checked)
        digest=hashlib.sha256(str(size).encode());digest.update("".join(label_paths+images).encode())
        cache={"labels":labels,"hash":digest.hexdigest(),"results":(len(labels),0,prepared["negatives"][split],0,len(labels)),
               "msgs":[],"version":old["version"]}
        classes=np.concatenate([label["cls"].reshape(-1) for label in labels]).astype(int)
        assert np.bincount(classes,minlength=12).tolist()==[prepared["objects"][split][name] for name in prepared["names"]]
        destination=target/"labels"/(split+".cache")
        created=False
        if not destination.exists():
            temporary=destination.with_name(split+".audited_candidate.npy")
            with temporary.open("xb") as stream:np.save(stream,cache,allow_pickle=True)
            # Exclusive link publishes a complete cache without overwriting any
            # framework cache that may have been created concurrently.
            try:
                os.link(temporary,destination)
                created=True
            except FileExistsError:
                pass
        actual=np.load(destination,allow_pickle=True).item()
        assert len(actual["labels"])==len(labels) and actual["hash"]==cache["hash"]
        proof["splits"][split]={"images":len(labels),"created_from_checked_arrays":created,
            "cache_sha256":file_sha(destination),"original_cache_sha256":file_sha(old_path),
            "boxes":len(classes),"framework_hash":cache["hash"],"version":cache["version"]}
        print(json.dumps(proof["splits"][split]),flush=True)
    save_json(target/"provenance/framework_cache_reuse.json",proof)


if __name__=="__main__":
    main()
