"""Save asset-corrected semantic previews while preserving native sensor evidence.

These are inspected development captures, not a new train/val/test dataset.
Unresolved procedural road pixels and sensor edge disagreement are ignored.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = {"asphalt":1,"indian_sign":8,"shop_board":3,"ground":10,"median":2}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(session,output):
    status = json.loads((session/"result.json").read_text())
    if not status["passed"] or len(status["worlds"]) != 5:
        raise ValueError("Require all five native development worlds to pass first")
    output.mkdir(exist_ok=False)
    results = []
    for result in status["worlds"]:
        key = result["profile"]
        source = session/key
        target = output/key
        target.mkdir()
        objects = json.loads((source/"scenery.json").read_text())["objects"]
        lookup = {row["actor_id"]:OVERRIDES[row["kind"]] for row in objects
            if row.get("kind") in OVERRIDES and "actor_id" in row}
        if not lookup or min(lookup) <= 0 or max(lookup) > 65535:
            raise ValueError("Instance IDs cannot be represented exactly")
        rows = [json.loads(line) for line in (source/"camera_frames.jsonl").read_text().splitlines()]
        class_counts = np.zeros(256,dtype=np.int64)
        overridden = {kind:0 for kind in OVERRIDES}
        ids_by_kind = {kind:[row["actor_id"] for row in objects if row.get("kind") == kind and "actor_id" in row]
                       for kind in OVERRIDES}
        records = []
        lut = np.full(65536,255,dtype=np.uint8)
        for actor_id,label in lookup.items():
            lut[actor_id] = label
        for row in rows:
            sem_path = source/row["streams"]["semantic_segmentation"]["file"]
            inst_path = source/row["streams"]["instance_segmentation"]["file"]
            assert sha(sem_path) == row["streams"]["semantic_segmentation"]["sha256"]
            assert sha(inst_path) == row["streams"]["instance_segmentation"]["sha256"]
            tags = np.asarray(Image.open(sem_path))[...,0]
            instance = np.asarray(Image.open(inst_path))
            ids = instance[...,1].astype(np.uint16)+instance[...,2].astype(np.uint16)*256
            disagreement = instance[...,0] != tags
            mask = tags.copy()
            mask[tags == 0] = 255
            mapped = lut[ids]
            owned = mapped != 255
            mask[owned] = mapped[owned]
            mask[disagreement] = 255
            for kind,actor_ids in ids_by_kind.items():
                pixels = np.isin(ids,actor_ids) & ~disagreement
                assert np.all(mask[pixels] == OVERRIDES[kind])
                overridden[kind] += int(pixels.sum())
            # All unrelated known native pixels remain unchanged.
            retained = ~owned & ~disagreement & (tags != 0)
            assert np.array_equal(mask[retained],tags[retained])
            path = target/f"{row['source_frame']:08d}.png"
            Image.fromarray(mask).save(path)
            assert np.array_equal(np.asarray(Image.open(path)),mask)
            class_counts += np.bincount(mask.ravel(),minlength=256)
            records.append({"source_frame":row["source_frame"],"mask":str(path.relative_to(output)),
                "sha256":sha(path),"source_rgb":str((source/row["streams"]["rgb"]["file"]).relative_to(ROOT)),
                "ignored_edge_disagreement_pixels":int(disagreement.sum())})
        if overridden["asphalt"] == 0 or overridden["indian_sign"] == 0:
            raise ValueError("Expected road/sign instances were never observed")
        (target/"pairs.json").write_text(json.dumps(records,indent=2)+"\n")
        results.append({"profile":key,"frames":len(rows),"visible_owned_pixels":overridden,
            "class_pixel_counts":{str(i):int(v) for i,v in enumerate(class_counts) if v},"asset_mapping_verified":True})
    report = {"schema":"indian_world_asset_semantics_v1","passed":True,"worlds":results,
        "native_semantics_unchanged":True,"source_session":str(session.relative_to(ROOT)),
        "training_allowed":False,"scope":"Inspected synthetic development previews; no dataset split or real model evaluation",
        "ids":{"1":"road","2":"sidewalk/median","3":"building/shop board","8":"traffic sign","10":"terrain","255":"ignore"},
        "method":"Exact owned instance IDs only; preserve unrelated native classes; ignore unresolved native class 0 and sensor disagreement",
        "limitation":"Not the complete depth/map road-boundary protocol; no detector-box export or autorickshaw runtime labels"}
    (output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":True,"worlds":len(results),"frames":sum(r["frames"] for r in results),"training_allowed":False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    session,output = args.session.resolve(),args.output.resolve()
    if not session.is_relative_to(ROOT/"runs") or not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Keep development evidence under runs/")
    prepare(session,output)
