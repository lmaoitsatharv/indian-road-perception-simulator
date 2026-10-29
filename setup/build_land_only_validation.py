"""Build a land-only validation copy inside the v4 view: <v4>/val_land/ and <v4>/data_land.yaml.

The original <v4>/images/val and labels/val (identical to v3) are NOT modified, so every earlier score stays comparable.
Rule (project decision, 29 Oct 2026): land animals and birds on the ground count; flying birds and birds on wires, poles, walls or
rooftops do not. Validation images containing an animal box too small to identify are left out of val_land.
Verdicts come from two model reviews: runs/animal_filter_review_20261029_01 (330 boxes) and runs/animal_val_review_20261029_01 (633).
"""
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "dataset/idd117k_detection_v4_landanimals"
R1 = ROOT / "runs/animal_filter_review_20261029_01"
R2 = ROOT / "runs/animal_val_review_20261029_01"
WALL_EGRETS_FIRST_REVIEW = {32, 33, 34, 35}  # egrets on a wall; excluded by the 29 Oct project decision
NAMES = ["person", "rider", "car", "motorcycle", "bicycle", "autorickshaw", "bus", "truck", "animal", "traffic_light", "traffic_sign", "other_vehicle"]


def key(stem, cx, cy):
    return stem, round(cx, 5), round(cy, 5)


def load_verdicts():
    out = {}
    first = {m["id"]: m for m in json.load(open(R1 / "val_boxes.json"))["items"]}
    for k in "EFGH":
        for v in json.load(open(R1 / ("val_verdicts_%s.json" % k), encoding="utf-8"))["verdicts"]:
            m = first[v["id"]]
            verdict = "wire_or_flying_bird" if v["id"] in WALL_EGRETS_FIRST_REVIEW else v["verdict"]
            out[key(m["stem"], m["cx"], m["cy"])] = verdict
    second = {m["id"]: m for m in json.load(open(R2 / "val_boxes_remaining.json"))["items"]}
    for k in range(1, 7):
        for v in json.load(open(R2 / ("verdicts_%d.json" % k), encoding="utf-8"))["verdicts"]:
            m = second[v["id"]]
            out[key(m["stem"], m["cx"], m["cy"])] = v["verdict"]
    assert len(out) == len(first) + len(second) == 963, len(out)
    return out


def main():
    target = V4 / "val_land"
    assert not target.exists()
    verdicts = load_verdicts()
    (target / "images").mkdir(parents=True)
    (target / "labels").mkdir()
    stats, excluded, edited = Counter(), [], []
    for label in sorted((V4 / "labels/val").glob("*.txt")):
        lines = [l.rstrip("\n") for l in open(label) if l.strip()]
        out, bad = [], False
        for l in lines:
            x = l.split()
            if x[0] != "8":
                out.append(l)
                continue
            verdict = verdicts[key(label.stem, float(x[1]), float(x[2]))]
            stats[verdict] += 1
            if verdict == "unsure_too_small":
                bad = True
            elif verdict == "land_animal":
                out.append(l)
        if bad:
            excluded.append(label.stem)
            continue
        os.link(V4 / "images/val" / (label.stem + ".jpg"), target / "images" / (label.stem + ".jpg"))
        if out == lines:
            shutil.copyfile(label, target / "labels" / label.name)
        else:
            (target / "labels" / label.name).write_text("\n".join(out) + ("\n" if out else ""))
            edited.append({"stem": label.stem, "animal_before": sum(1 for l in lines if l.startswith("8 ")),
                           "animal_after": sum(1 for l in out if l.startswith("8 "))})
    (V4 / "data_land.yaml").write_text("train: images/train\nval: val_land/images\nnames:\n" + "".join("  %d: %s\n" % (i, n) for i, n in enumerate(NAMES)))
    kept = sorted(p.stem for p in (target / "labels").glob("*.txt"))
    animal_kept = sum(1 for s in kept for l in open(target / "labels" / (s + ".txt")) if l.startswith("8 "))
    summary = {"rule": "animal = land animals and birds on the ground; flying, wire, pole, wall and roof birds removed; images with unidentifiable animal boxes left out",
               "reviews": "model visual review of all 963 validation animal boxes, not human verification",
               "original_validation_images": sum(1 for _ in (V4 / "labels/val").glob("*.txt")),
               "val_land_images": len(kept), "images_left_out": len(excluded), "images_with_animal_labels_edited": len(edited),
               "animal_box_verdicts": dict(stats), "animal_boxes_in_val_land": animal_kept,
               "protocol_note": "a different evaluation protocol from the original validation; report both, never compare a val_land score with an old all-box score"}
    (V4 / "provenance/val_land_summary.json").write_text(json.dumps(summary, indent=1))
    (V4 / "provenance/val_land_excluded.json").write_text(json.dumps(excluded, indent=1))
    (V4 / "provenance/val_land_edits.json").write_text(json.dumps(edited, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
