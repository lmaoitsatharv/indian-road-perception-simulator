"""Build dataset/idd117k_detection_v4_landanimals from the clean v3 view plus reviewed animal images.

Reads v3, v2 and the DATS animal subset; writes ONLY the new folder (v3, v2 and DATS stay untouched).
Rule (project decision, 29 Oct 2026): `animal` = land animals and birds on the ground. Birds in flight or on wires, poles, walls or
rooftops are not animals. Images containing an animal box too small to identify are left out entirely. Reviews are model
visual judgments (runs/animal_filter_review_20261029_01 and runs/animal_final_review_20261029_01).
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
V2, V3 = DATA / "idd117k_detection_v2", DATA / "idd117k_detection_v3"
DATS = DATA / "dats2022_raw/animal_subset"
R1 = ROOT / "runs/animal_filter_review_20261029_01"
R2 = ROOT / "runs/animal_final_review_20261029_01"
R0 = ROOT / "runs/animal_label_review_20261028_02"
ANIMAL = 8
NAMES = ["person", "rider", "car", "motorcycle", "bicycle", "autorickshaw", "bus", "truck", "animal", "traffic_light", "traffic_sign", "other_vehicle"]
DATS_ANIMALS = {"cattle", "dog", "goat", "camel", "horse"}
DATS_MAP = {"person": 0, "car": 2, "bike": 3, "cycle": 4, "rikshaw": 5, "bus": 6, "truck": 7, "traffic signal": 9, "signal": 9,
            "traffic sign board": 10, "road sign board": 10, "sign board": 10}  # bike = motorcycle (project decision, 29 Oct)
DATS_IGNORE = {"lamp post", "lamo post", "tree", "building", "road", "wall", "vegetation", "bridge", "zebra crossing", "footpath", "road divider",
               "electricity pole", "bus stop", "petrol pump", "tyre works", "flag", "gate", "barricade", "fuel station", "manhole", "pedestrian bridge",
               "digital display", "temple", "stall", "overbridge", "garbage bin", "metro station", "island stucture", "milestone", "water", "wa"}
USABLE = {"clean", "land_plus_removable_boxes"}  # images with poor-fit boxes are not added
HOLDOUT = 20


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_lines(path):
    return [l.rstrip("\n") for l in open(path) if l.strip()]


def final_train_verdicts():
    """(stem, 1-based box index in sorted animal order) -> final verdict for the 120 existing training images."""
    merged = json.load(open(R1 / "verdicts_merged.json"))["train_boxes"]
    boxes = {m["id"]: m for m in json.load(open(R2 / "boxes.json"))["items"]}
    vv = {}
    for k in range(1, 9):
        for v in json.load(open(R2 / ("verdicts_%d.json" % k), encoding="utf-8"))["verdicts"]:
            vv[v["id"]] = v
    rereview = [boxes[i] for i in sorted(boxes) if boxes[i]["source"] == "train_unsure"]
    out, k = {}, 0
    for r in merged:
        verdict = r["verdict"]
        if verdict in ("unsure_too_small", "other_not_animal"):
            item = rereview[k]
            assert item["group"] == "t%03d" % r["img"], (item["group"], r["img"])
            verdict = vv[item["id"]]["verdict"]
            k += 1
        out[(r["stem"], r["i"])] = verdict
    assert k == len(rereview)
    return out


def main(output):
    assert not output.exists(), "output must be new"
    pairs = [json.loads(l) for l in open(V3 / "pairs.jsonl", encoding="utf-8")]
    train_rows = [r for r in pairs if r["split"] == "train"]
    val_rows = [r for r in pairs if r["split"] == "val"]
    val_drives = {r["drive_id"] for r in val_rows}
    for d in ("images/train", "images/val", "labels/train", "labels/val", "provenance", "holdout_dats/images", "holdout_dats/labels"):
        (output / d).mkdir(parents=True)
    log = {"label_edits": [], "excluded_existing_images": [], "added_recovered": [], "skipped_recovered": [], "added_dats": [], "skipped_dats": [], "holdout_dats": []}

    # ---- 1. existing training images ----
    verdicts = final_train_verdicts()
    by_stem_rows = {Path(r["prepared_image"]).stem: r for r in train_rows}
    animal_stems = sorted({s for s, _ in verdicts})
    edited, excluded = {}, set()
    for stem in animal_stems:
        lines = read_lines(V3 / "labels/train" / (stem + ".txt"))
        animals = sorted(((tuple(map(float, l.split()[1:5])), l) for l in lines if l.split()[0] == str(ANIMAL)), key=lambda x: x[0])
        v = [verdicts[(stem, i + 1)] for i in range(len(animals))]
        assert len(v) == sum(1 for s, _ in verdicts if s == stem)
        if "unsure_too_small" in v:
            excluded.add(stem)
            log["excluded_existing_images"].append({"stem": stem, "reason": "contains an animal box too small to identify"})
            continue
        keep = {id(a[1]) for a, verdict in zip(animals, v) if verdict == "land_animal"}
        new = [l for l in lines if l.split()[0] != str(ANIMAL)] + [l for _, l in animals if id(l) in keep]
        # preserve original line order for kept lines
        order = {l: n for n, l in enumerate(lines)}
        new.sort(key=lambda l: order[l])
        edited[stem] = new
        log["label_edits"].append({"stem": stem, "animal_before": len(animals), "animal_kept": len(keep),
                                   "removed": dict(Counter(x for x in v if x != "land_animal"))})
    new_rows = []
    for r in train_rows:
        stem = Path(r["prepared_image"]).stem
        if stem in excluded:
            continue
        link(V3 / r["prepared_image"], output / r["prepared_image"])
        if stem in edited:
            (output / r["prepared_label"]).write_text("\n".join(edited[stem]) + ("\n" if edited[stem] else ""))
            r = {**r, "label_sha256": sha_file(output / r["prepared_label"]), "v4_change": "animal labels filtered by review"}
        else:
            shutil.copyfile(V3 / r["prepared_label"], output / r["prepared_label"])
        new_rows.append(r)

    # ---- 2. validation: identical to v3 ----
    for r in val_rows:
        link(V3 / r["prepared_image"], output / r["prepared_image"])
        shutil.copyfile(V3 / r["prepared_label"], output / r["prepared_label"])

    # ---- 3. recovered IDD images ----
    status = json.load(open(R2 / "image_status.json"))["res"]
    boxes = {m["id"]: m for m in json.load(open(R2 / "boxes.json"))["items"]}
    vv = {}
    for k in range(1, 9):
        for v in json.load(open(R2 / ("verdicts_%d.json" % k), encoding="utf-8"))["verdicts"]:
            vv[v["id"]] = v
    exclusions = {}
    for line in open(V3 / "provenance/training_exclusions.jsonl", encoding="utf-8"):
        row = json.loads(line)
        exclusions[row["prepared_image"]] = row
    cands = {"r%03d" % c["id"]: c for c in json.load(open(R0 / "candidates.json"))["candidates"]}
    existing_names = {Path(r["prepared_image"]).name for r in new_rows}
    for group, c in cands.items():
        st = status["idd_recovered|" + group]
        if st not in USABLE:
            log["skipped_recovered"].append({"group": group, "reason": st})
            continue
        if c["drive"] in val_drives:
            log["skipped_recovered"].append({"group": group, "reason": "recording overlaps validation"})
            continue
        assert Path(c["image"]).name not in existing_names
        v = {m["box_in_image"]: vv[i]["verdict"] for i, m in boxes.items() if m["source"] == "idd_recovered" and m["group"] == group}
        lines = read_lines(V2 / c["label"])
        out, j = [], 0
        for l in lines:
            if l.split()[0] == str(ANIMAL):
                j += 1
                if v[j] == "land_animal":
                    out.append(l)
            else:
                out.append(l)
        assert j == len(v)
        link(V2 / c["image"], output / c["image"])
        (output / c["label"]).write_text("\n".join(out) + "\n")
        row = {**exclusions[c["image"]], "split": "train", "label_sha256": sha_file(output / c["label"]), "v4_change": "added from recovered duplicate group"}
        row.pop("exclusion_reason", None); row.pop("canonical_prepared_image", None)
        new_rows.append(row)
        log["added_recovered"].append({"group": group, "image": c["image"], "land_boxes": sum(1 for x in v.values() if x == "land_animal")})

    # ---- 4. DATS ----
    selection = json.load(open(DATA / "dats2022_raw/dats_final_selection.json"))["kept"]
    pairs_dats = {p["image"]: p for p in json.load(open(DATS / "SUBSET.json"))["pairs"]}
    eligible = []
    for img in selection:
        st = status["dats|d_" + img]
        if st in USABLE:
            eligible.append(img)
        else:
            log["skipped_dats"].append({"image": img, "reason": st})
    holdout = set(random.Random(20261029).sample(sorted(eligible), HOLDOUT))
    for img in sorted(eligible):
        p = pairs_dats[img]
        root = ET.parse(DATS / p["xml"]).getroot()
        im = ImageOps.exif_transpose(Image.open(DATS / img)).convert("RGB")
        W, H = im.size
        xw, xh = float(root.findtext("size/width") or 0), float(root.findtext("size/height") or 0)
        sx, sy = (W / xw if xw else 1.0), (H / xh if xh else 1.0)
        v = {m["box_in_image"]: vv[i]["verdict"] for i, m in boxes.items() if m["source"] == "dats" and m["group"] == "d_" + img}
        lines, a, bad = [], 0, None
        for o in root.findall("object"):
            name = (o.findtext("name") or "").strip().lower()
            b = o.find("bndbox")
            x0, y0, x1, y1 = [float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            x0, x1, y0, y1 = max(0, x0 * sx), min(W, x1 * sx), max(0, y0 * sy), min(H, y1 * sy)
            if name in DATS_ANIMALS:
                a += 1
                if v[a] != "land_animal":
                    continue
                cls = ANIMAL
            elif name in DATS_MAP:
                cls = DATS_MAP[name]
            elif name in DATS_IGNORE:
                continue
            else:
                bad = name
                break
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (cls, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
        if bad:
            log["skipped_dats"].append({"image": img, "reason": "unmapped class " + bad})
            continue
        f = min(1.0, 1920 / max(W, H))
        im = im.resize((max(1, round(W * f)), max(1, round(H * f))), Image.LANCZOS)
        stem = "dats_" + Path(img).stem
        if img in holdout:
            ipath, lpath = output / "holdout_dats/images" / (stem + ".jpg"), output / "holdout_dats/labels" / (stem + ".txt")
        else:
            ipath, lpath = output / "images/train" / (stem + ".jpg"), output / "labels/train" / (stem + ".txt")
        im.save(ipath, "JPEG", quality=92)
        lpath.write_text("\n".join(lines) + "\n")
        entry = {"image": img, "new_name": stem, "land_animal_boxes": sum(1 for x in v.values() if x == "land_animal"), "size": [im.width, im.height]}
        if img in holdout:
            log["holdout_dats"].append(entry)
            continue
        new_rows.append({"image": "DATS_2022/" + img, "split": "train", "family": "DATS_2022", "drive_id": "DATS", "source_id": img,
                         "native_width": W, "native_height": H, "pixels_sha256": sha_file(ipath),
                         "prepared_image": "images/train/%s.jpg" % stem, "prepared_label": "labels/train/%s.txt" % stem,
                         "label_sha256": sha_file(lpath), "v4_change": "added from DATS_2022 (CC BY 4.0, rotated upright, resized to 1920 long side)"})
        log["added_dats"].append(entry)

    # ---- 5. metadata ----
    (output / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n" + "".join("  %d: %s\n" % (i, n) for i, n in enumerate(NAMES)))
    with open(output / "pairs.jsonl", "w", encoding="utf-8") as f:
        for r in new_rows + val_rows:
            f.write(json.dumps(r) + "\n")
    for name, value in log.items():
        (output / "provenance" / (name + ".json")).write_text(json.dumps(value, indent=1))
    counts = {"train": len(new_rows), "val": len(val_rows)}
    (output / "build_summary.json").write_text(json.dumps({
        "parent_view": "dataset/idd117k_detection_v3", "counts": counts, "excluded_existing_images": len(excluded),
        "edited_existing_images": len(edited), "added_recovered_images": len(log["added_recovered"]), "added_dats_images": len(log["added_dats"]),
        "holdout_dats_images": len(log["holdout_dats"]), "skipped_recovered": len(log["skipped_recovered"]), "skipped_dats": len(log["skipped_dats"]),
        "rule": "animal = land animals and birds on the ground; flying/wire/wall/roof birds excluded; images with unidentifiable animal boxes left out",
        "reviews": "model visual review, not human verification", "training_started": False}, indent=1))
    print(json.dumps(json.load(open(output / "build_summary.json")), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
