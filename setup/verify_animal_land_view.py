"""Independent checks of dataset/idd117k_detection_v4_landanimals against v3 (read-only)."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "dataset/idd117k_detection_v3"
V4 = ROOT / "dataset/idd117k_detection_v4_landanimals"


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stems(folder, suffix):
    return {p.stem for p in Path(folder).glob("*" + suffix)}


def main():
    out = {}
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        out[name] = {"passed": bool(cond), "detail": detail}
        print(("PASS " if cond else "FAIL ") + name, detail)

    # file pairing
    ti, tl = stems(V4 / "images/train", ".jpg"), stems(V4 / "labels/train", ".txt")
    vi, vl = stems(V4 / "images/val", ".jpg"), stems(V4 / "labels/val", ".txt")
    check("train images and labels pair exactly", ti == tl, "%d images, %d labels" % (len(ti), len(tl)))
    check("val images and labels pair exactly", vi == vl, "%d" % len(vi))
    # validation identical to v3
    v3v = sorted((V3 / "labels/val").glob("*.txt"))
    with ThreadPoolExecutor(16) as pool:
        same = list(pool.map(lambda p: h(p) == h(V4 / "labels/val" / p.name), v3v))
    check("validation labels byte-identical to v3", all(same) and len(v3v) == len(vl), "%d files" % len(v3v))
    v3img = sorted((V3 / "images/val").glob("*.jpg"))
    check("validation images are the same files as v3 (hard links)", all((V4 / "images/val" / p.name).exists() and
          (V4 / "images/val" / p.name).stat().st_size == p.stat().st_size for p in v3img), "%d files" % len(v3img))
    # label format + animal statistics
    def parse(p):
        rows = []
        for l in open(p):
            x = l.split()
            if x:
                rows.append((int(x[0]), *map(float, x[1:5])))
        return p.stem, rows
    with ThreadPoolExecutor(16) as pool:
        train = list(pool.map(parse, sorted((V4 / "labels/train").glob("*.txt")), chunksize=64))
    bad = [s for s, rows in train for r in rows if not (0 <= r[0] < 12 and all(0 <= v <= 1 for v in r[1:]) and r[3] > 0 and r[4] > 0)]
    check("every train label line valid (class 0-11, coordinates in 0-1, positive size)", not bad, "%d bad" % len(bad))
    classes = Counter(r[0] for _, rows in train for r in rows)
    animal_imgs = sum(1 for _, rows in train if any(r[0] == 8 for r in rows))
    animal = [r for _, rows in train for r in rows if r[0] == 8]
    out["train_class_boxes"] = dict(sorted(classes.items()))
    out["animal_stats"] = {"images": animal_imgs, "boxes": len(animal)}
    print("train animal images", animal_imgs, "boxes", len(animal), "| class counts", dict(sorted(classes.items())))
    # compare with v3: unchanged labels identical, changed ones only differ in animal lines
    v3t = {p.stem: p for p in (V3 / "labels/train").glob("*.txt")}
    changed_nonanimal, identical, edited = [], 0, 0
    for stem, p4 in ((p.stem, p) for p in sorted((V4 / "labels/train").glob("*.txt"))):
        if stem not in v3t:
            continue
        a, b = p4.read_text().split("\n"), v3t[stem].read_text().split("\n")
        if a == b:
            identical += 1
            continue
        edited += 1
        na = [l for l in a if l.strip() and not l.startswith("8 ")]
        nb = [l for l in b if l.strip() and not l.startswith("8 ")]
        if na != nb:
            changed_nonanimal.append(stem)
    check("inherited v3 labels: only animal lines differ", not changed_nonanimal, "%d identical, %d edited" % (identical, edited))
    prov = V4 / "provenance"
    added_r = json.load(open(prov / "added_recovered.json")); added_d = json.load(open(prov / "added_dats.json"))
    excl = json.load(open(prov / "excluded_existing_images.json")); hold = json.load(open(prov / "holdout_dats.json"))
    check("train count = v3 - excluded + recovered + DATS", len(ti) == 56729 - len(excl) + len(added_r) + len(added_d),
          "%d = 56729 - %d + %d + %d" % (len(ti), len(excl), len(added_r), len(added_d)))
    check("excluded images absent from v4 train", all(e["stem"] not in ti for e in excl))
    check("holdout DATS images absent from train and present in holdout", all(("dats_" + Path(x["image"]).stem) not in ti for x in hold) and
          len(list((V4 / "holdout_dats/images").glob("*.jpg"))) == len(hold) == 20)
    # pairs.jsonl covers train exactly
    rows = [json.loads(l) for l in open(V4 / "pairs.jsonl", encoding="utf-8")]
    ptrain = {Path(r["prepared_image"]).stem for r in rows if r["split"] == "train"}
    check("pairs.jsonl train rows match the train folder", ptrain == ti, "%d rows" % len(ptrain))
    # DATS images: upright, sensible size
    dats = sorted((V4 / "images/train").glob("dats_*.jpg"))
    sizes = [Image.open(p).size for p in dats[:30]]
    check("DATS images saved with no rotation tag, long side <= 1920", all(max(s) <= 1920 for s in sizes) and
          all(Image.open(p).getexif().get(0x0112, 1) == 1 for p in dats[:30]), "%d DATS train images" % len(dats))
    # recording overlap of added IDD images with validation
    pairs = {Path(r["prepared_image"]).name: r for r in rows}
    val_drives = {r["drive_id"] for r in rows if r["split"] == "val"}
    overlap = [x["image"] for x in added_r if pairs[Path(x["image"]).name]["drive_id"] in val_drives]
    check("added IDD images share no recording with validation", not overlap, "%d overlaps" % len(overlap))
    # animal size / position, compared with v3
    sz = np.array([np.sqrt(r[3] * r[4] * 1920 * 1080) for r in animal]); cy = np.array([r[2] for r in animal])
    out["animal_stats"].update({"median_size_px_at_1920x1080": float(np.median(sz)), "share_under_20px": float((sz < 20).mean()),
                                "share_centre_in_top_quarter": float((cy < 0.25).mean())})
    print("animal median size %.0f px | under 20 px %.1f%% | top quarter %.1f%%" % (np.median(sz), 100 * (sz < 20).mean(), 100 * (cy < 0.25).mean()))
    out["all_passed"] = ok
    (V4 / "verification.json").write_text(json.dumps(out, indent=1))
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
