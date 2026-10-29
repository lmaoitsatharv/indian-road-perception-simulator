"""Per-box review sheets for every animal source we have (read-only on datasets; writes a new runs/ folder).

Sources: (1) recovered IDD candidates, (2) existing training boxes the first review could not settle,
(3) DATS_2022 animal subset (rotated upright first). One tile per box, 20 tiles per sheet.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
V2 = DATA / "idd117k_detection_v2"
V3 = DATA / "idd117k_detection_v3"
DATS = DATA / "dats2022_raw/animal_subset"
REVIEW1 = ROOT / "runs/animal_label_review_20261028_02/candidates.json"
REVIEW2 = ROOT / "runs/animal_filter_review_20261029_01/verdicts_merged.json"
ANIMALS = {"cattle", "dog", "goat", "camel", "horse"}
TW, TH = 384, 288


def tile(im, box, label):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    cw = max(bw * 5, 160)
    ch = cw * TH / TW
    if ch < bh * 5:
        ch = bh * 5
        cw = ch * TW / TH
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    W, H = im.size
    left, top = min(max(0, cx - cw / 2), max(0, W - cw)), min(max(0, cy - ch / 2), max(0, H - ch))
    crop = im.crop((int(left), int(top), int(min(W, left + cw)), int(min(H, top + ch))))
    k = TW / cw
    t = Image.new("RGB", (TW, TH))
    t.paste(crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))), Image.LANCZOS), (0, 0))
    d = ImageDraw.Draw(t)
    d.rectangle([(x0 - left) * k, (y0 - top) * k, (x1 - left) * k, (y1 - top) * k], outline=(255, 0, 0), width=1)
    d.rectangle([0, 0, TW, 12], fill=(0, 0, 0))
    d.text((3, 0), "%s  %dx%d px" % (label, bw, bh), fill=(255, 255, 255))
    return t


def collect():
    items = []
    # 1. recovered IDD candidates (animal-labelled copy of each duplicate group)
    for c in json.load(open(REVIEW1))["candidates"]:
        im_path = V2 / c["image"]
        boxes = [tuple(map(float, l.split()[1:5])) for l in open(V2 / c["label"]) if l.split() and l.split()[0] == "8"]
        items.append({"source": "idd_recovered", "group": "r%03d" % c["id"], "path": str(im_path), "boxes": boxes, "norm": True, "stem": c["image"]})
    # 2. existing training boxes the first pass marked unsure or not-animal
    merged = json.load(open(REVIEW2))["train_boxes"]
    for r in merged:
        if r["verdict"] in ("unsure_too_small", "other_not_animal"):
            lab = V3 / "labels/train" / (r["stem"] + ".txt")
            boxes = sorted(tuple(map(float, l.split()[1:5])) for l in open(lab) if l.split() and l.split()[0] == "8")
            items.append({"source": "train_unsure", "group": "t%03d" % r["img"], "path": str(V3 / "images/train" / (r["stem"] + ".jpg")),
                          "boxes": [boxes[r["i"] - 1]], "norm": True, "stem": r["stem"], "first_pass": r["verdict"]})
    # 3. DATS animal subset, rotated upright; coordinates scaled if the XML size differs
    pairs = json.load(open(DATS / "SUBSET.json"))["pairs"]
    for p in pairs:
        root = ET.parse(DATS / p["xml"]).getroot()
        boxes = []
        for o in root.findall("object"):
            if (o.findtext("name") or "").strip().lower() in ANIMALS:
                b = o.find("bndbox")
                boxes.append([float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
        w, h = float(root.findtext("size/width") or 0), float(root.findtext("size/height") or 0)
        items.append({"source": "dats", "group": "d_" + p["image"], "path": str(DATS / p["image"]), "boxes": boxes, "norm": False,
                      "xml_size": [w, h], "stem": p["image"], "names": [(o.findtext("name") or "").strip().lower() for o in root.findall("object")
                                                                       if (o.findtext("name") or "").strip().lower() in ANIMALS]})
    return items


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "sheets").mkdir()
    tiles, meta = [], []
    for item in collect():
        im = ImageOps.exif_transpose(Image.open(item["path"])).convert("RGB")
        W, H = im.size
        for j, b in enumerate(item["boxes"]):
            if item["norm"]:
                cx, cy, w, h = b
                px = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
            else:
                sx = W / item["xml_size"][0] if item["xml_size"][0] else 1.0
                sy = H / item["xml_size"][1] if item["xml_size"][1] else 1.0
                px = (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)
            n = len(meta)
            tag = {"idd_recovered": "IDD-rec", "train_unsure": "train", "dats": "DATS"}[item["source"]]
            tiles.append(tile(im, px, "box %03d %s" % (n, tag)))
            meta.append({"id": n, "source": item["source"], "group": item["group"], "stem": item["stem"], "box_in_image": j + 1,
                         "w_px": round(px[2] - px[0]), "h_px": round(px[3] - px[1]), "image_size": [W, H],
                         **({"first_pass": item["first_pass"]} if "first_pass" in item else {}),
                         **({"dats_class": item["names"][j]} if "names" in item else {})})
    per = 20
    for s in range(0, len(tiles), per):
        sheet = Image.new("RGB", (5 * TW, 4 * TH))
        for i, t in enumerate(tiles[s:s + per]):
            sheet.paste(t, ((i % 5) * TW, (i // 5) * TH))
        sheet.save(output / "sheets" / ("final_%02d_ids_%03d-%03d.jpg" % (s // per, s, min(s + per, len(tiles)) - 1)), quality=90)
    (output / "boxes.json").write_text(json.dumps({"total": len(meta), "items": meta}, indent=1), encoding="utf-8")
    from collections import Counter
    print("boxes", len(meta), dict(Counter(m["source"] for m in meta)), "sheets", (len(tiles) + per - 1) // per)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
