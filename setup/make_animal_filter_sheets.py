"""Review sheets for the animal-label filter plan (read-only on datasets; writes into a new runs/ folder).

train: every training image that has an animal box (one tile per image, all its animal boxes numbered).
val:   the suspicious validation animal boxes (under 20 px, or centre in the top 35% of the image) plus a random
       60 of the other validation animal boxes (one tile per box).
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "dataset/idd117k_detection_v3"
ANIMAL = 8


def animal_boxes(label_path):
    rows = []
    for line in open(label_path):
        x = line.split()
        if x and int(x[0]) == ANIMAL:
            rows.append(tuple(map(float, x[1:5])))
    return label_path.stem, rows


def native(box, W, H):
    cx, cy, w, h = box
    return (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H


def train_tile(stem, boxes, n, tw, th):
    im = Image.open(V3 / "images/train" / (stem + ".jpg")).convert("RGB")
    W, H = im.size
    boxes = sorted(boxes)  # left to right, so numbers are stable
    px = [native(b, W, H) for b in boxes]
    x0, y0 = min(p[0] for p in px), min(p[1] for p in px)
    x1, y1 = max(p[2] for p in px), max(p[3] for p in px)
    cw, ch = max((x1 - x0) * 1.5, 240), max((y1 - y0) * 1.5, 180)
    s = max(cw / tw, ch / th)
    cw, ch = tw * s, th * s
    left = min(max(0, (x0 + x1) / 2 - cw / 2), max(0, W - cw))
    top = min(max(0, (y0 + y1) / 2 - ch / 2), max(0, H - ch))
    crop = im.crop((int(left), int(top), int(min(W, left + cw)), int(min(H, top + ch))))
    k = tw / cw
    tile = Image.new("RGB", (tw, th))
    tile.paste(crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))), Image.LANCZOS), (0, 0))
    d = ImageDraw.Draw(tile)
    for i, p in enumerate(px, 1):
        d.rectangle([(p[0] - left) * k, (p[1] - top) * k, (p[2] - left) * k, (p[3] - top) * k], outline=(255, 0, 0), width=1)
        d.text(((p[0] - left) * k, max(13, (p[1] - top) * k - 10)), str(i), fill=(255, 255, 0))
    sizes = [((p[2] - p[0]) * (p[3] - p[1])) ** 0.5 for p in px]
    d.rectangle([0, 0, tw, 12], fill=(0, 0, 0))
    d.text((3, 0), "img %03d  %d boxes  size px %d-%d" % (n, len(px), min(sizes), max(sizes)), fill=(255, 255, 255))
    meta = {"id": n, "stem": stem, "boxes": [{"i": i, "cx": b[0], "cy": b[1], "w_px": round((p[2] - p[0])), "h_px": round((p[3] - p[1]))}
                                             for i, (b, p) in enumerate(zip(boxes, px), 1)]}
    return tile, meta


def val_tile(stem, box, n, tw, th):
    im = Image.open(V3 / "images/val" / (stem + ".jpg")).convert("RGB")
    W, H = im.size
    p = native(box, W, H)
    bw, bh = p[2] - p[0], p[3] - p[1]
    cw = max(bw * 5, 160)
    ch = cw * th / tw
    if ch < bh * 5:
        ch = bh * 5
        cw = ch * tw / th
    cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
    left = min(max(0, cx - cw / 2), max(0, W - cw))
    top = min(max(0, cy - ch / 2), max(0, H - ch))
    crop = im.crop((int(left), int(top), int(min(W, left + cw)), int(min(H, top + ch))))
    k = tw / cw
    tile = Image.new("RGB", (tw, th))
    tile.paste(crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))), Image.LANCZOS), (0, 0))
    d = ImageDraw.Draw(tile)
    d.rectangle([(p[0] - left) * k, (p[1] - top) * k, (p[2] - left) * k, (p[3] - top) * k], outline=(255, 0, 0), width=1)
    d.rectangle([0, 0, tw, 12], fill=(0, 0, 0))
    d.text((3, 0), "box %03d  %dx%d px  y=%.2f" % (n, bw, bh, box[1]), fill=(255, 255, 255))
    return tile, {"id": n, "stem": stem, "cx": box[0], "cy": box[1], "w_px": round(bw), "h_px": round(bh)}


def sheets(tiles, cols, rows, tw, th, folder, prefix):
    per = cols * rows
    for s in range(0, len(tiles), per):
        sheet = Image.new("RGB", (cols * tw, rows * th))
        for i, t in enumerate(tiles[s:s + per]):
            sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
        sheet.save(folder / ("%s_%02d_ids_%03d-%03d.jpg" % (prefix, s // per, s, min(s + per, len(tiles)) - 1)), quality=90)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "train_sheets").mkdir()
    (output / "val_sheets").mkdir()
    with ThreadPoolExecutor(16) as pool:
        train = [(s, b) for s, b in pool.map(animal_boxes, sorted((V3 / "labels/train").glob("*.txt")), chunksize=64) if b]
        val = [(s, b) for s, b in pool.map(animal_boxes, sorted((V3 / "labels/val").glob("*.txt")), chunksize=64) if b]
    # training: one tile per image
    tw, th = 640, 480
    tiles, meta = [], []
    for n, (stem, boxes) in enumerate(train):
        t, m = train_tile(stem, boxes, n, tw, th)
        tiles.append(t)
        meta.append(m)
    sheets(tiles, 3, 3, tw, th, output / "train_sheets", "train")
    (output / "train_images.json").write_text(json.dumps({"images": len(meta), "items": meta}, indent=1))
    # validation: suspicious boxes + 60 random others, one tile per box
    flat = []
    for stem, boxes in val:
        flat.extend((stem, b) for b in boxes)
    def px_size(b):
        return (b[2] * 1920 * b[3] * 1080) ** 0.5
    suspicious = [x for x in flat if px_size(x[1]) < 20 or x[1][1] < 0.35]
    others = [x for x in flat if not (px_size(x[1]) < 20 or x[1][1] < 0.35)]
    picked = suspicious + random.Random(5).sample(others, 60)
    tw, th = 384, 288
    vtiles, vmeta = [], []
    for n, (stem, box) in enumerate(picked):
        t, m = val_tile(stem, box, n, tw, th)
        m["group"] = "suspicious" if n < len(suspicious) else "random_other"
        vtiles.append(t)
        vmeta.append(m)
    sheets(vtiles, 5, 4, tw, th, output / "val_sheets", "val")
    (output / "val_boxes.json").write_text(json.dumps({"total_val_animal_boxes": len(flat), "suspicious": len(suspicious), "random_other": 60, "items": vmeta}, indent=1))
    print("train images", len(meta), "train boxes", sum(len(m["boxes"]) for m in meta), "| val boxes", len(flat), "suspicious", len(suspicious), "picked", len(picked))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
