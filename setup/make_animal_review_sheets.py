"""Build numbered review sheets for IDD training images that were removed because duplicate copies disagreed.

Candidate = a duplicate-pixel group where exactly one copy carries animal boxes and every non-animal box
is identical across copies. Read-only on the dataset; writes only into a new runs/ folder.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
ANIMAL = 8
TILE_W, TILE_H, COLS, ROWS = 480, 360, 4, 4


def load_groups():
    groups = defaultdict(list)
    for line in open(DATA / "idd117k_detection_v3/provenance/training_exclusions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["exclusion_reason"] != "duplicate_pixels_with_differing_annotations":
            continue
        rows = [l.split() for l in open(DATA / "idd117k_detection_v2" / r["prepared_label"]) if l.strip()]
        boxes = [(int(x[0]), *map(float, x[1:5])) for x in rows]
        groups[r["pixels_sha256"]].append({"image": r["prepared_image"], "label": r["prepared_label"], "boxes": boxes,
                                           "drive": r["drive_id"], "source": r["source_id"]})
    return groups


def candidates(groups):
    out = []
    for sha, members in sorted(groups.items()):
        with_animal = [m for m in members if any(b[0] == ANIMAL for b in m["boxes"])]
        if len(with_animal) != 1:
            continue
        others = [sorted((b[0], round(b[1], 2), round(b[2], 2), round(b[3], 2), round(b[4], 2)) for b in m["boxes"] if b[0] != ANIMAL)
                  for m in members]
        if all(o == others[0] for o in others):
            out.append((sha, with_animal[0], [m["image"] for m in members if m is not with_animal[0]]))
    return out


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "sheets").mkdir()
    cands = candidates(load_groups())
    index, tiles = [], []
    for n, (sha, m, other_copies) in enumerate(cands):
        im = Image.open(DATA / "idd117k_detection_v2" / m["image"]).convert("RGB")
        W, H = im.size
        animals = [b for b in m["boxes"] if b[0] == ANIMAL]
        px = [(b[3] * W, b[4] * H) for b in animals]
        x0 = min((b[1] - b[3] / 2) * W for b in animals); x1 = max((b[1] + b[3] / 2) * W for b in animals)
        y0 = min((b[2] - b[4] / 2) * H for b in animals); y1 = max((b[2] + b[4] / 2) * H for b in animals)
        cw, ch = max((x1 - x0) * 2.2, 320), max((y1 - y0) * 2.2, 240)
        scale = max(cw / TILE_W, ch / TILE_H)
        cw, ch = TILE_W * scale, TILE_H * scale  # keep the tile aspect ratio
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        left = min(max(0, cx - cw / 2), max(0, W - cw)); top = min(max(0, cy - ch / 2), max(0, H - ch))
        crop = im.crop((int(left), int(top), int(min(W, left + cw)), int(min(H, top + ch))))
        k = TILE_W / (cw)  # source pixel -> tile pixel
        tile = Image.new("RGB", (TILE_W, TILE_H))
        tile.paste(crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k)))), (0, 0))
        d = ImageDraw.Draw(tile)
        for i, b in enumerate(animals):
            bx0, by0 = ((b[1] - b[3] / 2) * W - left) * k, ((b[2] - b[4] / 2) * H - top) * k
            bx1, by1 = ((b[1] + b[3] / 2) * W - left) * k, ((b[2] + b[4] / 2) * H - top) * k
            d.rectangle([bx0, by0, bx1, by1], outline=(255, 0, 0), width=2)
            d.text((bx0 + 2, max(0, by0 - 11)), str(i + 1), fill=(255, 255, 0))
        d.rectangle([0, TILE_H - 14, TILE_W, TILE_H], fill=(0, 0, 0))
        d.text((4, TILE_H - 12), "id %03d  %d box(es)  px(w x h): %s" % (n, len(animals),
               ", ".join("%dx%d" % p for p in px[:4])), fill=(255, 255, 255))
        tiles.append(tile)
        index.append({"id": n, "image": m["image"], "label": m["label"], "drive": m["drive"], "other_copies": other_copies,
                      "image_size": [W, H], "animal_boxes_px_wh": [[round(a), round(b)] for a, b in px]})
    per = COLS * ROWS
    for s in range(0, len(tiles), per):
        sheet = Image.new("RGB", (COLS * TILE_W, ROWS * TILE_H))
        for i, t in enumerate(tiles[s:s + per]):
            sheet.paste(t, ((i % COLS) * TILE_W, (i // COLS) * TILE_H))
        sheet.save(output / "sheets" / ("sheet_%02d_ids_%03d-%03d.jpg" % (s // per, s, min(s + per, len(tiles)) - 1)), quality=90)
    (output / "candidates.json").write_text(json.dumps({"count": len(cands), "candidates": index}, indent=1), encoding="utf-8")
    print("candidates", len(cands), "sheets", (len(tiles) + per - 1) // per)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
