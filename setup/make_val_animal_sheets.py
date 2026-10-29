"""Review sheets for the validation animal boxes not yet reviewed (read-only on datasets; writes a new runs/ folder).

The first review (runs/animal_filter_review_20261029_01) covered 330 of the 963 validation animal boxes.
This builds per-box tiles for the rest, using the same tile layout.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from make_animal_filter_sheets import V3, animal_boxes, sheets, val_tile

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "runs/animal_filter_review_20261029_01/val_boxes.json"


def key(stem, cx, cy):
    return stem, round(cx, 5), round(cy, 5)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "sheets").mkdir()
    reviewed = {key(m["stem"], m["cx"], m["cy"]) for m in json.load(open(FIRST))["items"]}
    with ThreadPoolExecutor(16) as pool:
        val = [(s, b) for s, b in pool.map(animal_boxes, sorted((V3 / "labels/val").glob("*.txt")), chunksize=64) if b]
    todo = [(stem, box) for stem, boxes in val for box in boxes if key(stem, box[0], box[1]) not in reviewed]
    total = sum(len(b) for _, b in val)
    tiles, meta = [], []
    for n, (stem, box) in enumerate(todo):
        t, m = val_tile(stem, box, n, 384, 288)
        tiles.append(t)
        meta.append(m)
    sheets(tiles, 5, 4, 384, 288, output / "sheets", "val2")
    (output / "val_boxes_remaining.json").write_text(json.dumps({"total_val_animal_boxes": total, "already_reviewed": total - len(todo),
                                                                  "items": meta}, indent=1))
    print("total", total, "already reviewed", total - len(todo), "to review", len(todo))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
