"""Audit the split PNG pairs and create display-only mask previews."""

import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent / "dataset"
OUTPUT = ROOT / "validation"


def main():
    with (ROOT / "splits" / "pair_splits.csv").open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    OUTPUT.mkdir(exist_ok=True)
    records, errors = [], []
    image_modes, mask_modes, dimensions = Counter(), Counter(), Counter()
    for index, pair in enumerate(pairs, 1):
        try:
            with Image.open(pair["split_image"]) as image:
                image.load()
                image_size, image_mode = image.size, image.mode
            with Image.open(pair["split_mask"]) as mask:
                mask.load()
                mask_size, mask_mode = mask.size, mask.mode
                pixels = np.asarray(mask)
                mask_values = np.unique(pixels).tolist()
                road_fraction = float(np.count_nonzero(pixels == 1) / pixels.size)
            problems = []
            if image_size != mask_size:
                problems.append("image and mask dimensions differ")
            if image_size != (512, 512) or mask_size != (512, 512):
                problems.append("dimensions are not 512 x 512")
            if pixels.ndim != 2:
                problems.append("mask is not a single channel")
            if not set(mask_values).issubset({0, 1}):
                problems.append(f"mask has unexpected values {mask_values}")
            if image_mode != "RGB":
                problems.append(f"image mode is {image_mode}, expected RGB")
            image_modes[image_mode] += 1
            mask_modes[mask_mode] += 1
            dimensions[f"{image_size[0]}x{image_size[1]}"] += 1
            records.append({
                "id": pair["id"], "split": pair["split"],
                "image_width": image_size[0], "image_height": image_size[1],
                "mask_width": mask_size[0], "mask_height": mask_size[1],
                "image_mode": image_mode, "mask_mode": mask_mode,
                "mask_values": ";".join(map(str, mask_values)),
                "road_fraction": road_fraction, "issues": "; ".join(problems),
            })
            if problems:
                errors.append({"id": pair["id"], "issues": problems})
        except Exception as error:
            errors.append({"id": pair["id"], "issues": [str(error)]})
        if index % 1000 == 0 or index == len(pairs):
            print(f"Decoded and audited {index}/{len(pairs)} pairs", flush=True)

    by_split = {}
    for split in ("train", "val", "test"):
        selected = [row for row in records if row["split"] == split]
        fractions = [row["road_fraction"] for row in selected]
        by_split[split] = {
            "pairs": len(selected),
            "all_zero_masks": sum(f == 0 for f in fractions),
            "all_one_masks": sum(f == 1 for f in fractions),
            "mean_road_fraction": float(np.mean(fractions)),
        }
    report = {
        "pairs_checked": len(pairs), "pairs_decoded": len(records),
        "issue_count": len(errors), "issues": errors,
        "image_dimensions": dict(dimensions),
        "image_modes": dict(image_modes), "mask_modes": dict(mask_modes),
        "mask_values_across_dataset": sorted({
            int(value) for row in records for value in row["mask_values"].split(";")
        }),
        "splits": by_split,
        "visual_review_scope": "Six seeded random pairs, two from each split; visual alignment is sampled, not verified for every pair.",
        "mask_preservation": "Source and split files are read only. White masks and green overlays are generated only for the preview.",
    }
    (OUTPUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if records:
        with (OUTPUT / "pair_validation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    rng = random.Random(42)
    samples = []
    for split in ("train", "val", "test"):
        samples.extend(rng.sample([pair for pair in pairs if pair["split"] == split], 2))
    tile, header, footer = 256, 38, 22
    panel_width, panel_height = tile * 3, header + tile + footer
    canvas = Image.new("RGB", (panel_width * 2, panel_height * 3), "#101722")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    for index, pair in enumerate(samples):
        x, y = (index % 2) * panel_width, (index // 2) * panel_height
        with Image.open(pair["split_image"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        with Image.open(pair["split_mask"]) as mask:
            road = np.asarray(mask) == 1
        overlay = rgb.copy()
        overlay[road] = (rgb[road] * 0.55 + np.array([40, 240, 110]) * 0.45).astype(np.uint8)
        views = [Image.fromarray(rgb), Image.fromarray(road.astype(np.uint8) * 255).convert("RGB"), Image.fromarray(overlay)]
        draw.text((x + 10, y + 9), f"{pair['split'].upper()} | Image_{pair['id']} | road: {road.mean():.1%}", fill="white", font=font)
        for column, (view, label) in enumerate(zip(views, ["Original image", "Mask: white = 1", "Road overlay"])):
            canvas.paste(view.resize((tile, tile), Image.Resampling.NEAREST if column == 1 else Image.Resampling.BILINEAR), (x + tile * column, y + header))
            draw.text((x + tile * column + 8, y + header + tile + 3), label, fill="#c3d1e6", font=small)
    canvas.save(OUTPUT / "sample_overlays.png")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Results: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
