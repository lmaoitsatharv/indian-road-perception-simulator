"""Independently reopen saved YOLO labels and compare sample boxes to their source."""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import file_sha, load_policy, read_manifest, save_json, source_path


def box_sort(box):
    return (box[0], *(round(v, 3) for v in box[1:]))


def saved_label_path(parents, split, relative):
    """Use checked fixed parents; still reject traversal and non-regular labels."""
    path=PurePosixPath(relative)
    if (split not in parents or len(path.parts)!=3 or path.parts[:2]!=("labels",split)
            or path.suffix!=".txt" or "\\" in relative or ":" in relative or ".." in path.parts):
        raise ValueError("Saved label paths must stay in the fixed prepared split folders")
    target=parents[split]/path.name
    if not stat.S_ISREG(target.lstat().st_mode):
        raise ValueError("Saved labels must be regular files")
    return target


def original_boxes(path, fmt, convention, width, height, policy):
    if fmt == "idd_json":
        objects = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        root = ET.fromstring(path.read_text(encoding="utf-8-sig"))
        objects = [{"name": obj.findtext("name"), "bbox": {k: obj.findtext("bndbox/"+k)
            for k in ("xmin", "ymin", "xmax", "ymax")}} for obj in root.findall("object")]
    result = set()
    for obj in objects:
        name = " ".join(obj["name"].strip().lower().replace("_", " ").split())
        name = policy.get("aliases", {}).get(name, name)
        if name in policy["background_labels"]:
            continue
        cls = policy["names"].index(policy["source_mapping"][name])
        x1, y1, x2, y2 = (float(obj["bbox"][k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        if convention == "voc_1based_inclusive":
            x1 -= 1; y1 -= 1
        result.add((cls, max(0, min(width, x1)), max(0, min(height, y1)),
                    max(0, min(width, x2)), max(0, min(height, y2))))
    return sorted(result, key=box_sort)


def main(dataset, source, manifest, output):
    dataset, source, output = Path(dataset).resolve(), Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    policy = load_policy()
    preparation = json.loads((dataset/"preparation.json").read_text())
    label_parents={split:source_path(dataset,"labels/"+split) for split in ("train","val")}
    assert all(parent==dataset/"labels"/split and parent.is_dir() for split,parent in label_parents.items())
    provider = read_manifest(manifest)
    assert preparation["source_manifest_sha256"] == file_sha(manifest)
    assert preparation["manifest_sha256"] == file_sha(dataset/"pairs.jsonl")
    assert preparation["data_yaml_sha256"] == file_sha(dataset/"data.yaml")
    assert preparation["policy_sha256"] == file_sha(ROOT/"detection_policy.json")
    expected_provider = {"train": 0, "val": 0, "test_metadata_only": 0}
    for row in provider:
        expected_provider["test_metadata_only" if row["split"] == "test" else row["split"]] += 1
    assert expected_provider == preparation.get("curated_manifest_counts", preparation["provider_counts"])
    if preparation.get("archive_provenance"):
        source_index = json.loads((dataset/preparation["archive_provenance"]["source_index"]["file"]).read_text())
        raw_counts = source_index.get("provider_counts", source_index["counts"])
        assert preparation["provider_counts"] == {"test_metadata_only" if s=="test" else s:n for s,n in raw_counts.items()}
    counts = Counter()
    objects = {s: Counter() for s in ("train", "val")}
    negatives = Counter()
    sizes, families, samples = Counter(), {}, []
    label_hash = hashlib.sha256()
    maximum_roundtrip_error = 0.
    source_samples_checked = 0
    for index, line in enumerate((dataset/"pairs.jsonl").open(encoding="utf-8")):
        row = json.loads(line)
        split = row["split"]
        assert split in {"train", "val"}, "Official unlabelled test entered prepared pairs"
        w, h = row["native_width"], row["native_height"]
        counts[split] += 1
        sizes[f"{w}x{h}"] += 1
        families.setdefault(row["family"], Counter())[split] += 1
        label_path = saved_label_path(label_parents,split,row["prepared_label"])
        data = label_path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["label_sha256"]
        label_hash.update(data)
        boxes = []
        for label in data.decode("utf-8").splitlines():
            values = label.split()
            assert len(values) == 5
            cls = int(values[0])
            cx, cy, bw, bh = map(float, values[1:])
            assert 0 <= cls < len(policy["names"])
            assert all(math.isfinite(v) for v in (cx, cy, bw, bh)) and bw > 0 and bh > 0
            assert min(cx-bw/2, cy-bh/2) >= -1e-8 and max(cx+bw/2, cy+bh/2) <= 1+1e-8
            boxes.append((cls, (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
            objects[split][policy["names"][cls]] += 1
        assert len(boxes) == row["foreground_objects"]
        negatives[split] += not boxes
        if (index+1)%5000==0:
            print(json.dumps({"saved_labels_independently_checked":index+1}),flush=True)
        # Hash-based selection spans the complete manifest without opening test pixels.
        selected = int(row["image_sha256"][:8], 16) % 503 == 0
        group = (row["family"], split)
        group_count = sum((x["family"], x["split"]) == group for x in samples)
        if group_count < 3 or selected:
            source_samples_checked += 1
            image_path = source_path(source, row["image"])
            assert file_sha(image_path) == row["image_sha256"]
            with Image.open(image_path) as image:
                assert image.size == (w, h) and image.mode == "RGB"
                image.load()
                digest = hashlib.sha256(f"RGB:{w}x{h}:".encode())
                digest.update(image.tobytes())
                assert digest.hexdigest() == row["pixels_sha256"]
            prepared_image = source_path(dataset, row["prepared_image"])
            assert prepared_image.samefile(image_path) or file_sha(prepared_image) == row["image_sha256"]
            annotation = source_path(source, row["annotation"])
            assert file_sha(annotation) == row["annotation_sha256"]
            original = original_boxes(annotation, row["format"], row["box_convention"], w, h, policy)
            converted = sorted(boxes, key=box_sort)
            assert len(original) == len(converted)
            for raw_box, prepared_box in zip(original, converted):
                assert raw_box[0] == prepared_box[0]
                error = max(abs(a-b) for a, b in zip(raw_box[1:], prepared_box[1:]))
                maximum_roundtrip_error = max(maximum_roundtrip_error, error)
                assert error < 0.001, (row["source_id"], raw_box, prepared_box)
            if group_count < 3:
                sample = {**row, "source_boxes": original, "saved_boxes": converted}
                samples.append(sample)
    assert all(source_path(dataset,"labels/"+split)==parent for split,parent in label_parents.items())
    assert dict(counts) == {s: preparation["counts"][s] for s in counts}
    assert all(dict(objects[s]) == {k: v for k, v in preparation["objects"][s].items() if v} for s in objects)
    assert all(negatives[s] == preparation["negatives"][s] for s in negatives)
    # Each family/split has a source-versus-saved annotation preview.
    previews = output/"previews"
    previews.mkdir()
    cards = []
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    for i, row in enumerate(samples):
        with Image.open(source_path(source, row["image"])) as image:
            image.thumbnail((960, 540))
            left, right = image.copy(), image.copy()
        scale_x, scale_y = left.width/row["native_width"], left.height/row["native_height"]
        for panel, key, colour in ((left, "source_boxes", (50, 255, 110)), (right, "saved_boxes", (255, 195, 40))):
            draw = ImageDraw.Draw(panel)
            for cls, x1, y1, x2, y2 in row[key]:
                box = (x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y)
                draw.rectangle(box, outline=colour, width=2)
                draw.text((box[0]+2, max(0, box[1]-17)), policy["names"][cls], font=font, fill=colour, stroke_width=1, stroke_fill="black")
        canvas = Image.new("RGB", (left.width*2, left.height+34), "#141820")
        canvas.paste(left, (0, 34)); canvas.paste(right, (left.width, 34))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "Original annotation", font=font, fill=(50,255,110))
        draw.text((left.width+8, 8), "Saved YOLO labels reconstructed in pixels", font=font, fill=(255,195,40))
        filename = f"{i:02}_{row['family']}_{row['split']}.jpg"
        canvas.save(previews/filename, quality=90)
        cards.append(f"<figure><figcaption>{html.escape(row['family']+' / '+row['split']+' / '+row['source_id'])}</figcaption><a href='previews/{filename}'><img src='previews/{filename}' loading='lazy'></a></figure>")
    (output/"index.html").write_text("<!doctype html><meta charset='utf-8'><title>IDD detection preprocessing checks</title><style>body{background:#10141b;color:#eee;font:18px system-ui;max-width:1920px;margin:30px auto}img{width:100%}figure{margin:28px 0}figcaption{margin:10px}</style><h1>IDD detection preprocessing checks</h1><p>Original boxes and saved training labels. No model predictions, training or official test images.</p>"+"".join(cards), encoding="utf-8")
    report = {"passed": True, "training_started": False, "official_test_pixels_opened": False,
        "counts": dict(counts), "objects": {s: dict(v) for s, v in objects.items()},
        "negatives": dict(negatives), "native_sizes": dict(sizes), "families": {k: dict(v) for k, v in families.items()},
        "all_saved_labels_reopened": True, "combined_saved_label_sha256": label_hash.hexdigest(),
        "source_sample_rule": "first three per family/split plus SHA256 mod 503 = 0",
        "source_images_and_annotations_rechecked": source_samples_checked,
        "maximum_sample_box_error_pixels": maximum_roundtrip_error, "preview_images": len(samples),
        "preparation_ready": preparation["passed"], "source_manifest_sha256": file_sha(manifest)}
    save_json(output/"verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.source, args.manifest, args.output)
