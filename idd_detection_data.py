"""Framework-independent IDD box conversion and dataset audit.

The observed 95K JSON is a list of {name, bbox:{xmin,ymin,xmax,ymax}}.
Legacy VOC XML is accepted separately. Coordinates are never inferred from a
filename or assumed to be 1920x1080. No road-model module is imported here.
"""
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parent
MANIFEST_FIELDS = ("image", "annotation", "format", "split", "family", "drive_id", "source_id", "box_convention")


class AnnotationError(ValueError):
    pass


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_policy(path=ROOT / "detection_policy.json"):
    policy = json.loads(Path(path).read_text())
    if (policy.get("schema") != "idd_detection_policy_v1" or policy.get("vehicle_control_allowed") is not False
            or len(set(policy["names"])) != len(policy["names"])):
        raise ValueError("Invalid observation-only detection policy")
    if not set(policy["source_mapping"].values()).issubset(policy["names"]):
        raise ValueError("Class mapping refers to an unknown output class")
    if set(policy["background_labels"]) & set(policy["source_mapping"]):
        raise ValueError("A label cannot be foreground and background")
    return policy


def normal_name(name, policy):
    name = " ".join(str(name).strip().lower().replace("_", " ").split())
    return policy.get("aliases", {}).get(name, name)


def annotation_objects(text, fmt, width, height):
    if len(text.encode("utf-8")) > 5 * 1024**2:
        raise AnnotationError("annotation_too_large")
    if fmt == "idd_json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise AnnotationError("Expected the observed IDD JSON list schema")
        if not all(isinstance(o, dict) and "name" in o and isinstance(o.get("bbox"), dict) for o in data):
            raise AnnotationError("Invalid IDD JSON object schema")
        return data
    if fmt != "voc_xml":
        raise AnnotationError("Unsupported annotation format")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise AnnotationError("XML entities/DTDs are not accepted")
    root = ET.fromstring(text)
    size = root.find("size")
    if size is None or (int(size.findtext("width", "0")), int(size.findtext("height", "0"))) != (width, height):
        raise AnnotationError("XML dimensions differ from actual image dimensions")
    output = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            raise AnnotationError("Object has no bounding box")
        output.append({"name": obj.findtext("name", ""),
            "bbox": {k: float(box.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")},
            **{flag: obj.findtext(flag, "0") for flag in ("difficult", "ignore", "iscrowd")}})
    return output


def convert_annotation(text, fmt, width, height, convention, policy=None):
    policy = policy or load_policy()
    if width <= 0 or height <= 0 or convention not in {"xyxy_edges", "voc_1based_inclusive"}:
        raise AnnotationError("Native dimensions and explicit coordinate convention are required")
    output, background, seen = [], {}, set()
    clipped = duplicates = 0
    for raw in annotation_objects(text, fmt, width, height):
        label = normal_name(raw["name"], policy)
        target = policy["source_mapping"].get(label)
        if target is None:
            if label not in policy["background_labels"]:
                raise AnnotationError("unknown_label:" + label)
            background[label] = background.get(label, 0) + 1
            continue
        for flag in ("difficult", "ignore", "iscrowd", "crowd"):
            value = raw.get(flag, 0)
            if value not in (False, 0, 0., "0", "false", "False", None):
                raise AnnotationError("flag_requires_explicit_ignore_policy:" + flag)
        try:
            coords = [float(raw["bbox"][k]) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (KeyError, TypeError, ValueError):
            raise AnnotationError("Invalid box coordinate fields") from None
        if not all(math.isfinite(v) for v in coords):
            raise AnnotationError("Non-finite box coordinate")
        if convention == "voc_1based_inclusive":
            coords[0] -= 1
            coords[1] -= 1
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            raise AnnotationError("Degenerate or reversed box")
        visible = [max(0., min(float(width), x1)), max(0., min(float(height), y1)),
                   max(0., min(float(width), x2)), max(0., min(float(height), y2))]
        if visible[2] <= visible[0] or visible[3] <= visible[1]:
            raise AnnotationError("Foreground box is entirely outside the image")
        clipped += int(visible != coords)
        cls = policy["names"].index(target)
        key = (cls, *visible)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        x1, y1, x2, y2 = visible
        xywh = [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
        output.append({"class_id": cls, "class_name": target, "source_name": label,
                       "xyxy": visible, "xywh": xywh})
    return {"objects": output, "background_labels": background, "clipped_boxes": clipped,
            "duplicate_boxes_removed": duplicates, "negative": not output}


def yolo_lines(converted):
    return "".join(str(obj["class_id"]) + " " + " ".join(f"{v:.9f}" for v in obj["xywh"]) + "\n"
                   for obj in converted["objects"])


def source_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
        raise ValueError("Manifest paths must be relative to the designated source folder")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes the dataset root")
    return path


def read_manifest(path, *, allow_drive_conflicts=False):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(MANIFEST_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires: " + ", ".join(MANIFEST_FIELDS))
        records = list(reader)
    if not records:
        raise ValueError("Empty source manifest")
    identities, drives = {}, {}
    for row in records:
        if any(not row[k].strip() for k in MANIFEST_FIELDS):
            raise ValueError("Manifest metadata may not be empty")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Keep the provider train/val/test identities")
        if row["format"] not in {"idd_json", "voc_xml"} or row["box_convention"] not in {"xyxy_edges", "voc_1based_inclusive"}:
            raise ValueError("Annotation format/convention must be explicit")
        identity = (row["family"], row["source_id"])
        if identity in identities:
            raise ValueError("Duplicate source identity in manifest")
        identities[identity] = row["split"]
        drives.setdefault((row["family"], row["drive_id"]), set()).add(row["split"])
    conflicts = [{"family": k[0], "drive_id": k[1], "splits": sorted(v)} for k, v in drives.items() if len(v) > 1]
    if conflicts and not allow_drive_conflicts:
        raise ValueError("Drive sequence crosses provider splits; resolve before export: " + json.dumps(conflicts[:5]))
    return records


def inspect_pair(row, source_root, policy):
    """Read one labelled source pair; workers never write dataset files."""
    try:
        image_path = source_path(source_root, row["image"])
        annotation_path = source_path(source_root, row["annotation"])
        with Image.open(image_path) as image:
            if image.getexif().get(274, 1) != 1 or image.mode != "RGB":
                raise AnnotationError("Unsupported orientation/colour mode; coordinate review required")
            width, height = image.size
            image.load()
            pixel_digest = hashlib.sha256(f"RGB:{width}x{height}:".encode())
            pixel_digest.update(image.tobytes())
            pixel_sha = pixel_digest.hexdigest()
        image_sha = file_sha(image_path)
        if annotation_path.stat().st_size > 5 * 1024**2:
            raise AnnotationError("annotation_too_large")
        annotation_sha = file_sha(annotation_path)
        converted = convert_annotation(annotation_path.read_text(encoding="utf-8-sig"), row["format"],
            width, height, row["box_convention"], policy)
        return {"image_path": image_path, "width": width, "height": height, "pixel_sha": pixel_sha,
            "image_sha": image_sha, "annotation_sha": annotation_sha, "converted": converted}
    except (AnnotationError, OSError, ValueError, KeyError, ET.ParseError) as error:
        return {"error": str(error)}


def inspected_records(records, source_root, policy, workers):
    labelled = iter(r for r in records if r["split"] != "test")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while batch := list(itertools.islice(labelled, max(32, workers * 16))):
            # Bounded batches avoid queuing 117K futures and keep memory bounded.
            inspected = executor.map(lambda row: inspect_pair(row, source_root, policy), batch)
            yield from zip(batch, inspected)


def prepare_dataset(source_root, manifest, output, *, policy_path=ROOT / "detection_policy.json", download_record=None,
                    workers=1, source_index=None):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    policy_path = Path(policy_path).resolve()
    if not 1 <= workers <= 16:
        raise ValueError("Use 1 to 16 bounded image-decoding workers")
    if download_record is None:
        if not output.is_relative_to(ROOT / "runs"):
            raise ValueError("Unverified fixtures must stay under runs/")
        scope = "fixture_only_not_for_training"
    else:
        record = json.loads(Path(download_record).read_text())
        if record.get("passed") is not True or len(record.get("files", [])) != 5:
            raise ValueError("The complete five-part download must pass verification first")
        if source_index is None:
            raise ValueError("Real exports require the full verified source index")
        index = json.loads(Path(source_index).read_text())
        if (index.get("schema") != "idd117k_source_index_v1" or not index.get("passed") or
            index.get("manifest_sha256") != file_sha(manifest) or Path(index["source_root"]).resolve() != source_root):
            raise ValueError("Source index is incomplete or differs from this manifest/source directory")
        evidence = {"source_index": Path(source_index), "archive_integrity": Path(index["archive_integrity_file"]),
                    "extraction": Path(index["extraction_report_file"])}
        for key, expected_key in (("archive_integrity", "archive_integrity_sha256"), ("extraction", "extraction_report_sha256")):
            if file_sha(evidence[key]) != index[expected_key] or not json.loads(evidence[key].read_text()).get("passed"):
                raise ValueError("Full archive/extraction evidence changed")
        scope = "real_idd_detection"
    records = read_manifest(manifest)
    policy = load_policy(policy_path)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}
    if download_record is not None:
        (output/"provenance").mkdir()
        for key, path in evidence.items():
            shutil.copy2(path, output/"provenance"/(key+".json"))
            provenance[key] = {"file": "provenance/"+key+".json", "sha256": file_sha(path)}
    report = {"schema": "idd_detection_dataset_v1", "scope": scope, "passed": False,
        "training_started": False, "names": policy["names"], "source_manifest_sha256": file_sha(manifest),
        "policy_sha256": file_sha(policy_path), "policy_file": str(policy_path),
        "download_verification_sha256": file_sha(download_record) if download_record else None,
        "counts": {"train": 0, "val": 0, "test_metadata_only": 0},
        "objects": {s: {n: 0 for n in policy["names"]} for s in ("train", "val")},
        "negatives": {"train": 0, "val": 0}, "quarantine": [], "cross_split_duplicates": [],
        "clipped_boxes": 0, "duplicate_boxes_removed": 0, "link_count": 0, "copy_count": 0,
        "source_images_resized": False, "model_control": False}
    report["archive_provenance"] = provenance
    report["provider_counts"] = {"train": sum(r["split"]=="train" for r in records),
        "val":sum(r["split"]=="val" for r in records), "test_metadata_only":sum(r["split"]=="test" for r in records)}
    report["curated_manifest_counts"] = dict(report["provider_counts"])
    if download_record is not None and "provider_counts" in index:
        report["provider_counts"] = {"test_metadata_only" if s=="test" else s:n for s,n in index["provider_counts"].items()}
        report["source_exclusions"] = index.get("curation", {})
    save_json(output / "preparation.json", report)
    report["counts"]["test_metadata_only"] = report["curated_manifest_counts"]["test_metadata_only"]
    seen_bytes, seen_pixels = {}, {}
    started = updated = time.monotonic()
    processed = 0
    pair_stream = (output / "pairs.jsonl").open("w", encoding="utf-8")
    for row, inspection in inspected_records(records, source_root, policy, workers):
        processed += 1
        if time.monotonic() - updated >= 20:
            progress = {"processed_labelled_images": processed, "total_labelled_images": len(records)-report["counts"]["test_metadata_only"],
                "counts": report["counts"], "quarantined_images": len(report["quarantine"]),
                "cross_split_duplicate_detections": len(report["cross_split_duplicates"]),
                "elapsed_seconds": round(time.monotonic()-started, 1), "training_started": False}
            save_json(output / "progress.json", progress)
            pair_stream.flush()
            print(json.dumps(progress), flush=True)
            updated = time.monotonic()
        try:
            if "error" in inspection:
                raise AnnotationError(inspection["error"])
            image_path = inspection["image_path"]
            width, height = inspection["width"], inspection["height"]
            image_sha, pixel_sha = inspection["image_sha"], inspection["pixel_sha"]
            annotation_sha, converted = inspection["annotation_sha"], inspection["converted"]
            for kind, seen, digest in (("encoded_bytes", seen_bytes, image_sha), ("decoded_pixels", seen_pixels, pixel_sha)):
                if digest in seen and seen[digest]["split"] != row["split"]:
                    report["cross_split_duplicates"].append({"kind": kind, "sha256": digest,
                        "first": seen[digest], "second": {k: row[k] for k in ("family", "source_id", "split", "image")}})
                seen[digest] = {k: row[k] for k in ("family", "source_id", "split", "image")}
            key = hashlib.sha256((row["family"] + "/" + row["source_id"]).encode()).hexdigest()[:24]
            image_target = output / "images" / row["split"] / (key + image_path.suffix.lower())
            label_target = output / "labels" / row["split"] / (key + ".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(image_path, image_target)
                report["link_count"] += 1
            except OSError:
                shutil.copy2(image_path, image_target)
                report["copy_count"] += 1
            label_target.write_text(yolo_lines(converted), encoding="utf-8")
            report["counts"][row["split"]] += 1
            report["negatives"][row["split"]] += int(converted["negative"])
            report["clipped_boxes"] += converted["clipped_boxes"]
            report["duplicate_boxes_removed"] += converted["duplicate_boxes_removed"]
            for obj in converted["objects"]:
                report["objects"][row["split"]][obj["class_name"]] += 1
            pair = {**row, "native_width": width, "native_height": height,
                "image_sha256": image_sha, "annotation_sha256": annotation_sha, "pixels_sha256": pixel_sha,
                "prepared_image": image_target.relative_to(output).as_posix(),
                "prepared_label": label_target.relative_to(output).as_posix(),
                "label_sha256": file_sha(label_target), "foreground_objects": len(converted["objects"]),
                "image_bytes":image_target.stat().st_size,
                "image_mtime_ms":image_target.stat().st_mtime_ns // 1000000,
                "background_labels": converted["background_labels"]}
            pair_stream.write(json.dumps(pair, allow_nan=False) + "\n")
        except (AnnotationError, OSError, ValueError, KeyError, ET.ParseError) as error:
            report["quarantine"].append({"family": row["family"], "source_id": row["source_id"],
                "image": row["image"], "annotation": row["annotation"], "split": row["split"], "reason": str(error)})
    pair_stream.close()
    # JSON is valid YAML. Omitting an absolute path keeps this portable between
    # Windows preparation and WSL training; Ultralytics resolves its YAML parent.
    data = {"train": "images/train", "val": "images/val", "names": dict(enumerate(policy["names"]))}
    save_json(output / "data.yaml", data)
    missing = [n for n in policy["names"] if not report["objects"]["train"][n] or not report["objects"]["val"][n]]
    report.update(missing_train_or_validation_classes=missing, manifest_file="pairs.jsonl",
        manifest_sha256=file_sha(output / "pairs.jsonl"), data_yaml_sha256=file_sha(output / "data.yaml"))
    report["passed"] = (not report["quarantine"] and not report["cross_split_duplicates"]
                        and not missing and report["counts"]["train"] > 0 and report["counts"]["val"] > 0)
    report["training_allowed"] = report["passed"] and scope == "real_idd_detection"
    report["elapsed_seconds"] = round(time.monotonic()-started, 1)
    report["decode_workers"] = workers
    save_json(output / "progress.json", {"status": "completed", "passed": report["passed"],
        "counts": report["counts"], "elapsed_seconds": report["elapsed_seconds"], "training_started": False})
    save_json(output / "preparation.json", report)
    return report
