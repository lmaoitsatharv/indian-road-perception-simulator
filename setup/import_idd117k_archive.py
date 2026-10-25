"""Validate and safely stream the five-part IDD archive without a second big copy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 4 * 1024**2


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or not path.parts or any(p.endswith((" ", ".")) for p in path.parts)):
        raise ValueError("Unsafe archive path: " + name)
    if path.parts[0] != "IDD117K_Detection":
        raise ValueError("Unexpected archive root: " + name)
    return path.as_posix().rstrip("/")


class SplitReader(io.RawIOBase):
    """Sequential compressed bytes, with an independent hash for every part."""
    def __init__(self, folder, records):
        super().__init__()
        self.folder, self.records = Path(folder), records
        self.index, self.stream = -1, None
        self.total, self.current = 0, 0
        self.digest = None
        self.verified = []
        for row in records:
            if Path(row["name"]).name != row["name"]:
                raise ValueError("Part name must be a filename")
            if (self.folder / row["name"]).stat().st_size != row["bytes"]:
                raise ValueError("Part size differs from the completed download")

    def readable(self):
        return True

    def _advance(self):
        if self.stream is not None:
            row = self.records[self.index]
            self.stream.close()
            self.stream = None
            actual = self.digest.hexdigest()
            if actual != row["sha256"] or self.current != row["bytes"]:
                raise ValueError("Part hash/size mismatch: " + row["name"])
            self.verified.append({**row, "verified": True})
        self.index += 1
        if self.index == len(self.records):
            return False
        self.stream = (self.folder / self.records[self.index]["name"]).open("rb")
        self.digest, self.current = hashlib.sha256(), 0
        return True

    def readinto(self, buffer):
        if self.stream is None and self.index >= len(self.records):
            return 0
        if self.stream is None and not self._advance():
            return 0
        count = self.stream.readinto(buffer)
        if count == 0:
            if not self._advance():
                return 0
            return self.readinto(buffer)
        self.digest.update(memoryview(buffer)[:count])
        self.current += count
        self.total += count
        return count

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()


def metadata_stats(name, data, stats, examples):
    parts = PurePosixPath(name).parts
    family = parts[1] if len(parts) > 1 else "root"
    key = family + "/" + PurePosixPath(name).suffix.lower()
    section = stats.setdefault(key, {"files": 0, "objects": 0, "labels": {}, "flags": {},
        "object_keys": {}, "box_keys": {}, "coordinate_min": {}, "xml_sizes": {}, "errors": []})
    section["files"] += 1
    try:
        text = data.decode("utf-8-sig")
        if name.lower().endswith(".json"):
            objects = json.loads(text)
            if not isinstance(objects, list):
                raise ValueError("JSON is not an object list")
        else:
            if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
                raise ValueError("XML DTD/entity")
            root = ET.fromstring(text)
            size = root.find("size")
            if size is not None:
                dims = "x".join(size.findtext(k, "?") for k in ("width", "height", "depth"))
                section["xml_sizes"][dims] = section["xml_sizes"].get(dims, 0) + 1
            objects = [{"name": obj.findtext("name", ""),
                "bbox": {k: obj.findtext("bndbox/"+k, "nan") for k in ("xmin", "ymin", "xmax", "ymax")},
                **{k: obj.findtext(k, "0") for k in ("difficult", "ignore", "iscrowd", "truncated")}}
                for obj in root.findall("object")]
        for obj in objects:
            label = str(obj.get("name", ""))
            section["labels"][label] = section["labels"].get(label, 0) + 1
            section["objects"] += 1
            for k in obj:
                section["object_keys"][k] = section["object_keys"].get(k, 0) + 1
            for k, v in obj.get("bbox", {}).items():
                section["box_keys"][k] = section["box_keys"].get(k, 0) + 1
                try:
                    number = float(v)
                    section["coordinate_min"][k] = min(section["coordinate_min"].get(k, number), number)
                except (ValueError, TypeError):
                    pass
            for flag in ("difficult", "ignore", "iscrowd", "crowd", "truncated"):
                if flag in obj:
                    flag_key = flag + "=" + str(obj[flag])
                    section["flags"][flag_key] = section["flags"].get(flag_key, 0) + 1
        split = next((p for p in parts if p.lower() in {"train", "val", "test"}), "unknown")
        example_key = key + "/" + split
        if len(examples.setdefault(example_key, [])) < 2:
            examples[example_key].append({"path": name, "text": text})
    except (ValueError, TypeError, ET.ParseError, UnicodeError) as exc:
        if len(section["errors"]) < 20:
            section["errors"].append({"path": name, "error": str(exc)})


def run(folder, run_dir, mode, destination=None):
    folder, run_dir = Path(folder).resolve(), Path(run_dir).resolve()
    if not run_dir.is_relative_to(ROOT / "runs"):
        raise ValueError("Evidence belongs in this project's runs folder")
    record_path = folder / "download_verification.json"
    record = json.loads(record_path.read_text())
    records = sorted(record["files"], key=lambda r: r["part"])
    if not record.get("passed") or [r["part"] for r in records] != [1, 2, 3, 4, 5]:
        raise ValueError("Need the verified five-part download")
    run_dir.mkdir(parents=True, exist_ok=True)
    if mode == "inventory":
        if (run_dir / "inventory.jsonl").exists():
            raise FileExistsError("Preserve the previous inventory")
    else:
        inventory = json.loads((run_dir / "archive_integrity.json").read_text())
        if not inventory.get("passed") or inventory["part_records"] != records:
            raise ValueError("Inventory must verify this exact archive before extraction")
        destination = Path(destination).resolve()
        if not destination.is_relative_to(ROOT / "dataset"):
            raise ValueError("Extract into a fresh project dataset folder")
        reserve = 8 * 1024**3
        required = inventory["regular_file_bytes"] + reserve
        if shutil.disk_usage(destination.parent).free < required:
            raise OSError(f"Need {required} free bytes including reserve before extraction")
        destination.mkdir(parents=True, exist_ok=False)
    source = SplitReader(folder, records)
    names, groups, suffixes, stats, examples = set(), {}, Counter(), {}, {}
    count = regular = total = 0
    start = last_update = time.monotonic()
    progress_path = run_dir / (mode + "_progress.json")
    inventory_stream = (run_dir / "inventory.jsonl").open("w", encoding="utf-8") if mode == "inventory" else None
    report = {"mode": mode, "started_utc": utc(), "passed": False, "training_started": False,
        "image_pixels_decoded": False, "part_records": records}
    try:
        with io.BufferedReader(source, BLOCK) as raw, gzip.GzipFile(fileobj=raw, mode="rb") as uncompressed:
            with tarfile.open(fileobj=uncompressed, mode="r|", bufsize=BLOCK) as archive:
                for member in archive:
                    name = safe_name(member.name)
                    if not (member.isdir() or member.isfile()):
                        raise ValueError("Only regular files/directories accepted: " + name)
                    if member.size < 0:
                        raise ValueError("Negative member length")
                    if member.isfile():
                        folded = name.casefold()
                        if folded in names:
                            raise ValueError("Duplicate/case-colliding archive file: " + name)
                        names.add(folded)
                        regular += 1
                        total += member.size
                        suffixes[PurePosixPath(name).suffix.lower()] += 1
                    count += 1
                    parts = PurePosixPath(name).parts
                    group_key = "/".join(parts[:min(4, len(parts))])
                    group = groups.setdefault(group_key, {"files": 0, "bytes": 0, "examples": []})
                    if member.isfile():
                        group["files"] += 1
                        group["bytes"] += member.size
                        if len(group["examples"]) < 2:
                            group["examples"].append(name)
                    if inventory_stream:
                        inventory_stream.write(json.dumps({"path": name, "bytes": member.size,
                            "kind": "file" if member.isfile() else "directory"}) + "\n")
                    if mode == "extract":
                        target = destination.joinpath(*parts)
                        if not target.resolve().is_relative_to(destination):
                            raise ValueError("Extraction path escapes destination")
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with archive.extractfile(member) as src, target.open("xb") as dst:
                                shutil.copyfileobj(src, dst, BLOCK)
                            if target.stat().st_size != member.size:
                                raise ValueError("Extracted file length mismatch")
                    elif (member.isfile() and member.size <= 5 * 1024**2
                          and PurePosixPath(name).suffix.lower() in {".json", ".xml"}
                          and "test" not in [p.lower() for p in parts]):
                        metadata_stats(name, archive.extractfile(member).read(), stats, examples)
                    now = time.monotonic()
                    if now - last_update >= 20:
                        progress = {"mode": mode, "updated_utc": utc(), "members": count, "files": regular,
                            "uncompressed_file_bytes": total, "compressed_bytes_read": source.total,
                            "compressed_total_bytes": sum(r["bytes"] for r in records),
                            "compressed_percent": round(100 * source.total / sum(r["bytes"] for r in records), 2),
                            "elapsed_seconds": round(now - start, 1), "current_path": name}
                        write_json(progress_path, progress)
                        print(json.dumps(progress), flush=True)
                        last_update = now
            # Tar stops at its end marker. Drain gzip to EOF to verify CRC and ISIZE,
            # and finish hashing every compressed part, including archive padding.
            while uncompressed.read(BLOCK):
                pass
        if len(source.verified) != 5:
            raise ValueError("The gzip stream did not consume every downloaded part")
        if mode == "extract" and (regular != inventory["regular_files"] or total != inventory["regular_file_bytes"]):
            raise ValueError("Extraction differs from the verified inventory")
        report.update({"passed": True, "completed_utc": utc(), "elapsed_seconds": round(time.monotonic()-start, 1),
            "members": count, "regular_files": regular, "regular_file_bytes": total,
            "compressed_bytes": source.total, "verified_parts": source.verified,
            "gzip_crc_and_length_verified": True, "suffix_counts": dict(suffixes), "groups": groups})
        if destination:
            report["destination"] = str(destination)
        else:
            write_json(run_dir / "annotation_survey.json", stats)
            write_json(run_dir / "annotation_examples.json", examples)
    except Exception as exc:
        report.update({"failed_utc": utc(), "error": type(exc).__name__+": "+str(exc),
            "members": count, "regular_file_bytes": total, "compressed_bytes": source.total})
        raise
    finally:
        if inventory_stream:
            inventory_stream.close()
        write_json(run_dir / ("archive_integrity.json" if mode == "inventory" else "extraction.json"), report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"groups", "verified_parts", "part_records"}}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "dataset/idd117k_raw")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=("inventory", "extract"), required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--wait-for-inventory", action="store_true")
    args = parser.parse_args()
    if args.mode == "extract" and args.destination is None:
        parser.error("Extraction requires a fresh destination")
    if args.wait_for_inventory:
        if args.mode != "extract":
            parser.error("Only extraction can wait for its prerequisite inventory")
        print("Waiting for full archive verification before extraction", flush=True)
        while not (args.run/"archive_integrity.json").exists():
            time.sleep(10)
    run(args.folder, args.run, args.mode, args.destination)
