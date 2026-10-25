"""Finish the official CARLA package installation without changing ML environments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "setup/carla/downloads/CARLA_0.9.16.zip"
TARGET = ROOT / "tools/carla/0.9.16"
EXPECTED_BYTES = 7812252810
URL = "https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip"


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="Wait for the already-running download")
    args = parser.parse_args()
    if TARGET.exists():
        record = ROOT / "setup/carla/installation.json"
        if record.exists() and json.loads(record.read_text(encoding="utf-8")).get("status") == "installed":
            print("Already installed; leaving this installation unchanged.", flush=True)
            return
        raise RuntimeError(f"Existing unverified installation requires inspection: {TARGET}")
    if args.wait:
        import time
        deadline = time.monotonic() + 3600
        while not ARCHIVE.exists() or ARCHIVE.stat().st_size < EXPECTED_BYTES:
            if time.monotonic() > deadline:
                raise TimeoutError("Download not complete after one hour; resume it before retrying.")
            time.sleep(5)
        # A completed file must remain stable before opening it.
        time.sleep(3)
    if not ARCHIVE.exists() or ARCHIVE.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("Official 7,812,252,810-byte archive has not finished downloading.")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print("Hashing downloaded archive...", flush=True)
    with ARCHIVE.open("rb") as archive_stream:
        digest = hashlib.file_digest(archive_stream, "sha256").hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = TARGET.parent / f"0.9.16.extracting_{stamp}"
    with zipfile.ZipFile(ARCHIVE) as package:
        members = package.infolist()
        required = sum(item.file_size for item in members)
        if shutil.disk_usage(TARGET.parent).free < required + 2 * 1024**3:
            raise RuntimeError(f"Need {required / 1024**3:.1f} GiB plus 2 GiB free for extraction.")
        staging.mkdir()
        for item in members:
            name = item.filename.replace("\\", "/")
            dest = (staging / name).resolve()
            if not dest.is_relative_to(staging.resolve()) or stat.S_ISLNK(item.external_attr >> 16):
                raise RuntimeError(f"Unsafe archive entry: {name}")
        print(f"Extracting {len(members)} entries ({required / 1024**3:.1f} GiB); CRC checked while reading...", flush=True)
        package.extractall(staging)
    candidates = [p for p in staging.rglob("CarlaUE4.exe") if "Binaries" not in p.parts]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one top-level CARLA launcher, found {candidates}")
    relative_exe = candidates[0].relative_to(staging)
    os.replace(staging, TARGET)
    record = {
        "status": "installed", "version": "0.9.16", "platform": "Windows",
        "source_url": URL, "archive_bytes": EXPECTED_BYTES, "archive_sha256": digest,
        "hash_note": "Locally computed integrity record; no publisher SHA256 was supplied.",
        "archive_crc_checked_during_extraction": True,
        "uncompressed_bytes": required, "entries": len(members),
        "root": str(TARGET), "executable": str(TARGET / relative_exe),
        "installed_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(ROOT / "setup/carla/installation.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
