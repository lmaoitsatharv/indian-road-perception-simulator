"""Verify a downloaded DATS_2022 zip against the SHA-256 values saved from the Mendeley metadata (read-only).

Seven filenames appear twice in the published file list with different contents (for example N_0.xml), so
matching is by content hash, not by name.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset/dats2022_raw"


def main(zip_path, report_path):
    meta = json.loads((RAW / "mendeley_metadata_v2.json").read_text(encoding="utf-8"))
    expected = {f["content_details"]["sha256_hash"]: (f["filename"], f["size"]) for f in meta["files"]}
    result = {"zip": str(zip_path), "zip_bytes": zip_path.stat().st_size, "expected_files": len(meta["files"]),
              "expected_unique_contents": len(expected), "matched": 0, "not_in_published_list": [], "missing": []}
    found = set()
    with zipfile.ZipFile(zip_path) as archive:
        result["zip_crc_error_first_bad_member"] = archive.testzip()  # the zip's own CRC check
        for info in archive.infolist():
            if info.is_dir():
                continue
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            hexdigest = digest.hexdigest()
            if hexdigest in expected:
                found.add(hexdigest)
            else:
                result["not_in_published_list"].append(info.filename)
    result["matched"] = len(found)
    result["missing"] = sorted(expected[h][0] for h in set(expected) - found)
    result["passed"] = not result["missing"] and not result["not_in_published_list"] and result["zip_crc_error_first_bad_member"] is None
    report_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=RAW / "download_verification.json")
    args = parser.parse_args()
    main(args.zip, args.report)
