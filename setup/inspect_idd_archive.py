"""Inspect a verified, already downloaded prefix without extracting the archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 16 * 1024 * 1024


def inspect(folder, output, prefix_mib=32, max_members=500):
    state = json.loads((folder / "download_state.json").read_text())
    first = state["files"]["1"]
    size = min(prefix_mib * 1024**2, first["size"])
    blocks = []
    with (folder / first["name"]).open("rb") as stream:
        for index in range((size + CHUNK - 1) // CHUNK):
            expected = first["chunks"].get(str(index))
            if not expected:
                break
            data = stream.read(min(CHUNK, first["size"] - index * CHUNK))
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("An archive prefix range differs from its saved hash")
            blocks.append(data)
    if not blocks:
        raise ValueError("No committed prefix is available yet")
    prefix = b"".join(blocks)[:size]
    members, texts = [], {}
    stop = "member_limit"
    try:
        with tarfile.open(fileobj=io.BytesIO(prefix), mode="r|gz") as archive:
            for member in archive:
                members.append({"name": member.name, "bytes": member.size,
                                "regular_file": member.isfile(), "directory": member.isdir()})
                lower = member.name.lower()
                # Read annotation/schema examples, including validation metadata;
                # never decode images or inspect official test annotations here.
                training_or_metadata = "test" not in lower.split("/")
                if (member.isfile() and member.size <= 2 * 1024**2 and
                    training_or_metadata and lower.endswith((".xml", ".json", ".txt", ".md", ".yaml"))
                    and len(texts) < 12):
                    texts[member.name] = archive.extractfile(member).read().decode("utf-8", errors="replace")
                if len(members) >= max_members:
                    break
    except (EOFError, tarfile.ReadError, OSError):
        stop = "end_of_verified_prefix_expected"
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "verified_archive_prefix_only",
              "full_archive_verified": False, "training_started": False, "prefix_bytes": len(prefix),
              "prefix_sha256": hashlib.sha256(prefix).hexdigest(), "stop_reason": stop,
              "member_sample": members, "annotation_schema_samples": texts,
              "image_pixels_opened": False, "performance_evaluated": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prefix_bytes": len(prefix), "members": len(members), "stop_reason": stop,
                      "first_members": members[:14], "text_files": list(texts)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "dataset/idd117k_raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-mib", type=int, choices=range(16, 129, 16), default=32)
    parser.add_argument("--max-members", type=int, choices=range(1, 50001), default=500)
    args = parser.parse_args()
    inspect(args.folder.resolve(), args.output.resolve(), args.prefix_mib, args.max_members)
