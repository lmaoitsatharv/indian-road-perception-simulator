"""Check the Git index for common accidental disclosures before sharing.

This intentionally reads staged bytes, not potentially different working files.
It is a conservative sharing guard, not a complete secret scanner or security audit.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BLOCKED_ROOTS = {"dataset", "media", "runs", "archive", "tools", "share", "preprocessing", "output", "tmp", "private",
                 "motion-film", "null-breakdown",
                 ".venv", ".venv-carla", ".aws"}
BLOCKED_SUFFIXES = {".keras", ".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".pem", ".key",
                    ".p12", ".pfx", ".exe", ".dll", ".zip", ".mp4", ".avi", ".log", ".pyc"}
PATTERNS = {
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "Service API token": re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{35,}"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Private key": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "Credential in URL": re.compile(r"https?://[^\s/:'\"]+:[^\s/@'\"]+@"),
    "Literal credential": re.compile(r'''(?i)\b(?:api_key|access_token|password|client_secret)\s*[:=]\s*["'](?!REPLACE|YOUR_|example|placeholder)[A-Za-z0-9_+/=-]{20,}["']'''),
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    files = git("ls-files", "--stage", "-z").split(b"\0")
    failures, count, total = [], 0, 0
    ids = [record.split(b"\t", 1)[0].split()[1] for record in files if record]
    packed = subprocess.check_output(["git", "cat-file", "--batch"],
                                     input=b"\n".join(ids) + b"\n", cwd=ROOT)
    blobs, offset = {}, 0
    for oid in ids:
        end = packed.index(b"\n", offset)
        header = packed[offset:end].split()
        size = int(header[2])
        offset = end + 1
        blobs[oid.decode()] = packed[offset:offset + size]
        offset += size + 1
    staged_by_path = {record.split(b"\t", 1)[1].decode("utf-8"):
                      blobs[record.split(b"\t", 1)[0].split()[1].decode()]
                      for record in files if record}
    texture_hashes = {}
    asset_manifest = "integration/indian_world_assets/manifest.json"
    if asset_manifest in staged_by_path:
        manifest = json.loads(staged_by_path[asset_manifest].decode("utf-8"))
        for item in manifest.get("textures", []):
            for name_key, hash_key in (("file", "sha256"), ("banner_file", "banner_sha256")):
                if name_key in item:
                    name = item[name_key]
                    if Path(name).name != name or not name.endswith(".png"):
                        failures.append({"path": asset_manifest, "reason": "Invalid texture path"})
                        continue
                    texture_hashes["integration/indian_world_assets/textures/" + name] = item[hash_key]
        for name, digest in texture_hashes.items():
            if name not in staged_by_path or hashlib.sha256(staged_by_path[name]).hexdigest() != digest:
                failures.append({"path": name, "reason": "Staged texture does not match staged asset manifest"})
    for record in files:
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        mode, oid, stage = meta.decode().split()
        relative = raw_path.decode("utf-8")
        path = Path(relative)
        count += 1
        if stage != "0" or mode not in ("100644", "100755"):
            failures.append({"path": relative, "reason": "Unmerged, linked or special entry"})
            continue
        hidden_local_dir = len(path.parts) > 1 and path.parts[0].startswith(".") and path.parts[0] != ".github"
        if (path.parts[0] in BLOCKED_ROOTS or hidden_local_dir or path.suffix.lower() in BLOCKED_SUFFIXES
                or any(p == "__pycache__" for p in path.parts)
                or (path.name.startswith(".env") and path.name != ".env.example")):
            failures.append({"path": relative, "reason": "Excluded asset or credential path"})
        data = blobs[oid]
        size = len(data)
        total += size
        if size > 8 * 1024**2:
            failures.append({"path": relative, "reason": "Over 8 MiB; review large assets separately"})
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            documented_texture = (relative in texture_hashes or relative ==
                                  "integration/indian_world_assets/textures/uv_grid.png")
            allowed_png = (path.suffix.lower() == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n")
                           and (path.parts[:2] == ("docs", "results") or documented_texture))
            if not allowed_png:
                failures.append({"path": relative, "reason": "Unexpected binary file"})
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                failures.append({"path": relative, "line": text.count("\n", 0, match.start()) + 1,
                                 "reason": label})
        if path.suffix == ".ipynb":
            notebook = json.loads(text)
            if any(cell.get("outputs") or cell.get("execution_count") is not None
                   for cell in notebook["cells"] if cell["cell_type"] == "code"):
                failures.append({"path": relative, "reason": "Notebook has saved execution output"})
    print(json.dumps({"passed": bool(count) and not failures, "staged_files": count,
                      "staged_bytes": total, "findings": failures}, indent=2))
    raise SystemExit(0 if count and not failures else 1)


if __name__ == "__main__":
    main()
