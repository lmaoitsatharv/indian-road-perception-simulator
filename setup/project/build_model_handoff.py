"""Package the three retained road checkpoints without any image or mask data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    retained = json.loads((ROOT / "models/retained_models.json").read_text(encoding="utf-8"))
    items = {}
    for model in retained["models"]:
        items[model["checkpoint"]] = model["checkpoint_sha256"]
        items[model["run_config"]] = None
    encoder = retained["pretrained_encoder_dependency"]
    items[encoder["path"]] = encoder["sha256"]
    items["dataset/splits/pair_splits.csv"] = retained["manifest_sha256"]
    for name in ("retained_models.json", "segmentation_final_lock.json", "model_config.json"):
        items["models/" + name] = None
    manifest = []
    for relative, expected in items.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Missing or invalid handoff input: {relative}")
        actual = sha256(path)
        if expected and actual != expected:
            raise ValueError(f"Protected asset changed: {relative}")
        manifest.append({"path": relative, "sha256": actual, "bytes": path.stat().st_size})
    destination = ROOT / "share/road-models-private.zip"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError("A handoff already exists; preserve it or explicitly move it before rebuilding.")
    temporary = destination.with_suffix(".zip.partial")
    if temporary.exists():
        raise FileExistsError("A previous partial handoff exists; inspect it before retrying.")
    note = ("Private model handoff. Extract into the cloned project folder.\n"
            "Contains three checkpoints, encoder weights, configs and an unchanged split CSV.\n"
            "No images, masks or downloaded footage are included.\n"
            "The CSV contains original PC paths for reproducibility; inference checks only its hash, IDs and split counts.\n"
            "Do not edit the locked model, normalization, threshold or split.\n"
            "See docs/GETTING_STARTED.md for the environment and prediction command.\n")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as archive:
        for item in manifest:
            archive.write(ROOT / item["path"], arcname=item["path"])
        archive.writestr("HANDOFF_README.txt", note)
        archive.writestr("HANDOFF_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(temporary) as archive:
        for item in manifest:
            with archive.open(item["path"]) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                    raise ValueError(f"Archive verification failed: {item['path']}")
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError(f"Source changed during packaging: {item['path']}")
    temporary.replace(destination)
    digest = sha256(destination)
    destination.with_suffix(".zip.sha256").write_text(digest + "  " + destination.name + "\n", encoding="utf-8")
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file": str(destination), "bytes": destination.stat().st_size,
                      "sha256": digest, "protected_files_verified": len(manifest)}))


if __name__ == "__main__":
    main()
