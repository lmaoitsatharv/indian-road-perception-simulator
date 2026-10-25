"""Exercise split boundaries, gzip corruption and contained extraction on fixtures."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import prepare_dataset
from setup.import_idd117k_archive import SplitReader, run, safe_name, write_json


def main():
    output = ROOT / "runs/idd_detection_import_20261023_01/archive_checks"
    output.mkdir(exist_ok=False)
    raw = output / "parts"
    raw.mkdir()
    payload = io.BytesIO()
    contents = {"IDD117K_Detection/fixture/train/label.json": b"[]",
        "IDD117K_Detection/fixture/test/image.jpg": b"unlabelled-test-metadata-fixture"}
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name, data in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    compressed = gzip.compress(payload.getvalue())
    records = []
    for i in range(5):
        data = compressed[len(compressed)*i//5:len(compressed)*(i+1)//5]
        name = f"fixture.part{i}"
        (raw/name).write_bytes(data)
        records.append({"part": i+1, "name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    write_json(raw/"download_verification.json", {"passed": True, "files": records})
    with io.BufferedReader(SplitReader(raw, records)) as stream:
        assert gzip.decompress(stream.read()) == payload.getvalue()
    inventory = run(raw, output/"inventory", "inventory")
    assert inventory["passed"] and inventory["regular_files"] == 2
    destination = ROOT / "dataset/idd117k_import_fixture_20261023_01"
    extracted = run(raw, output/"inventory", "extract", destination)
    assert extracted["passed"]
    assert all((destination/name).read_bytes() == data for name, data in contents.items())
    for name in ("../escape", "/absolute", "IDD117K_Detection/../escape", "IDD117K_Detection/E:bad",
                 "IDD117K_Detection/evil\\outside", "another_root/file", "IDD117K_Detection/name."):
        try:
            safe_name(name)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe path accepted")
    corrupted = bytearray((raw/records[2]["name"]).read_bytes())
    corrupted[len(corrupted)//2] ^= 1
    (raw/records[2]["name"]).write_bytes(corrupted)
    try:
        with io.BufferedReader(SplitReader(raw, records)) as stream:
            stream.read()
    except ValueError as error:
        assert "hash/size mismatch" in str(error)
    else:
        raise AssertionError("Changed archive part accepted")
    # Gzip's own checksum is checked in addition to local part checksums.
    damaged = bytearray(compressed)
    damaged[-8] ^= 1
    try:
        gzip.decompress(damaged)
    except gzip.BadGzipFile:
        pass
    else:
        raise AssertionError("Incorrect gzip CRC accepted")
    fixtures = ROOT / "runs/idd_detection_import_20261023_01/converter_checks_pass"
    parallel = prepare_dataset(fixtures/"fixture_sources", fixtures/"fixture_manifest.csv",
        output/"parallel_export", workers=6)
    serial = json.loads((fixtures/"export/preparation.json").read_text())
    assert parallel["counts"] == serial["counts"] and parallel["objects"] == serial["objects"]
    for name in ("pairs.jsonl", "data.yaml"):
        assert (output/"parallel_export"/name).read_bytes() == (fixtures/"export"/name).read_bytes()
    result = {"passed": True, "training_started": False, "real_image_pixels_opened": False,
        "checks": ["five_part_sequential_bytes_and_hashes", "combined_gzip_crc_and_tar_inventory",
            "safe_contained_extraction_preserves_bytes", "unsafe_archive_paths_rejected",
            "changed_part_hash_rejected", "gzip_crc_corruption_rejected",
            "six_worker_conversion_identical_to_serial_including_test_exclusion"]}
    write_json(output/"verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
