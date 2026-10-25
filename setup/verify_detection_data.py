"""Independent numerical and failure-case checks for detection preparation."""
import csv
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import (AnnotationError, MANIFEST_FIELDS, convert_annotation,
    file_sha, load_policy, prepare_dataset, read_manifest, save_json, source_path, yolo_lines)


def rejected(function, expected=None):
    try:
        function()
    except (ValueError, OSError, KeyError) as error:
        if expected is not None:
            assert expected in str(error), str(error)
        return True
    raise AssertionError("Invalid input was accepted")


def main(output):
    output.mkdir(exist_ok=False)
    policy = load_policy()
    checks = []
    def box(name="car", coords=(10, 20, 110, 70), **extra):
        return {"name": name, "bbox": dict(zip(("xmin", "ymin", "xmax", "ymax"), coords)), **extra}
    def convert(objects, width=200, height=100):
        return convert_annotation(json.dumps(objects), "idd_json", width, height, "xyxy_edges", policy)
    reference = convert([box()])
    assert np.allclose(reference["objects"][0]["xywh"], [.3, .45, .5, .5], atol=1e-12)
    # Independent reconstruction from the serialized trainer input.
    label = yolo_lines(reference).split()
    cls, cx, cy, w, h = map(float, label)
    assert cls == policy["names"].index("car")
    assert np.allclose([(cx-w/2)*200, (cy-h/2)*100, (cx+w/2)*200, (cy+h/2)*100], [10,20,110,70])
    checks.append("non_square_native_dimensions_and_yolo_roundtrip")
    xml = "<annotation><size><width>200</width><height>100</height></size><object><name>car</name><bndbox><xmin>11</xmin><ymin>21</ymin><xmax>110</xmax><ymax>70</ymax></bndbox></object></annotation>"
    equivalent = convert_annotation(xml, "voc_xml", 200, 100, "voc_1based_inclusive", policy)
    assert equivalent["objects"][0]["xyxy"] == [10,20,110,70]
    rejected(lambda: convert_annotation(xml, "voc_xml", 400, 100, "voc_1based_inclusive", policy), "dimensions")
    checks.append("explicit_voc_origin_and_dimension_mismatch")
    clipped = convert([box(coords=(-10,20,220,70))])
    assert clipped["clipped_boxes"] == 1 and clipped["objects"][0]["xyxy"] == [0,20,200,70]
    assert convert([box("vehicle fallback")])["objects"][0]["class_name"] == "other_vehicle"
    assert convert([box("auto-rickshaw")])["objects"][0]["class_name"] == "autorickshaw"
    assert convert([box("ego vehicle")])["negative"] and convert([])["negative"]
    assert len(convert([box(),box()])["objects"]) == 1
    assert len(convert([box("rider"),box("motorcycle")])["objects"]) == 2
    checks.append("clipping_fallback_alias_negative_and_overlapping_classes")
    for raw, message in [([box("mystery")], "unknown_label"), ([box(coords=(1,2,1,8))], "Degenerate"),
                         ([box(coords=(300,2,400,8))], "outside"), ([box(coords=(float("nan"),2,4,8))], "Non-finite"),
                         ([box(ignore=1)], "ignore_policy"), ([box(difficult=1)], "ignore_policy"),
                         ([box(iscrowd=1)], "ignore_policy")]:
        rejected(lambda raw=raw: convert(raw), message)
    checks.append("unknown_invalid_and_ignore_labels_do_not_become_negatives")
    rejected(lambda: source_path(output, "../outside"), "relative")
    rejected(lambda: convert_annotation("<!DOCTYPE x><annotation/>", "voc_xml", 200, 100, "xyxy_edges"), "entities")
    checks.append("source_path_and_xml_entity_rejection")
    # Actual downloaded annotations are used only to check schema/class mapping.
    # Their image pixels are not opened and no model performance is evaluated.
    sample = json.loads((output.parent / "archive_prefix_schema.json").read_text())
    metadata_count = sum(len(json.loads(text)) for text in sample["annotation_schema_samples"].values())
    source_names = sorted({obj["name"] for text in sample["annotation_schema_samples"].values() for obj in json.loads(text)})
    assert set(source_names).issubset(set(policy["source_mapping"]) | set(policy["background_labels"]))
    checks.append("actual_downloaded_json_schema_and_source_names")
    source = output / "fixture_sources"
    source.mkdir()
    records = []
    raw_names = [next(k for k,v in policy["source_mapping"].items() if v==name) for name in policy["names"]]
    for number, (split, negative) in enumerate((("train",False),("train",True),("val",False))):
        image = Image.new("RGB", (200,100), (20+number*60,40,70))
        name = f"fixture_{number}"
        image.save(source / (name+".png"))
        labels = [] if negative else [box(n, (5+i*14,10,16+i*14,85)) for i,n in enumerate(raw_names)]
        (source / (name+".json")).write_text(json.dumps(labels))
        records.append(dict(zip(MANIFEST_FIELDS, [name+".png",name+".json","idd_json",split,"fixture",split,name,"xyxy_edges"])))
    records.append(dict(zip(MANIFEST_FIELDS,["nonexistent_test.jpg","-","idd_json","test","fixture","test","heldout","xyxy_edges"])))
    manifest = output / "fixture_manifest.csv"
    def write_manifest(rows, path=manifest):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            writer.writeheader(); writer.writerows(rows)
    write_manifest(records)
    report = prepare_dataset(source, manifest, output / "export")
    assert report["passed"] and report["training_allowed"] is False and report["scope"] == "fixture_only_not_for_training"
    assert report["counts"] == {"train":2,"val":1,"test_metadata_only":1} and report["negatives"]["train"] == 1
    for row in map(json.loads,(output/"export/pairs.jsonl").read_text().splitlines()):
        assert file_sha(output/"export"/row["prepared_image"]) == row["image_sha256"]
        lines=(output/"export"/row["prepared_label"]).read_text().splitlines()
        assert len(lines) == row["foreground_objects"]
        for line in lines:
            values = list(map(float,line.split()))
            assert len(values)==5 and 0<=values[0]<12 and values[0].is_integer()
            assert all(0<=v<=1 for v in values[1:]) and values[3]>0 and values[4]>0
    checks.append("full_fixture_export_preserves_negatives_bytes_and_test_exclusion")
    conflicted = [dict(row) for row in records]
    conflicted[2]["drive_id"]="train"
    conflict = output / "conflict.csv"; write_manifest(conflicted, conflict)
    rejected(lambda: read_manifest(conflict), "Drive sequence crosses")
    checks.append("drive_level_leakage_rejected")
    duplicate = [dict(row) for row in records]
    duplicate[2]["image"] = duplicate[0]["image"]
    dup = output / "duplicate.csv"; write_manifest(duplicate,dup)
    dup_report = prepare_dataset(source,dup,output/"duplicate_export")
    assert not dup_report["passed"] and dup_report["cross_split_duplicates"] and not dup_report["training_allowed"]
    # Different encodings of identical decoded pixels must also be caught.
    with Image.open(source/records[0]["image"]) as duplicate_image:
        duplicate_image.save(source/"reencoded.bmp")
    pixel_duplicate=[dict(row) for row in records]
    pixel_duplicate[2]["image"]="reencoded.bmp"
    pixel_path=output/"pixel_duplicate.csv"; write_manifest(pixel_duplicate,pixel_path)
    assert file_sha(source/"reencoded.bmp") != file_sha(source/records[0]["image"])
    pixel_report=prepare_dataset(source,pixel_path,output/"pixel_duplicate_export")
    assert not pixel_report["passed"] and pixel_report["cross_split_duplicates"]
    checks.append("cross_split_byte_and_pixel_duplicates_block_readiness")
    missing = [dict(row) for row in records]
    missing[0]["annotation"]="absent.json"
    path = output/"missing.csv"; write_manifest(missing,path)
    missing_report=prepare_dataset(source,path,output/"missing_export")
    assert not missing_report["passed"] and missing_report["quarantine"] and missing_report["counts"]["train"]==1
    checks.append("missing_annotation_quarantined_not_a_negative")
    image = Image.new("RGB", (800,400), "white")
    draw=ImageDraw.Draw(image)
    draw.rectangle((40,80,440,280),outline=(20,150,60),width=4)
    draw.text((40,55),"Expected car rectangle: (10,20)-(110,70), native 200x100",fill="black")
    draw.text((40,310),"Serialized YOLO: class=car, center=(0.30,0.45), size=(0.50,0.50)",fill="black")
    image.save(output/"coordinate_check.png")
    result={"passed":True,"checks":checks,"check_groups":len(checks),"training_started":False,
        "real_annotation_schema_objects_inspected":metadata_count,"source_names_observed":source_names,
        "real_image_pixels_opened":False,"accuracy_evaluated":False,"fixture_training_allowed":False}
    save_json(output/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/detection_readiness_20261023_01/data_checks")
    main(parser.parse_args().output.resolve())
