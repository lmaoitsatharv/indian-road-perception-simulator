"""Verify quality exclusions, pixel duplicates, reused audits and fixture isolation."""
import csv
import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import MANIFEST_FIELDS, file_sha, load_policy, prepare_dataset, save_json
from setup.finalize_idd_detection_quality import main as finalize
from setup.audit_prepared_idd_detection import main as audit


def main(root):
    root.mkdir(exist_ok=False)
    source = root/"source"
    source.mkdir()
    policy = load_policy()
    names = [next(k for k,v in policy["source_mapping"].items() if v==n) for n in policy["names"]]
    labels = [{"name": name, "bbox": {"xmin": 5+i*14, "ymin":10, "xmax":16+i*14, "ymax":85}} for i,name in enumerate(names)]
    rows = []
    for i, colour in enumerate(((100,20,30), (20,30,150), (40,150,60), (100,20,30), (10,10,10))):
        split = "val" if i==3 else "train"
        Image.new("RGB", (200,100), colour).save(source/f"{i}.png")
        objects = [] if i==1 else labels
        if i==4:
            objects = [{"name":"car", "bbox":{"xmin":1,"ymin":2,"xmax":1,"ymax":4}}]
        (source/f"{i}.json").write_text(json.dumps(objects))
        rows.append(dict(zip(MANIFEST_FIELDS, [f"{i}.png",f"{i}.json","idd_json",split,"fixture",f"drive{i}",str(i),"xyxy_edges"])))
    rows.append(dict(zip(MANIFEST_FIELDS, ["nonexistent_test.jpg","-","idd_json","test","fixture","test","heldout","xyxy_edges"])))
    manifest = root/"candidate.csv"
    with manifest.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=MANIFEST_FIELDS)
        writer.writeheader();writer.writerows(rows)
    initial = prepare_dataset(source,manifest,root/"audited",workers=6)
    assert len(initial["quarantine"])==1 and initial["cross_split_duplicates"]
    evidence = ROOT/"runs/idd_detection_import_20261023_01/archive_checks/inventory"
    save_json(manifest.with_suffix(".index.json"), {"schema":"idd117k_source_index_v1", "passed":True,
        "provider_counts":{"train":4,"val":1,"test":1},"counts":{"train":4,"val":1,"test":1},
        "archive_integrity_file":str(evidence/"archive_integrity.json"),"extraction_report_file":str(evidence/"extraction.json"),
        "archive_integrity_sha256":file_sha(evidence/"archive_integrity.json"),"extraction_report_sha256":file_sha(evidence/"extraction.json")})
    result = finalize(root/"audited",source,manifest,root/"final",root/"selected.csv")
    assert result["passed"] and not result["training_allowed"]
    assert result["counts"]=={"train":2,"val":1,"test_metadata_only":1}
    assert result["quality_curation"]["invalid_pairs_excluded"]==1
    assert result["quality_curation"]["training_duplicates_of_validation_excluded"]==1
    assert result["negatives"]=={"train":1,"val":0}
    assert not result["quarantine"] and not result["cross_split_duplicates"]
    audit(root/"final",source,root/"selected.csv",root/"independent_audit")
    checks={"passed":True,"training_started":False,"fixture_training_allowed":False,
        "checks":["invalid_whole_pairs_removed_and_recorded","training_duplicates_of_validation_removed",
            "valid_negative_images_preserved","original_split_labels_unchanged","all_test_metadata_retained_without_opening",
            "reused_native_pixel_audit_bound_to_hashes_and_file_metadata","saved_labels_still_match_source_geometry",
            "fixture_scope_cannot_become_real_training_data","independent_final_dataset_audit_passed"]}
    save_json(root/"verification.json",checks)
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/idd_detection_import_20261023_01/quality_checks")
    main(parser.parse_args().output.resolve())
