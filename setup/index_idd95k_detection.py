"""Index the observed 95K layout; do not guess the older release's drive IDs."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import MANIFEST_FIELDS, read_manifest, save_json

EXPECTED={"train":65328,"val":9977,"test":19850}


def index_release(source_root, output, convention, *, expected=EXPECTED):
    source_root,output=Path(source_root).resolve(),Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve the previous source manifest")
    if convention not in {"xyxy_edges","voc_1based_inclusive"}:
        raise ValueError("Specify the reviewed box convention")
    candidates=[source_root] if source_root.name=="IDD_95kDetection" else [source_root/"IDD_95kDetection",source_root/"IDD117K_Detection/IDD_95kDetection"]
    candidates=[p for p in candidates if p.is_dir()]
    if len(candidates)!=1:
        raise ValueError("Expected one observed IDD_95kDetection root")
    base=candidates[0]
    rows=[]
    counts={s:0 for s in EXPECTED}
    for split in EXPECTED:
        image_root=base/split/"leftImg8bit"
        if not image_root.is_dir():
            raise ValueError("The observed leftImg8bit split folder is missing: "+split)
        images={}
        for path in sorted(image_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg",".jpeg",".png"}:
                continue
            key=path.relative_to(image_root).with_suffix("").as_posix()
            if len(Path(key).parts)!=2 or not Path(key).parts[0].isdigit():
                raise ValueError("Drive identity is not explicit; do not infer it from a flat filename")
            if key in images:
                raise ValueError("Ambiguous duplicate image key")
            images[key]=path
        annotations={}
        if split!="test":
            # The full archive uses labelsJSON for train, Labels_json for val.
            candidates=[base/split/name for name in ("labelsJSON", "Labels_json") if (base/split/name).is_dir()]
            if len(candidates)!=1:
                raise ValueError("Expected one inspected annotation directory: "+split)
            label_root=candidates[0]
            annotations={p.relative_to(label_root).with_suffix("").as_posix():p for p in label_root.rglob("*.json")}
            if set(annotations)!=set(images):
                raise ValueError(f"{split}: missing images or annotations; never create negatives for missing labels")
        for key,path in sorted(images.items()):
            rows.append({"image":path.relative_to(source_root).as_posix(),
                "annotation":annotations[key].relative_to(source_root).as_posix() if split!="test" else "-",
                "format":"idd_json","split":split,"family":"IDD_95kDetection",
                "drive_id":Path(key).parts[0],"source_id":key,"box_convention":convention})
            counts[split]+=1
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=MANIFEST_FIELDS)
        writer.writeheader();writer.writerows(rows)
    # This also rejects cross-split drive reuse; no image pixels are opened.
    read_manifest(output)
    result={"schema":"idd95k_source_index_v1","scope":"95K_part_only_not_full_117K",
        "counts":counts,"expected_provider_counts":expected,"passed":expected is not None and counts==expected,
        "image_pixels_opened":False,"box_convention":convention,"training_started":False,
        "legacy_original_detection_part_indexed":False}
    save_json(output.with_suffix(".index.json"),result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--box-convention",choices=("xyxy_edges","voc_1based_inclusive"),required=True)
    args=parser.parse_args()
    result=index_release(args.source_root,args.output,args.box_convention)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["passed"] else 2)
