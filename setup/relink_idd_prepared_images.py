"""Replace only this import's verified duplicate copies with equivalent hard links."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import file_sha,save_json


def relink(row, raw, prepared):
    source=(raw/row["image"]).resolve()
    target=(prepared/row["prepared_image"]).resolve()
    if not source.is_relative_to(raw) or not target.is_relative_to(prepared/"images"):
        raise ValueError("Refuse paths outside this import's raw/prepared image folders")
    if row["split"] not in {"train","val"}:
        raise ValueError("Do not operate on unlabelled test images")
    if source.samefile(target):
        return 0
    expected=(row["image_bytes"],row["image_mtime_ms"])
    for path in (source,target):
        info=path.stat()
        if (info.st_size,info.st_mtime_ns//1000000)!=expected or file_sha(path)!=row["image_sha256"]:
            raise ValueError("Image differs from its complete native-image audit")
    temporary=target.with_name(target.name+".verified-hardlink.tmp")
    if temporary.exists():
        raise FileExistsError("Preserve an unexpected prior link attempt")
    os.link(source,temporary)
    if not source.samefile(temporary):
        raise ValueError("Hard link does not refer to the verified source")
    # Both resolved paths were checked above. Replace only the completed copy
    # named in this import's audit; its raw source remains untouched.
    for attempt in range(6):
        try:
            os.replace(temporary,target)
            break
        except PermissionError:
            if attempt==5:
                raise
            time.sleep(0.5)
    if not source.samefile(target):
        raise ValueError("Replacement did not preserve the verified image")
    return row["image_bytes"]


def main():
    raw=(ROOT/"dataset/idd117k_extracted").resolve()
    prepared=(ROOT/"dataset/idd117k_detection_v1").resolve()
    run=ROOT/"runs/idd_detection_import_20261023_01"
    report={"status":"running","relinked_images":0,"duplicate_bytes_reclaimed":0,"images_checked":0,
        "raw_images_deleted":False,"pixel_content_changed":False,"training_started":False}
    result=run/"hardlink_recovery.json"
    started=updated=time.monotonic()
    try:
        with (prepared/"pairs.jsonl").open(encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=4) as executor:
            finished=False
            while not finished:
                batch=[]
                for _ in range(192):
                    position=stream.tell()
                    line=stream.readline()
                    if not line:
                        break
                    # The exporter flushes complete JSON lines after each batch.
                    if not line.endswith("\n"):
                        stream.seek(position)
                        break
                    batch.append(json.loads(line))
                for saved in executor.map(lambda row:relink(row,raw,prepared),batch):
                    report["images_checked"]+=1
                    report["relinked_images"]+=saved>0
                    report["duplicate_bytes_reclaimed"]+=saved
                if not batch:
                    try:
                        progress=json.loads((prepared/"progress.json").read_text())
                    except json.JSONDecodeError:
                        progress={}
                    finished=progress.get("status")=="completed"
                    if not finished:
                        time.sleep(2)
                if time.monotonic()-updated>=20:
                    report["elapsed_seconds"]=round(time.monotonic()-started,1)
                    save_json(result,report)
                    print(json.dumps(report),flush=True)
                    updated=time.monotonic()
        report["status"]="completed"
    except BaseException as error:
        report.update(status="failed_preserved",error=type(error).__name__+": "+str(error))
        raise
    finally:
        report["elapsed_seconds"]=round(time.monotonic()-started,1)
        save_json(result,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
