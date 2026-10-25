"""Fetch three public, version-pinned MOTOR front-camera development clips."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'media/real_driving_20261017'
REPO='varunpaturkar/MOTOR'
REVISION='3aad7f943951bd1ed59fd51ae835388c9140e343'
BASE=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
SELECTED={
    '03_070':('01_unmarked_low_traffic.mp4','Low-traffic unmarked paved road'),
    '12_096':('02_marked_busy_traffic.mp4','Marked paved road, high traffic; lane-change annotation'),
    '17_030':('03_marked_overtaking.mp4','Marked paved road, high traffic; overtake annotation'),
}


def atomic_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.headers['User-Agent']='TeslaCollegeRoadPerception/1.0 (public research clip download)'
    readme=session.get(BASE+'README.md',timeout=30);readme.raise_for_status()
    metadata=session.get(BASE+'metadata.csv',timeout=30);metadata.raise_for_status()
    for name,body in [('SOURCE_README.md',readme.content),('SOURCE_metadata.csv',metadata.content)]:
        p=OUT/name
        if p.exists() and p.read_bytes()!=body:raise RuntimeError(f'Existing source record differs: {p}')
        if not p.exists():p.write_bytes(body)
    rows=list(csv.DictReader(io.StringIO(metadata.content.decode('utf-8-sig'))))
    manifest=dict(source_repository=f'https://huggingface.co/datasets/{REPO}',
        source_project='https://varuniiith.github.io/MOTOR-Dataset/',revision=REVISION,
        acquired_at_utc=datetime.now(timezone.utc).isoformat(),license='CC BY-NC 4.0',
        attribution='Varun A. Paturkar, Shankar Gangisetty, C. V. Jawahar; MOTOR, ICRA 2026',
        camera='front-mounted motorcycle camera, not a calibrated car dashcam',
        purpose='External real-video development and visual diagnostics; not an IDD benchmark split',
        source_binary_road_masks_provided=False,clips=[])
    for clip_id,(filename,description) in SELECTED.items():
        matches=[r for r in rows if r['clip_id']==clip_id and r['camera']=='front']
        if len(matches)!=1:raise RuntimeError(f'Expected one front-camera clip for {clip_id}')
        row=matches[0];url=BASE+row['file_name'];path=OUT/filename
        if not path.exists():
            part=path.with_suffix('.mp4.part')
            if part.exists():raise FileExistsError(f'Inspect interrupted download first: {part}')
            with session.get(url,stream=True,timeout=(30,60)) as response:
                response.raise_for_status();size=0
                with part.open('xb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        if not chunk:continue
                        size+=len(chunk)
                        if size>100*1024*1024:raise RuntimeError('Clip exceeded the 100 MiB download bound')
                        stream.write(chunk)
            with part.open('rb') as stream:header=stream.read(64)
            if b'ftyp' not in header:raise RuntimeError('Downloaded content is not an MP4 container')
            part.rename(path)
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        record=dict(file=filename,clip_id=clip_id,description=description,
            source_url=url,source_path=row['file_name'],source_metadata=row,
            source_duration_s=float(row['duration_s']),bytes=path.stat().st_size,sha256=digest,
            modifications='None; original MP4 bytes downloaded without trimming, resizing or concatenation')
        manifest['clips'].append(record);atomic_json(OUT/'clips.json',manifest)
        print(json.dumps({k:record[k] for k in ['file','clip_id','source_duration_s','bytes','sha256']}),flush=True)
    print('Downloaded all three original clips',flush=True)


if __name__=='__main__':main()
