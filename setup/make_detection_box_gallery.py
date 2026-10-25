"""Show actual validation labels beside frozen-model detections; no distance claims."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import file_sha, save_json


def matches(truth, predictions, threshold=.25):
    pairs = []
    for gi, gt in enumerate(truth):
        a = gt["box"]
        for pi, pred in enumerate(predictions):
            if pred["confidence"] < threshold or pred["class_id"] != gt["class_id"]:
                continue
            b = pred["box"]
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            iou = intersection / max(union, 1e-9)
            if iou >= .5:
                pairs.append((iou, gi, pi))
    used_gt, used_pred, accepted = set(), set(), []
    for iou, gi, pi in sorted(pairs, reverse=True):
        if gi not in used_gt and pi not in used_pred:
            accepted.append({"gt": gi, "prediction": pi, "iou": iou})
            used_gt.add(gi)
            used_pred.add(pi)
    return {"pairs": accepted, "missed_labels": [i for i in range(len(truth)) if i not in used_gt],
        "unmatched_predictions": [i for i, p in enumerate(predictions) if p["confidence"] >= threshold and i not in used_pred],
        "shown_predictions": sum(p["confidence"] >= threshold for p in predictions)}


def prepare(output):
    import numpy as np
    assert output.parent == ROOT / "runs" and output.name.startswith("detection_box_examples_")
    output.mkdir(exist_ok=False)
    checkpoint = ROOT / "runs/detection_train_20261025_02/weights/last_resume_fp32.pt"
    before = checkpoint.stat()
    shutil.copy2(checkpoint, output / "snapshot.pt")
    after = checkpoint.stat()
    assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)
    assert file_sha(checkpoint) == file_sha(output / "snapshot.pt")
    dataset = ROOT / "dataset/idd117k_detection_v3"
    rows = np.load(dataset / "labels/val.cache", allow_pickle=True).item()["labels"]
    assert len(rows) == 17675
    rng = np.random.default_rng(20261026)
    selected = set(int(i) for i in rng.choice(len(rows), 12, replace=False))
    crowded, foreground_person, foreground_vehicle, animals = [], [], [], []
    for index, row in enumerate(rows):
        cls = row["cls"].reshape(-1).astype(int)
        boxes = row["bboxes"]
        crowded.append((int((cls == 0).sum()), index))
        for c, b in zip(cls, boxes):
            area, bottom = float(b[2]*b[3]), float(b[1]+b[3]/2)
            if int(c) == 0 and b[3] >= .15 and bottom >= .55:
                foreground_person.append((area, index))
            if int(c) in (2, 3, 5, 6, 7) and b[3] >= .25 and bottom >= .65:
                foreground_vehicle.append((area, index))
            if int(c) == 8:
                animals.append((area, index))
    for pool, count in ((crowded, 16), (foreground_person, 24), (foreground_vehicle, 12), (animals, 12)):
        unique = list(dict.fromkeys(i for score, i in sorted(pool, reverse=True)))
        selected.update(unique[:count])
    if len(selected) < 80:
        remainder = sorted(set(range(len(rows)))-selected)
        selected.update(int(i) for i in rng.choice(remainder, 80-len(selected), replace=False))
    names = json.loads((ROOT / "detection_policy.json").read_text())["names"]
    records = []
    for index in sorted(selected):
        basename = Path(rows[index]["im_file"]).name
        label = dataset / "labels/val" / (Path(basename).stem + ".txt")
        records.append({"image": basename, "source_index": index, "label_sha256": file_sha(label)})
    save_json(output / "selection.json", {"names": names, "records": records,
        "checkpoint_source": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": file_sha(output / "snapshot.pt"),
        "selection": "Fixed random examples plus label-geometry-selected crowd, larger person/vehicle and animal cases; qualitative illustrations only",
        "official_test_used": False, "measured_distances_available": False})


def render_pair(record, image, destination, *, crop=False):
    from PIL import Image, ImageDraw, ImageFont
    width, height = image.size
    if crop:
        a = record["truth"][record["focus_gt"]]["box"]
        cx, cy = (a[0]+a[2])/2, (a[1]+a[3])/2
        cw, ch = max(300, (a[2]-a[0])*2.1), max(260, (a[3]-a[1])*1.5)
        cw = max(cw, ch*1.6)
        ch = cw/1.6
        x1 = max(0, min(width-cw, cx-cw/2))
        y1 = max(0, min(height-ch, cy-ch/2))
        region = [x1, y1, min(width, x1+cw), min(height, y1+ch)]
    else:
        region = [0, 0, width, height]
    record["zoom_region"] = region if crop else record.get("zoom_region")
    part = image.crop(tuple(int(v) for v in region))
    pane_width = 1040
    pane_height = round(part.height * pane_width / part.width)
    part = part.resize((pane_width, pane_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (2144, pane_height+154), "#101722")
    draw = ImageDraw.Draw(canvas)
    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(fontpath, 25)
    small = ImageFont.truetype(fontpath, 20)
    label_font = ImageFont.truetype(fontpath, 15)
    draw.text((24, 12), record["title"] + (" - zoom" if crop else ""), fill="#ffffff", font=font)
    draw.text((24, 48), "Dataset labels (green)", fill="#6df0a4", font=small)
    draw.text((1088, 48), "Model output: blue; red = unmatched to labels", fill="#87d5ff", font=small)
    sx, sy = pane_width/(region[2]-region[0]), pane_height/(region[3]-region[1])
    default = record["comparison"]
    for side in (0, 1):
        xoffset, yoffset = 24+side*1064, 82
        canvas.paste(part, (xoffset, yoffset))
        boxes = record["truth"] if side == 0 else record["predictions"]
        for index, item in enumerate(boxes):
            if side == 1 and item["confidence"] < .25:
                continue
            box = item["box"]
            if box[2] <= region[0] or box[0] >= region[2] or box[3] <= region[1] or box[1] >= region[3]:
                continue
            a = [xoffset+max(0, box[0]-region[0])*sx, yoffset+max(0, box[1]-region[1])*sy,
                xoffset+min(region[2]-region[0], box[2]-region[0])*sx,
                yoffset+min(region[3]-region[1], box[3]-region[1])*sy]
            color = "#3df38a" if side == 0 else ("#ff6464" if index in default["unmatched_predictions"] else "#36bfff")
            draw.rectangle(a, outline=color, width=3 if crop else 2)
            if crop or a[3]-a[1] >= 25:
                text = item["name"] + (f" {item['confidence']:.2f}" if side else "")
                bounds = draw.textbbox((0, 0), text, font=label_font)
                tw = bounds[2]-bounds[0]+6
                tx = min(a[0], xoffset+pane_width-tw)
                ty = max(yoffset, a[1]-21)
                draw.rectangle([tx, ty, tx+tw, ty+21], fill="#101722")
                draw.text((tx+3, ty+1), text, fill=color, font=label_font)
    footer = f"Labels: {len(record['truth'])} | Predictions shown: {default['shown_predictions']} | Matched: {len(default['pairs'])} | Missed labels: {len(default['missed_labels'])}"
    draw.text((24, pane_height+92), footer, fill="#e5ecf5", font=small)
    draw.text((24, pane_height+121), "Confidence >= 0.25; class-matched IoU >= 0.50. Box size is not measured distance.", fill="#b8c6d8", font=small)
    canvas.save(destination)


def worker(output):
    os.nice(10)
    os.environ.update(CUDA_VISIBLE_DEVICES="", YOLO_CONFIG_DIR=str(output/"settings"),
        MPLCONFIGDIR=str(output/"matplotlib"), MPLBACKEND="Agg", YOLO_AUTOINSTALL="False", WANDB_DISABLED="true")
    for folder in ("settings", "matplotlib"):
        (output/folder).mkdir(exist_ok=True)
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO, settings
    from setup.verify_detection_runtime import protection
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    settings.update({k: False for k in ("sync", "wandb", "comet", "clearml", "mlflow", "neptune",
        "raytune", "tensorboard", "dvc", "hub") if k in settings})
    selected = json.loads((output/"selection.json").read_text())
    protected = protection()
    model = YOLO(str(output/"snapshot.pt"), task="detect")
    names = [model.names[i] for i in range(12)]
    assert names == selected["names"]
    epoch = int(model.ckpt["epoch"])+1
    assert all(bool(torch.isfinite(v).all()) for v in model.model.state_dict().values())
    records = []
    for index, source in enumerate(selected["records"]):
        image_path = ROOT/"dataset/idd117k_detection_v3/images/val"/source["image"]
        label_path = ROOT/"dataset/idd117k_detection_v3/labels/val"/(image_path.stem+".txt")
        assert file_sha(label_path) == source["label_sha256"]
        result = model.predict(str(image_path), imgsz=1280, conf=.001, iou=.7, max_det=300,
            device="cpu", half=False, verbose=False, save=False)[0]
        height, width = result.orig_shape
        truth = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            c, x, y, w, h = [float(v) for v in line.split()]
            truth.append({"class_id": int(c), "name": names[int(c)],
                "box": [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]})
        predictions = [{"class_id": int(c), "name": names[int(c)], "confidence": float(score), "box": [float(x) for x in box]}
            for box, c, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]
        record = {"image": source["image"], "width": width, "height": height, "truth": truth, "predictions": predictions}
        record["comparison"] = matches(truth, predictions)
        records.append(record)
        save_json(output/"progress.json", {"phase": "predicting", "images": index+1, "expected": len(selected["records"]),
            "cpu_only": True, "checkpoint_epoch_completed": epoch})
    save_json(output/"all_candidate_predictions.json", records)
    del model
    gc.collect()
    def area(gt, record):
        b = gt["box"]
        return (b[2]-b[0])*(b[3]-b[1])/(record["width"]*record["height"])
    def large(gt, record):
        b = gt["box"]
        return (b[3]-b[1])/record["height"] >= .15 and b[3]/record["height"] >= .55
    used = set()
    cases = []
    definitions = [("foreground", "Large foreground vehicle: labels and predictions"),
        ("person_miss", "A labelled person without a matching prediction"),
        ("crowded", "Crowded traffic: larger and smaller objects"),
        ("animal", "Animal example: labels and predictions")]
    for kind, title in definitions:
        ranked = []
        for record in records:
            if record["image"] in used:
                continue
            pairs = record["comparison"]["pairs"]
            matched = {p["gt"] for p in pairs}
            missed = record["comparison"]["missed_labels"]
            eligible = []
            for gi, gt in enumerate(record["truth"]):
                b = gt["box"]
                if kind == "foreground" and gi in matched and gt["class_id"] in (2,3,5,6,7) and large(gt,record):
                    eligible.append((area(gt,record)/(1+len(record["truth"])/60), gi))
                elif kind == "person_miss" and gi in missed and gt["class_id"] == 0 and large(gt,record):
                    eligible.append((area(gt,record), gi))
                elif kind == "crowded" and gi in missed and gt["class_id"] in (0,1):
                    bh = b[3]-b[1]
                    if bh >= 15 and area(gt,record) < .003:
                        eligible.append((sum(record["truth"][i]["class_id"] in (0,1) for i in missed)+len(matched)/20, gi))
                elif kind == "animal" and gi in missed and gt["class_id"] == 8:
                    eligible.append((area(gt,record), gi))
            if eligible:
                score, gi = max(eligible)
                ranked.append((score, record["image"], gi, record))
        if not ranked:
            continue
        score, image_name, focus, record = max(ranked, key=lambda x:(x[0],x[1]))
        chosen = dict(record, kind=kind, title=title, focus_gt=focus)
        cases.append(chosen)
        used.add(image_name)
    assert len(cases) >= 3
    public = output/"public"
    (public/"images").mkdir(parents=True)
    for index, record in enumerate(cases):
        source = ROOT/"dataset/idd117k_detection_v3/images/val"/record["image"]
        record["image_sha256"] = file_sha(source)
        record["asset"] = f"images/case_{index+1}.jpg"
        shutil.copy2(source, public/record["asset"])
        image = Image.fromarray(cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB))
        render_pair(record, image, public/f"case_{index+1}_full.png")
        render_pair(record, image, public/f"case_{index+1}_zoom.png", crop=True)
    report = {"checkpoint_epoch_completed": epoch, "checkpoint_sha256": selected["checkpoint_sha256"],
        "source_run": "detection_train_20261025_02", "resolution": 1280, "display_confidence": .25,
        "prediction_floor": .001, "nms_iou": .7, "matching_iou": .5,
        "candidates_inferred": len(records), "selection": selected["selection"], "cases": cases,
        "measured_distances_available": False, "not_a_new_dataset_score": True, "official_test_used": False,
        "protected_files_unchanged": protection() == protected,
        "snapshot_unchanged": file_sha(output/"snapshot.pt") == selected["checkpoint_sha256"]}
    assert report["protected_files_unchanged"] and report["snapshot_unchanged"]
    save_json(output/"report.json", report)
    save_json(public/"examples.json", report)
    (public/"index.html").write_text(HTML, encoding="utf-8")
    save_json(output/"progress.json", {"phase": "completed", "examples": len(cases), "checkpoint_epoch_completed": epoch,
        "checked_utc": datetime.now(timezone.utc).isoformat()})


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Actual labels vs our detector</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0d1420;color:#eaf2ff;font:16px/1.5 Segoe UI,sans-serif}main{max-width:1500px;margin:auto;padding:26px}h1{margin:0}p{color:#bac9dc}.controls,.card{background:#182334;border:1px solid #34445b;border-radius:14px;padding:18px;margin:18px 0}.row{display:flex;gap:20px;flex-wrap:wrap;align-items:center}.panes{display:grid;grid-template-columns:1fr 1fr;gap:18px}svg{display:block;width:100%;background:#05090f;border-radius:10px}select{max-width:100%}.panes>div{min-width:0}label{max-width:100%}button,select{padding:10px 14px;background:#253b54;color:white;border:1px solid #526b88;border-radius:8px}label{display:block}a{color:#95ceff}.green{color:#3df38a}.blue{color:#36bfff}.red{color:#ff6464}h2{font-size:19px}@media(max-width:850px){.panes{grid-template-columns:1fr 1fr}main{padding:16px}}.small{font-size:13px}</style><main>
<h1>Dataset labels vs our model</h1><p id="model">Loading the frozen checkpoint…</p><p>These are selected real validation images. They illustrate successes and misses; they are not a new accuracy score.</p>
<div class="controls"><div class="row"><label>Example <select id="example"></select></label><label>Show model predictions above confidence <strong id="confidenceValue">0.25</strong><br><input id="confidence" type="range" min="0.05" max="0.9" step="0.05" value="0.25"></label><button id="zoom">Zoom to example object</button></div><p class="small">Green: supplied labels. Blue: model boxes matched to labels. Red: model boxes without a matching label. An unmatched box can also reflect a missing or ambiguous dataset annotation.</p></div>
<h2 id="title"></h2><div class="panes"><div><h2 class="green">Supplied dataset labels</h2><svg id="truth"></svg></div><div><h2 class="blue">Actual model predictions</h2><svg id="predictions"></svg></div></div><div class="card"><strong id="counts"></strong><p id="focus"></p><p id="review"></p><p class="small">Matching requires the same class and box overlap (IoU) of at least 0.50. Lower confidence reveals more uncertain predictions and may add false alarms. Hover over a box for its full label. Tiny-box text is hidden for readability; the boxes remain visible.</p><a id="full" target="_blank">Open full-size side-by-side figure</a> · <a id="crop" target="_blank">Open zoomed figure</a></div>
<div class="card"><h2>Does a small box mean a distant object?</h2><p>No measured distances in metres are available in these labels. Box size and foreground position are visual proxies. A distant bus can have a larger box than a nearby bicycle.</p><p>Nearby hazards deserve priority, but collision risk also depends on whether an object is in our path and how quickly the gap is closing. At 50 km/h the car travels about 14 metres each second, so waiting until an obstacle is very close can leave too little time to react and brake.</p><p><a href="https://www.gov.uk/guidance/the-highway-code/general-rules-techniques-and-advice-for-all-drivers-and-riders-103-to-158#rule126" target="_blank">Stopping-distance guidance</a> · <a href="https://www.nhtsa.gov/vehicle-safety/driver-assistance-technologies" target="_blank">Collision-warning and braking principles</a></p><p>These examples evaluate detection only. They do not validate a collision-avoidance controller.</p></div></main><script>
let data,index=0,zoomed=false;const $=id=>document.getElementById(id);function overlap(a,b){const area=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));return area/Math.max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)}
function draw(){const r=data.cases[index],threshold=+$('confidence').value,kept=r.predictions.map((p,i)=>({...p,i})).filter(p=>p.confidence>=threshold),pairs=[];r.truth.forEach((g,gi)=>kept.forEach(p=>{const iou=overlap(g.box,p.box);if(g.class_id===p.class_id&&iou>=.5)pairs.push({gi,pi:p.i,iou})}));pairs.sort((a,b)=>b.iou-a.iou||b.gi-a.gi||b.pi-a.pi);const gs=new Set(),ps=new Set();pairs.forEach(p=>{if(!gs.has(p.gi)&&!ps.has(p.pi)){gs.add(p.gi);ps.add(p.pi)}});$('title').textContent=r.title;$('confidenceValue').textContent=threshold.toFixed(2);const view=zoomed?r.zoom_region:[0,0,r.width,r.height],viewbox=zoomed?[view[0],view[1],view[2]-view[0],view[3]-view[1]]:view;for(const [id,boxes]of [['truth',r.truth],['predictions',kept]]){const svg=$(id);svg.setAttribute('viewBox',viewbox.join(' '));svg.innerHTML='<image href="'+r.asset+'" width="'+r.width+'" height="'+r.height+'"/>';boxes.forEach((b,j)=>{const prediction=id==='predictions',color=prediction?(ps.has(b.i)?'#36bfff':'#ff6464'):'#3df38a',x=b.box[0],y=b.box[1],w=b.box[2]-x,h=b.box[3]-y,label=b.name+(prediction?' '+b.confidence.toFixed(2):'');const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');Object.entries({x,y,width:w,height:h,fill:'transparent',stroke:color,'stroke-width':2,'vector-effect':'non-scaling-stroke'}).forEach(([k,v])=>rect.setAttribute(k,v));const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=label;rect.appendChild(title);svg.appendChild(rect);if(h>=50||zoomed){const text=document.createElementNS('http://www.w3.org/2000/svg','text');Object.entries({x:x+3,y:Math.max(25,y-5),fill:color,'font-size':Math.max(18,viewbox[2]/Math.max(svg.clientWidth,240)*11),'font-weight':700,stroke:'#101722','stroke-width':1,'paint-order':'stroke'}).forEach(([k,v])=>text.setAttribute(k,v));text.textContent=label;svg.appendChild(text)}})}$('counts').textContent=r.truth.length+' supplied labels · '+kept.length+' shown predictions · '+gs.size+' matched labels · '+(r.truth.length-gs.size)+' labels without a matching shown prediction';const focus=r.truth[r.focus_gt];$('review').textContent=r.review_note||'';$('focus').textContent='Example object is labelled '+focus.name+'. '+(gs.has(r.focus_gt)?'It has a matching prediction at this threshold.':'It has no same-class matching prediction at this threshold.');$('full').href='case_'+(index+1)+'_full.png';$('crop').href='case_'+(index+1)+'_zoom.png';$('zoom').textContent=zoomed?'Show entire image':'Zoom to example object'}
fetch('examples.json').then(r=>r.json()).then(d=>{data=d;$('model').textContent='Frozen '+d.source_run+' checkpoint after '+d.checkpoint_epoch_completed+' completed epoch(s) · 1280-pixel input · CPU FP32 inference';d.cases.forEach((r,i)=>{const option=document.createElement('option');option.value=i;option.textContent=(i+1)+'. '+r.title;$('example').appendChild(option)});$('example').onchange=e=>{index=+e.target.value;zoomed=false;draw()};$('confidence').oninput=draw;$('zoom').onclick=()=>{zoomed=!zoomed;draw()};draw()});</script></html>'''


def launch(output):
    assert sys.platform == "linux"
    prepare(output)
    shutil.copy2(Path(__file__), output/"runner_source.py")
    with (output/"console.log").open("xb", buffering=0) as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--output", str(output)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(output/"owner.json", {"pid": process.pid,
        "linux_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "linux_process_start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split(") ",1)[1].split()[19],
        "started_utc": datetime.now(timezone.utc).isoformat(), "cpu_only": True})
    print(json.dumps({"launched": True, "pid": process.pid, "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        code = 1
        try:
            worker(output)
            code = 0
        finally:
            save_json(output/"exit.json", {"returncode": code, "finished_utc": datetime.now(timezone.utc).isoformat()})
    else:
        launch(output)
