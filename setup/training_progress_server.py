"""Read-only, localhost dashboard for an existing detection training run."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
# One Ultralytics progress update: "1/1  4.46G  1.043  0.6934  0.9623  51  1280: 0% ... 52/28448 2.3it/s"
PROGRESS = re.compile(r"(\d+)/(\d+)\s+([\d.]+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+:\s+\d+%.*?(\d+)/(\d+)\s+([\d.]+)(it/s|s/it)")
_curve_cache = {"key": None, "value": {}}


def curves(log_path, points=400):
    """Running-average losses, speed and GPU memory parsed from the training log, thinned to about `points`."""
    try:
        stat = log_path.stat()
    except OSError:
        return {}
    key = (str(log_path), stat.st_size, stat.st_mtime)
    if _curve_cache["key"] == key:
        return _curve_cache["value"]
    rows = {}
    with log_path.open("rb") as stream:
        text = ANSI.sub("", stream.read().decode("utf-8", errors="replace"))
    for match in PROGRESS.finditer(text):
        epoch, gpu, box, cls, dfl, done, total, rate, unit = (int(match[1]), float(match[3]), float(match[4]), float(match[5]),
                                                              float(match[6]), int(match[7]), int(match[8]), float(match[9]), match[10])
        speed = rate if unit == "it/s" else (1 / rate if rate else 0.0)
        rows[(epoch, done)] = (epoch, done, total, gpu, box, cls, dfl, speed)
    ordered = [rows[k] for k in sorted(rows)]
    step = max(1, len(ordered) // points)
    thin = ordered[::step] + ([ordered[-1]] if ordered and (len(ordered) - 1) % step else [])
    value = {"x": [(r[0] - 1) * r[2] + r[1] for r in thin], "batches_per_epoch": thin[-1][2] if thin else None,
             "gpu_gb": [r[3] for r in thin], "box": [r[4] for r in thin], "cls": [r[5] for r in thin],
             "dfl": [r[6] for r in thin], "speed": [r[7] for r in thin]}
    _curve_cache.update(key=key, value=value)
    return value


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def snapshot(run):
    launcher = run.with_name(run.name + "_launcher")
    recipe = read_json(run / "recipe.json")
    if not recipe:
        command = read_json(launcher / "trainer_process.json").get("command", [])
        if "--recipe" in command and command.index("--recipe") + 1 < len(command):
            recipe_path = Path(command[command.index("--recipe") + 1]).resolve()
            if recipe_path.is_relative_to(ROOT):
                recipe = read_json(recipe_path)
    batch = read_json(run / "batch_progress.json")
    status = read_json(run / "status.json")
    progress = read_json(run / "progress.json")
    finished = read_json(launcher / "exit.json")
    history = []
    try:
        rows = csv.DictReader(io.StringIO((run / "results.csv").read_text(encoding="utf-8-sig")))
        for row in rows:
            values = {}
            for key, value in row.items():
                try:
                    values[key.strip()] = float(value)
                except (ValueError, TypeError, AttributeError):
                    pass
            if "epoch" in values:
                history.append(values)
    except OSError:
        pass
    log_path = launcher / "training.log"
    log_age = None
    log_lines = []
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 12000))
            raw = stream.read().decode("utf-8", errors="replace")
        log_lines = [line.strip() for line in ANSI.sub("", raw).replace("\r", "\n").splitlines() if line.strip()][-12:]
        log_age = max(0, datetime.now().timestamp() - log_path.stat().st_mtime)
    except OSError:
        pass
    return {
        "run": run.name,
        "recipe": recipe,
        "now_utc": datetime.now(timezone.utc).isoformat(),
        "batch": batch,
        "status": status.get("status", "checking_dataset" if (launcher / "owner.json").is_file() and not run.exists() else "waiting"),
        "epochs_completed": status.get("epochs_completed", len(history)),
        "progress": progress,
        "history": history,
        "exit": finished,
        "log": "\n".join(log_lines),
        "log_age_seconds": log_age,
        "curves": curves(log_path),
        "lr_probe": None if run.name != "detection_train_20261025_02" else {
            "progress": read_json(ROOT / "runs/detection_lr_probe_20261025_01/progress.json"),
            "baseline": read_json(ROOT / "runs/detection_lr_probe_20261025_01/baseline.json"),
            "control": read_json(ROOT / "runs/detection_lr_probe_20261025_01/control_result.json"),
            "higher": read_json(ROOT / "runs/detection_lr_probe_20261025_01/higher_result.json"),
            "review": read_json(ROOT / "runs/detection_lr_probe_20261025_01/review.json"),
            "exit": read_json(ROOT / "runs/detection_lr_probe_20261025_01/exit.json"),
        },
    }


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tesla · Live training</title>
<style>
:root{color-scheme:dark;--bg:#0d1118;--panel:#171e29;--muted:#a0b1c7;--line:#2b3647;--ink:#edf3ff;--green:#74e0b1;--blue:#84b7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",sans-serif}main{max-width:1150px;margin:auto;padding:32px 26px 50px}
header{display:flex;gap:20px;align-items:center;justify-content:space-between;margin-bottom:28px}.eyebrow{color:var(--green);font-size:12px;letter-spacing:2px;font-weight:700}h1{font-size:32px;line-height:1.2;margin:7px 0}p{margin:7px 0;color:var(--muted)}.badge{white-space:nowrap;background:#16342d;border:1px solid #28594b;color:var(--green);padding:8px 15px;border-radius:30px}.badge.warn{background:#423519;border-color:#6d562a;color:#ffdb89}.badge.error{background:#411f26;color:#ffb2bd;border-color:#6a323e}
.panel,.card{border:1px solid var(--line);background:var(--panel);border-radius:16px;padding:24px}.hero{margin-bottom:18px}.row{display:flex;justify-content:space-between;align-items:baseline;gap:16px}.epoch{font-size:30px;font-weight:650}.percent{font-size:29px;color:var(--green)}progress{appearance:none;width:100%;height:13px;border:0;border-radius:20px;overflow:hidden;margin:18px 0 8px;background:#293447}progress::-webkit-progress-bar{background:#293447}progress::-webkit-progress-value{background:linear-gradient(90deg,#4fc9ab,#91e6bd);border-radius:20px}progress::-moz-progress-bar{background:var(--green)}.small{font-size:13px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}.label{font-size:13px;color:var(--muted)}.value{font-size:27px;font-weight:650;margin-top:7px}.sub{font-size:12px;color:var(--muted);margin-top:3px}.columns{display:grid;grid-template-columns:1.25fr 1fr;gap:18px;margin-bottom:18px}h2{margin:0 0 12px;font-size:19px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:18px}.metric strong{display:block;font-size:25px;margin-top:4px}.note{color:var(--muted);font-size:13px;margin-top:15px}.loss{display:flex;gap:24px;flex-wrap:wrap}.loss strong{display:block;font-size:23px}.plot{width:100%;height:115px;display:block;margin-top:16px}.plot text{fill:var(--muted);font-size:10px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}.chart h3{margin:0;font-size:15px}.chart .help{font-size:12px;color:var(--muted);margin:2px 0 8px}.chart svg{width:100%;height:150px;display:block;overflow:visible}.chart svg text{fill:var(--muted);font-size:10px}.chart .now{font-size:13px;color:var(--ink);font-weight:600}.tip{position:fixed;pointer-events:none;background:#0b0f15;border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px;color:var(--ink);display:none;z-index:5}
@media(max-width:780px){.charts{grid-template-columns:1fr}}details summary{cursor:pointer;font-weight:600}pre{font:12px/1.7 Consolas,monospace;white-space:pre-wrap;word-break:break-word;color:#bfcfe6;max-height:255px;overflow:auto;margin-bottom:0}.footer{margin-top:18px;color:var(--muted);font-size:12px}.live{color:var(--green)}a{color:var(--blue)}
@media(max-width:780px){main{padding:22px 16px}.grid{grid-template-columns:1fr 1fr}.columns{grid-template-columns:1fr}header{align-items:flex-start}h1{font-size:27px}.panel,.card{padding:18px}}
</style></head><body><main>
<header><div><div class="eyebrow">TESLA / INDIAN TRAFFIC</div><h1>Live model training</h1><p>People and vehicle detector · YOLO11s · 12 object categories</p></div><span id="state" class="badge">Connecting…</span></header>
<section class="panel hero"><div class="row"><div><div class="label">Current epoch</div><div id="epoch" class="epoch">—</div></div><div id="percent" class="percent">—</div></div>
<progress id="bar" max="100" value="0" aria-label="Training batches in current epoch"></progress><div class="row small"><span id="batches">Waiting for progress…</span><span>Training batches this epoch</span></div>
<p class="note" id="phase">The display refreshes automatically. Training runs independently of this page.</p></section>
<div class="grid"><section class="card"><div class="label">Training elapsed</div><div class="value" id="elapsed">—</div><div class="sub">Since the first training batch</div></section><section class="card"><div class="label">Epochs completed</div><div class="value" id="completed">—</div><div class="sub" id="maxEpochs">Maximum epochs for this run</div></section><section class="card"><div class="label">Input resolution</div><div class="value" id="resolution">—</div><div class="sub" id="batchsize">—</div></section><section class="card"><div class="label">Peak GPU memory</div><div class="value" id="gpu">—</div><div class="sub">RTX 4060 Laptop · 8 GB</div></section></div>
<section class="panel" style="margin-bottom:18px"><h2>How training is going</h2><p class="small">Each line is the framework's running average since the epoch started, plotted against batches completed. Lower loss means the model fits the training images better; it is not an accuracy score. Hover a chart to read exact values.</p></section>
<div class="charts"><section class="card chart"><div class="row"><h3>Box loss</h3><span class="now" id="nowBox">—</span></div><p class="help">How far predicted boxes are from the labelled boxes.</p><svg id="cBox" viewBox="0 0 480 150" role="img" aria-label="Box loss by batch"></svg></section>
<section class="card chart"><div class="row"><h3>Class loss</h3><span class="now" id="nowCls">—</span></div><p class="help">How often the object type (car, person, animal…) is wrong.</p><svg id="cCls" viewBox="0 0 480 150" role="img" aria-label="Class loss by batch"></svg></section>
<section class="card chart"><div class="row"><h3>Detail loss</h3><span class="now" id="nowDfl">—</span></div><p class="help">How precisely box edges are placed (distribution focal loss).</p><svg id="cDfl" viewBox="0 0 480 150" role="img" aria-label="Detail loss by batch"></svg></section>
<section class="card chart"><div class="row"><h3>Training speed</h3><span class="now" id="nowSpeed">—</span></div><p class="help">Batches per second (2 images per batch). Drops can mean the GPU is busy or throttling.</p><svg id="cSpeed" viewBox="0 0 480 150" role="img" aria-label="Batches per second by batch"></svg></section></div>
<div class="tip" id="tip"></div>
<section class="panel" id="lrPanel" style="margin-bottom:18px;display:none"><h2>Small learning-rate comparison</h2><p class="small">Same starting model and images. Current rate: 0.000030 · Test rate: 0.0000375 (+25%).</p><p id="lrPhase">Waiting for the comparison…</p><div class="metrics"><div class="metric"><span class="label">Current rate · sample mAP50</span><strong id="lrControl">—</strong></div><div class="metric"><span class="label">Higher rate · sample mAP50</span><strong id="lrHigher">—</strong></div></div><p class="note" id="lrDecision">This uses a 192-image screening sample. Full training continues separately.</p></section><div class="columns"><section class="panel"><div class="row"><h2>Latest validation scores</h2><span id="scoreEpoch" class="small">Pending</span></div><div class="metrics"><div class="metric"><span class="label">mAP50</span><strong id="map50">—</strong></div><div class="metric"><span class="label">mAP50–95</span><strong id="map95">—</strong></div><div class="metric"><span class="label">Precision</span><strong id="precision">—</strong></div><div class="metric"><span class="label">Recall</span><strong id="recall">—</strong></div></div><p class="note" id="scoresNote">Scores appear after an epoch finishes and validation completes.</p><svg id="plot" class="plot" viewBox="0 0 480 115" role="img" aria-label="Validation mAP50 by epoch"></svg></section>
<section class="panel"><h2>Current learning signals</h2><p class="small">Sampled losses from the latest progress update. These measure training error, rather than accuracy.</p><div class="loss"><div><span class="label">Box loss</span><strong id="box">—</strong></div><div><span class="label">Class loss</span><strong id="cls">—</strong></div><div><span class="label">Detail loss</span><strong id="dfl">—</strong></div></div><p class="note" id="steps">—</p><p class="note" id="policyNote">56,729 unique training images · 17,675 validation images.</p></section></div>
<section class="panel"><details open><summary>Live training log</summary><pre id="log">Loading…</pre></details></section>
<div class="footer"><span id="fresh" class="live">Connecting…</span> · Page checks every 5 seconds; batch details update about every 20 seconds.<br><span id="run">—</span> · You can close this page and reopen it while training continues.</div>
</main><script>
const $=id=>document.getElementById(id), fmt=n=>Number(n||0).toLocaleString('en-IN');
const duration=s=>{s=Math.max(0,Number(s)||0);return s>=3600?`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m`:`${Math.floor(s/60)}m ${Math.floor(s%60)}s`};
function chart(rows){const pts=rows.filter(r=>Number.isFinite(r['metrics/mAP50(B)']));if(!pts.length){$('plot').innerHTML='<text x="0" y="55">Validation history will appear here.</text>';return}const maxX=Math.max(2,...pts.map(p=>p.epoch));const xy=pts.map(p=>[20+(p.epoch-1)/(maxX-1)*440,93-p['metrics/mAP50(B)']*78]);$('plot').innerHTML='<path d="M20 15V93H465" fill="none" stroke="#344154"/><text x="22" y="110">Epoch 1</text><text x="400" y="110">Epoch '+maxX+'</text><polyline fill="none" stroke="#74e0b1" stroke-width="3" points="'+xy.map(p=>p.join(',')).join(' ')+'"/>'+xy.map(p=>'<circle r="3.5" fill="#74e0b1" cx="'+p[0]+'" cy="'+p[1]+'"/>').join('')}
function line(id,xs,ys,fmtY,total){const svg=$(id);if(!ys||ys.length<2){svg.innerHTML='<text x="0" y="75">Waiting for enough training output…</text>';return}
const L=44,R=470,T=8,B=126,lo=Math.min(...ys),hi=Math.max(...ys),pad=(hi-lo)*0.08||Math.abs(hi)*0.05||1,y0=lo-pad,y1=hi+pad,xmax=Math.max(total||0,xs.at(-1)||1);
const sx=x=>L+x/xmax*(R-L),sy=y=>B-(y-y0)/(y1-y0)*(B-T),pts=xs.map((x,i)=>[sx(x),sy(ys[i])]);
let g='';for(const f of [0,.5,1]){const v=y0+f*(y1-y0),yy=sy(v);g+=`<line x1="${L}" x2="${R}" y1="${yy}" y2="${yy}" stroke="#2b3647" stroke-width="1"/><text x="${L-6}" y="${yy+3}" text-anchor="end">${fmtY(v)}</text>`}
g+=`<text x="${L}" y="${B+16}">0</text><text x="${R}" y="${B+16}" text-anchor="end">${fmt(xmax)} batches</text>`;
const last=pts.at(-1);g+=`<polyline fill="none" stroke="#3987e5" stroke-width="2" stroke-linejoin="round" points="${pts.map(p=>p.map(v=>v.toFixed(1)).join(',')).join(' ')}"/><circle cx="${last[0]}" cy="${last[1]}" r="4" fill="#3987e5" stroke="#171e29" stroke-width="2"/>`;
g+=`<line id="${id}x" x1="0" x2="0" y1="${T}" y2="${B}" stroke="#a0b1c7" stroke-width="1" stroke-dasharray="3 3" visibility="hidden"/><rect x="${L}" y="${T}" width="${R-L}" height="${B-T}" fill="transparent"/>`;svg.innerHTML=g;
svg.onmousemove=e=>{const r=svg.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*480;let i=0,best=1e9;pts.forEach((p,k)=>{const d=Math.abs(p[0]-px);if(d<best){best=d;i=k}});const c=$(id+'x');c.setAttribute('x1',pts[i][0]);c.setAttribute('x2',pts[i][0]);c.setAttribute('visibility','visible');const t=$('tip');t.style.display='block';t.style.left=(e.clientX+12)+'px';t.style.top=(e.clientY-28)+'px';t.textContent=`Batch ${fmt(xs[i])} · ${fmtY(ys[i])}`};
svg.onmouseleave=()=>{$(id+'x').setAttribute('visibility','hidden');$('tip').style.display='none'}}
function charts(c,b){if(!c||!c.x){return}const total=(b.epochs_max||1)*(c.batches_per_epoch||b.batches_per_epoch||0),f3=v=>v.toFixed(3);
line('cBox',c.x,c.box,f3,total);line('cCls',c.x,c.cls,f3,total);line('cDfl',c.x,c.dfl,f3,total);line('cSpeed',c.x,c.speed,v=>v.toFixed(2)+'/s',total);
for(const [id,k,f] of [['nowBox','box',f3],['nowCls','cls',f3],['nowDfl','dfl',f3],['nowSpeed','speed',v=>v.toFixed(2)+' batches/s']])if(c[k]&&c[k].length)$(id).textContent='now '+f(c[k].at(-1))}
async function update(){try{const response=await fetch('/api/progress',{cache:'no-store'});if(!response.ok)throw Error(response.status);const d=await response.json(),b=d.batch||{},rows=d.history||[],last=rows.at(-1)||{},metrics={...last,...(d.progress.metrics||{})};const ep=b.epoch||1,max=b.epochs_max||d.recipe?.epochs||30,pct=Math.min(100,(b.batches_completed_in_epoch||0)/Math.max(1,b.batches_per_epoch||1)*100);$('epoch').textContent=`${ep} / ${max}`;$('percent').textContent=pct.toFixed(1)+'%';$('bar').value=pct;$('batches').textContent=`${fmt(b.batches_completed_in_epoch)} / ${fmt(b.batches_per_epoch)} batches`;$('elapsed').textContent=duration(b.elapsed_training_seconds);$('completed').textContent=fmt(Math.max(d.epochs_completed||0,last.epoch||0));$('resolution').textContent=(b.imgsz||d.recipe?.imgsz||960)+' px';$('batchsize').textContent='Batch size '+(b.batch||d.recipe?.batch||4);$('gpu').textContent=b.gpu_peak_allocated_bytes?(b.gpu_peak_allocated_bytes/1024**3).toFixed(2)+' GB':'—';
for(const [id,key] of [['map50','metrics/mAP50(B)'],['map95','metrics/mAP50-95(B)'],['precision','metrics/precision(B)'],['recall','metrics/recall(B)']])$(id).textContent=Number.isFinite(metrics[key])?(metrics[key]*100).toFixed(2)+'%':'—';
$('maxEpochs').textContent='Maximum '+max+' epochs in this run';const counts=d.recipe?.expected_prepared_counts||{};$('policyNote').textContent=(counts.train?fmt(counts.train)+' training images · '+fmt(counts.val)+' validation images':'Dataset: '+(d.recipe?.dataset||'—'))+' · dataset '+(d.recipe?.dataset||'').split('/').pop()+'. Early stopping patience: '+(d.recipe?.patience||8)+' epochs without improvement.';charts(d.curves,b);$('lrPanel').style.display=d.lr_probe?'block':'none';const scored=d.progress.epoch_completed||last.epoch;if(scored){$('scoreEpoch').textContent='Epoch '+scored;$('scoresNote').textContent='Measured on the unchanged validation split. Scores update after each completed validation.'}for(const [id,key] of [['box','box_loss'],['cls','cls_loss'],['dfl','dfl_loss']])$(id).textContent=Number.isFinite(b.losses?.[key])?b.losses[key].toFixed(4):'—';$('steps').textContent=fmt(b.optimizer_steps_completed)+' learning updates completed.';$('log').textContent=d.log||'Waiting for training output…';$('run').textContent=d.run;chart(rows);
if(d.lr_probe){const lp=d.lr_probe||{},lpp=lp.progress||{},lrLabels={baseline_validation:'Scoring the starting model',control_training:'Training with the current rate',control_matched_validation:'Scoring the current-rate model',higher_training:'Training with the slightly higher rate',higher_matched_validation:'Scoring the higher-rate model',completed:'Comparison completed'};if(lpp.phase){$('lrPhase').textContent=(lrLabels[lpp.phase]||'Preparing the comparison')+(lpp.batches!=null?' · '+lpp.batches+'/'+lpp.batches_expected+' training images':'')}for(const [id,arm] of [['lrControl',lp.control],['lrHigher',lp.higher]]){const value=arm?.validation?.metrics?.['metrics/mAP50(B)'];$(id).textContent=Number.isFinite(value)?(100*value).toFixed(2)+'%':'Pending'}if(lp.review?.decision){$('lrDecision').textContent=lp.review.decision+'. This is a small screening comparison; the main training rate is unchanged.'}else if(lp.exit?.returncode){$('lrDecision').textContent='The comparison was interrupted; its evidence is saved. The main training job is separate.'}}
const finished=Object.keys(d.exit||{}).length>0,stale=d.log_age_seconds==null||d.log_age_seconds>180;const badge=$('state');badge.className='badge';if(finished){const failed=d.exit.returncode!==0;badge.textContent=failed?'Training interrupted':'Training completed';if(failed)badge.classList.add('error');$('phase').textContent=failed?'An interruption was recorded. The latest saved progress is displayed.':'Training completed. Final results are shown below.'}else if(['checking_dataset','starting','preparing_training'].includes(d.status)){badge.textContent='Preparing training';badge.classList.add('warn');$('phase').textContent='Verifying the dataset and preparing the current training run. Previous models and results are preserved.'}else if(d.status==='completed'){badge.textContent='Training completed';$('phase').textContent='Training has completed; final reporting may still be running.'}else if(stale){badge.textContent='Waiting for fresh output';badge.classList.add('warn');$('phase').textContent='The training output has not updated recently. The latest saved progress is displayed.'}else{badge.textContent='Training live';$('phase').textContent=pct>=99?'Training batches are almost finished. Validation runs next before the next epoch.':'Training is active. Validation scores appear after each epoch and its validation pass.'}
$('fresh').textContent='Page refreshed '+new Date().toLocaleTimeString()+ (d.log_age_seconds!=null?' · Training output '+Math.floor(d.log_age_seconds)+'s ago':'');
}catch(error){$('state').textContent='Connection interrupted';$('state').className='badge error';$('fresh').textContent='Reconnecting automatically…';}finally{setTimeout(update,5000)}}update();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    run = args.run.resolve()
    launcher = run.with_name(run.name + "_launcher")
    if run.parent != (ROOT / "runs").resolve() or not (run.is_dir() or (launcher / "owner.json").is_file()):
        parser.error("Choose an existing training run directly inside this project's runs folder.")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                body, content_type = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            elif route == "/api/progress":
                body = json.dumps(snapshot(run), ensure_ascii=False, allow_nan=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Training dashboard: http://127.0.0.1:{args.port} · {run.name}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
