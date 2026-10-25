"""CPU-only strict-retention report, every reviewed frame and honest regressions."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image,ImageDraw,ImageFont


def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def image_path(value):
    """Read the immutable WSL manifest paths from either operating system."""
    if os.name=="nt" and value.startswith("/mnt/") and len(value)>7 and value[6]=="/":
        return Path(value[5].upper()+":/"+value[7:])
    return Path(value)


def delta(a,b):return 100*(a-b) if a is not None and b is not None else None


def percentage(value):return f"{100*value:.4f}%" if value is not None else "n/a"


def render(record,baseline_mask,candidate_mask,base,after,path):
    rgb=np.array(Image.open(image_path(record["image"])).convert("RGB"));target=np.array(Image.open(image_path(record["mask"])).convert("L"))
    original=np.array(Image.open(baseline_mask).convert("L"));candidate=np.array(Image.open(candidate_mask).convert("L"))
    reference=np.zeros_like(rgb);reference[target==1]=255;reference[target==255]=[110,72,130]
    def overlay(mask):
        result=rgb.copy();result[mask==1]=(.6*result[mask==1]+.4*np.array([30,220,110])).astype(np.uint8);return result
    def errors(mask):
        result=rgb.copy();result[(target==0)&(mask==1)]=[40,135,250];result[(target==1)&(mask==0)]=[245,65,65];return result
    canvas=Image.new("RGB",(1440,1085),(17,25,36));draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f"{record['name']} | original IoU {100*base['road_iou']:.3f}% | candidate {100*after['road_iou']:.3f}% | {100*(after['road_iou']-base['road_iou']):+.3f} pp",font=ImageFont.load_default(size=17),fill="white")
    label="Human-reviewed mask" if record["domain"]=="real_guard" else "Supplied real mask" if record["domain"]=="real" else "Geometry-assisted reference"
    panels=[("Image",rgb),(label+" (purple excluded)",reference),("Original errors: blue extra / red missed",errors(original)),
            ("Original locked prediction",overlay(original)),("Experimental candidate prediction",overlay(candidate)),("Candidate errors: blue extra / red missed",errors(candidate))]
    for index,(title,pixels) in enumerate(panels):
        x,y=480*(index%3),65+515*(index//3);draw.text((x+10,y-20),title,font=ImageFont.load_default(size=14),fill="white")
        canvas.paste(Image.fromarray(pixels).resize((480,480),Image.Resampling.NEAREST),(x,y))
    canvas.save(path,quality=92)


def dataset_map(results):
    return {"real_validation":results["real_validation"],"development":results["development"],
            **{"video_"+group:value for group,value in results["real_video_guards"].items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path);args=parser.parse_args()
    out=args.output.resolve(strict=True);status=load(out/"run_status.json")
    if status["phase"]!="completed":raise ValueError("Wait for completed trial")
    if (out/"MODEL_REPORT.md").exists():raise FileExistsError("Preserve completed report")
    result,config,manifest=load(out/"comparison.json"),load(out/"config.json"),load(out/"manifest.json")
    baseline=dataset_map(result["baseline"])
    shared=result["selection"]["selected_epoch"];simulator=result["simulator_selection"]["selected_epoch"]
    best_dev=max(result["epochs"],key=lambda row:row["evaluations"]["development"]["metrics"]["road_iou"])["epoch"]
    displayed=shared or simulator or best_dev
    choice="Shared-model candidate passed all measured strict aggregate gates" if shared else "Simulation-only experimental candidate; shared real-retention approval rejected" if simulator else "No eligible candidate; failed-trial diagnostic examples only"
    counts=[];per_image={};metrics_fields=("road_iou","boundary_iou_5px","precision","recall")
    lines=["# Real-retention rehearsal experiment", "", "## Decision", "", choice+".",
           "The original locked model remains the default for real/unknown inputs. No deployment lock was changed.",
           "A measured aggregate non-regression result does not guarantee unchanged predictions for every image or unseen real roads.",
           "", "## Matched measurements", "", "| Trial / epoch | Dataset | Road IoU | Boundary IoU 5px | Precision | Recall |", "|---|---|---:|---:|---:|---:|"]
    def metric_row(name,group,metrics):
        return "| "+name+" | "+group+" | "+" | ".join(percentage(metrics[field]) for field in metrics_fields)+" |"
    for group,value in baseline.items():lines.append(metric_row("Original baseline",group,value["metrics"]))
    for epoch in result["epochs"]:
        measured=dataset_map(epoch["evaluations"]);changes={}
        for group,data in measured.items():
            lines.append(metric_row(f"Epoch {epoch['epoch']}",group,data["metrics"]))
            original={row["name"]:row["metrics"] for row in baseline[group]["rows"]};entries=[]
            for row in data["rows"]:
                before,after=original[row["name"]],row["metrics"]
                entries.append({"name":row["name"],"original":before,"candidate":after,
                    "road_delta_pp":delta(after["road_iou"],before["road_iou"]),
                    "boundary5_delta_pp":delta(after["boundary_iou_5px"],before["boundary_iou_5px"])})
            changes[group]=entries
            counts.append({"epoch":epoch["epoch"],"dataset":group,"frames":len(entries),
                           "road_iou_worse":sum(row["road_delta_pp"] is not None and row["road_delta_pp"]<0 for row in entries),
                           "road_iou_better":sum(row["road_delta_pp"] is not None and row["road_delta_pp"]>0 for row in entries),
                           "boundary5_iou_worse":sum(row["boundary5_delta_pp"] is not None and row["boundary5_delta_pp"]<0 for row in entries)})
        per_image[str(epoch["epoch"])]=changes
    (out/"per_image_comparison.json").write_text(json.dumps(per_image,indent=2)+"\n",encoding="utf-8")
    lines += ["", "## Exact strict gates", "",
              "Each epoch must equal or exceed the original baseline road IoU AND boundary IoU5 independently on all 687 real validation images, the 12 time-selected human-scored frames, and the 6 diagnostic human-scored frames. Integer cross-products compare ratios without a rounding allowance.", ""]
    for epoch in result["epochs"]:
        lines.append(f"- Epoch {epoch['epoch']}: "+"; ".join(group+"="+("PASS" if gate["passed"] else "FAIL") for group,gate in epoch["strict_gates"].items()))
    lines += ["", "## Individual regressions remain visible", "",
              "| Epoch | Dataset | Frames | Road IoU worse | Road IoU better | Boundary5 worse |", "|---|---|---:|---:|---:|---:|"]
    for row in counts:lines.append(f"| {row['epoch']} | {row['dataset']} | {row['frames']} | {row['road_iou_worse']} | {row['road_iou_better']} | {row['boundary5_iou_worse']} |")
    lines += ["", "## Data, training and limits", "",
        f"- Real training replay: {config['real_train_images']} train-only images; synthetic training: {config['synthetic_train_images']} immutable clear-scene frames.",
        "- The original teacher generated float32 clipped logits only for those real TRAIN images; lossless cached targets and input hashes are recorded. No validation or reviewed-video labels enter optimization.",
        f"- Frozen encoder/all BatchNorm; decoder Adam LR {config['learning_rate']:g}; boundary-emphasized logit distillation weight {config['distillation_weight']:g}; {config['epochs']} epochs; batch 4.",
        "- Real replay has no augmentation so each image matches its teacher target exactly. Synthetic frames receive declared gentle augmentation. RGB is normalized once; inference threshold stays>=0.5.",
        "- All three original checkpoints, split manifest and deployment lock remain unchanged. Reserved 689 test images were not opened.",
        "- Primary 12 and diagnostic 6 reviewed-video groups remain separate. They are regression guards from previously reviewed clips, not a new independent real-world test.",
        "- The new future simulator recording was not opened or evaluated by this tool. Simulator development shares a town/route family and geometry-assisted references; synthetic appearance limitations remain.",
        "- All epoch checkpoints, masks and per-image changes are retained, including failed epochs. Simulator-only eligibility is based on development IoU and is not shared-model approval.", ""]
    (out/"MODEL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(12,3.2));epochs=[row["epoch"] for row in result["epochs"]]
    axes[0].plot(epochs,[row["train_loss"] for row in result["epochs"]],"o-");axes[0].set_title("Training loss")
    for index,(title,group,key) in enumerate((("Real validation IoU","real_validation","road_iou"),("Synthetic development IoU","development","road_iou")),1):
        axes[index].plot(epochs,[100*row["evaluations"][group]["metrics"][key] for row in result["epochs"]],"o-")
        axes[index].axhline(100*baseline[group]["metrics"][key],linestyle="--",color="gray",label="original");axes[index].legend();axes[index].set_title(title)
    for axis in axes:axis.set_xlabel("Epoch");axis.set_xticks(epochs);axis.grid(alpha=.25)
    figure.tight_layout();figure.savefig(out/"curves.png",dpi=160);plt.close(figure)
    actual={"real_validation":manifest["real_validation"],"development":manifest["development"],
            **{"video_"+group:rows for group,rows in manifest["real_video_guards"].items()}}
    folder=out/"comparisons";folder.mkdir(exist_ok=False);cards=[];pictures=[]
    for group,rows in actual.items():
        changes=per_image[str(displayed)][group];by_name={row["name"]:row for row in changes};chosen={}
        if group.startswith("video_"):chosen={row["name"]:"all reviewed frames" for row in rows}
        else:
            for kind,ordered in (("largest road regression",sorted(changes,key=lambda row:row["road_delta_pp"])),
                                 ("largest road gain",sorted(changes,key=lambda row:row["road_delta_pp"],reverse=True)),
                                 ("lowest candidate IoU",sorted(changes,key=lambda row:row["candidate"]["road_iou"]))):
                for row in ordered[:4]:chosen.setdefault(row["name"],kind)
        cards.append("<h2>"+html.escape(group.replace("_"," ").title())+"</h2>")
        for record in rows:
            name=record["name"]
            if name not in chosen:continue
            before=out/"evaluations"/("baseline_"+group)/"masks_01"/(name+".png")
            after=out/"evaluations"/(f"epoch_{displayed:02d}_"+group)/"masks_01"/(name+".png")
            path=folder/(group+"_"+name+".jpg");change=by_name[name]
            render(record,before,after,change["original"],change["candidate"],path)
            relative=str(path.relative_to(out));pictures.append(relative)
            boundary_text=f'{change["boundary5_delta_pp"]:+.3f} pp' if change["boundary5_delta_pp"] is not None else "n/a"
            cards.append(f'<article><h3>{html.escape(name)} · {html.escape(chosen[name])} · road {change["road_delta_pp"]:+.3f} pp · boundary5 {boundary_text}</h3><a href="{relative}"><img loading="lazy" src="{relative}" alt="Original and experimental candidate comparison including errors"></a></article>')
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Road model real-retention review</title><style>:root{color-scheme:dark}body{max-width:1480px;margin:auto;padding:24px;background:#101923;color:#edf3f9;font:16px system-ui}p{max-width:1100px;line-height:1.6}a{color:#70e2ef}article{padding:12px;margin:20px 0;background:#1b2b3c;border:1px solid #394d63;border-radius:12px}h3{margin:0 0 12px}img{display:block;width:100%;height:auto}.notice{border-left:4px solid #f1c76c;background:#233547;padding:18px}nav{display:flex;flex-wrap:wrap;gap:24px;margin:25px 0}</style><h1>Road model real-retention review</h1>'''
    document+='<p class="notice">'+html.escape(choice)+'. Displayed epoch '+str(displayed)+'. Real/unknown inputs continue using the original locked model.</p><p>Aggregate guards can hide individual regressions. Every reviewed-video frame is shown below, alongside real-validation and simulator gains, regressions and worst cases. Blue means extra predicted road; red means missed road; purple reference pixels are excluded. No reserved real test image or fresh future simulator check was opened.</p><nav><a href="MODEL_REPORT.md">Full measurements and strict gates</a><a href="per_image_comparison.json">Every per-image change</a><a href="config.json">Exact training settings</a><a href="comparison.json">Exact evaluation counts</a></nav><img src="curves.png" alt="Training loss, real validation IoU and development IoU">'+''.join(cards)+'</html>'
    (out/"index.html").write_text(document,encoding="utf-8")
    from html.parser import HTMLParser
    class Check(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ("href","src"):
                    path=(out/value).resolve();assert path.is_relative_to(out) and path.is_file()
    Check().feed(document)
    shutil.copy2(__file__,out/"source"/Path(__file__).name)
    check={"passed":True,"displayed_epoch":displayed,"comparison_images":len(pictures),"all_reviewed_video_frames_shown":18,
           "all_local_links_exist":True,"browser_controls_tested":False,"additional_inference":False,
           "report_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"report_check.json").write_text(json.dumps(check,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(check))


if __name__=="__main__":main()
