"""CPU-only baseline/candidate gallery from completed saved masks; no inference."""
import argparse
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def picture(record, baseline_path, candidate_path, base_iou, candidate_iou, path):
    rgb=np.array(Image.open(record["image"]).convert("RGB"))
    truth=np.array(Image.open(record["mask"]).convert("L"))
    baseline=np.array(Image.open(baseline_path).convert("L"))
    candidate=np.array(Image.open(candidate_path).convert("L"))
    assert set(np.unique(baseline)) <= {0,1} and set(np.unique(candidate)) <= {0,1}
    assert set(np.unique(truth)) <= {0,1,255}
    target=np.zeros_like(rgb);target[truth==1]=255;target[truth==255]=[110,72,130]
    def overlay(mask):
        view=rgb.copy();yes=mask==1;view[yes]=(.6*view[yes]+.4*np.array([30,220,110])).astype(np.uint8)
        return view
    def error(mask):
        view=rgb.copy();view[(truth==1)&(mask==0)]=[245,65,65];view[(truth==0)&(mask==1)]=[40,135,250]
        return view
    output=Image.new("RGB",(1440,626),(17,25,36));draw=ImageDraw.Draw(output)
    font=ImageFont.load_default(size=17);small=ImageFont.load_default(size=14)
    draw.text((12,8),f"{record['name']} | {record.get('sim_s',0):.2f}s | baseline {100*base_iou:.2f}% | candidate {100*candidate_iou:.2f}% | {100*(candidate_iou-base_iou):+.2f} pp",font=font,fill="white")
    panels=[("Camera",rgb),("Geometry-assisted reference (purple excluded)",target),
            ("Baseline errors: blue extra / red missed",error(baseline)),
            ("Locked baseline prediction",overlay(baseline)),("Selected candidate prediction",overlay(candidate)),
            ("Candidate errors: blue extra / red missed",error(candidate))]
    for index,(title,view) in enumerate(panels):
        x,y=(index%3)*480,55+(index//3)*298
        draw.text((x+10,y-20),title,font=small,fill="white")
        output.paste(Image.fromarray(view).resize((480,270),Image.Resampling.NEAREST),(x,y))
    output.save(path,quality=92)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    output=args.output.resolve(strict=True)
    if (output/"index.html").exists():
        raise FileExistsError("Preserve existing gallery")
    comparison,state=load(output/"comparison.json"),load(output/"run_status.json")
    if state["phase"]!="completed":
        raise ValueError("Wait for completed model evaluation")
    selected=comparison["selection"]["selected_epoch"]
    selected_label=f"Epoch {selected}" if selected else "No qualifying candidate"
    table_rows=[]
    def table(label,base,candidate):
        if base is None or candidate is None:
            return
        table_rows.append(f"<tr><th>{html.escape(label)}</th><td>{base['road_iou']*100:.3f}%</td><td>{candidate['road_iou']*100:.3f}%</td><td>{100*(candidate['road_iou']-base['road_iou']):+.3f} pp</td></tr>")
    epoch=(selected or 2)-1
    table("Real validation · 687 images",comparison["baseline_real"],comparison["epochs"][epoch]["real_validation"])
    table("Synthetic development · 60 frames",comparison["baseline_development"],comparison["epochs"][epoch]["synthetic_development"])
    table("Withheld synthetic · 110 frames",comparison["baseline_heldout"],comparison["candidate_heldout"])
    cards=[]
    entries=[]
    if selected:
        records=load(output/"data/heldout_after_selection/manifest.json")["records"]
        base=load(output/"evaluations/baseline_heldout/metrics.json")
        candidate=load(output/"evaluations/candidate_heldout/metrics.json")
        base={row["name"]:row["metrics"] for row in base["rows"]}
        candidate={row["name"]:row["metrics"] for row in candidate["rows"]}
        folder=output/"withheld_comparisons";folder.mkdir(exist_ok=False)
        for record in records:
            name=record["name"]
            path=folder/(name+".jpg")
            picture(record,output/"evaluations/baseline_heldout/masks_01"/(name+".png"),
                    output/"evaluations/candidate_heldout/masks_01"/(name+".png"),
                    base[name]["road_iou"],candidate[name]["road_iou"],path)
            entry={"name":name,"sim_s":record["sim_s"],"comparison":str(path.relative_to(output)),
                   "baseline_iou":base[name]["road_iou"],"candidate_iou":candidate[name]["road_iou"]}
            entries.append(entry)
            cards.append(f'<article><h3>{record["sim_s"]:.2f}s · {html.escape(name)}</h3><a href="{entry["comparison"]}"><img loading="lazy" src="{entry["comparison"]}" alt="Baseline and candidate road predictions at {record["sim_s"]:.2f} seconds"></a></article>')
        (folder/"manifest.json").write_text(json.dumps(entries,indent=2)+"\n",encoding="utf-8")
    evidence=[]
    for name in ("real_validation","synthetic_development"):
        examples=load(output/"examples"/name/"manifest.json")
        evidence.append(f'<h2>{html.escape(name.replace("_"," ").title())}: gains, regressions and worst cases</h2>')
        for entry in examples:
            evidence.append(f'<article><h3>{html.escape(entry["name"])} · {html.escape(entry["kind"])} · {entry["delta_pp"]:+.2f} pp</h3><a href="{entry["comparison"]}"><img loading="lazy" src="{entry["comparison"]}" alt="Saved baseline and candidate comparison"></a></article>')
    status=("Qualified on development and real-validation retention guards; original deployed lock preserved."
            if selected else "Neither epoch met all selection guards. These are failed candidate diagnostics; the locked original remains preferred. Withheld imagery was not opened.")
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Road-boundary model comparison</title>
<style>:root{color-scheme:dark}body{font:16px system-ui;max-width:1480px;margin:0 auto;padding:24px;background:#101923;color:#edf3f9}h1,h2{line-height:1.2}p{max-width:1050px;line-height:1.6}a{color:#6fe1ee}table{border-collapse:collapse;margin:20px 0;width:100%;max-width:950px}th,td{padding:12px;border-bottom:1px solid #394c61;text-align:left}article{background:#192838;border:1px solid #394c61;border-radius:12px;padding:12px;margin:18px 0}article h3{margin:0 0 12px}img{width:100%;height:auto;display:block;border-radius:5px}.notice{padding:16px;border-left:4px solid #f4cb72;background:#233246}nav{display:flex;gap:20px;flex-wrap:wrap;margin:24px 0}code{background:#263749;padding:2px 5px}</style>
<h1>Road-boundary model comparison</h1>'''
    document+=f'<p class="notice"><strong>{selected_label}.</strong> {status}</p><p>Baseline is the locked morning-best model. Both use the same U-Net/MobileNetV2 architecture, one normalization and a 0.5 threshold. Blue means extra predicted road; red means missed road. Synthetic reference labels use aligned simulator depth and driving-lane geometry; purple pixels are excluded.</p>'
    document+='<table><thead><tr><th>Dataset</th><th>Locked baseline IoU</th><th>Candidate IoU</th><th>Difference</th></tr></thead><tbody>'+''.join(table_rows)+'</tbody></table>'
    document+='<p>This is an offline comparison. The candidate does not control the car. The synthetic recordings share one town and route family, so these scores do not establish general real-road performance. No reserved test images were opened.</p><nav><a href="MODEL_REPORT.md">Full report</a><a href="comparison.json">Exact counts</a><a href="config.json">Training settings</a><a href="history.csv">Training history</a></nav>'
    if comparison.get("heldout_appearance_limit"):
        document+='<p class="notice">'+html.escape(comparison["heldout_appearance_limit"])+"</p>"
    if (output/"curves.png").exists():document+='<h2>Two-epoch training and validation curves</h2><img src="curves.png" alt="Training loss, real validation IoU and development IoU">'
    fixed=output/"fixed_case_comparison.json"
    if fixed.exists():
        document+='<h2>Your 40.5-second example</h2><p>These are known development examples scored after the candidate was frozen. They were not used for candidate selection. The two rows use different scene appearances; within each row, both models see the exact same RGB and reference.</p>'
        for entry in load(fixed)["cases"]:
            document+=f'<article><h3>{html.escape(entry["name"].replace("_"," ").title())}: baseline {entry["baseline"]["road_iou"]*100:.2f}% → candidate {entry["candidate"]["road_iou"]*100:.2f}% ({entry["delta_pp"]:+.2f} pp)</h3><a href="{entry["comparison"]}"><img src="{entry["comparison"]}" alt="Known user example before and after model adaptation"></a></article>'
    document+='<h2>All withheld synthetic frames</h2>'+(''.join(cards) if cards else '<p>Not opened because no candidate qualified.</p>')
    document+=''.join(evidence)+'</html>'
    (output/"index.html").write_text(document,encoding="utf-8")
    # Static artifact verification; this does not claim browser-control testing.
    from html.parser import HTMLParser
    class Check(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ("href","src"):
                    assert value and not value.startswith(("http:","https:","file:","javascript:"))
                    assert (output/value).resolve().is_relative_to(output)
                    assert (output/value).is_file(),value
    Check().feed(document)
    result={"passed":True,"withheld_frames":len(entries),"full_comparison_images":len(entries),
            "all_local_links_exist":True,"browser_controls_tested":False,"additional_inference":False}
    (output/"gallery_check.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    shutil.copy2(__file__,output/"source"/Path(__file__).name)
    print(json.dumps(result))


if __name__=="__main__":
    main()
