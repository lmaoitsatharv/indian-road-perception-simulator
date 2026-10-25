"""Build a portable, complete validation gallery from the two saved mask exports."""
import argparse
import csv
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Original image', 'Supplied mask', 'Original model', 'New model: trial 13', 'New model vs supplied mask']
FOOTER = 'Red: labelled road missed. Blue: extra predicted road. Disagreement can include annotation/class-definition issues. Masks remain 0/1; white is display-only.'


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def load_metrics(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return {int(row['id']): float(row['road_iou']) for row in csv.DictReader(stream)}


def font(size):
    for path in [Path('C:/Windows/Fonts/arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def render_pair(key, output, baseline, candidate):
    rgb = np.asarray(Image.open(ROOT / f'dataset/splits/val/images/Image_{key}.png').convert('RGB'))
    truth = np.asarray(Image.open(ROOT / f'dataset/splits/val/masks/Mask_{key}.png'))
    before = np.asarray(Image.open(output / f'predicted_masks/baseline/Prediction_{key}.png'))
    after = np.asarray(Image.open(output / f'predicted_masks/trial13/Prediction_{key}.png'))
    assert rgb.shape == (512, 512, 3)
    assert all(mask.shape == (512, 512) and np.isin(mask, [0, 1]).all() for mask in [truth, before, after])
    errors = rgb.copy()
    errors[(truth == 1) & (after == 0)] = [240, 55, 55]
    errors[(truth == 0) & (after == 1)] = [35, 125, 245]
    panels = [Image.fromarray(rgb), Image.fromarray(truth * 255), Image.fromarray(before * 255),
              Image.fromarray(after * 255), Image.fromarray(errors)]
    sheet = Image.new('RGB', (2560, 596), 'white')
    draw = ImageDraw.Draw(sheet)
    captions = [f'ID {key}: original image', 'Supplied mask (unchanged)',
                f'Original model: IoU {baseline[key]:.2%}', f'New model: IoU {candidate[key]:.2%}',
                'New model: label disagreement']
    for column, (panel, caption) in enumerate(zip(panels, captions)):
        draw.text((column * 512 + 12, 17), caption, font=font(22), fill='#202731')
        sheet.paste(panel.convert('RGB'), (column * 512, 58))
    draw.text((12, 576), FOOTER, font=font(16), fill='#333333')
    sheet.save(output / f'comparisons/Comparison_{key}.png', compress_level=3)
    thumbnail = sheet.resize((1280, 298), Image.Resampling.LANCZOS)
    thumbnail.save(output / f'previews/Comparison_{key}.jpg', quality=88, subsampling=0)
    return dict(id=key, baseline_iou=baseline[key], candidate_iou=candidate[key],
                change_pp=(candidate[key] - baseline[key]) * 100,
                image=f'comparisons/Comparison_{key}.png', preview=f'previews/Comparison_{key}.jpg')


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IDD • original vs new model • all validation images</title>
<style>
:root{color-scheme:dark;--bg:#0d1219;--panel:#161e2a;--line:#2a3747;--muted:#abb9c9;--accent:#82d9c5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#f2f6fa;font:15px/1.55 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px 24px 60px}h1{font-size:clamp(25px,3vw,38px);line-height:1.2;margin:9px 0 15px}
.eyebrow{color:var(--accent);letter-spacing:.12em;font-size:12px;text-transform:uppercase}.intro{color:var(--muted);max-width:1040px}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 19px;min-width:170px}.stat strong{display:block;font-size:23px}.stat span{color:var(--muted);font-size:12px}
.toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:end;padding:16px;background:var(--panel);border:1px solid var(--line);border-radius:14px;position:sticky;top:8px;z-index:2}
label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:5px}input,select,button,.link{border:1px solid #40516a;border-radius:8px;background:#0d1724;color:#f1f6fc;padding:10px 12px;font:inherit}
input{width:185px}button,.link{cursor:pointer;text-decoration:none}button:hover,.link:hover{border-color:var(--accent)}button:disabled{opacity:.35;cursor:default}.info{margin:16px 0;color:var(--muted)}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0 23px;color:var(--muted);font-size:13px}.legend b{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px}.red{background:#f03737}.blue{background:#237df5}
.card{border:1px solid var(--line);background:var(--panel);border-radius:13px;overflow:hidden;margin:0 0 20px}.cardhead{display:flex;flex-wrap:wrap;gap:13px;align-items:center;padding:13px 16px}.cardhead h2{font-size:18px;margin:0 auto 0 0}.pill{font-size:12px;padding:4px 10px;background:#253448;border-radius:20px}.better{color:#9ce2b6}.worse{color:#ffb0ad}.card img{width:100%;height:auto;display:block;background:white}.caption{padding:10px 16px;font-size:12px;color:var(--muted)}
.pagination{display:flex;gap:12px;align-items:center;justify-content:center;padding:12px 0}a{color:#9fdbc9}.empty{padding:40px;text-align:center;color:var(--muted)}footer{margin-top:30px;border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:12px}
dialog{border:1px solid var(--line);border-radius:14px;background:#101721;color:white;width:96vw;max-width:none;height:94vh;padding:12px}dialog::backdrop{background:#000b}.zoomtop{display:flex;align-items:center;gap:14px;margin-bottom:10px}.zoomtop strong{margin-right:auto}.zoomarea{overflow:auto;height:calc(100% - 60px);background:#e6e6e6}.zoomarea img{display:block;width:2560px;max-width:none;height:auto}#fit:checked~.zoomarea img{width:100%}.notice{border-left:3px solid var(--accent);padding:8px 14px;background:#15241f;margin:14px 0}
@media(max-width:650px){main{padding:20px 12px}.toolbar{position:static}input{width:140px}.cardhead{gap:7px}.stat{min-width:140px}.zoomarea img{width:2560px}}
</style></head><body><main>
<div class="eyebrow">IDD road segmentation • complete validation comparison</div>
<h1>Original model & new model</h1>
<p class="intro">All <strong>687 validation images</strong>, each shown with its unchanged supplied mask, the original model prediction, trial 13 prediction and the new model's disagreement overlay. Click a row to inspect its full 512 × 512 panels.</p>
<div class="stats"><div class="stat"><strong>687 / 687</strong><span>Validation pairs included</span></div><div class="stat"><strong>__OLD_IOU__</strong><span>Original model · global road IoU</span></div><div class="stat"><strong>__NEW_IOU__</strong><span>New model · global road IoU</span></div><div class="stat"><strong>__IMPROVED__ ↑ · __WORSENED__ ↓</strong><span>Per-image IoU changes</span></div></div>
<div class="notice">Colours show disagreement with the supplied label. A low IoU or blue area can also reflect an annotation or class-definition issue; visual inspection is needed. The test set is reserved.</div>
<div class="toolbar">
<label>Find image ID<input id="search" inputmode="numeric" placeholder="e.g. 4160 or Image_4160"></label>
<label>Order<select id="sort"><option value="id">Image ID</option><option value="new-low">New model: lowest IoU first</option><option value="new-high">New model: highest IoU first</option><option value="old-low">Original: lowest IoU first</option><option value="gain">Largest improvement first</option><option value="loss">Largest regression first</option></select></label>
<label>Show<select id="filter"><option value="all">All images</option><option value="better">New model improved</option><option value="worse">New model worsened</option><option value="weak">Either model below 50% IoU</option></select></label>
<label>Rows per page<select id="size"><option>3</option><option selected>6</option><option>12</option><option>24</option></select></label>
<button id="reset">Reset</button><a class="link" href="comparison_metrics.csv" download>All scores CSV</a>
</div>
<div class="legend"><span><b class="red"></b>Labelled road missed by new model</span><span><b class="blue"></b>New model predicts road outside supplied label</span><span>White = road · Black = non-road · threshold 0.5</span></div>
<div class="pagination"><button class="prev">← Previous</button><span class="pageinfo"></span><button class="next">Next →</button><label>Go to page<input id="page" type="number" min="1" value="1" style="width:85px"></label><button id="go">Go</button></div>
<div class="info" id="count" aria-live="polite"></div><section id="cards"></section>
<div class="pagination"><button class="prev">← Previous</button><span class="pageinfo"></span><button class="next">Next →</button></div>
<footer>Original: frozen_01 / best.keras · New: trial_13_shadow_fixed_bn_low_lr / best.keras.<br>Validation only. No augmentation; image normalization applied once. Original dataset masks stay 0/1. PNG rows use 0/255 solely for display. Saved checkpoints and split assignments were preserved.<br><a href="README.md">Read the export notes</a> · <a href="gallery_manifest.json">Export verification</a></footer>
</main>
<dialog id="zoom"><div class="zoomtop"><strong id="zoomtitle"></strong><a id="download" download>Save full PNG</a><button id="close">Close ✕</button></div><input type="checkbox" id="fit" checked><label for="fit" style="display:inline">Fit width (uncheck to inspect original pixels)</label><div class="zoomarea"><img id="zoomimage" alt="Full comparison row"></div></dialog>
<script>
const data=__DATA__;let currentPage=1;const $=id=>document.getElementById(id);let visible=[];
function render(){const query=$('search').value.toLowerCase().replace(/image_|mask_|id\s*/g,'').trim();visible=data.filter(r=>!query||String(r.id)===query);const filter=$('filter').value;visible=visible.filter(r=>filter==='all'||filter==='better'&&r.change_pp>0||filter==='worse'&&r.change_pp<0||filter==='weak'&&Math.min(r.baseline_iou,r.candidate_iou)<.5);
const sort=$('sort').value;visible.sort((a,b)=>sort==='id'?a.id-b.id:sort==='new-low'?a.candidate_iou-b.candidate_iou:sort==='new-high'?b.candidate_iou-a.candidate_iou:sort==='old-low'?a.baseline_iou-b.baseline_iou:sort==='gain'?b.change_pp-a.change_pp:a.change_pp-b.change_pp);
const size=+$('size').value,pages=Math.max(1,Math.ceil(visible.length/size));currentPage=Math.min(Math.max(1,currentPage),pages);$('page').value=currentPage;$('page').max=pages;
document.querySelectorAll('.pageinfo').forEach(el=>el.textContent=`Page ${currentPage} of ${pages}`);document.querySelectorAll('.prev').forEach(el=>el.disabled=currentPage===1);document.querySelectorAll('.next').forEach(el=>el.disabled=currentPage===pages);
const start=(currentPage-1)*size;$('count').textContent=visible.length?`Showing ${start+1}–${Math.min(start+size,visible.length)} of ${visible.length} matching images · 687 total`:'No matching image. Try another ID or reset the filters.';
$('cards').innerHTML=visible.slice(start,start+size).map(r=>`<article class="card"><div class="cardhead"><h2>Image ${r.id}</h2><span class="pill">Original ${(r.baseline_iou*100).toFixed(2)}%</span><span class="pill">New ${(r.candidate_iou*100).toFixed(2)}%</span><span class="pill ${r.change_pp>=0?'better':'worse'}">${r.change_pp>=0?'+':''}${r.change_pp.toFixed(3)} pp</span><a href="${r.image}" download>Save PNG</a></div><a href="${r.image}" data-id="${r.id}"><img src="${r.preview}" loading="lazy" width="1280" height="298" alt="Image ${r.id}: original image, supplied mask, original prediction, new prediction, new disagreement overlay"></a><div class="caption">Original → supplied mask → original prediction → new prediction → new-model disagreement. Click to zoom.</div></article>`).join('');
document.querySelectorAll('[data-id]').forEach(el=>el.addEventListener('click',event=>{event.preventDefault();openZoom(data.find(r=>r.id===+el.dataset.id));}));}
function openZoom(r){$('zoomtitle').textContent=`Image ${r.id} · Original ${(r.baseline_iou*100).toFixed(2)}% · New ${(r.candidate_iou*100).toFixed(2)}%`;$('zoomimage').src=r.image;$('download').href=r.image;$('fit').checked=true;$('zoom').showModal();}
['search','sort','filter','size'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',()=>{currentPage=1;render();}));function jumpToPage(){currentPage=+$('page').value||1;render();}$('go').onclick=jumpToPage;$('page').addEventListener('change',jumpToPage);$('page').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();jumpToPage();}});
document.querySelectorAll('.prev').forEach(el=>el.onclick=()=>{currentPage--;render();});document.querySelectorAll('.next').forEach(el=>el.onclick=()=>{currentPage++;render();});$('reset').onclick=()=>{$('search').value='';$('sort').value='id';$('filter').value='all';currentPage=1;render();};$('close').onclick=()=>$('zoom').close();
render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    export = json.loads((output / 'inference_status.json').read_text())
    assert export['status'] == 'completed' and export['test_images_opened'] is False
    baseline, candidate = load_metrics(output / 'baseline_metrics.csv'), load_metrics(output / 'trial13_metrics.csv')
    assert baseline.keys() == candidate.keys() and len(baseline) == 687
    for directory in ['comparisons', 'previews']:
        (output / directory).mkdir(exist_ok=False)
    status = dict(status='rendering', completed=0, total=687)
    atomic(output / 'gallery_status.json', status)
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(lambda key: render_pair(key, output, baseline, candidate), sorted(baseline)):
            records.append(record)
            if len(records) % 50 == 0 or len(records) == 687:
                status['completed'] = len(records)
                atomic(output / 'gallery_status.json', status)
                print(f'Rendered {len(records)}/687 comparison rows', flush=True)
    for directory, pattern in [('comparisons', 'Comparison_*.png'), ('previews', 'Comparison_*.jpg'),
                               ('predicted_masks/baseline', 'Prediction_*.png'), ('predicted_masks/trial13', 'Prediction_*.png')]:
        paths = list((output / directory).glob(pattern))
        assert len(paths) == 687
        for path in paths:
            with Image.open(path) as im:
                im.verify()
    csv_path = output / 'comparison_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    html = (HTML.replace('__DATA__', json.dumps(records))
            .replace('__OLD_IOU__', f"{export['models']['baseline']['global_road_iou']:.4%}")
            .replace('__NEW_IOU__', f"{export['models']['trial13']['global_road_iou']:.4%}")
            .replace('__IMPROVED__', str(sum(r['change_pp'] > 0 for r in records)))
            .replace('__WORSENED__', str(sum(r['change_pp'] < 0 for r in records))))
    (output / 'index.html').write_text(html, encoding='utf-8')
    (output / 'README.md').write_text('''# All validation images: original versus new model

Open index.html in a browser. The gallery includes all 687 validation pairs, pagination,
exact image-ID search, sorting by IoU or improvement, and full-resolution image zoom.
Each comparisons/Comparison_<id>.png contains five 512x512 panels:
original RGB, unchanged supplied ground truth, original frozen_01 prediction,
new trial13 prediction, and new-model disagreement against the supplied mask.
Red means labelled road missed; blue means road predicted outside the supplied label.
These are disagreements with dataset labels, not proof that the label is correct.

predicted_masks/baseline and predicted_masks/trial13 each contain all 687 inferred
binary masks with original values 0/1. Display panels alone scale 1 to white (255).
comparison_metrics.csv records every ID, both IoUs and percentage-point change.
No training, augmentation, label editing or test-image inference occurred.
The fixed threshold is 0.5. The existing RGB normalization is used once.
inference_status.json records checkpoint hashes, GPU execution and agreement with
the previous verified validation results. Every image and exported PNG was checked.

The adjacent ZIP contains this complete gallery, masks, comparisons and score files.
Extract it before opening index.html so all relative image paths remain available.
''', encoding='utf-8')
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), split='val', images=687,
                    original_model=export['models']['baseline'], new_model=export['models']['trial13'],
                    comparison_rows=687, mask_files_per_model=687, test_images_opened=False,
                    all_output_images_readable=True,
                    images_improved=sum(r['change_pp'] > 0 for r in records),
                    images_worsened=sum(r['change_pp'] < 0 for r in records),
                    supplied_labels_unchanged=True, threshold=0.5)
    atomic(output / 'gallery_manifest.json', manifest)
    status.update(status='packaging', completed=687)
    atomic(output / 'gallery_status.json', status)
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    # PNG/JPEG files are already compressed; store them without costly recompression.
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'gallery_status.json' and not path.name.startswith('local_server'):
                z.write(path, path.relative_to(output.parent))
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        assert bad is None, bad
        assert sum('/comparisons/Comparison_' in name for name in z.namelist()) == 687
    status.update(status='completed', archive=str(archive), archive_bytes=archive.stat().st_size)
    atomic(output / 'gallery_status.json', status)
    print(json.dumps(dict(gallery=str(output / 'index.html'), rows=687, archive=str(archive),
                          archive_megabytes=archive.stat().st_size / 1024**2)), flush=True)


if __name__ == '__main__':
    main()
