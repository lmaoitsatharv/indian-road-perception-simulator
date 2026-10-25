"""Audit three already-known validation cases without inference or label edits."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/validation_label_audit_20261003'
IDS=[48,606,6514]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    with (ROOT/'dataset/splits/pair_splits.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows={int(row['id']):row for row in csv.DictReader(stream)}
    fig,axes=plt.subplots(3,4,figsize=(15,11),layout='constrained')
    report=[]
    for line,i in enumerate(IDS):
        row=rows[i];assert row['split']=='val'
        image_path=ROOT/f'dataset/splits/val/images/Image_{i}.png'
        mask_path=ROOT/f'dataset/splits/val/masks/Mask_{i}.png'
        assert sha(image_path)==sha(row['source_image'])==row['image_sha256']
        assert sha(mask_path)==sha(row['source_mask'])==row['mask_sha256']
        image=np.array(Image.open(image_path));truth=np.array(Image.open(mask_path))
        assert set(np.unique(truth))=={0,1}
        prediction_path=ROOT/f'runs/model_comparisons/baseline_vs_trial13_validation/predicted_masks/trial13/Prediction_{i}.png'
        pred=np.array(Image.open(prediction_path));assert set(np.unique(pred)).issubset({0,1})
        truth=truth==1;pred=pred==1
        tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum())
        iou=tp/(tp+fp+fn)
        overlay=image.copy();overlay[truth&~pred]=[235,65,80];overlay[pred&~truth]=[45,130,235]
        images=[image,truth.astype(np.uint8)*255,pred.astype(np.uint8)*255,overlay]
        titles=[f'Image {i}: original',f'Supplied mask: {truth.sum():,} positive pixels',f'Locked morning model: IoU {100*iou:.2f}%', 'Red: missed label / blue: extra prediction']
        for ax,values,title in zip(axes[line],images,titles):
            ax.imshow(values,cmap='gray',vmin=0,vmax=255);ax.set_title(title,fontsize=10);ax.axis('off')
        report.append({'id':i,'split':'val','supplied_positive_pixels':int(truth.sum()),'supplied_positive_fraction':float(truth.mean()),
            'saved_batch16_road_iou':iou,'prediction_positive_pixels':int(pred.sum()),'tp':tp,'fp':fp,'fn':fn,
            'source_and_split_image_hash_match':True,'source_and_split_mask_hash_match':True,
            'mask_sha256':row['mask_sha256'],'label_changed':False,'ignored_or_excluded_from_metrics':False})
    fig.suptitle('Known validation label questions · saved predictions only\nBlack/white scaling is for display; original labels remain 0/1. Visual disagreement alone does not prove annotation error.',fontsize=12)
    fig.savefig(OUT/'label_questions.png',dpi=140);plt.close(fig)
    audit={'status':'completed','ids':IDS,'rows':report,'new_model_inference':False,'test_contents_read':False,
           'labels_edited':False,'metric_exclusions_changed':False,'original_to_binary_class_conversion':'unconfirmed',
           'potential_dataset_source':'https://www.kaggle.com/datasets/mitanshuchakrawarty/new-idd-dataset/data',
           'matching_layout_description':'https://github.com/Dalageo/road-segmentation-idd',
           'source_identity_confirmed_by_remote_hash':False,
           'original_idd_definition_reference':'https://idd.insaan.iiit.ac.in/media/publications/idd-650.pdf'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    text='''# Supplied label audit — validation only

## Confirmed locally

The three supplied masks and images match the preserved original files and manifest hashes exactly. Binary values are still 0/1. This audit reads existing morning-model batch-16 predictions; it does not call a model or read test image contents. No label was changed, excluded or ignored.

| Validation ID | Supplied road pixels | Fraction of image | Saved morning IoU |
|---|---:|---:|---:|
'''
    for row in report:
        text+=f"| {row['id']} | {row['supplied_positive_pixels']:,} | {100*row['supplied_positive_fraction']:.4f}% | {100*row['saved_batch16_road_iou']:.2f}% |\n"
    text+='''
## Interpretation and unresolved questions

- Image 48 remains a difficult segmentation example. Review actual road edges, the model's extra regions, supplied class definitions and image/mask pairing before assigning a cause.
- Image 606's mask has just 103 positive pixels. The rough visible surface may belong to a different original class; an incorrect conversion or annotation is another possibility. The binary files alone cannot distinguish these explanations.
- Image 6514 contains a visible paved foreground where the model and supplied target disagree strongly. Being visually plausible does not prove the prediction matches the intended target definition.

The original IDD taxonomy distinguishes road, parking and drivable fallback. Its paper discusses ambiguity on dirt/mud and unmarked road boundaries. That makes class-definition review relevant, but it does not identify which original classes were retained in these binary masks. [Original IDD paper](https://idd.insaan.iiit.ac.in/media/publications/idd-650.pdf).

A public repository describes exactly 6,993 RGB `Image_{num}.png` files and binary `Mask_{num}.png` files in `image_archive`/`mask_archive`, at 512×512, matching the local layout. It links a Kaggle dataset. This is a plausible provenance lead, not a remote-hash confirmation of the local archive or its conversion. [Author's dataset description](https://github.com/Dalageo/road-segmentation-idd), [linked dataset](https://www.kaggle.com/datasets/mitanshuchakrawarty/new-idd-dataset/data).

## Next evidence to obtain

1. Original download URL/version and the binary-mask conversion script.
2. The exact positive-class mapping, including parking/drivable-fallback classes.
3. Ignore/void-label handling and original image-to-sequence/filename mapping.
4. A small reviewed record of the original multiclass annotations for these IDs.

Keep the current locked model and supplied-mask score unchanged. Any reviewed alternative labels should live in a separate versioned audit dataset, with documented rationale and independent evaluation; do not silently repair labels to match predictions.

## Method summary

“Our score measures agreement with the supplied binary road labels. Some visible surfaces have very small or unusual target regions. We verified that the files were paired and preserved, and documented the cases. We still need the original class-conversion procedure before calling those labels incorrect.”
'''
    (OUT/'LABEL_AUDIT.md').write_text(text,encoding='utf-8')
    print(json.dumps({'status':'completed','ids':IDS,'positive_pixels':[r['supplied_positive_pixels'] for r in report],'new_inference':False}))


if __name__=='__main__':
    main()
