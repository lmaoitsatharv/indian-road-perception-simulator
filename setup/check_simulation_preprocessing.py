"""Check actual simulation train/val batches without constructing a model or reading test pixels."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation_data import read_pairs,make_dataset,decode_pair
from idd_experiments import light_augment
from prepare_simulation_dataset import DEFAULT, save, sha, protected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    pairs = read_pairs(output/"splits")
    assert {k:len(v) for k,v in pairs.items()} == {"train":4000,"val":500,"test":500}
    summaries = {}
    tf.keras.utils.set_random_seed(42)
    for split in ("train","val"):
        records = pairs[split]
        dataset = make_dataset(records,training=split=="train",batch_size=4)
        seen,minimum,maximum = 0,1.,-1.
        valid_pixels = 0
        for images,masks,valid in dataset:
            x,y,w = images.numpy(),masks.numpy(),valid.numpy()
            assert x.dtype==y.dtype==w.dtype==np.float32
            assert x.shape[1:] == (512,512,3) and y.shape[1:]==w.shape[1:]==(512,512,1)
            assert np.isfinite(x).all() and x.min()>=-1 and x.max()<=1
            assert set(np.unique(y)) <= {0.,1.} and set(np.unique(w)) <= {0.,1.}
            assert not (y[w==0] != 0).any()
            seen += len(x)
            valid_pixels += int(w.sum())
            minimum,maximum = min(minimum,float(x.min())),max(maximum,float(x.max()))
        assert seen == len(records)
        assert int(dataset.cardinality()) == math.ceil(seen/4)
        summaries[split] = {"pairs_checked":seen,"batch_count":math.ceil(seen/4),"image_min":minimum,
                            "image_max":maximum,"valid_pixels":valid_pixels,"mask_labels":[0,1]}
    # Exact validation normalization and repeatability, independent of batch iterator shuffling.
    r = pairs["val"][0]
    raw,mask,valid = decode_pair(r["image"],r["mask"],r["valid"])
    x,y,w = next(iter(make_dataset([r])))
    assert np.array_equal(x.numpy()[0],raw.numpy()/127.5-1.)
    assert np.array_equal(y.numpy()[0],mask.numpy()) and np.array_equal(w.numpy()[0],valid.numpy())
    # An asymmetric fixture verifies that the existing paired flip moves validity
    # with the target, including ignored pixels, and shadow changes RGB only.
    image = tf.reshape(tf.cast(tf.range(512*512*3)%256,tf.float32),[512,512,3])
    target = tf.concat([tf.cast(tf.range(512)[None,:,None]>200,tf.float32)+tf.zeros([512,512,1]),
                        tf.cast(tf.range(512)[None,:,None]>80,tf.float32)+tf.zeros([512,512,1])],axis=-1)
    saw_flip=False
    for i in range(8):
        changed,paired = light_augment(image,target,tf.constant([42,i]),"shadow")
        same=np.array_equal(paired.numpy(),target.numpy())
        flip=np.array_equal(paired.numpy(),tf.image.flip_left_right(target).numpy())
        assert same or flip
        assert np.isfinite(changed).all() and float(tf.reduce_min(changed))>=0 and float(tf.reduce_max(changed))<=255
        saw_flip |= flip
    assert saw_flip
    protocol=json.loads((output/"plan.json").read_text())
    assert protected()==protocol["protected_sha256"]
    report={"passed":True,"splits":summaries,"test_pixels_read_by_this_check":False,"training_started":False,
            "model_constructed":False,"normalization_exactly_once":True,"paired_target_and_validity_flip":True,
            "existing_shadow_augmentation_reused":True,"protected_unchanged":True,
            "source_sha256":{name:sha(ROOT/name) for name in ("simulation_data.py","idd_data.py","idd_experiments.py")}}
    save(output/"preprocessing_check.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
