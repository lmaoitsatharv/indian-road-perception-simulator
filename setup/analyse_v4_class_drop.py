"""Why did bicycle and traffic_light AP50 drop slightly from the f7bb7e4c checkpoint to the v4 (806a7163) checkpoint?

CPU only. Reads saved evaluation files (analysis.json, per_image_counts.npz written by
setup/analyse_detection_size_brightness.py), training sampling records and label text files. Runs no model, trains nothing,
and writes only to runs/detection_v4_class_drop_20261029/.

Usage:  python setup/analyse_v4_class_drop.py [--bootstrap 2000] [--skip-label-diff]
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/detection_v4_class_drop_20261029"
NAMES = ["person", "rider", "car", "motorcycle", "bicycle", "autorickshaw", "bus", "truck", "animal", "traffic_light",
         "traffic_sign", "other_vehicle"]
SIZES = ("small", "medium", "large")
THRESHOLDS = ("0.10", "0.25", "0.50")
OLD_SHA, NEW_SHA = "f7bb7e4c", "806a7163"
PAIRS = {  # same validation images and settings, old checkpoint vs new checkpoint
    "full_val_1536_nms060": ("runs/detection_setting_probe_20261028_1536_nms060", "runs/detection_v4_eval_20261029_1536_nms060"),
    "full_val_1280_nms070": ("runs/detection_error_review_20261028_01", "runs/detection_v4_eval_20261029_1280_nms070"),
    "land_val_1536_nms060": ("runs/detection_v4_valland_20261029_old", "runs/detection_v4_valland_20261029_new"),
}
OLD_1280_AP = "runs/detection_pilot_recovery_20261025_01/matched_final_1280.json"  # older analysis.json has no per_class_ap
PARENT_1280_AP = "runs/detection_pilot_recovery_20261025_01/baseline_1280.json"  # epoch-30 parent of f7bb7e4c, same 1280 protocol
FOCUS = ["bicycle", "traffic_light", "car", "motorcycle", "rider", "traffic_sign"]
BRIGHT = {"terciles": ([1 / 3, 2 / 3], ("darker_third", "middle_third", "brighter_third")),
          "tails": ([0.10, 0.90], ("darkest_10pct", "middle_80pct", "brightest_10pct"))}


def rate(a, b):
    return None if b == 0 else float(a) / float(b)


def f1(p, r):
    return None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r)


def load_run(folder):
    folder = ROOT / folder
    analysis = json.loads((folder / "analysis.json").read_text())
    z = np.load(folder / "per_image_counts.npz")
    return analysis, {k: z[k] for k in z.files}


def ap_table(name, old_a, new_a):
    old_ap = old_a.get("per_class_ap")
    if old_ap is None and name == "full_val_1280_nms070":
        ref = json.loads((ROOT / OLD_1280_AP).read_text())
        assert ref["checkpoint_sha256"] == old_a["checkpoint_sha256"] and ref["resolution"] == 1280
        old_ap = {k: {"ap50": v["ap50"], "ap50_95": v["ap50_95"]} for k, v in ref["per_class"].items()}
    new_ap = new_a["per_class_ap"]
    return {c: {"old_ap50": old_ap[c]["ap50"], "new_ap50": new_ap[c]["ap50"], "delta_ap50_pts": 100 * (new_ap[c]["ap50"] - old_ap[c]["ap50"]),
                "old_ap50_95": old_ap[c]["ap50_95"], "new_ap50_95": new_ap[c]["ap50_95"],
                "delta_ap50_95_pts": 100 * (new_ap[c]["ap50_95"] - old_ap[c]["ap50_95"])} for c in NAMES}


def counts(z, mask, ti, ci):
    g = z["gt"][mask][:, ci].sum(0).astype(int)
    f = z["found"][mask][:, ti, ci].sum(0).astype(int)
    p = z["pred_tp"][mask][:, ti, ci].sum(0).astype(int)
    q = z["pred_fp"][mask][:, ti, ci].sum(0).astype(int)
    return g, f, p, q


def cell(g, f, p, q):
    g, f, p, q = int(g), int(f), int(p), int(q)
    r, pr = rate(f, g), rate(p, p + q)
    return {"targets": g, "found_TP": f, "missed_FN": g - f, "predictions": p + q, "pred_TP": p, "pred_FP": q,
            "recall": r, "precision": pr, "f1": f1(pr, r)}


def both(old, new):
    out = {"old": old, "new": new}
    for k in ("recall", "precision", "f1"):
        out["delta_" + k + "_pts"] = None if old[k] is None or new[k] is None else 100 * (new[k] - old[k])
    for k in ("found_TP", "missed_FN", "predictions", "pred_FP"):
        out["delta_" + k] = new[k] - old[k]
    return out


def class_breakdown(zo, zn, ci, bins):
    n = len(zo["files"])
    everything = np.ones(n, dtype=bool)
    res = {}
    for ti, thr in enumerate(THRESHOLDS):
        go, fo, po, qo = counts(zo, everything, ti, ci)
        gn, fn, pn, qn = counts(zn, everything, ti, ci)
        assert (go == gn).all()
        entry = {"all": both(cell(go.sum(), fo.sum(), po.sum(), qo.sum()), cell(gn.sum(), fn.sum(), pn.sum(), qn.sum())), "by_size": {},
                 "by_brightness": {}}
        for si, s in enumerate(SIZES):
            entry["by_size"][s] = both(cell(go[si], fo[si], po[si], qo[si]), cell(gn[si], fn[si], pn[si], qn[si]))
        for scheme, (_, labels) in BRIGHT.items():
            entry["by_brightness"][scheme] = {}
            for bi, lab in enumerate(labels):
                m = bins[scheme] == bi
                a, b = counts(zo, m, ti, ci), counts(zn, m, ti, ci)
                entry["by_brightness"][scheme][lab] = both(cell(*(x.sum() for x in a)), cell(*(x.sum() for x in b)))
        res["conf_" + thr] = entry
    return res


def confusion_rows(z, ci):
    """Ground-truth boxes of class ci at conf 0.25: found, or best-overlapping (IoU>=0.5) other-class prediction, or nothing."""
    conf = z["confusion"]  # (size, true class, pred class + nothing + same-class-already-used)
    cols = NAMES + ["nothing_overlapping (background miss)", "same_class_box_already_used"]
    row = {}
    for si, s in enumerate(SIZES + ("all",)):
        vec = conf[:, ci].sum(0) if s == "all" else conf[si, ci]
        row[s] = {cols[j]: int(vec[j]) for j in range(len(cols)) if vec[j]}
    column = {NAMES[t]: int(conf[:, t, ci].sum()) for t in range(len(NAMES)) if t != ci and conf[:, t, ci].sum()}
    return {"row_true_class": row, "column_other_true_classes_whose_best_overlap_was_this_class": column}


def concentration(zo, zn, ci, ti=1):
    d_found = zn["found"][:, ti, ci].sum(1).astype(int) - zo["found"][:, ti, ci].sum(1).astype(int)
    d_fp = zn["pred_fp"][:, ti, ci].sum(1).astype(int) - zo["pred_fp"][:, ti, ci].sum(1).astype(int)
    has = zo["gt"][:, ci].sum(1) > 0
    order = np.argsort(d_found)
    net = int(d_found.sum())
    worst = {k: int(d_found[order[:k]].sum()) for k in (5, 10, 25, 50)}
    return {"threshold": THRESHOLDS[ti], "images_with_class": int(has.sum()),
            "found_change": {"net": net, "images_gained": int((d_found > 0).sum()), "boxes_gained": int(d_found[d_found > 0].sum()),
                             "images_lost": int((d_found < 0).sum()), "boxes_lost": int(-d_found[d_found < 0].sum()),
                             "largest_single_image_loss": int(-d_found.min()), "largest_single_image_gain": int(d_found.max()),
                             "sum_of_k_worst_images": worst,
                             "churn_ratio_(gained+lost)/|net|": None if net == 0 else float(np.abs(d_found).sum() / abs(net)),
                             # sign-test style check on image-level changes (a lower bound on box-level churn): net / sqrt(changed boxes)
                             "z_net_over_sqrt_changed": None if not np.abs(d_found).sum() else float(net / np.sqrt(np.abs(d_found).sum()))},
            "false_positive_change": {"net": int(d_fp.sum()), "images_more_fp": int((d_fp > 0).sum()), "images_fewer_fp": int((d_fp < 0).sum()),
                                      "largest_single_image_increase": int(d_fp.max())},
            "worst_images": [{"file": str(zo["files"][i]), "found_change": int(d_found[i]), "fp_change": int(d_fp[i]),
                              "targets": int(zo["gt"][i, ci].sum())} for i in order[:10] if d_found[i] < 0]}


def bootstrap(zo, zn, reps, seed=0):
    """Paired bootstrap over images: resample image indices, recompute recall/precision/F1 for both models, take differences."""
    n = len(zo["files"])
    feats = []  # per image: gt, found_old, tp_old, fp_old, found_new, tp_new, fp_new  for each threshold and class
    for ti in range(3):
        for ci in range(len(NAMES)):
            feats += [zo["gt"][:, ci].sum(1), zo["found"][:, ti, ci].sum(1), zo["pred_tp"][:, ti, ci].sum(1), zo["pred_fp"][:, ti, ci].sum(1),
                      zn["found"][:, ti, ci].sum(1), zn["pred_tp"][:, ti, ci].sum(1), zn["pred_fp"][:, ti, ci].sum(1)]
    x = np.stack(feats, 1).astype(np.float64)
    rng = np.random.default_rng(seed)
    sums = []
    for start in range(0, reps, 200):
        k = min(200, reps - start)
        w = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(k)]).astype(np.float64)
        sums.append(w @ x)
    s = np.concatenate(sums).reshape(reps, 3, len(NAMES), 7)
    g, fo, po, qo, fn, pn, qn = (s[..., i] for i in range(7))
    with np.errstate(invalid="ignore", divide="ignore"):
        ro, rn = fo / g, fn / g
        pro, prn = po / (po + qo), pn / (pn + qn)
        f1o, f1n = 2 * pro * ro / (pro + ro), 2 * prn * rn / (prn + rn)
    out = {}
    for ti, thr in enumerate(THRESHOLDS):
        out["conf_" + thr] = {}
        for ci, c in enumerate(NAMES):
            e = {}
            for key, d in (("recall", rn - ro), ("precision", prn - pro), ("f1", f1n - f1o)):
                v = 100 * d[:, ti, ci]
                v = v[np.isfinite(v)]
                e["delta_" + key + "_pts"] = {"mean": float(v.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                               "share_of_resamples_below_zero": float((v < 0).mean())}
            out["conf_" + thr][c] = e
    return out


def label_diff():
    def load(d):
        return {p.stem: p.read_text() for p in (ROOT / d).glob("*.txt")}

    def cnt(texts):
        c, imgs = Counter(), Counter()
        for t in texts:
            seen = set()
            for line in t.split("\n"):
                if line.strip():
                    k = int(line.split()[0])
                    c[k] += 1
                    seen.add(k)
            imgs.update(seen)
        return c, imgs

    v3, v4 = load("dataset/idd117k_detection_v3/labels/train"), load("dataset/idd117k_detection_v4_landanimals/labels/train")
    only3, only4 = sorted(set(v3) - set(v4)), sorted(set(v4) - set(v3))
    changed = [s for s in set(v3) & set(v4) if v3[s] != v4[s]]
    (b3, i3), (b4, i4) = cnt(v3.values()), cnt(v4.values())
    rem, _ = cnt(v3[s] for s in only3)
    dats, _ = cnt(v4[s] for s in only4 if s.startswith("dats"))
    rec, _ = cnt(v4[s] for s in only4 if not s.startswith("dats"))
    ch3, _ = cnt(v3[s] for s in changed)
    ch4, _ = cnt(v4[s] for s in changed)
    named = lambda c: {NAMES[k]: int(c[k]) for k in range(len(NAMES)) if c[k]}
    return {"v3_train_images": len(v3), "v4_train_images": len(v4), "images_only_in_v3": len(only3), "images_only_in_v4": len(only4),
            "inherited_images_with_changed_labels": len(changed),
            "boxes_per_class": {NAMES[k]: {"v3": int(b3[k]), "v4": int(b4[k]), "change": int(b4[k] - b3[k])} for k in range(len(NAMES))},
            "images_per_class": {NAMES[k]: {"v3": int(i3[k]), "v4": int(i4[k]), "change": int(i4[k] - i3[k])} for k in range(len(NAMES))},
            "boxes_in_removed_images": named(rem), "boxes_in_added_dats_images": named(dats),
            "boxes_in_added_recovered_idd_images": named(rec),
            "changed_images_box_change": {NAMES[k]: int(ch4[k] - ch3[k]) for k in range(len(NAMES)) if ch4[k] != ch3[k]}}


def sampler_compare():
    old = json.loads((ROOT / "runs/detection_train_20261025_02/training_sampling.json").read_text())
    new = json.loads((ROOT / "runs/detection_train_20261029_01/training_sampling.json").read_text())
    ro = json.loads((ROOT / "runs/detection_train_20261025_02/recipe.json").read_text())
    rn = json.loads((ROOT / "runs/detection_train_20261029_01/recipe.json").read_text())
    keys = ("imgsz", "batch", "epochs", "optimizer", "lr0", "lrf", "cos_lr", "warmup_epochs", "mosaic", "scale", "close_mosaic", "amp", "seed")
    exp = {}
    for c in NAMES:
        a, b = old["expected_class_exposure"][c], new["expected_class_exposure"][c]
        exp[c] = {"old_images": a["original_images"], "new_images": b["original_images"],
                  "old_expected_draws_per_epoch": round(a["expected_primary_samples_per_epoch"], 1),
                  "new_expected_draws_per_epoch": round(b["expected_primary_samples_per_epoch"], 1),
                  "draw_share_change_pct": round(100 * ((b["expected_primary_samples_per_epoch"] / new["draws_per_epoch"]) /
                                                        (a["expected_primary_samples_per_epoch"] / old["draws_per_epoch"]) - 1), 2)}
    return {"sampler_config_identical": old["configuration"] == new["configuration"], "sampler_config": new["configuration"],
            "draws_per_epoch": {"old": old["draws_per_epoch"], "new": new["draws_per_epoch"]},
            "recipe_differences": {k: {"old": ro.get(k), "new": rn.get(k)} for k in keys if ro.get(k) != rn.get(k)},
            "recipe_same": {k: rn.get(k) for k in keys if ro.get(k) == rn.get(k)},
            "old_run_note": "old run = 3 epochs from the epoch-30 parent (lr 3e-5 cosine); new run = 1 further epoch from its best (epoch 3)",
            "expected_class_exposure": exp}


def ap_history():
    """AP50 at the same 1280/NMS0.7 protocol: epoch-30 parent -> f7bb7e4c (3 epochs) -> 806a7163 (+1 epoch on v4)."""
    p = json.loads((ROOT / PARENT_1280_AP).read_text())
    o = json.loads((ROOT / OLD_1280_AP).read_text())
    n = json.loads((ROOT / "runs/detection_train_20261029_01/final_validation.json").read_text())
    return {"checkpoints": {"parent": p["checkpoint_sha256"][:8], "old": o["checkpoint_sha256"][:8], "new": NEW_SHA},
            "ap50": {c: {"parent": p["per_class"][c]["ap50"], "old": o["per_class"][c]["ap50"], "new": n["per_class"][c]["ap50"]} for c in NAMES},
            "framework_P_R_at_max_F1": {c: {k: {"precision": d["per_class"][c]["precision"], "recall": d["per_class"][c]["recall"]}
                                             for k, d in (("old", o), ("new", n))} for c in NAMES}}


def main(reps, skip_label_diff):
    OUT.mkdir(exist_ok=True)
    report = {"schema": "idd_v4_class_drop_review_v1", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "validation only; saved evaluation files; no inference, no training", "old_checkpoint": OLD_SHA, "new_checkpoint": NEW_SHA,
              "bootstrap_reps": reps,
              "notes": ["recall/precision are at fixed confidence thresholds with IoU>=0.5 one-to-one matching, as in "
                        "setup/analyse_detection_size_brightness.py; they are not AP",
                        "AP itself cannot be bootstrapped: prediction scores were not saved per box",
                        "bootstrap resamples images independently; frames from one recording are correlated, so true intervals are "
                        "likely somewhat wider"],
              "pairs": {}}
    for name, (old_dir, new_dir) in PAIRS.items():
        oa, zo = load_run(old_dir)
        na, zn = load_run(new_dir)
        assert oa["checkpoint_sha256"].startswith(OLD_SHA) and na["checkpoint_sha256"].startswith(NEW_SHA)
        assert oa["imgsz"] == na["imgsz"] and oa.get("nms_iou", 0.7) == na.get("nms_iou", 0.7)
        assert (zo["files"] == zn["files"]).all() and (zo["gt"] == zn["gt"]).all()
        luma_gap = float(np.abs(zo["luma"] - zn["luma"]).max())
        bins = {s: np.digitize(zo["luma"], np.quantile(zo["luma"], q)) for s, (q, _) in BRIGHT.items()}
        pair = {"old_run": old_dir, "new_run": new_dir, "imgsz": oa["imgsz"], "nms_iou": oa.get("nms_iou", 0.7), "images": len(zo["files"]),
                "max_luma_difference_between_runs": luma_gap, "ap": ap_table(name, oa, na), "classes": {}}
        for c in FOCUS:
            ci = NAMES.index(c)
            pair["classes"][c] = {"metrics": class_breakdown(zo, zn, ci, bins),
                                  "confusion_conf_0_25": {"old": confusion_rows(zo, ci), "new": confusion_rows(zn, ci)},
                                  **{"concentration_conf_" + t: concentration(zo, zn, ci, ti) for ti, t in enumerate(THRESHOLDS)}}
        pair["all_class_overview"] = {c: {"ap50_delta_pts": pair["ap"][c]["delta_ap50_pts"],
                                          **{"conf_" + t: {k: v for k, v in class_breakdown(zo, zn, NAMES.index(c), bins)["conf_" + t]["all"].items()
                                                           if k.startswith("delta")} for t in THRESHOLDS}} for c in NAMES}
        pair["bootstrap"] = bootstrap(zo, zn, reps)
        report["pairs"][name] = pair
        print("done", name)
    report["sampler_and_recipe"] = sampler_compare()
    report["ap50_history_1280"] = ap_history()
    (OUT / "class_drop_analysis.json").write_text(json.dumps(report, indent=1))
    if not skip_label_diff:
        (OUT / "train_label_diff_v3_v4.json").write_text(json.dumps(label_diff(), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--skip-label-diff", action="store_true")
    a = ap.parse_args()
    main(a.bootstrap, a.skip_label_diff)
