"""Independent NumPy validation metrics; no TensorFlow or data mutation."""
import numpy as np


def morph(mask, radius, dilation=False, border=False):
    """Separable square boolean morphology, equivalent to 3x3 iterations."""
    padded = np.pad(np.asarray(mask, bool), radius, constant_values=border)
    horizontal = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=1)
    reduced = horizontal.max(axis=-1) if dilation else horizontal.min(axis=-1)
    vertical = np.lib.stride_tricks.sliding_window_view(reduced, 2 * radius + 1, axis=0)
    return vertical.max(axis=-1) if dilation else vertical.min(axis=-1)


def boundary(mask, radius=5, include_frame=True):
    return mask & ~morph(mask, radius, border=not include_frame)


def counts(truth, pred):
    t, p = np.asarray(truth, bool), np.asarray(pred, bool)
    out = {"tp": int((t & p).sum()), "fp": int((~t & p).sum()), "fn": int((t & ~p).sum())}
    for d in (5, 14):
        tb, pb = boundary(t, d), boundary(p, d)
        out[f"boundary_intersection_{d}px"] = int((tb & pb).sum())
        out[f"boundary_union_{d}px"] = int((tb | pb).sum())
    # Approximate many-to-one contour matching; 3px Chebyshev tolerance.
    # Frame is not a physical observed contour and is excluded in this metric.
    tc, pc = boundary(t, 1, False), boundary(p, 1, False)
    tc[[0, -1], :] = False
    tc[:, [0, -1]] = False
    pc[[0, -1], :] = False
    pc[:, [0, -1]] = False
    out.update(contour_pred_pixels=int(pc.sum()), contour_true_pixels=int(tc.sum()),
               contour_precision_matches=int((pc & morph(tc, 3, dilation=True)).sum()),
               contour_recall_matches=int((tc & morph(pc, 3, dilation=True)).sum()))
    return out


def scores(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    ratio = lambda n, d: n / d if d else 1.
    precision = ratio(c["contour_precision_matches"], c["contour_pred_pixels"])
    recall = ratio(c["contour_recall_matches"], c["contour_true_pixels"])
    # One empty contour with one nonempty contour must score zero.
    if not c["contour_pred_pixels"] and c["contour_true_pixels"]:
        precision = 0.
    if not c["contour_true_pixels"] and c["contour_pred_pixels"]:
        recall = 0.
    return {"road_iou": ratio(tp, tp + fp + fn), "road_dice": ratio(2 * tp, 2 * tp + fp + fn),
            "road_precision": ratio(tp, tp + fp), "road_recall": ratio(tp, tp + fn),
            **{f"boundary_iou_{d}px": ratio(c[f"boundary_intersection_{d}px"], c[f"boundary_union_{d}px"]) for d in (5, 14)},
            "visible_boundary_precision_3px": precision, "visible_boundary_recall_3px": recall,
            "visible_boundary_f1_3px": 2 * precision * recall / (precision + recall) if precision + recall else 0.}


def summarize(rows):
    keys = counts(np.zeros((2, 2), bool), np.zeros((2, 2), bool)).keys()
    total = {key: sum(row[key] for row in rows) for key in keys}
    return {"global": scores(total), "counts": total,
            "mean_per_image": {key: float(np.mean([row[key] for row in rows])) for key in scores(total)},
            "road_iou_below_50pct": sum(row["road_iou"] < .5 for row in rows),
            "road_iou_below_75pct": sum(row["road_iou"] < .75 for row in rows), "images": len(rows)}
