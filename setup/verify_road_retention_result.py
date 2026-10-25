"""Recount rehearsal masks independently, including the strict boundary guard."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local(value):
    return Path("E:/" + value[7:]) if value.startswith("/mnt/e/") else Path(value)


def channel(path):
    return pygame.surfarray.array3d(pygame.image.load(str(path)))[:, :, 0].T


def erode(mask, radius, border):
    # Independent summed-area implementation, with explicit image-border policy.
    padded = np.pad(mask.astype(np.int32), radius, constant_values=int(border))
    integral = np.pad(padded.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1, 0), (1, 0)))
    size = 2 * radius + 1
    count = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return count == size * size


def recount(truth, prediction):
    assert truth.shape == prediction.shape == (512, 512)
    assert set(np.unique(truth)) <= {0, 1, 255} and set(np.unique(prediction)) <= {0, 1}
    valid, t, p = truth != 255, truth == 1, prediction == 1
    tb, pb = t & ~erode(t, 5, False), p & ~erode(p, 5, False)
    safe = erode(valid, 5, True)
    return {"tp": int((valid & t & p).sum()), "fp": int((valid & ~t & p).sum()),
            "fn": int((valid & t & ~p).sum()), "tn": int((valid & ~t & ~p).sum()),
            "ignored": int((~valid).sum()), "total": int(valid.size),
            "boundary_intersection_5": int((tb & pb & safe).sum()),
            "boundary_union_5": int(((tb | pb) & safe).sum())}


def retention(a, b):
    ad, bd = a["tp"] + a["fp"] + a["fn"], b["tp"] + b["fp"] + b["fn"]
    road = ad > 0 and bd > 0 and a["tp"] * bd >= b["tp"] * ad
    edge = (a["boundary_union_5"] > 0 and b["boundary_union_5"] > 0
            and a["boundary_intersection_5"] * b["boundary_union_5"] >= b["boundary_intersection_5"] * a["boundary_union_5"])
    return {"road_iou_not_lower": road, "boundary5_iou_not_lower": edge, "passed": road and edge}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    state, manifest, history = (load(output / name) for name in ("run_status.json", "manifest.json", "history.json"))
    assert state["phase"] == "completed" and state["protected_files_unchanged"]
    for name, expected in manifest["protected_files"].items():
        assert sha(ROOT / name) == expected
    # Border behavior matters to the selection guard: all-road touches the image frame.
    assert erode(np.ones((15, 15), bool), 5, True).all()
    expected = np.zeros((15, 15), bool); expected[5:10, 5:10] = True
    np.testing.assert_array_equal(erode(np.ones((15, 15), bool), 5, False), expected)
    sets = {"real_validation": manifest["real_validation"], "development": manifest["development"],
            **{"video_" + group: rows for group, rows in manifest["real_video_guards"].items()}}
    checked = {}
    for prefix in ["baseline", *(f"epoch_{row['epoch']:02d}" for row in history)]:
        for dataset, records in sets.items():
            label = prefix + "_" + dataset
            folder = output / "evaluations" / label
            report = load(folder / "metrics.json")
            by_name = {row["name"]: row for row in report["rows"]}
            assert len(by_name) == len(records) == report["images"]
            totals = None
            for row in records:
                truth_file, image_file = local(row["mask"]), local(row["image"])
                assert sha(truth_file) == row["mask_sha256"] and sha(image_file) == row["image_sha256"]
                prediction_file = folder / "masks_01" / (row["name"] + ".png")
                assert sha(prediction_file) == by_name[row["name"]]["mask_sha256"]
                actual = recount(channel(truth_file), channel(prediction_file))
                for key, value in actual.items():
                    assert by_name[row["name"]]["metrics"][key] == value, (label, row["name"], key)
                totals = actual if totals is None else {key: totals[key] + actual[key] for key in actual}
            for key, value in totals.items():
                assert report["metrics"][key] == value, (label, key)
            checked[label] = {"images": len(records), "counts": totals}
            print(json.dumps({"verified": label, "images": len(records)}), flush=True)
    decisions = {}
    for row in history:
        prefix = f"epoch_{row['epoch']:02d}"
        decisions[prefix] = {dataset: retention(checked[prefix + "_" + dataset]["counts"], checked["baseline_" + dataset]["counts"])
                             for dataset in sets if dataset != "development"}
        assert decisions[prefix] == row["strict_gates"]
        assert all(item["passed"] for item in decisions[prefix].values()) == row["real_retention_passed"]
    result = {"passed": True, "independently_recounted": checked, "strict_decisions": decisions,
              "method": "Direct PNG confusion plus independent integral-image erosion for 5px boundary IoU; exact integer gate comparisons",
              "protected_files_unchanged": True, "reserved_test_images_read": False}
    (output / "independent_recount.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "masks": sum(item["images"] for item in checked.values())}), flush=True)


if __name__ == "__main__":
    main()
