"""Synthetic label/scoring checks; does not create human labels for real clips."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from idd_inference import file_sha256, write_json_atomic
from road_video_labels import RUBRIC, decode_rle, score_pack


def rle(a):
    flat = a.ravel()
    cuts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1])+1, flat.size]
    return [x for i, j in zip(cuts[:-1], cuts[1:]) for x in (int(flat[i]), int(j-i))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pack = out / "synthetic_pack"
    pack.mkdir()
    image = np.zeros((512, 512, 3), np.uint8)
    prediction = np.zeros((512, 512), np.uint8)
    prediction[:, :256] = 1
    labels = np.zeros((512, 512), np.uint8)
    labels[:, :128] = 1
    labels[:, :32] = 255
    Image.fromarray(image).save(pack / "rgb.png")
    Image.fromarray(prediction).save(pack / "prediction.png")
    frames = [{"id": f"synthetic_{i}", "image": "rgb.png", "prediction": "prediction.png",
               "image_sha256": file_sha256(pack / "rgb.png"),
               "prediction_sha256": file_sha256(pack / "prediction.png"),
               "group": "time_selected_scoring" if i < 12 else "diagnostic_only"} for i in range(18)]
    manifest = {"schema": "road_label_pack_v1", "pack_id": pack.name, "rubric_id": RUBRIC,
                "image_shape": [512, 512], "minimum_reviewed_coverage": .8,
                "scoring_frame_count": 12, "diagnostic_frame_count": 6,
                "evaluation_scope": "SYNTHETIC TEST ONLY. Not human labels or video accuracy.", "frames": frames}
    write_json_atomic(pack / "manifest.json", manifest)
    data = {"schema": "road_annotations_v1", "pack_id": pack.name, "rubric_id": RUBRIC,
            "pack_manifest_sha256": file_sha256(pack / "manifest.json"),
            "reviewer": "SYNTHETIC CONTRACT FIXTURE; NOT HUMAN LABELS",
            "frames": [{"id": f["id"], "image_sha256": f["image_sha256"], "reviewed": True,
                        "mask_rle": rle(labels if i < 12 else prediction)} for i, f in enumerate(frames)]}
    json_path = out / "synthetic_annotations.json"
    write_json_atomic(json_path, data)
    result = score_pack(pack, json_path, out / "synthetic_scores")
    m = result["primary_score"]["metrics"]
    # Hand-counted columns: TP96, FP128, FN0, TN256; 32 ignored, each x512.
    assert abs(m["road_iou"] - 3/7) < 1e-12
    assert abs(m["road_dice"] - 3/5) < 1e-12
    assert abs(m["pixel_accuracy"] - 11/15) < 1e-12
    assert m["label_coverage"] == 480/512
    assert result["diagnostic_only_score"]["metrics"]["road_iou"] == 1
    assert result["whole_video_accuracy_established"] is False
    partial = dict(data, frames=data["frames"][:11])
    write_json_atomic(json_path, partial)
    assert score_pack(pack, json_path, out / "synthetic_partial")["primary_score"] is None
    invalids = [[0, 1], [1, 262145], [255, 262143, 1, 2], [True, 262144], [2, 262144], [0, 262144.0]]
    for value in invalids:
        try:
            decode_rle(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid label RLE was accepted: {value}")
    bad = json.loads(json.dumps(data))
    bad["frames"][0]["mask_rle"] = [255, 262144]
    write_json_atomic(json_path, bad)
    try:
        score_pack(pack, json_path, out / "must_not_exist_low_coverage")
    except ValueError as error:
        assert "80%" in str(error)
    else:
        raise AssertionError("All-ignore reviewed frame was accepted")
    bad = json.loads(json.dumps(data))
    bad["frames"][0]["image_sha256"] = "0"*64
    write_json_atomic(json_path, bad)
    try:
        score_pack(pack, json_path, out / "must_not_exist_wrong_image")
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong-image annotation was accepted")
    write_json_atomic(json_path, data)
    # Check the actual editor's JS syntax and cross-language mask encoding.
    template = (Path(__file__).resolve().parents[1] / "setup/road_label_editor_template.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", template, flags=re.S)
    editor_js = out / "editor_syntax.js"
    editor_js.write_text(scripts[-1], encoding="utf-8")
    subprocess.run(["node", "--check", str(editor_js)], check=True, capture_output=True, text=True)
    functions = "\n".join(line for line in scripts[-1].splitlines() if line.startswith(("function encodeRle(", "function decodeRle(")))
    unit = out / "editor_rle_contract.js"
    unit.write_text("const size=512*512;\n"+functions+"\nconst m=new Uint8Array(size);m.fill(255,0,32*512);m.fill(1,32*512,128*512);const runs=encodeRle(m);const n=decodeRle(runs);if(!m.every((x,i)=>x===n[i]))throw Error('RLE mismatch');process.stdout.write(JSON.stringify(runs));", encoding="utf-8")
    js_runs = json.loads(subprocess.run(["node", str(unit)], check=True, capture_output=True, text=True).stdout)
    decoded = decode_rle(js_runs)
    assert np.all(decoded.ravel()[:32*512] == 255) and np.all(decoded.ravel()[32*512:128*512] == 1)
    evidence = {"state": "passed", "synthetic_only": True, "real_human_annotations_created": False,
                "actual_editor_controls_tested_in_browser": False,
                "checks": ["hand-counted IoU/Dice/accuracy", "ignored pixels and coverage", "diagnostic separation",
                           "partial labels do not publish primary score", "invalid RLE rejection", "wrong image rejection",
                           "reviewed coverage gate", "actual editor JavaScript syntax", "JavaScript/Python RLE roundtrip"]}
    write_json_atomic(out / "verification.json", evidence)
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
