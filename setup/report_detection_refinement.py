"""Verify a finished refinement and write its matched validation comparison."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import file_sha, save_json
from setup.verify_detection_runtime import protection


def main(name):
    if Path(name).name != name or not name.startswith("detection_train_"):
        raise ValueError("Expected a local detector run name")
    run = ROOT / "runs" / name
    owner = run.with_name(name + "_launcher")
    if (run / "refinement_result.json").exists():
        raise FileExistsError("Preserve the previous completion report")
    recipe = json.loads((run / "recipe.json").read_text())
    assert recipe["initialization"] == "verified_idd_detector"
    before = json.loads((owner / "protected_before.json").read_text())
    assert protection() == before
    provenance = json.loads((ROOT / recipe["initial_weights_metadata"]).read_text())
    assert file_sha(ROOT / recipe["initial_weights"]) == provenance["sha256"]
    analysed_path = ROOT / recipe["baseline_analysis"]
    assert file_sha(analysed_path) == recipe["baseline_analysis_sha256"]
    baseline = json.loads((analysed_path.parent / ("diagnostics_" + str(recipe["imgsz"]) + ".json")).read_text())
    with (run / "results.csv").open(newline="") as stream:
        rows = [{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    assert [row["epoch"] for row in rows] == list(range(1,len(rows)+1)) and len(rows) <= recipe["epochs"]
    best_epoch = int(max(rows,key=lambda row:row["metrics/mAP50-95(B)"])["epoch"])
    exit_record = json.loads((owner / "trainer_exit.json").read_text())
    if (run / "final_validation.json").exists():
        final = json.loads((run / "final_validation.json").read_text())
        precision_note = "full-precision framework final-validation metrics"
    else:
        # SIGKILL may bypass the final callback: accept only a complete final table.
        log = owner / "training.log"
        with log.open("rb") as stream:
            stream.seek(max(0,log.stat().st_size-80000))
            tail = stream.read().decode("utf-8",errors="replace").replace("\r","\n")
        assert "2210/2210" in tail and "Results saved to" in tail
        final = {"metrics":json.loads((run / "progress.json").read_text())["metrics"],"per_class":{}}
        assert "val/box_loss" not in final["metrics"], "A final saved-checkpoint validation is required"
        pattern=re.compile(r"^\s*(\w+)\s+(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$")
        for line in tail.splitlines():
            match=pattern.fullmatch(line)
            if match and match[1] in baseline["per_class"]:
                final["per_class"][match[1]]=dict(zip(("precision","recall","ap50","ap50_95"),map(float,match.groups()[3:])))
        precision_note="full-precision aggregate; per-class values from rounded final log after interrupted callback"
    assert set(final["per_class"]) == set(baseline["per_class"])
    output = run / "completion_check"
    output.mkdir(exist_ok=False)
    os.environ.update(YOLO_CONFIG_DIR=str(output/"settings"),MPLCONFIGDIR=str(output/"matplotlib"),YOLO_AUTOINSTALL="False")
    for folder in ("settings","matplotlib"):
        (output/folder).mkdir()
    import torch
    checkpoint_checks={}
    for label in ("best","last"):
        path=run/"weights"/(label+".pt")
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        assert all(bool(torch.isfinite(tensor).all()) for tensor in checkpoint["model"].state_dict().values())
        assert [checkpoint["model"].names[i] for i in range(12)] == list(final["per_class"])
        checkpoint_checks[label]={"path":path.relative_to(ROOT).as_posix(),"sha256":file_sha(path),
            "all_tensors_finite":True,"bytes":path.stat().st_size}
        del checkpoint
    deltas={key:100*(value-baseline["metrics"][key]) for key,value in final["metrics"].items() if key in baseline["metrics"]}
    class_deltas={key:{"ap50_delta_percentage_points":100*(value["ap50"]-baseline["per_class"][key]["ap50"]),
        "ap50_95_delta_percentage_points":100*(value["ap50_95"]-baseline["per_class"][key]["ap50_95"])}
        for key,value in final["per_class"].items()}
    result={"schema":"idd_detector_refinement_result_v1","checked_utc":datetime.now(timezone.utc).isoformat(),
        "report_evidence_verified":True,"run":run.relative_to(ROOT).as_posix(),"epochs_completed":len(rows),
        "best_epoch":best_epoch,"trainer_exit":exit_record,"process_clean_exit":exit_record["returncode"]==0,
        "checkpoint_checks":checkpoint_checks,"validation_images":baseline["images"],"resolution":recipe["imgsz"],
        "metrics":final["metrics"],"per_class":final["per_class"],"metric_precision_note":precision_note,
        "baseline_at_same_resolution":baseline["metrics"],"metric_delta_percentage_points":deltas,
        "class_delta_percentage_points":class_deltas,
        "all_classes_ap50_non_decreasing":all(value["ap50_delta_percentage_points"]>=0 for value in class_deltas.values()),
        "overall_map50_improved":deltas["metrics/mAP50(B)"]>0,
        "overall_map50_95_improved":deltas["metrics/mAP50-95(B)"]>0,
        "person_ap50_improved":class_deltas["person"]["ap50_delta_percentage_points"]>0,
        "targets":{"map50_70_percent_met":final["metrics"]["metrics/mAP50(B)"]>=0.70,
                   "map50_95_45_percent_met":final["metrics"]["metrics/mAP50-95(B)"]>=0.45},
        "protected_files_unchanged":protection()==before,"initial_detector_unchanged":True,
        "official_test_used":False,"control_allowed":False,"automatically_deployed":False}
    assert result["protected_files_unchanged"]
    save_json(run/"refinement_result.json",result)
    save_json(ROOT/"docs/results/detection_refinement.json",result)
    table=["| Class | AP50 before | AP50 after | Change, percentage points |",
           "| --- | ---: | ---: | ---: |"]
    for label,value in final["per_class"].items():
        table.append(f"| {label} | {100*baseline['per_class'][label]['ap50']:.2f}% | {100*value['ap50']:.2f}% | {class_deltas[label]['ap50_delta_percentage_points']:+.2f} |")
    text=(f"# Detector refinement result — {name}\n\n"
        f"Completed {len(rows)} epochs; selected epoch {best_epoch}. "
        f"Training process exit code: {exit_record['returncode']}. Both saved checkpoints reopen with finite tensors.\n\n"
        f"Same {baseline['images']:,}-image project validation set at {recipe['imgsz']} input. "
        "Original detector weights and all 17 protected road/inference/driver fingerprints are unchanged.\n\n"
        f"mAP50: **{100*final['metrics']['metrics/mAP50(B)']:.2f}%** "
        f"({deltas['metrics/mAP50(B)']:+.2f} percentage points against the original detector at the same resolution).\n\n"
        f"mAP50–95: **{100*final['metrics']['metrics/mAP50-95(B)']:.2f}%** "
        f"({deltas['metrics/mAP50-95(B)']:+.2f} points).\n\n"+"\n".join(table)+"\n\n"
        f"Metrics: {precision_note}. Baseline standalone validation used FP32; framework final validation may use AMP. "
        "Small numerical changes can reflect this difference. These are development validation scores, "
        "not an independent test or nearby-distance/tracking evaluation. No automatic deployment or vehicle control occurred.\n\n"
        "Read [the machine-readable result](results/detection_refinement.json) for individual gains/regressions and checkpoint hashes.\n")
    (ROOT/"docs/DETECTION_REFINEMENT_RESULT.md").write_text(text,encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",required=True)
    main(parser.parse_args().run_name)
