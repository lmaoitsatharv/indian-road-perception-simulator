"""Finish the local LR experiment with a conservative, predeclared validation gate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/detection_lr_probe_20261025_01"


def review():
    owner = json.loads((OUTPUT / "owner.json").read_text())
    while not (OUTPUT / "exit.json").exists():
        proc = Path("/proc") / str(owner["pid"]) / "stat"
        if not proc.exists() or proc.read_text().split(") ", 1)[1].split()[19] != owner["linux_process_start_ticks"]:
            raise RuntimeError("CPU study disappeared without a completion record")
        time.sleep(10)
    assert json.loads((OUTPUT / "exit.json").read_text())["returncode"] == 0, "The CPU study failed; preserve logs"
    raw = json.loads((OUTPUT / "comparison.json").read_text())
    assert raw["passed"] and raw["identical_augmented_inputs"] and raw["identical_optimizer_step_count"]
    initial = raw["baseline"]["metrics"]
    control = raw["arms"]["control"]["validation"]
    higher = raw["arms"]["higher"]["validation"]
    improvement = {key: 100 * (higher["metrics"][key] - value) for key, value in initial.items()}
    gate = {"strict_gain_vs_control_at_least_0_1_point": raw["higher_minus_control_percentage_points"]["metrics/mAP50-95(B)"] >= .1,
        "map50_not_below_control": raw["higher_minus_control_percentage_points"]["metrics/mAP50(B)"] >= 0,
        "strict_gain_vs_starting_checkpoint": improvement["metrics/mAP50-95(B)"] > 0,
        "map50_not_below_starting_checkpoint": improvement["metrics/mAP50(B)"] >= 0,
        "person_not_below_control": raw["class_ap50_deltas_percentage_points"]["person"] >= 0,
        "animal_not_below_control": raw["class_ap50_deltas_percentage_points"]["animal"] >= 0}
    promising = all(gate.values())
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "gates": gate,
        "higher_minus_starting_checkpoint_percentage_points": improvement,
        "promising_for_larger_confirmation": promising, "automatically_promoted": False,
        "decision": "Higher rate is promising in this small screen; confirm on a larger comparison" if promising else "No convincing higher-rate win under the declared gate; retain the current rate",
        "scope": "256 training selections / 192 stratified validation images, CPU FP32; not a full-dataset result"}
    (OUTPUT / "review.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if (OUTPUT / "REPORT.md").exists():
        shutil.copy2(OUTPUT / "REPORT.md", OUTPUT / "SCREENING_REPORT.md")
    lines = ["# Modest learning-rate comparison", "", result["decision"] + ".", "",
        "Only the learning rate differs between the paired arms: 0.000030 versus 0.0000375 (+25%). Identical augmented input streams and optimizer-update counts passed verification.", "",
        "| Model | mAP50 | mAP50-95 |", "| --- | ---: | ---: |"]
    for name, metrics in (("Starting checkpoint", initial), ("Current rate", control["metrics"]), ("25% higher rate", higher["metrics"])):
        lines.append(f"| {name} | {100*metrics['metrics/mAP50(B)']:.3f}% | {100*metrics['metrics/mAP50-95(B)']:.3f}% |")
    lines += ["", result["scope"] + ".", "",
        "The decision gate was registered before candidate results were available. It requires improvement over both the control and the starting model, with person/animal checks. A screened win is not a confirmed full-validation improvement. The main GPU job and its learning rate were not modified. No schedule was created."]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        try:
            review()
        except Exception as error:
            (OUTPUT / "review_error.json").write_text(json.dumps({"error": str(error),
                "checked_utc": datetime.now(timezone.utc).isoformat()}) + "\n", encoding="utf-8")
            raise
    else:
        assert not (OUTPUT / "review_owner.json").exists()
        assert not (OUTPUT / "higher_result.json").exists(), "Declare the gate before seeing candidate results"
        (OUTPUT / "review_policy.json").write_text(json.dumps({"declared_utc": datetime.now(timezone.utc).isoformat(),
            "strict_control_gain_minimum_percentage_points": .1, "must_improve_over_starting_model": True,
            "person_and_animal_must_not_regress_vs_control": True, "full_confirmation_required": True}, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(Path(__file__), OUTPUT / "review_source.py")
        with (OUTPUT / "review.log").open("xb", buffering=0) as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (OUTPUT / "review_owner.json").write_text(json.dumps({"pid": process.pid,
            "scope": "Local experiment completion/report generation only; no scheduled agent calls"}) + "\n")
        print(json.dumps({"review_started": True, "pid": process.pid}), flush=True)
