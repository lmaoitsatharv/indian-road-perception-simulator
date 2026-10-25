"""Run three bounded fresh checks sequentially after model selection is frozen."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/model_retention_20261021_01"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local(value):
    if len(value) > 2 and value[1] == ":" and ROOT.as_posix().startswith("/mnt/"):
        return Path("/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/"))
    return Path(value)


def main():
    trial = load(ROOT / "runs/road_rehearsal_20261021_01/run_status.json")
    if trial["phase"] != "completed":
        raise ValueError("Finish the bounded training job before fresh evaluation")
    selection = load(OUT / "simulation_selection.json")
    source = load(ROOT / "runs/retention_holdout_capture_20261021_01/capture.json")
    if source["state"] != "captured_unscored":
        raise ValueError("Use the predeclared fresh capture")
    episode = local(source["episode"])
    journal = OUT / "confirmation_status.json"
    if journal.exists():
        raise FileExistsError("Preserve earlier confirmations; never duplicate or overwrite")
    state = {"phase": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
             "selection_frozen_utc": selection["frozen_utc"], "episode": str(episode), "models": []}

    def save():
        temp = journal.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temp.replace(journal)

    save()
    for key, spec in (("original", None), ("previous03", OUT / "previous03_simulation_model.json"),
                      ("selected", OUT / "simulation_model.json")):
        argv = [sys.executable, "-u", str(ROOT / "evaluate_carla_road.py"), "--episode", str(episode),
                "--output", str(OUT / ("fresh_" + key)), "--device", "gpu", "--max-wall-seconds", "300"]
        if spec:
            argv += ["--simulation-model", str(spec)]
        record = {"model": key, "started_utc": datetime.now(timezone.utc).isoformat(), "argv": argv}
        state["models"].append(record); state["phase"] = "evaluating_" + key; save()
        try:
            process = subprocess.run(argv, cwd=ROOT, timeout=360)
            record["exit_code"] = process.returncode
            record["finished_utc"] = datetime.now(timezone.utc).isoformat(); save()
            if process.returncode:
                raise RuntimeError("Fresh model check failed: " + key)
        except BaseException as error:
            state["phase"] = "failed_preserved"; state["error"] = str(error); save()
            raise
    state["phase"] = "completed"; state["finished_utc"] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == "__main__":
    main()
