"""Compare the frozen new simulation model and locked baseline on fresh captures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "setup"))
from prepare_simulation_dataset import save,sha,protected


def local(value):
    if sys.platform == "linux" and len(value)>2 and value[1]==":":
        return Path("/mnt/"+value[0].lower()+"/"+value[3:].replace("\\","/"))
    return Path(value)


def confirm(output):
    capture=json.loads((output / "capture.json").read_text())
    plan=json.loads((output / "plan.json").read_text())
    if capture["status"] != "captured_unscored" or len(capture["episodes"])!=8:
        raise RuntimeError("Finish all eight predeclared native captures before inference")
    if (output / "evaluation_status.json").exists():
        raise FileExistsError("Preserve previous evaluation attempts")
    assert protected()==plan["protected_sha256"]
    checkpoint=ROOT / "runs/simulation_frozen_20261022_01/best.keras"
    assert sha(checkpoint)==plan["simulation_checkpoint_sha256"]
    assert sha(output / "selection.json")==plan["selection_sha256"]
    assert sha(output / "simulation_model.json")==plan["model_manifest_sha256"]
    sources=[local(name) for name in capture["episodes"]]
    state={"status":"starting","started_utc":datetime.now(timezone.utc).isoformat(),
           "checkpoint_fixed_before_capture":True,"episodes":len(sources),"models":[]}
    save(output / "evaluation_status.json",state)
    try:
        for name,manifest in (("fresh_model",output / "simulation_model.json"),("locked_baseline",None)):
            argv=[sys.executable,"-u",str(ROOT / "evaluate_carla_road.py"),
                  "--output",str(output / name),"--device","gpu","--max-wall-seconds","900"]
            for episode in sources:
                argv += ["--episode",str(episode)]
            if manifest:
                argv += ["--simulation-model",str(manifest)]
            record={"name":name,"started_utc":datetime.now(timezone.utc).isoformat(),"argv":argv}
            state["models"].append(record)
            state["status"]="evaluating_"+name
            save(output / "evaluation_status.json",state)
            with (output / (name+".log")).open("w") as log:
                process=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=960)
            record["exit_code"]=process.returncode
            save(output / "evaluation_status.json",state)
            if process.returncode:
                raise RuntimeError("Fresh comparison failed: "+name+"; inspect its preserved log")
            report=json.loads((output / name / "report.json").read_text())
            assert report["status"]=="completed" and report["model_weights_unchanged"] and report["protected_files_unchanged"]
            record.update(frames=sum(e["frames"] for e in report["episodes"]),metrics=report["metrics"])
            print(json.dumps({"model":name,"frames":record["frames"],"metrics":record["metrics"]}),flush=True)
        state.update(status="completed",finished_utc=datetime.now(timezone.utc).isoformat(),
                     protected_unchanged=protected()==plan["protected_sha256"] and sha(checkpoint)==plan["simulation_checkpoint_sha256"])
        assert state["protected_unchanged"]
    except BaseException as error:
        state.update(status="failed_preserved",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(output / "evaluation_status.json",state)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT / "runs/simulation_confirmation_20261023_01")
    args=parser.parse_args()
    confirm(args.output.resolve())
