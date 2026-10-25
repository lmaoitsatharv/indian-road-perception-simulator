"""Wait for this dataset build, run two independent checks, then publish evidence. No training."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time
import traceback

from prepare_simulation_dataset import DEFAULT,ROOT,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT)
    output=parser.parse_args().output.resolve()
    state={"status":"waiting_for_preprocessing","training_started":False,
           "started_utc":datetime.now(timezone.utc).isoformat()}
    path=output/"completion_status.json"
    save(path,state)
    started=time.monotonic()
    try:
        while True:
            capture=json.loads((output/"capture.json").read_text())
            if capture["status"]=="failed_preserved":
                raise RuntimeError("Capture failed; preserve the version for inspection")
            if (output/"dataset_report.json").exists() and capture["status"]=="captured":
                break
            if time.monotonic()-started>3600:
                raise TimeoutError("Preparation did not finish within the bounded wait")
            time.sleep(5)
        state["status"]="running_independent_and_preprocessing_checks"
        save(path,state)
        commands={"independent_verification":[str(ROOT/".venv-carla/Scripts/python.exe"),"-u","setup/verify_simulation_dataset.py","--output",str(output)],
                  "tensorflow_preprocessing":[str(ROOT/".venv/Scripts/python.exe"),"-u","setup/check_simulation_preprocessing.py","--output",str(output)]}
        def run_check(name,argv):
            with (output/f"{name}.log").open("w",encoding="utf-8") as log:
                result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            return name,result.returncode
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(run_check,name,argv) for name,argv in commands.items()]
            results=dict(f.result() for f in futures)
        state["checks_exit_codes"]=results
        save(path,state)
        if any(results.values()):
            raise RuntimeError(f"A preparation check failed: {results}; read the corresponding log")
        with (output/"finalization.log").open("w",encoding="utf-8") as log:
            result=subprocess.run([str(ROOT/".venv/Scripts/python.exe"),"setup/finalize_simulation_dataset.py","--output",str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Finalization failed; inspect finalization.log")
        state.update(status="completed",finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-started)
        save(path,state)
        print(json.dumps(state,indent=2),flush=True)
        return 0
    except BaseException:
        state.update(status="failed_preserved",error=traceback.format_exc(),elapsed_s=time.monotonic()-started)
        save(path,state)
        raise


if __name__=="__main__":
    raise SystemExit(main())
