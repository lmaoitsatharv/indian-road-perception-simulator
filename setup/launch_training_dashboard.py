"""Start the local, read-only training dashboard independently of the command window."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    launcher = run.with_name(run.name + "_launcher")
    if run.parent != (ROOT/"runs").resolve() or not (run.is_dir() or (launcher/"owner.json").is_file()):
        parser.error("Choose an existing project training run")
    url = f"http://127.0.0.1:{args.port}"
    try:
        with urlopen(url+"/api/progress", timeout=2) as response:
            current = json.load(response)
    except OSError:
        current = None
    if current is not None:
        if current.get("run") != run.name:
            raise RuntimeError("This port is displaying a different run; select another port")
        print(json.dumps({"url":url,"already_running":True}))
    else:
        options = {"cwd":ROOT, "stdin":subprocess.DEVNULL, "close_fds":True}
        if sys.platform == "win32":
            options["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
                                        subprocess.CREATE_BREAKAWAY_FROM_JOB)
        else:
            options["start_new_session"] = True
        env=os.environ.copy();env["PYTHONIOENCODING"]="utf-8";options["env"]=env
        dashboard_directory = run if run.is_dir() else launcher
        with (dashboard_directory/"dashboard.stdout.log").open("ab",buffering=0) as stdout, (dashboard_directory/"dashboard.stderr.log").open("ab",buffering=0) as stderr:
            process=subprocess.Popen([sys.executable,"-u",str(ROOT/"setup/training_progress_server.py"),
                "--run",str(run),"--port",str(args.port)],stdout=stdout,stderr=stderr,**options)
        ready=False
        for _ in range(30):
            if process.poll() is not None:
                raise RuntimeError("Dashboard exited; inspect dashboard.stderr.log")
            try:
                with urlopen(url+"/api/progress",timeout=1) as response:
                    ready=json.load(response).get("run")==run.name
                if ready:
                    break
            except OSError:
                time.sleep(.1)
        if not ready:
            raise RuntimeError("Dashboard has not answered its startup check")
        record={"process_id":process.pid,"url":url,"run":str(run),"platform":sys.platform,
            "detached_from_command_job":True,"started_utc":datetime.now(timezone.utc).isoformat()}
        (dashboard_directory/"dashboard.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(record,indent=2))
    if args.open:
        webbrowser.open(url)


if __name__ == "__main__":
    main()
