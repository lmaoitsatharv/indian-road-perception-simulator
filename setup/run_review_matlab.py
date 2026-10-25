"""Run one bounded, hidden review batch in its own kill-on-close Windows job."""
import argparse
import sys
import time
from pathlib import Path
from traffic_session import OwnedJob, ROOT, atomic, stamp
sys.path.insert(0, str(ROOT))
from project_runtime import matlab_executable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',required=True)
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--matlab',help='MATLAB executable; otherwise MATLAB_EXECUTABLE or PATH')
    args=parser.parse_args()
    if not 1<=args.timeout<=900: parser.error('Timeout must be between 1 and 900 seconds')
    log=args.log.resolve()
    if not log.is_relative_to(ROOT/'setup/review_20261013'):
        parser.error('Use the review log directory')
    status=log.with_suffix('.job.json')
    if log.exists() or status.exists(): parser.error('Choose a fresh log path')
    log.parent.mkdir(parents=True,exist_ok=True)
    try:
        matlab=matlab_executable(args.matlab)
    except FileNotFoundError as error:
        parser.error(str(error))
    job=None;record=dict(status='starting',started_at_utc=stamp(),timeout_seconds=args.timeout,
        command=args.code,log=str(log),unrelated_desktop_attached=False,keep_awake_requested=False)
    atomic(status,record)
    try:
        job=OwnedJob([str(matlab),'-wait','-batch',args.code,'-logfile',str(log)],ROOT)
        record.update(status='running',pid=job.pid);atomic(status,record)
        end=time.monotonic()+args.timeout
        while True:
            code=job.poll()
            if code is not None:
                record.update(status='completed' if code==0 else 'failed',exit_code=code)
                break
            if time.monotonic()>=end:
                record.update(status='timed_out',exit_code=None)
                break
            time.sleep(.25)
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        if job is not None:job.close()
        record.update(ended_at_utc=stamp(),owned_job_closed=True)
        atomic(status,record)
    print(record['status'],flush=True)
    raise SystemExit(0 if record['status']=='completed' else 1)


if __name__=='__main__':main()
