"""Append a trial to a historical traffic session's lease-bound queue.

Requires an existing active session; it does not create a general job scheduler.
"""
import argparse
import sys
from pathlib import Path
import traffic_session as session
sys.path.insert(0, str(session.ROOT))
from project_runtime import matlab_executable


def main():
    p=argparse.ArgumentParser()
    p.add_argument('job_id')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--duration',type=float,default=300)
    p.add_argument('--scenario',default='mixed')
    p.add_argument('--parked-clearance',action='store_true')
    p.add_argument('--independent-directions',action='store_true')
    p.add_argument('--overtaking',action='store_true')
    p.add_argument('--smooth-following',action='store_true')
    p.add_argument('--pedestrian-passing',action='store_true')
    p.add_argument('--matlab',help='MATLAB executable; otherwise MATLAB_EXECUTABLE or PATH')
    a=p.parse_args()
    assert a.job_id.replace('_','').replace('-','').isalnum()
    assert a.scenario.replace('_','').isalnum()
    assert 0<a.duration<=900 and a.seed>=0
    assert session.permission()==(None,False),'Fresh allowed monitor lease required'
    state=session.read(session.SESSION/'session_status.json',{})
    assert state.get('status') in ('idle','running'),'Controller must be active'
    output=session.ROOT/'runs/traffic_overnight_20261004'/a.job_id
    queue=session.read(session.SESSION/'queue.json',[])
    assert not output.exists() and not (session.SESSION/'jobs'/(a.job_id+'.json')).exists()
    assert not any(x['id']==a.job_id for x in queue)
    matlab=lambda value:'true' if value else 'false'
    matlab_path=lambda path:path.as_posix().replace("'", "''")
    code=(f"addpath('{matlab_path(session.ROOT/'integration/matlab')}'); "
          f"run_indian_traffic_trial('{matlab_path(output)}',{a.seed},{a.duration},"
          f"'Scenario','{a.scenario}','Overtaking',{matlab(a.overtaking)},"
          f"'IndependentDirections',{matlab(a.independent_directions)},"
          f"'ParkedClearance',{matlab(a.parked_clearance)},"
          f"'SmoothFollowing',{matlab(a.smooth_following)},"
          f"'PedestrianPassing',{matlab(a.pedestrian_passing)});")
    queue.append(dict(id=a.job_id,argv=[str(matlab_executable(a.matlab)),
        '-wait','-batch',code,'-logfile',str(session.SESSION/(a.job_id+'.log'))],
        timeout_seconds=min(900,120+a.duration*1.5)))
    session.atomic(session.SESSION/'queue.json',queue)
    print('Queued '+a.job_id)


if __name__=='__main__':main()
