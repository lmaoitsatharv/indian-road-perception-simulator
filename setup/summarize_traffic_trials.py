"""Build a compact evidence index for the bounded traffic session."""
import contextlib
import csv
import io
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from audit_traffic_rollout import audit

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/traffic_overnight_20261004'

def atomic_text(path,text):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(text,encoding='utf-8');os.replace(temp,path)

def main():
    rows=[]
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or not (folder/'result.json').exists():continue
        result=json.loads((folder/'result.json').read_text(encoding='utf-8-sig'))
        if 'simulation_time_s' not in result:continue
        audit_path=folder/'audit_motion.json'
        if not audit_path.exists() and (folder/'replay.json').exists():
            with contextlib.redirect_stdout(io.StringIO()):audit(folder,audit_path)
        independent=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        motion=independent.get('ego_motion',{})
        config=json.loads((folder/'world_config.json').read_text())
        rows.append(dict(run=folder.name,seed=result['seed'],scenario=config.get('scenario','mixed'),
            revision=config.get('revision','unknown'),status=result['status'],duration_s=result['simulation_time_s'],
            progress_m=result['ego_progress_m'],ego_overlap_samples=independent.get('ego_overlap_samples'),
            other_overlap_samples=independent.get('other_actor_overlap_samples'),
            distinct_overlap_incidents=independent.get('distinct_pair_incidents'),road_departure=result['road_departure'],
            longest_stop_s=independent.get('longest_stopped_interval_s'),
            strong_braking_seconds=motion.get('braking_at_least_2_5_mps2_seconds'),
            jerk_p95_mps3=motion.get('percentile_95_absolute_jerk_mps3'),
            max_absolute_jerk_mps3=motion.get('maximum_absolute_jerk_mps3'),
            overtakes_started=result.get('overtakes_started',0),overtakes_completed=result.get('overtakes_completed',0),
            crossings_completed=result['crossings_completed']))
    stamp=datetime.now(timezone.utc).isoformat()
    atomic_text(OUT/'comparison.json',json.dumps(dict(updated_at_utc=stamp,trials=rows),indent=2)+'\n')
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['run'])
    writer.writeheader();writer.writerows(rows);atomic_text(OUT/'comparison.csv',stream.getvalue())
    lines=['# Saved traffic comparisons','',f'Updated {stamp}.','',
        'Overlap counts are body overlaps at recorded 0.1-second frames. Ego also has separate 0.02-second checks in each result. An overlap-free sample does not establish real-road safety.',
        '', '| Run | Progress (m) | Ego / other overlap samples | Longest stop (s) | Strong braking (s) | Jerk p95 (m/s³) | Overtakes | Crossings |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda n:'n/a' if n is None else f'{n:.2f}'
    for r in rows:
        lines.append(f"| [{r['run']}]({r['run']}/result.json) | {r['progress_m']:.2f} | {r['ego_overlap_samples']} / {r['other_overlap_samples']} | {fmt(r['longest_stop_s'])} | {fmt(r['strong_braking_seconds'])} | {fmt(r['jerk_p95_mps3'])} | {r['overtakes_completed']} | {r['crossings_completed']} |")
    lines+=['','Strong braking means sampled acceleration at or below −2.5 m/s². Jerk is the rate of acceleration change; these values are finite differences at 0.1 s and include controller switching. A failed run can have a successful batch exit code; use its result and independent audit.',
        '', 'Each trial retains exact configuration, MATLAB source snapshot, complete recorded actor states, events, trajectory CSV and independent audit. The audit includes each overlap incident and all actors\' motion metrics. New v6 trials also retain per-step decision reasons and the responsible actor IDs.',
        '', 'Repeated seeds are development comparisons, not independent safety validation. NPC motion and pedestrian rules are illustrative; RL and calibrated camera driving are not used.']
    atomic_text(OUT/'COMPARISON.md','\n'.join(lines)+'\n')
    print(json.dumps({'trials_indexed':len(rows),'updated_at_utc':stamp}))

if __name__=='__main__':main()
