"""Independent NumPy audit of recorded 0.1 s actor states, with incident times."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

def validated_replay(data):
    """Reject damaged logs rather than reporting NaNs as zero collisions."""
    if not isinstance(data, dict):
        raise ValueError('Replay must be an object')
    specs = data.get('actors')
    if not isinstance(specs, list) or not specs:
        raise ValueError('Replay must contain actor specifications')
    names = set()
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get('name'), str) or not spec['name']:
            raise ValueError('Every actor requires a name')
        if spec['name'] in names or spec['name'] == 'ego':
            raise ValueError('Actor names must be unique and cannot reserve ego')
        names.add(spec['name'])
        if spec.get('kind') not in ('car', 'auto_rickshaw', 'two_wheeler', 'pedestrian', 'parked_car'):
            raise ValueError('Unsupported actor kind')
        for key in ('length_m', 'width_m', 'rear_overhang_m'):
            value = spec.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'Invalid actor dimension: {key}')
        if spec['length_m'] <= 0 or spec['width_m'] <= 0 or not 0 <= spec['rear_overhang_m'] <= spec['length_m']:
            raise ValueError('Actor dimensions must describe a positive body')
    ego = np.asarray(data.get('ego'), dtype=float)
    actors = np.asarray(data.get('actor_states'), dtype=float)
    if ego.ndim != 2 or len(ego) < 2 or ego.shape[1] < 9:
        raise ValueError('Ego log requires at least two frames and nine columns')
    if actors.ndim != 3 or actors.shape[:2] != (len(ego), len(specs)) or actors.shape[2] < 3:
        raise ValueError('Actor state dimensions do not match frames/specifications')
    # MATLAB encodes an infinite 'no lead actor' gap as JSON null. That optional
    # diagnostic column is not geometry or a motion input to this audit.
    finite_columns = [i for i in range(ego.shape[1]) if i != 11]
    if not np.isfinite(ego[:,finite_columns]).all() or not np.isfinite(actors).all():
        raise ValueError('Replay contains non-finite states')
    if ego.shape[1] > 11:
        for row in data['ego']:
            gap = row[11]
            if gap is not None and (isinstance(gap,bool) or not isinstance(gap,(int,float)) or not np.isfinite(gap)):
                raise ValueError('Optional lead gap must be finite or JSON null')
    if ego[0, 0] < 0 or not (np.diff(ego[:, 0]) > 0).all():
        raise ValueError('Replay timestamps must be nonnegative and strictly increasing')
    if (ego[:, 4] < 0).any():
        raise ValueError('Recorded ego speed cannot be negative')
    return ego, actors, specs

def intervals(times, mask, frame_offset=0):
    """Recorded-frame intervals; endpoints are samples, not continuous bounds."""
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return [dict(start_s=float(times[b]),end_s=float(times[e]),
                 duration_s=float(times[e]-times[b]),replay_frame=int(b+frame_offset))
            for b,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1)]

def motion_metrics(times, positions, speed=None):
    dt=np.diff(times)
    velocity=np.diff(positions,axis=0)/dt[:,None]
    travelled=np.linalg.norm(np.diff(positions,axis=0),axis=1)
    actual_speed=np.linalg.norm(velocity,axis=1) if speed is None else speed[1:]
    stationary=actual_speed<.15
    stops=intervals(times[1:],stationary,1)
    acceleration=np.diff(actual_speed)/dt[1:]
    jerk=np.diff(acceleration)/dt[2:]
    hard=acceleration<=-2.5
    return dict(distance_along_recorded_positions_m=float(travelled.sum()),
        maximum_speed_mps=float(actual_speed.max(initial=0)),
        stopped_seconds_at_speed_below_0_15=float(dt[stationary].sum()),
        longest_stopped_interval_s=max((x['duration_s'] for x in stops),default=0),
        long_stops_over_10s=[x for x in stops if x['duration_s']>=10],
        minimum_acceleration_mps2=float(acceleration.min(initial=0)),
        maximum_acceleration_mps2=float(acceleration.max(initial=0)),
        maximum_absolute_jerk_mps3=float(np.abs(jerk).max(initial=0)),
        percentile_95_absolute_jerk_mps3=float(np.percentile(np.abs(jerk),95)) if len(jerk) else 0,
        braking_at_least_2_5_mps2_seconds=float(dt[1:][hard].sum()),
        strong_braking_episodes=intervals(times[2:],hard,2))

def hits(a, sa, b, sb):
    pa, pb = sa['kind']=='pedestrian', sb['kind']=='pedestrian'
    if pa and pb:
        return np.linalg.norm(a[:,:2]-b[:,:2],axis=1) <= (sa['width_m']+sb['width_m'])/2
    if pb:
        return hits(b,sb,a,sa)
    if pa:
        u=np.stack([np.cos(b[:,2]),np.sin(b[:,2])],axis=1)
        n=np.stack([-u[:,1],u[:,0]],axis=1)
        d=a[:,:2]-b[:,:2]
        q=np.stack([np.sum(d*u,axis=1),np.sum(d*n,axis=1)],axis=1)
        closest=np.clip(q,[-sb['rear_overhang_m'],-sb['width_m']/2],
                          [sb['length_m']-sb['rear_overhang_m'],sb['width_m']/2])
        return np.linalg.norm(q-closest,axis=1)<=sa['width_m']/2
    u=np.stack([np.cos(a[:,2]),np.sin(a[:,2])],axis=1);v=np.stack([-u[:,1],u[:,0]],axis=1)
    w=np.stack([np.cos(b[:,2]),np.sin(b[:,2])],axis=1);z=np.stack([-w[:,1],w[:,0]],axis=1)
    ac=a[:,:2]+(sa['length_m']/2-sa['rear_overhang_m'])*u
    bc=b[:,:2]+(sb['length_m']/2-sb['rear_overhang_m'])*w
    overlap=np.ones(len(a),dtype=bool)
    for axis in (u,v,w,z):
        distance=np.abs(np.sum((bc-ac)*axis,axis=1))
        radii=(sa['length_m']/2*np.abs(np.sum(u*axis,axis=1))+
            sa['width_m']/2*np.abs(np.sum(v*axis,axis=1))+
            sb['length_m']/2*np.abs(np.sum(w*axis,axis=1))+
            sb['width_m']/2*np.abs(np.sum(z*axis,axis=1)))
        overlap &= distance<=radii
    return overlap

def audit(run_dir, output):
    data=json.loads((Path(run_dir)/'replay.json').read_text())
    ego,actors,specs=validated_replay(data);times=ego[:,0]
    ego_spec=dict(name='ego',kind='car',length_m=4.5,width_m=1.8,rear_overhang_m=.9)
    all_specs=specs+[ego_spec]
    all_states=np.concatenate([actors[:,:,:3],ego[:,None,1:4]],axis=1)
    incidents=[];pair_counts={};ego_count=0;other_count=0
    for i,sa in enumerate(all_specs):
        for j in range(i+1,len(all_specs)):
            sb=all_specs[j];hit=hits(all_states[:,i,:],sa,all_states[:,j,:],sb)
            count=int(hit.sum())
            if not count:continue
            key=f"{sa['name']} / {sb['name']}";pair_counts[key]=count
            if j==len(specs):ego_count+=count
            else:other_count+=count
            edges=np.diff(np.r_[False,hit,False].astype(int))
            for begin,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1):
                incidents.append(dict(actor_a=sa['name'],actor_b=sb['name'],
                    start_s=float(times[begin]),end_s=float(times[end]),samples=int(end-begin+1),
                    replay_frame=int(begin),kind='sampled_body_overlap'))
    incidents.sort(key=lambda e:e['start_s'])
    speed=ego[:,4];stopped=speed<.15
    edges=np.diff(np.r_[False,stopped,False].astype(int))
    stop_durations=[float(times[e]-times[b]) for b,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1)]
    ego_motion=motion_metrics(times,ego[:,1:3],speed)
    actor_motion={s['name']:motion_metrics(times,actors[:,i,:2]) for i,s in enumerate(specs)}
    result=dict(run_dir=str(run_dir),audit='independent_NumPy_OBB_and_pedestrian_disc_at_recorded_frames',
        audit_revision='v3_validated_all_actor_motion_and_braking',
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sample_period_s=float(np.median(np.diff(times))),
        ego_overlap_samples=ego_count,other_actor_overlap_samples=other_count,
        distinct_pair_incidents=len(incidents),incidents=incidents,
        pair_counts=dict(sorted(pair_counts.items(),key=lambda x:-x[1])),
        longest_stopped_interval_s=max(stop_durations,default=0),
        stopped_sample_fraction=float(stopped.mean()),
        maximum_speed_kmh=float(speed.max()*3.6),
        progress_m=float(ego[-1,8]),
        ego_motion=ego_motion,actor_motion=actor_motion,
        motion_metric_limits='Finite differences of rounded 0.1 s poses/speeds; jerk peaks depend on sample period. NPC lateral paths are prescribed; these are descriptive metrics, not a human-comfort or dynamics certification.',
        limits='Discrete recorded-frame audit, not continuous collision proof; no traffic-realism validation.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    # Historical evidence is immutable; callers choose a fresh audit filename.
    if output.exists():
        raise FileExistsError(f'Choose a new audit output: {output}')
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=output.parent,
                prefix=output.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);stream.write(payload)
        # On Windows rename refuses an existing destination, preserving evidence.
        os.rename(temporary,output)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    summary={k:v for k,v in result.items() if k not in ('incidents','actor_motion','ego_motion')}
    summary['ego_motion']={k:v for k,v in ego_motion.items() if not isinstance(v,list)}
    summary['strong_braking_episode_count']=len(ego_motion['strong_braking_episodes'])
    print(json.dumps(summary,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('output');a=p.parse_args()
    audit(a.run_dir,a.output)
