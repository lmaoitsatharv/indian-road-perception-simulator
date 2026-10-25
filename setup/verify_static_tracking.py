"""Independently audit saved MATLAB motion, bounds, and goal using NumPy."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/static_tracking_20261003_feedforward'


def main():
    config = json.loads((RUN/'tracking_config.json').read_text(encoding='utf-8'))
    result = json.loads((RUN/'tracking_result.json').read_text(encoding='utf-8'))
    assert result['status']=='completed' and result['goal_reached']
    fine = np.genfromtxt(RUN/'integration_trace.csv',delimiter=',',names=True,encoding='utf-8-sig')
    control = np.genfromtxt(RUN/'tracking_trace.csv',delimiter=',',names=True,encoding='utf-8-sig')
    assert np.all(np.diff(fine['time_s'])>0)
    times=fine['time_s'];dt=np.diff(times);theta=fine['heading_rad'];speed=fine['speed_mps']
    assert np.allclose(dt,config['integration_dt_s'],atol=1e-10)
    assert np.all(speed>=0) and np.isfinite(np.column_stack([fine[name] for name in fine.dtype.names])).all()
    # Each fine interval uses the steering command of that control interval.
    rows=np.rint(times[1:]/config['controller_dt_s']+0.499999).astype(int)
    rows=np.clip(rows,1,len(control)-1)
    steering=control['steering_rad'][rows]
    moved=(speed[1:]+speed[:-1])/2*dt
    heading_delta=np.arctan2(np.sin(np.diff(theta)),np.cos(np.diff(theta)))
    expected_delta=moved*np.tan(steering)/config['scene']['vehicle']['wheelbase_m']
    assert np.max(np.abs(heading_delta-expected_delta))<1e-9
    # Check movement independently using the midpoint arc chord identity.
    chord=moved*np.sinc(heading_delta/(2*np.pi))
    expected_x=chord*np.cos(theta[:-1]+heading_delta/2)
    expected_y=chord*np.sin(theta[:-1]+heading_delta/2)
    displacement_error=np.hypot(np.diff(fine['rear_axle_x_m'])-expected_x,np.diff(fine['rear_axle_y_m'])-expected_y)
    assert displacement_error.max()<1e-9, 'Pose replay does not obey the stored bicycle speed and steering.'
    accelerations=np.diff(speed)/dt
    assert accelerations.max()<=config['maximum_acceleration_mps2']+1e-8
    assert accelerations.min()>=-config['maximum_deceleration_mps2']-1e-8
    assert np.abs(control['steering_rad']).max()<=config['maximum_steering_rad']+1e-10
    rates=np.diff(control['steering_rad'])/config['controller_dt_s']
    assert np.abs(rates).max()<=config['maximum_steering_rate_radps']+1e-10
    scene=config['scene'];vehicle=scene['vehicle']
    local=np.array([[-vehicle['rear_overhang_m'],-vehicle['width_m']/2],
                    [vehicle['front_of_rear_axle_m'],-vehicle['width_m']/2],
                    [vehicle['front_of_rear_axle_m'],vehicle['width_m']/2],
                    [-vehicle['rear_overhang_m'],vehicle['width_m']/2]])
    cosine=np.cos(theta);sine=np.sin(theta)
    corners=np.empty((len(theta),4,2))
    corners[:,:,0]=local[None,:,0]*cosine[:,None]-local[None,:,1]*sine[:,None]+fine['rear_axle_x_m'][:,None]
    corners[:,:,1]=local[None,:,0]*sine[:,None]+local[None,:,1]*cosine[:,None]+fine['rear_axle_y_m'][:,None]
    r=scene['road_bounds_xyxy_m'];minimum_road=float(np.min(np.concatenate([
        corners[:,:,0]-r[0],r[2]-corners[:,:,0],corners[:,:,1]-r[1],r[3]-corners[:,:,1]],axis=1)))
    assert minimum_road>=scene['safety_margin_m']
    assert abs(minimum_road-result['minimum_road_footprint_clearance_m'])<1e-9
    # A square-expanded obstacle is conservative for the required Euclidean margin.
    o=np.array(scene['obstacle_bounds_xyxy_m'])+np.array([-1,-1,1,1])*scene['safety_margin_m']
    obstacle=np.array([[o[0],o[1]],[o[2],o[1]],[o[2],o[3]],[o[0],o[3]]])
    axes=np.stack([np.tile([1.,0.],(len(theta),1)),np.tile([0.,1.],(len(theta),1)),
                   np.stack([cosine,sine],axis=1),np.stack([-sine,cosine],axis=1)],axis=1)
    body_proj=np.einsum('nki,nji->nkj',axes,corners)
    obstacle_proj=np.einsum('nki,ji->nkj',axes,obstacle)
    separated=(body_proj.max(axis=2)<obstacle_proj.min(axis=2))|(obstacle_proj.max(axis=2)<body_proj.min(axis=2))
    assert np.all(separated.any(axis=1)), 'A recorded body overlaps the safety-expanded static obstacle.'
    goal=np.array(scene['goal_pose']);final=np.array([fine['rear_axle_x_m'][-1],fine['rear_axle_y_m'][-1],theta[-1]])
    position_error=float(np.linalg.norm(final[:2]-goal[:2]))
    heading_error=float(abs(np.arctan2(np.sin(final[2]-goal[2]),np.cos(final[2]-goal[2]))))
    assert position_error<=config['goal_position_tolerance_m'] and heading_error<=config['goal_heading_tolerance_rad']
    assert speed[-1]<=config['goal_speed_tolerance_mps']
    verification={'status':'passed','recorded_poses':len(fine),'controller_records':len(control),
        'independent_rear_axle_kinematics_checked':True,'maximum_displacement_error_m':float(displacement_error.max()),
        'maximum_heading_update_error_rad':float(np.max(np.abs(heading_delta-expected_delta))),
        'speed_steering_acceleration_rate_bounds_passed':True,
        'road_body_margin_recount_m':minimum_road,'no_body_overlap_with_conservative_margin_expanded_obstacle':True,
        'goal_position_error_m':position_error,'goal_heading_error_rad':heading_error,'final_speed_mps':float(speed[-1]),
        'scope':'One static ground-truth development trace, sampled every 0.01s. No AEB, RL, camera projection or general driving evaluation.'}
    (RUN/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    reference=np.genfromtxt(ROOT/'runs/static_reference_20261003/reference_path.csv',delimiter=',',names=True)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axs[0,0];ax.plot(reference['rear_axle_x_m'],reference['rear_axle_y_m'],'--',label='Planned rear axle route',color='#3976ba')
    ax.plot(fine['rear_axle_x_m'],fine['rear_axle_y_m'],label='Simulated rear axle trace',color='#0b936e')
    from matplotlib.patches import Rectangle
    original=scene['obstacle_bounds_xyxy_m'];ax.add_patch(Rectangle(original[:2],original[2]-original[0],original[3]-original[1],facecolor='#f2b253',edgecolor='#895918'))
    ax.set(xlabel='World x (m)',ylabel='World y (m)',title='Planned route versus actual simulated motion',xlim=(5,56),ylim=(6,19));ax.set_aspect('equal');ax.legend(fontsize=8)
    axs[0,1].plot(control['time_s'],control['rear_path_distance_m'],color='#0b936e');axs[0,1].set(xlabel='Simulation time (s)',ylabel='Distance to nearest rear path sample (m)',title='Rear axle tracking deviation')
    axs[1,0].plot(control['time_s'],control['target_speed_mps'],label='Target',linestyle='--',color='#3976ba');axs[1,0].plot(times,speed,label='Simulated',color='#0b936e');axs[1,0].set(xlabel='Simulation time (s)',ylabel='Speed (m/s)',title='Acceleration, cruise and goal braking');axs[1,0].legend()
    axs[1,1].plot(control['time_s'],np.rad2deg(control['steering_rad']),color='#0b936e');bound=np.rad2deg(config['maximum_steering_rad']);axs[1,1].axhline(bound,color='#b46d2b',linestyle='--');axs[1,1].axhline(-bound,color='#b46d2b',linestyle='--');axs[1,1].set(xlabel='Simulation time (s)',ylabel='Front wheel steering (degrees)',title='Steering angle and configured limits')
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.suptitle('Static closed-loop development reference · Stanley steering + PI with braking feedforward\nGround-truth map; no RL, moving actors, AEB or learned perception map',fontsize=13)
    fig.savefig(RUN/'tracking_summary.png',dpi=160)
    plt.close(fig)
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()
