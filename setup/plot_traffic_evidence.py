"""Reproducible diagnostic plots from saved runs; never edits rollouts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
import numpy as np

ROOT=Path(__file__).resolve().parent.parent/'runs/traffic_overnight_20261004'

def body(ax,state,spec,color):
    if spec['kind']=='pedestrian':
        ax.add_patch(Circle(state[:2],.25,color=color));return
    r=spec['rear_overhang_m'];length=spec['length_m'];w=spec['width_m']/2
    local=np.array([[-r,-w],[length-r,-w],[length-r,w],[-r,w]])
    h=state[2];rotation=np.array([[np.cos(h),np.sin(h)],[-np.sin(h),np.cos(h)]])
    ax.add_patch(Polygon(local@rotation+state[:2],color=color,alpha=.85))

def main():
    out=ROOT/'evidence';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True,constrained_layout=True)
    names=['mixed_seed42_v3','mixed_seed42_v4','mixed_seed42_v5','mixed_seed42_v6']
    colors=['#cc5244','#747b86','#ca921c','#087f8c']
    sources=[]
    for name,color in zip(names,colors):
        path=ROOT/name/'replay.json'
        if not path.exists():continue
        d=json.loads(path.read_text());e=np.array(d['ego'],float);label=name.split('_')[-1]
        axs[0].plot(e[:,0],e[:,8],label=label,color=color,lw=2)
        axs[1].plot(e[:,0],e[:,4]*3.6,color=color,label=label,lw=1,alpha=.85)
        acc=np.diff(e[:,4])/np.diff(e[:,0]);axs[2].plot(e[1:,0],acc,color=color,lw=.7,alpha=.8)
        sources.append(str(path.relative_to(ROOT)))
    axs[0].set(ylabel='Route progress (m)',title='Seed 42: collision repair, queue clearance and following comparison')
    axs[0].legend(ncol=4);axs[1].set(ylabel='Actual ego speed (km/h)')
    axs[2].set(ylabel='Sampled acceleration (m/s²)',xlabel='Simulation time (s)')
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(out/'seed42_comparison.png',dpi=150);plt.close(fig)
    d=json.loads((ROOT/'mixed_seed42_v4/replay.json').read_text());a=np.array(d['actor_states'],float)
    frame=800;specs=d['actors'];fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    ax.axhspan(94.5,105.5,color='#eff1f4');ax.axhline(100,color='#c49a25',ls='--')
    for j,spec in enumerate(specs):
        state=a[frame,j]
        if 52<state[0]<80 and 93<state[1]<105:
            body(ax,state,spec,'#708293' if spec['kind']=='parked_car' else '#147d92')
            if state[1]<100:
                ax.text(state[0],state[1]+1.4,spec['name'],fontsize=8,ha='center')
    ax.annotate('Car 8: stopped with its weave offset frozen\nPhysical lateral clearance ≈ 0.514 m\nFollowing buffer = 0.55 m',xy=(61.5,96.5),xytext=(69,94),arrowprops={'arrowstyle':'->'},fontsize=10)
    ax.set(xlim=(53,80),ylim=(93,104),xlabel='World x (m)',ylabel='World y (m)',title='Observed queue cause before the v5 parked-clearance repair (v4, t = 80 s)')
    ax.set_aspect('equal');fig.savefig(out/'parked_queue_diagnosis.png',dpi=150);plt.close(fig)
    (out/'plot_sources.json').write_text(json.dumps({'sources':sources,'diagnosis_source':'mixed_seed42_v4/replay.json','diagnosis_frame':frame,'limits':'Synthetic recorded trajectories; sampled metrics only.'},indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
