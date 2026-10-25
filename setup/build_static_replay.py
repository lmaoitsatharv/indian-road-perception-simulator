"""Build a standalone geometric replay from the actual MATLAB path output."""
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/static_reference_20261003'

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tesla · Static planning reference</title><style>
:root{color-scheme:dark;--bg:#0e1624;--panel:#172235;--line:#344258;--text:#eef3f9;--muted:#b5c2d3;--blue:#62b5ff;--green:#75e2b3;--amber:#f7bc62}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,sans-serif}main{max-width:1260px;margin:auto;padding:30px 24px 42px}header{display:flex;gap:18px;justify-content:space-between;align-items:center;margin-bottom:20px}h1{margin:4px 0 8px;font-size:clamp(24px,3vw,36px);letter-spacing:-.8px}p{margin:0;color:var(--muted)}.eyebrow{color:var(--green);text-transform:uppercase;font-size:12px;font-weight:700;letter-spacing:1.5px}.badge{border:1px solid #806542;background:#382b1c;padding:7px 12px;border-radius:20px;color:#ffda9e;white-space:nowrap;font-size:13px}.stage{border:1px solid var(--line);background:var(--panel);border-radius:16px;overflow:hidden}.canvas-wrap{height:clamp(280px,43vw,540px);background:#111c2b}canvas{width:100%;height:100%;display:block}.controls{display:flex;gap:12px;align-items:center;padding:18px;flex-wrap:wrap;border-top:1px solid var(--line)}button,select{font:inherit;border-radius:9px;border:1px solid #52647e;background:#22334c;color:var(--text);padding:9px 16px;cursor:pointer}button.primary{background:var(--blue);border-color:var(--blue);color:#082039;font-weight:700}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid var(--green);outline-offset:3px}.seek{flex:1;min-width:220px;display:flex;align-items:center;gap:12px}input[type=range]{width:100%;accent-color:var(--blue)}output{font-variant-numeric:tabular-nums;min-width:98px;font-size:14px;color:var(--muted)}.state{color:var(--green);font-size:14px;min-width:92px}.legend{display:flex;gap:22px;flex-wrap:wrap;padding:12px 18px;color:var(--muted);font-size:13px;border-top:1px solid var(--line)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.stat{border:1px solid var(--line);border-radius:12px;padding:16px;background:var(--panel)}.stat strong{display:block;font-size:24px;color:var(--text);letter-spacing:-.6px}.stat span{font-size:13px;color:var(--muted)}.explain{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{padding:20px;border:1px solid var(--line);border-radius:12px}.card h2{font-size:18px;margin:0 0 8px}.card p{font-size:14px}.foot{margin-top:20px;font-size:12px;color:#93a3b9}details{margin-top:18px;color:var(--muted);font-size:14px}summary{cursor:pointer;color:var(--text)}dl{display:grid;grid-template-columns:180px 1fr;gap:6px 12px}dt{color:#93a3b9}dd{margin:0}a{color:var(--blue)}@media(max-width:700px){main{padding:20px 12px}header{align-items:flex-start;flex-direction:column}.stats{grid-template-columns:1fr 1fr}.explain{grid-template-columns:1fr}.controls{gap:10px}.seek{order:5;flex-basis:100%}.canvas-wrap{height:310px}dl{grid-template-columns:1fr}dd{margin-bottom:10px}}
</style></head><body><main>
<header><div><div class="eyebrow">Tesla project · MATLAB reference</div><h1>A car-sized path around an obstacle</h1><p>Play the saved Hybrid A* route and inspect the car's position.</p></div><span class="badge">Ground-truth map · geometric replay</span></header>
<section class="stage" aria-label="Static planned car replay"><div class="canvas-wrap"><canvas id="scene" aria-label="Top-down metre map with ego car, parked obstacle, and planned path"></canvas></div>
<div class="controls"><button id="play" class="primary">Play</button><button id="reset">Reset</button><label for="speed">Playback</label><select id="speed"><option value="1">1×</option><option value="4">4×</option><option value="16">16×</option></select><span class="state" id="state" role="status">Ready</span><div class="seek"><label for="seek" class="sr">Position</label><input id="seek" type="range" min="0" max="1" step="0.001" value="0" aria-label="Replay position"><output id="clock">0.0 / 15.3 s</output></div></div>
<div class="legend"><span><i class="dot" style="background:var(--green)"></i>Ego car</span><span><i class="dot" style="background:var(--amber)"></i>Parked obstacle</span><span><i class="dot" style="background:var(--blue)"></i>Planned rear axle path</span><span>Map: 60 × 30 metres</span></div></section>
<section class="stats" aria-label="Measured reference results"><div class="stat"><strong id="length"></strong><span>Path length</span></div><div class="stat"><strong id="planning"></strong><span>Single planning call</span></div><div class="stat"><strong id="obstacle"></strong><span>Minimum sampled obstacle clearance</span></div><div class="stat"><strong id="road"></strong><span>Minimum sampled road clearance</span></div></section>
<section class="explain"><article class="card"><h2>What this proves</h2><p>MATLAB found a forward path that respects the car's turning radius. Its saved poses were checked against the full rotated vehicle rectangle and the road boundary.</p></article><article class="card"><h2>What happens next</h2><p>This animation places the car on the planned poses at a nominal 10.8 km/h. Steering and speed controllers, moving actors, and RL STOP / SLOW / PROCEED still need implementation and testing.</p></article></section>
<details><summary>Scene settings and scope</summary><dl><dt>Pose reference</dt><dd>Rear axle centre. World x/y in metres; heading in radians.</dd><dt>Vehicle</dt><dd>4.5 m long, 1.8 m wide, 2.7 m wheelbase. Minimum turning radius 6 m.</dd><dt>Checks</dt><dd>0.25 m requested safety margin; planning and exported-pose checks at about 0.05 m spacing. Sampled checks are not a continuous-time driving proof.</dd><dt>Map source</dt><dd>Declared development geometry. The U-Net camera mask is not connected to this map; camera calibration is still required.</dd><dt>Replay speed</dt><dd>Playback timing only. This is not measured controlled vehicle speed.</dd><dt>Evidence</dt><dd><a href="result.json">Measured results</a> · <a href="reference_path.csv">Path CSV</a> · <a href="scene_config.json">Exact scene settings</a></dd></dl></details>
<p class="foot">One static development scene. No RL training or final multi-scenario driving evaluation has run. The three trained segmentation models and their final test result remain fixed.</p>
</main><script>
'use strict';
const replayData=__REPLAY_DATA__, measuredResults=__RESULT_DATA__;
const config=replayData.config, poses=replayData.poses, distances=replayData.distance_m;
const nominalSpeed=replayData.nominal_replay_speed_mps||3;
const times=replayData.time_s||distances.map(d=>d/nominalSpeed), duration=times.at(-1), plannedPoses=replayData.reference_poses||poses;
const canvas=document.getElementById('scene'), ctx=canvas.getContext('2d');
const playButton=document.getElementById('play'), seek=document.getElementById('seek'), speed=document.getElementById('speed');
let replayTime=0, playing=false, lastTick=null, cssWidth=0, cssHeight=0;
document.getElementById('length').textContent=measuredResults.path_length_m.toFixed(2)+' m';
document.getElementById('planning').textContent=(replayData.controller_used?measuredResults.simulation_time_s.toFixed(1):measuredResults.plan_wall_seconds.toFixed(3))+' s';
document.getElementById('obstacle').textContent=measuredResults.minimum_obstacle_footprint_clearance_m.toFixed(2)+' m';
document.getElementById('road').textContent=measuredResults.minimum_road_footprint_clearance_m.toFixed(2)+' m';
function poseAt(time){
 const queryTime=Math.max(0,Math.min(time,duration));
 let low=0, high=distances.length-1;
 while(low<high){let mid=Math.floor((low+high)/2); if(times[mid]<queryTime)low=mid+1;else high=mid;}
 const index=low;if(index===0)return [...poses[0]];const a=index-1, span=times[index]-times[a];
 const fraction=span>0?(queryTime-times[a])/span:0;
 const dh=Math.atan2(Math.sin(poses[index][2]-poses[a][2]),Math.cos(poses[index][2]-poses[a][2]));
 return [poses[a][0]+fraction*(poses[index][0]-poses[a][0]),poses[a][1]+fraction*(poses[index][1]-poses[a][1]),poses[a][2]+fraction*dh];
}
function setPlaying(value){playing=value;lastTick=null;playButton.textContent=value?'Pause':replayTime>=duration?'Replay':'Play';document.getElementById('state').textContent=value?'Playing':replayTime>=duration?'Finished':replayTime>0?'Paused':'Ready';}
function draw(){
 if(!cssWidth)return;const padding=42, scale=Math.min((cssWidth-padding*2)/60,(cssHeight-padding*2)/30);
 const left=(cssWidth-60*scale)/2, top=(cssHeight-30*scale)/2;
 const px=x=>left+x*scale, py=y=>top+(30-y)*scale;
 ctx.clearRect(0,0,cssWidth,cssHeight);ctx.fillStyle='#111c2b';ctx.fillRect(0,0,cssWidth,cssHeight);
 ctx.fillStyle='#244236';ctx.fillRect(px(0),py(30),60*scale,30*scale);
 let [x1,y1,x2,y2]=config.road_bounds_xyxy_m;ctx.fillStyle='#334052';ctx.fillRect(px(x1),py(y2),(x2-x1)*scale,(y2-y1)*scale);
 ctx.strokeStyle='#c7d2df';ctx.lineWidth=1.4;ctx.strokeRect(px(x1),py(y2),(x2-x1)*scale,(y2-y1)*scale);
 ctx.font='11px system-ui';ctx.fillStyle='#a9b9cb';ctx.textAlign='center';ctx.strokeStyle='rgba(170,194,212,.12)';ctx.lineWidth=.7;
 for(let x=0;x<=60;x+=10){ctx.beginPath();ctx.moveTo(px(x),py(0));ctx.lineTo(px(x),py(30));ctx.stroke();ctx.fillText(x+' m',px(x),py(0)+19);}
 ctx.textAlign='right';for(let y=0;y<=30;y+=5){ctx.beginPath();ctx.moveTo(px(0),py(y));ctx.lineTo(px(60),py(y));ctx.stroke();ctx.fillText(y+' m',px(0)-8,py(y)+4);}
 [x1,y1,x2,y2]=config.obstacle_bounds_xyxy_m;ctx.fillStyle='#f7bc62';ctx.fillRect(px(x1),py(y2),(x2-x1)*scale,(y2-y1)*scale);ctx.strokeStyle='#694717';ctx.lineWidth=2;ctx.strokeRect(px(x1),py(y2),(x2-x1)*scale,(y2-y1)*scale);
 ctx.fillStyle='#e9d4b3';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText('Parked car',px((x1+x2)/2),py(y2)-12);
 ctx.strokeStyle='#62b5ff';ctx.lineWidth=2.3;ctx.setLineDash([6,4]);ctx.beginPath();plannedPoses.forEach((p,i)=>i?ctx.lineTo(px(p[0]),py(p[1])):ctx.moveTo(px(p[0]),py(p[1])));ctx.stroke();ctx.setLineDash([]);
 const current=poseAt(replayTime);
 ctx.strokeStyle='#75e2b3';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(px(poses[0][0]),py(poses[0][1]));for(let i=1;i<poses.length&&times[i]<=replayTime;i++)ctx.lineTo(px(poses[i][0]),py(poses[i][1]));ctx.lineTo(px(current[0]),py(current[1]));ctx.stroke();
 for(const [label,p,color] of [['Start',config.start_pose,'#75e2b3'],['Goal',config.goal_pose,'#62b5ff']]){ctx.fillStyle=color;ctx.beginPath();ctx.arc(px(p[0]),py(p[1]),4,0,Math.PI*2);ctx.fill();ctx.fillText(label,px(p[0]),py(p[1])+30);}
 ctx.save();ctx.translate(px(current[0]),py(current[1]));ctx.rotate(-current[2]);ctx.fillStyle='#75e2b3';ctx.strokeStyle='#c9ffe9';ctx.lineWidth=1.5;ctx.fillRect(-.9*scale,-.9*scale,4.5*scale,1.8*scale);ctx.strokeRect(-.9*scale,-.9*scale,4.5*scale,1.8*scale);ctx.fillStyle='#203e42';ctx.fillRect(1.9*scale,-.67*scale,.75*scale,1.34*scale);ctx.fillStyle='#092b27';ctx.beginPath();ctx.arc(0,0,2.8,0,Math.PI*2);ctx.fill();ctx.restore();
 ctx.textAlign='left';ctx.fillStyle='#edf4fb';ctx.font='12px system-ui';
 const speedIndex=Math.min(times.length-1,Math.round(replayTime/(times[1]-times[0])));
 const speedText=replayData.speed_mps?' · Speed '+(replayData.speed_mps[speedIndex]*3.6).toFixed(1)+' km/h':'';
 ctx.fillText('Rear axle: x '+current[0].toFixed(2)+' m · y '+current[1].toFixed(2)+' m'+speedText,16,22);
 seek.value=(replayTime/duration).toString();document.getElementById('clock').textContent=replayTime.toFixed(1)+' / '+duration.toFixed(1)+' s';
}
function resize(){const rect=canvas.getBoundingClientRect(), ratio=window.devicePixelRatio||1;cssWidth=rect.width;cssHeight=rect.height;canvas.width=Math.round(cssWidth*ratio);canvas.height=Math.round(cssHeight*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);draw();}
playButton.addEventListener('click',()=>{if(replayTime>=duration)replayTime=0;setPlaying(!playing);draw();});
document.getElementById('reset').addEventListener('click',()=>{replayTime=0;setPlaying(false);draw();});
seek.addEventListener('input',()=>{replayTime=Number(seek.value)*duration;setPlaying(false);draw();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)setPlaying(false);});
function tick(timestamp){if(playing){if(lastTick!==null)replayTime=Math.min(duration,replayTime+(timestamp-lastTick)/1000*Number(speed.value));if(replayTime>=duration)setPlaying(false);else lastTick=timestamp;draw();}requestAnimationFrame(tick);}
new ResizeObserver(resize).observe(canvas);resize();requestAnimationFrame(tick);
</script></body></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    args=parser.parse_args();run=args.run_dir.resolve()
    tracking=(run/'tracking_replay.json').is_file()
    replay = json.loads((run / ('tracking_replay.json' if tracking else 'replay.json')).read_text(encoding='utf-8'))
    results = json.loads((run / ('tracking_result.json' if tracking else 'result.json')).read_text(encoding='utf-8'))
    assert results['status'] == 'completed' and results['goal_reached' if tracking else 'path_found']
    assert not replay['rl_used'] and not replay['learned_mask_used']
    template=HTML
    if tracking:
        import math
        reference=json.loads((RUN/'replay.json').read_text(encoding='utf-8'))
        replay['reference_poses']=reference['poses']
        distances=[0.0]
        for before,after in zip(replay['poses'],replay['poses'][1:]):
            distances.append(distances[-1]+math.hypot(after[0]-before[0],after[1]-before[1]))
        replay['distance_m']=distances
        results['path_length_m']=distances[-1]
        template=template.replace('Static planning reference','Static controller reference').replace('A car-sized path around an obstacle','The car follows its planned route')
        template=template.replace("Play the saved Hybrid A* route and inspect the car's position.","Watch the recorded steering and speed controller follow the Hybrid A* route.")
        template=template.replace('Ground-truth map · geometric replay','Ground-truth map · closed-loop trace')
        template=template.replace('Path length</span>','Driven trace length</span>').replace('Single planning call','Simulated drive time')
        template=template.replace("MATLAB found a forward path that respects the car's turning radius. Its saved poses were checked against the full rotated vehicle rectangle and the road boundary.","The Stanley steering and PI speed controller reached the goal in this one static scene. Saved motion obeyed the kinematic bicycle equations, and the full vehicle body passed the sampled clearance checks.")
        template=template.replace('This animation places the car on the planned poses at a nominal 10.8 km/h. Steering and speed controllers, moving actors, and RL STOP / SLOW / PROCEED still need implementation and testing.', 'The blue line is planned; the green trail is recorded simulated motion. Moving actors, independent emergency braking, and RL STOP / SLOW / PROCEED remain the next stages. Camera-mask projection still needs calibration.')
        template=template.replace('0.25 m requested safety margin; planning and exported-pose checks at about 0.05 m spacing. Sampled checks are not a continuous-time driving proof.', '0.25 m required margin. Controller updates every 0.1 s; actual body checked every 0.01 s. Sampled checks are not a continuous-time driving proof.')
        template=template.replace('Playback timing only. This is not measured controlled vehicle speed.','Playback can run faster than the recorded simulation clock. The on-map speed comes from the actual saved controller trace.')
        template=template.replace('result.json','tracking_result.json').replace('reference_path.csv','tracking_trace.csv').replace('scene_config.json','tracking_config.json')
        template=template.replace('One static development scene. No RL training or final multi-scenario driving evaluation has run.', 'One static development drive, not a multi-scenario safety result. No RL training has run.')
    else:
        assert not replay['controller_used']
        assert len(replay['poses']) == len(replay['distance_m']) == results['path_poses']
    html = template.replace('__REPLAY_DATA__', json.dumps(replay).replace('</', '<\\/')).replace('__RESULT_DATA__', json.dumps(results).replace('</', '<\\/'))
    output = run / 'replay.html'
    assert not output.exists(), 'Preserve the existing replay artifact.'
    output.write_text(html, encoding='utf-8')
    print(json.dumps({'saved':str(output),'poses':len(replay['poses']),'duration_s':replay['time_s'][-1] if tracking else replay['distance_m'][-1]/3,'controller_used':tracking}))


if __name__ == '__main__':
    main()
