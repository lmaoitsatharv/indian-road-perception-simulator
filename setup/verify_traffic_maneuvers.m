function verify_traffic_maneuvers(outputDir)
% Deterministic preflight gates at the real recorded overtake start state.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'integration','matlab'));
assert(~isfolder(outputDir),'Preserve earlier verification output.');mkdir(outputDir);
d=load(fullfile(root,'runs','traffic_overnight_20261004','overtake_clear_v2','world.mat'));
eventIndex=find(strcmp({d.events.event},'overtake_started'),1);
time=d.events(eventIndex).time_s;[~,frame]=min(abs(d.egoLog(:,1)-time));
e=d.egoLog(frame,:);actors=squeeze(d.actorLog(frame,:,:));specs=d.specs;cfg=d.config;
m=struct('mode',0,'started',0,'completed',0,'cooldown',0,'last_rejection','', ...
    'start_progress',0,'out_length',16,'return_start',0,'total_length',0,'pass_offset',-1, ...
    'lead_id',0,'start_time',0);
s=d.events(eventIndex).s_m;pose=e(2:4);v=e(5);progress=e(9);
checks=struct();
[accepted,note]=traffic_overtake_plan(pose,s,progress,v,actors,specs,cfg,m,time);
assert(strcmp(note,'overtake_started'));checks.clear_straight_accepts=true;
altered=actors;[xy,h]=indian_loop_pose(s+55,-2.5);
altered(2,:)=[xy,h+pi,-3*cos(h),-3*sin(h),0,s+55];
[state,~]=traffic_overtake_plan(pose,s,progress,v,altered,specs,cfg,m,time);
assert(state.mode==0 && strcmp(state.last_rejection,'oncoming_traffic'));
checks.oncoming_rejected=true;
altered=actors;[xy,h]=indian_loop_pose(s+25,6.5);
altered(cfg.vehicle_count+1,:)=[xy,h,0,-1,1,s+25];
[state,~]=traffic_overtake_plan(pose,s,progress,v,altered,specs,cfg,m,time);
assert(state.mode==0 && strcmp(state.last_rejection,'pedestrian_near_manoeuvre'));
checks.pedestrian_rejected=true;
altered=actors;q=accepted.return_start+8;offset=traffic_maneuver_offset(progress+q,accepted);
[xy,h]=indian_loop_pose(s+q,offset);altered(end,:)=[xy,h,0,0,0,s+q];
[state,~]=traffic_overtake_plan(pose,s,progress,v,altered,specs,cfg,m,time);
assert(state.mode==0 && strcmp(state.last_rejection,'predicted_vehicle_or_return_conflict'));
checks.blocked_return_rejected=true;
altered=actors;q=accepted.out_length;[xy,h]=indian_loop_pose(s+q,accepted.pass_offset);
altered(end,:)=[xy,h,0,0,0,s+q];
[state,~]=traffic_overtake_plan(pose,s,progress,v,altered,specs,cfg,m,time);
assert(state.mode==0 && strcmp(state.last_rejection,'predicted_vehicle_or_return_conflict'));
checks.occupied_passing_corridor_rejected=true;
[xy,h]=indian_loop_pose(95,2.5);
[state,~]=traffic_overtake_plan([xy,h],95,95,v,actors,specs,cfg,m,time);
assert(state.mode==0 && strcmp(state.last_rejection,'curve'));checks.curve_rejected=true;
altered=actors;[xy,h]=indian_loop_pose(70,2.5);poseEnd=[xy,h];
[xy,h]=indian_loop_pose(80,2.5);altered(1,:)=[xy,h,.85,0,0,80];
[state,~]=traffic_overtake_plan(poseEnd,70,70,v,altered,specs,cfg,m,time);
assert(state.mode==0 && strcmp(state.last_rejection,'insufficient_straight_distance'));
checks.short_straight_rejected=true;
[xy,h]=indian_loop_pose(s+accepted.total_length,accepted.pass_offset);
[state,note]=traffic_overtake_plan([xy,h],s+accepted.total_length,progress+accepted.total_length,3,actors,specs,cfg,accepted,time+30);
assert(state.completed==0 && isempty(note));checks.completion_requires_actual_return=true;
[xy,h]=indian_loop_pose(s+accepted.total_length,2.5);
[state,note]=traffic_overtake_plan([xy,h],s+accepted.total_length,progress+accepted.total_length,3,actors,specs,cfg,accepted,time+30);
assert(state.completed==1 && strcmp(note,'overtake_completed'));checks.actual_pass_and_return_completes=true;
result=struct('status','passed','release',version('-release'),'checks',checks, ...
    'limits','Snapshot gate checks, not a proof for changing actors during execution. Dynamic rollout evidence is reported separately.');
fid=fopen(fullfile(outputDir,'verification.json'),'w','n','UTF-8');fprintf(fid,'%s\n',jsonencode(result));fclose(fid);
copyfile(mfilename('fullpath')+".m",outputDir);
copyfile(fullfile(root,'integration','matlab','traffic_overtake_plan.m'),outputDir);
disp(jsonencode(result));
end
