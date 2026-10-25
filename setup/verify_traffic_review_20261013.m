function verify_traffic_review_20261013(outputDir,runTrials)
% Regression evidence on historical traffic failures; no segmentation data.
arguments
    outputDir (1,:) char
    runTrials (1,1) logical=false
end
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'integration','matlab'));
assert(~isfolder(outputDir),'Choose a unique verification directory.');mkdir(outputDir);
checks=struct();
d=load(fullfile(root,'runs','traffic_overnight_20261004','mixed_seed2026_v6','world.mat'));
frame=find(abs(d.egoLog(:,1)-68)<.001,1);states=squeeze(d.actorLog(frame,:,:));
[~,h]=indian_loop_pose(states(2,7));[anchor,~]=indian_loop_pose(states(2,7));
offset=dot(states(2,1:2)-anchor,[-sin(h),cos(h)]);
speed=norm(states(2,4:5));
[target,blocker,gap]=traffic_npc_speed_target(2,states,d.specs,d.config,states(2,7),offset,speed,d.specs(2).nominal_speed_mps);
assert(target<speed-.2 && blocker==11 && gap>0,'Curved-road stopped leader did not constrain the trailing full body.');
checks.bend_body_following=struct('passed',true,'saved_time_s',68,'old_speed_mps',speed, ...
    'new_target_mps',target,'blocking_actor_id',blocker,'full_body_path_gap_m',gap);
before=squeeze(d.actorLog(find(abs(d.egoLog(:,1)-70)<.001,1),[2,11],:));
proposal=squeeze(d.actorLog(find(abs(d.egoLog(:,1)-72)<.001,1),[2,11],:));
assert(traffic_body_gap(before(1,:),d.specs(2),before(2,:),d.specs(11),0)>0);
assert(traffic_body_gap(proposal(1,:),d.specs(2),proposal(2,:),d.specs(11),0)<=0);
[accepted,held,~]=traffic_npc_step_guard(before,proposal,d.specs([2,11]),2);
assert(held(1) && traffic_body_gap(accepted(1,:),d.specs(2),accepted(2,:),d.specs(11),0)>0);
checks.prospective_overlap_held_and_not_relocated=true;
withEgo=[d.specs,d.specs(1)];withEgo(end).id=0;withEgo(end).name='ego';
entry=traffic_npc_intervention(1,2,numel(withEgo),withEgo,2,.04,.002);
assert(entry.actor_id==2 && entry.blocking_actor_id==0 && entry.accepted_route_step_m==0 && entry.accepted_lateral_step_m==0);
checks.npc_intervention_uses_ego_id_zero=true;
% Invalid options must fail before creating even a partial output folder.
invalidDir=fullfile(outputDir,'invalid_inputs_must_not_create');
didFail=false;try,run_indian_traffic_trial(invalidDir,42,1,'Scenario','typo');catch,didFail=true;end
assert(didFail && ~isfolder(invalidDir));checks.unknown_scenario_rejected=true;
didFail=false;try,run_indian_traffic_trial(invalidDir,42,inf);catch,didFail=true;end
assert(didFail && ~isfolder(invalidDir));checks.nonfinite_duration_rejected=true;
didFail=false;try,run_indian_traffic_trial(invalidDir,42,.15);catch,didFail=true;end
assert(didFail && ~isfolder(invalidDir));checks.partial_control_step_duration_rejected=true;
% The public compatibility entrypoint must execute the same reviewed policy,
% rather than retain a second divergent traffic implementation.
smokeDir=fullfile(outputDir,'public_wrapper_smoke');
smoke=run_indian_traffic_world(smokeDir,42,.2);
smokeConfig=jsondecode(fileread(fullfile(smokeDir,'world_config.json')));
assert(strcmp(smokeConfig.revision,'traffic_trial_v8_body_following_persistent_detours') && ...
    smokeConfig.independent_vehicle_directions && smokeConfig.npc_parked_clearance_enabled && ...
    smokeConfig.smooth_following_enabled && smokeConfig.directional_pedestrian_passing && ...
    ~smokeConfig.overtaking_enabled && abs(smoke.simulation_time_s-.2)<1e-8);
checks.public_world_entrypoint_uses_reviewed_policy=true;
configFile=fullfile(smokeDir,'world_config.json');
fid=fopen(configFile,'rb');assert(fid>=0);originalBytes=fread(fid,inf,'*uint8');fclose(fid);
didFail=false;try,run_indian_traffic_world(smokeDir,42,.2);catch,didFail=true;end
fid=fopen(configFile,'rb');assert(fid>=0);afterBytes=fread(fid,inf,'*uint8');fclose(fid);
assert(didFail && isequal(originalBytes,afterBytes));
checks.public_wrapper_refuses_existing_output_without_overwrite=true;
% A pedestrian disc at a rectangle corner must agree with the physical circle
% rule, rather than report a collision from an enclosing square alone.
car=struct('kind','car','length_m',4.2,'width_m',1.75,'rear_overhang_m',.84);
ped=struct('kind','pedestrian','width_m',.5);
assert(traffic_body_gap([3.56,1.075,0],ped,[0,0,0],car,0)>0);
checks.pedestrian_disc_corner_clearance=true;
q=load(fullfile(root,'runs','traffic_overnight_20261004','mixed_seed7_v6','world.mat'));
frame=find(abs(q.egoLog(:,1)-68)<.001,1);all=squeeze(q.actorLog(frame,:,:));
ids=[1:q.config.vehicle_count,q.config.vehicle_count+q.config.pedestrian_count+(1:3)];
pedState=all(q.config.vehicle_count+6,:);[xy,h]=indian_loop_pose(pedState(7));
offset=dot(pedState(1:2)-xy,[-sin(h),cos(h)]);
[targetS,found]=traffic_pedestrian_detour_target(pedState(7),offset,-7,all(ids,:),q.specs(ids),q.config.loop_length_m);
delta=mod(targetS-pedState(7)+q.config.loop_length_m/2,q.config.loop_length_m)-q.config.loop_length_m/2;
expected=q.actorLog(frame,3,7)+q.specs(3).length_m-q.specs(3).rear_overhang_m+.85;
assert(found && delta>.1 && abs(targetS-expected)<.01, ...
    'Seed-7 cluster must clear the stopped car front before crossing inward.');
checks.seed7_cluster_detour=struct('passed',true,'from_s_m',pedState(7),'target_s_m',targetS,'detour_distance_m',delta);
% Preserve the existing initiation, body-clearance and completion gate suite.
verify_traffic_maneuvers(fullfile(outputDir,'maneuver_gates'));
clearRun=load(fullfile(root,'runs','traffic_overnight_20261004','overtake_clear_v2','world.mat'));
eventIndex=find(strcmp({clearRun.events.event},'overtake_started'),1);
time=clearRun.events(eventIndex).time_s;routeS=clearRun.events(eventIndex).s_m;
[~,frame]=min(abs(clearRun.egoLog(:,1)-time));e=clearRun.egoLog(frame,:);
actors=squeeze(clearRun.actorLog(frame,:,:));
m=struct('mode',0,'started',0,'completed',0,'cooldown',0,'last_rejection','', ...
    'start_progress',0,'out_length',16,'return_start',0,'total_length',0,'pass_offset',-1, ...
    'lead_id',0,'start_time',0);
[active,note]=traffic_overtake_plan(e(2:4),routeS,e(9),e(5),actors,clearRun.specs,clearRun.config,m,time);
assert(strcmp(note,'overtake_started'));
[xy,h]=indian_loop_pose(routeS+active.total_length,2.5);
actors(end,:)=[xy+4.5*[cos(h),sin(h)],h,0,0,0,routeS+active.total_length+4.5];
[blocked,note]=traffic_overtake_plan([xy,h],routeS+active.total_length,e(9)+active.total_length,3, ...
    actors,clearRun.specs,clearRun.config,active,time+30);
assert(blocked.completed==0 && isempty(note),'Completion must verify return clearance to actors other than the lead.');
checks.actual_return_gap_against_nonlead_actor=true;
result=struct('status','passed','release',version('-release'),'checks',checks, ...
    'limits','Historical-state checks and sampled proposal constraints. They do not establish general traffic safety.');
fid=fopen(fullfile(outputDir,'verification.json'),'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(result));clear cleanup
copyfile(mfilename('fullpath')+".m",outputDir);
if runTrials
    for seed=[42,7,2026]
        run_indian_traffic_trial(fullfile(outputDir,['mixed_seed',num2str(seed),'_v8']),seed,300, ...
            'IndependentDirections',true,'ParkedClearance',true,'SmoothFollowing',true,'PedestrianPassing',true);
    end
    run_indian_traffic_trial(fullfile(outputDir,'overtake_clear_v8'),42,80,'Scenario','clear','Overtaking',true,'SmoothFollowing',true);
    run_indian_traffic_trial(fullfile(outputDir,'overtake_oncoming_v8'),42,80,'Scenario','oncoming','Overtaking',true,'SmoothFollowing',true);
    run_indian_traffic_trial(fullfile(outputDir,'overtake_crossing_v8'),42,80,'Scenario','crossing','Overtaking',true,'SmoothFollowing',true);
end
disp(jsonencode(result));
end
