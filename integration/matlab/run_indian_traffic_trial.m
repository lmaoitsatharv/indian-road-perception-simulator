function result=run_indian_traffic_trial(outputDir,seed,duration,options)
%RUN_INDIAN_TRAFFIC_WORLD Ground-truth mixed-traffic loop development world.
% Ego: actual Stanley steering + bounded speed feedback + braking override.
% Other vehicles: prescribed loop positions with car-following speed rules.
% Pedestrians: seeded footpath walking / waiting / crossing state machine.
% Not a calibrated Indian traffic model, RL training, or learned-mask planner.
arguments
    outputDir (1,:) char = ''
    seed (1,1) double {mustBeInteger,mustBeNonnegative,mustBeFinite}=42
    duration (1,1) double {mustBePositive,mustBeFinite}=300
    options.Scenario (1,1) string = "mixed"
    options.Overtaking (1,1) logical = false
    options.IndependentDirections (1,1) logical = false
    options.ParkedClearance (1,1) logical = false
    options.SmoothFollowing (1,1) logical = false
    options.PedestrianPassing (1,1) logical = false
end
assert(any(options.Scenario==["mixed","clear","oncoming","crossing","opposing_pedestrians"]), ...
    'Unknown traffic Scenario. Choose mixed, clear, oncoming, crossing or opposing_pedestrians.');
assert(duration>=.1 && abs(duration/.1-round(duration/.1))<1e-8, ...
    'Duration must use at least one whole 0.1-second control step.');
projectDir=fileparts(fileparts(fileparts(mfilename('fullpath'))));
if isempty(outputDir),outputDir=fullfile(projectDir,'runs',['indian_traffic_',datestr(now,'yyyymmdd_HHMMSS'),'_seed',num2str(seed)]);end
assert(~exist(outputDir,'dir'),'Choose a new run directory; earlier results are preserved.');
assert(exist('lateralControllerStanley','file')==2,'MATLAB Stanley controller is unavailable.');
mkdir(outputDir);rng(seed,'twister');
config=struct('kind','synthetic_indian_traffic_inspired_development_world', ...
    'seed',seed,'duration_s',duration,'control_dt_s',.1,'integration_dt_s',.02, ...
    'world_size_m',[160,120],'road_width_m',11,'footpath_width_m',2.5, ...
    'ego_lane_offset_m',2.5,'ego_cruise_mps',3,'slow_speed_mps',1.2, ...
    'wheelbase_m',2.7,'max_acceleration_mps2',1.5,'max_deceleration_mps2',3, ...
    'max_steering_rad',atan(2.7/6),'max_steering_rate_radps',pi/4, ...
    'prediction_horizon_s',4,'prediction_sample_s',.2,'stop_ttc_s',2.8, ...
    'safety_margin_m',.45,'vehicle_count',16,'pedestrian_count',10, ...
    'traffic_side','left','world_frame','fixed_x_right_y_up_metres', ...
    'pose_reference','vehicle_rear_axle_pedestrian_center','heading_units','radians', ...
    'route_source','fixed_ground_truth_loop_reference; no Hybrid A star search in this world', ...
    'behavior_policy','rule_baseline_STOP_SLOW_PROCEED_with_independent_braking_override', ...
    'actor_prediction','constant_velocity_for_local_ego_decisions', ...
    'rl_used',false,'learned_mask_used',false, ...
    'note','Synthetic moderate-speed development scene; traffic distributions are illustrative, not fitted to Indian-road data.');
config.revision='traffic_trial_v8_body_following_persistent_detours';
config.scenario=char(options.Scenario);config.overtaking_enabled=options.Overtaking;
config.independent_vehicle_directions=options.IndependentDirections;
config.npc_parked_clearance_enabled=options.ParkedClearance;
config.npc_parked_lateral_clearance_m=.8;
config.smooth_following_enabled=options.SmoothFollowing;
config.directional_pedestrian_passing=options.PedestrianPassing;
config.follow_standstill_gap_m=2.8;config.follow_headway_s=1.3;
config.follow_gap_gain=.5;config.nominal_jerk_limit_mps3=2;
config.npc_following='full_body_route_corridor_with_0_55m_margin';
config.npc_step_guard='prospective_middle_and_end_body_checks_with_0_08m_margin; holds are explicitly logged';
config.pedestrian_detours='persistent_target_clearing_full_stopped_vehicle_cluster';
if options.Scenario~="mixed",config.vehicle_count=2;config.pedestrian_count=1;end
if options.Scenario=="opposing_pedestrians",config.pedestrian_count=2;end
writeJSON(fullfile(outputDir,'world_config.json'),config);
sourceDir=fullfile(outputDir,'source_snapshot');mkdir(sourceDir);
copyfile(mfilename('fullpath')+".m",fullfile(sourceDir,'run_indian_traffic_trial.m'));
copyfile(fullfile(fileparts(mfilename('fullpath')),'indian_loop_pose.m'),sourceDir);
copyfile(fullfile(fileparts(mfilename('fullpath')),'traffic_overtake_plan.m'),sourceDir);
copyfile(fullfile(fileparts(mfilename('fullpath')),'traffic_maneuver_offset.m'),sourceDir);
copyfile(fullfile(fileparts(mfilename('fullpath')),'traffic_follow_speed.m'),sourceDir);
copyfile(fullfile(fileparts(mfilename('fullpath')),'traffic_pedestrian_step_score.m'),sourceDir);
for helper={'traffic_body_gap.m','traffic_npc_speed_target.m','traffic_npc_step_guard.m','traffic_npc_intervention.m','traffic_pedestrian_detour_target.m'}
    copyfile(fullfile(fileparts(mfilename('fullpath')),helper{1}),sourceDir);
end
[~,~,~,loopLength]=indian_loop_pose(0);config.loop_length_m=loopLength;
routeS=(0:.25:loopLength-.001)';
[routeXY,routeHeading]=indian_loop_pose(routeS,config.ego_lane_offset_m);
frontXY=routeXY+config.wheelbase_m*[cos(routeHeading),sin(routeHeading)];
frontTangent=circshift(frontXY,-1)-circshift(frontXY,1);
frontHeading=atan2(frontTangent(:,2),frontTangent(:,1));
route=[routeS,routeXY,routeHeading];
% Fixed numbered actor slots with seeded type/direction/speed variation.
specs=repmat(struct('id',0,'name','','kind','','length_m',0,'width_m',0, ...
    'rear_overhang_m',0,'height_m',0,'color',[0,0,0],'direction',0, ...
    'base_offset_m',0,'nominal_speed_mps',0),1,config.vehicle_count+config.pedestrian_count+3);
order=randperm(config.vehicle_count); kinds={'car','auto_rickshaw','two_wheeler'};
directions=ones(1,config.vehicle_count);
if options.IndependentDirections,directions(randperm(config.vehicle_count,ceil(config.vehicle_count/3)))=-1;end
npcS=zeros(1,config.vehicle_count);npcV=zeros(size(npcS));npcD=zeros(size(npcS));
phase=2*pi*rand(size(npcS));amplitude=zeros(size(npcS));
for i=1:config.vehicle_count
    kind=kinds{mod(order(i)-1,3)+1};direction=1;if mod(order(i),3)==0,direction=-1;end
    if options.IndependentDirections,direction=directions(i);end
    if strcmp(kind,'car'),dims=[4.2,1.75,1.5];color=[.22,.57,.94];base=2.5;cruise=2.5+rand*1.4;amp=.12;
    elseif strcmp(kind,'auto_rickshaw'),dims=[2.6,1.35,1.75];color=[.96,.73,.14];base=2.8;cruise=1.8+rand*1.2;amp=.18;
    else,dims=[1.9,.65,1.3];color=[.76,.38,.93];base=1.05+rand*2;cruise=3+rand*1.6;amp=.25;end
    specs(i)=makeSpec(i,[kind,'_',num2str(i)],kind,dims,.2*dims(1),color,direction,direction*base,cruise);
    npcS(i)=mod(25+(i-1)*loopLength/config.vehicle_count+(rand-.5)*4,loopLength);
    if options.IndependentDirections
        % Reserve an empty start corridor. The former evenly spaced ring could
        % wrap the last positive-direction vehicle directly onto the ego car.
        npcS(i)=32+(i-1)*(loopLength-64)/max(1,config.vehicle_count-1)+(rand-.5)*4;
    end
    npcV(i)=cruise;npcD(i)=direction*base;amplitude(i)=amp;
end
np=config.pedestrian_count;pedS=zeros(1,np);pedD=zeros(1,np);pedSpeed=.85+.65*rand(1,np);
pedDir=2*(rand(1,np)>.5)-1;pedMode=zeros(1,np);nextCross=6+35*rand(1,np);
crossTarget=zeros(1,np);pauseUntil=zeros(1,np);crossCount=0;completedCrossCount=0;
pedVelocity=zeros(np,2);pedDetouring=false(1,np);pedDetours=0;pedBlockedSteps=0;
pedSeparationWaits=0;
pedDetourTarget=nan(1,np);
npcInterventions=struct('time_s',{},'actor_id',{},'blocking_actor_id',{}, ...
    'requested_speed_mps',{},'accepted_speed_mps',{},'requested_route_step_m',{}, ...
    'accepted_route_step_m',{},'requested_lateral_step_m',{},'accepted_lateral_step_m',{},'reason',{});
previousNpcHeld=false(1,config.vehicle_count);
events=struct('time_s',{},'actor_id',{},'event',{},'s_m',{},'offset_m',{});
for i=1:np
    id=config.vehicle_count+i;
    pedS(i)=mod(15+(i-1)*loopLength/np+(rand-.5)*10,loopLength);
    pedD(i)=(6.65+.35*pedDir(i))*(2*(rand>.5)-1);
    specs(id)=makeSpec(id,['pedestrian_',num2str(i)],'pedestrian',[.5,.5,1.7],.25,[1,.5,.25],0,pedD(i),pedSpeed(i));
end
parkS=[62,155,248];parkD=[4.7,-4.7,4.7];
for i=1:3
    id=config.vehicle_count+np+i;
    specs(id)=makeSpec(id,['parked_vehicle_',num2str(i)],'parked_car',[3.2,1.4,1.5],.64,[.58,.63,.67],1,parkD(i),0);
end
if options.Scenario~="mixed"
    specs(1)=makeSpec(1,'slow_lead_auto','auto_rickshaw',[2.6,1.35,1.75],.52,[.96,.73,.14],1,2.5,.85);
    specs(2)=makeSpec(2,'oncoming_car','car',[4.2,1.75,1.5],.84,[.22,.57,.94],-1,-2.5,3);
    npcS=[18,200];npcV=[.85,3];npcD=[2.5,-2.5];amplitude=[0,0];
    pedS(:)=220;pedD(:)=7;nextCross(:)=10000;
    parkS=[150,230,320];
    if options.Scenario=="oncoming",npcS(2)=65;end
    if options.Scenario=="crossing",pedS(1)=35;nextCross(1)=6;end
    if options.Scenario=="opposing_pedestrians"
        npcS=[100,200];pedSpeed(:)=1.2;pedDir(:)=1;
        pedS(:)=35-2*1.2;pedD=[-6.65,6.65];nextCross(:)=2;
    end
end
nActors=numel(specs);steps=ceil(duration/config.control_dt_s);
egoLog=zeros(steps+1,13);actorLog=zeros(steps+1,nActors,7);modes=zeros(steps+1,1);overrides=false(steps+1,1);
egoS=0;[xy,angle]=indian_loop_pose(egoS,config.ego_lane_offset_m);pose=[xy,angle];v=0;delta=0;
progress=0;oldProgressS=egoS;stopSteps=0;slowSteps=0;overrideSteps=0;
previousAcceleration=0;
decisionLog=repmat(struct('time_s',0,'decision',0,'reason','','blocking_actor_id',0, ...
    'following_actor_id',0,'following_gap_m',inf,'target_speed_mps',0,'ttc_s',inf),steps,1);
egoCollisions=0;actorCollisionSamples=0;minGap=inf;minRoad=inf;bodyChecks=0;reason='initial';
maneuver=struct('mode',0,'started',0,'completed',0,'cooldown',0,'last_rejection','', ...
    'start_progress',0,'out_length',16,'return_start',0,'total_length',0,'pass_offset',-1, ...
    'lead_id',0,'start_time',0);
states=actorStates(0);egoLog(1,:)=[0,pose,v,delta,0,0,0,0,0,inf,0];actorLog(1,:,:)=reshape(states,[1,nActors,7]);
writeJSON(fullfile(outputDir,'world_config.json'),config);
[initialGap,initialCollision]=egoClearance(pose,states,specs,0);initialPairs=[];
for first=1:nActors
    for second=first+1:nActors
        if pairOverlap(states(first,:),specs(first),states(second,:),specs(second)),initialPairs(end+1,:)=[first,second];end %#ok<AGROW>
    end
end
writeJSON(fullfile(outputDir,'initial_validation.json'),struct('passed',~initialCollision && isempty(initialPairs), ...
    'ego_clearance_lower_bound_m',initialGap,'other_actor_overlap_pairs',initialPairs));
assert(~initialCollision && isempty(initialPairs),'Invalid initial actor placement; simulation has not advanced.');
started=tic;
try
    for step=1:steps
        t=(step-1)*config.control_dt_s;
        front=pose(1:2)+config.wheelbase_m*[cos(pose(3)),sin(pose(3))];
        [~,rearIdx]=min(sum((routeXY-pose(1:2)).^2,2));egoS=routeS(rearIdx);
        advanceS=mod(egoS-oldProgressS+loopLength/2,loopLength)-loopLength/2;
        progress=progress+advanceS;oldProgressS=egoS;
        [maneuver,overtakeEvent]=traffic_overtake_plan(pose,egoS,progress,v,states,specs,config,maneuver,t);
        if ~isempty(overtakeEvent)
            events(end+1)=struct('time_s',t,'actor_id',maneuver.lead_id,'event',overtakeEvent,'s_m',egoS,'offset_m',0); %#ok<AGROW>
        end
        controlXY=frontXY;controlHeading=frontHeading;
        if maneuver.mode>0
            query=(-5:.25:35)';d=traffic_maneuver_offset(progress+query,maneuver);
            localXY=indian_loop_pose(egoS+query,d);tangent=[diff(localXY);localXY(end,:)-localXY(end-1,:)];
            localH=atan2(tangent(:,2),tangent(:,1));
            controlXY=localXY+config.wheelbase_m*[cos(localH),sin(localH)];
            tangent=[diff(controlXY);controlXY(end,:)-controlXY(end-1,:)];controlHeading=atan2(tangent(:,2),tangent(:,1));
        end
        [~,idx]=min(sum((controlXY-front).^2,2));
        desired=deg2rad(lateralControllerStanley([controlXY(idx,:),rad2deg(controlHeading(idx))], ...
            [pose(1:2),rad2deg(pose(3))],v,'PositionGain',2.5,'Wheelbase',config.wheelbase_m, ...
            'MaxSteeringAngle',rad2deg(config.max_steering_rad),'Direction',1));
        change=config.max_steering_rate_radps*config.control_dt_s;
        delta=delta+max(-change,min(change,desired-delta));
        [decision,override,ttc,reason,blockingActor]=behavior(pose,v,egoS,states);
        target=config.ego_cruise_mps;if decision==1,target=config.slow_speed_mps;elseif decision==2,target=0;end
        followId=0;followGap=inf;
        if config.smooth_following_enabled
            [followTarget,followId,followGap]=traffic_follow_speed(egoS,progress,v,states,specs,config,maneuver);
            target=min(target,followTarget);
            if decision==0 && target<config.ego_cruise_mps-.05
                decision=1;if target<.05,decision=2;end
                reason='same_lane_headway';blockingActor=followId;
            end
        end
        acceleration=max(-config.max_deceleration_mps2,min(config.max_acceleration_mps2,2*(target-v)));
        if decision==2 && v>.02,acceleration=-min(2,max(.5,2*v));end
        if config.smooth_following_enabled
            maxChange=config.nominal_jerk_limit_mps3*config.control_dt_s;
            acceleration=max(previousAcceleration-maxChange,min(previousAcceleration+maxChange,acceleration));
        end
        if override,acceleration=-config.max_deceleration_mps2;end
        oldControlSpeed=v;
        decisionLog(step)=struct('time_s',t,'decision',decision,'reason',reason,'blocking_actor_id',blockingActor, ...
            'following_actor_id',followId,'following_gap_m',followGap,'target_speed_mps',target,'ttc_s',ttc);
        stopSteps=stopSteps+(decision==2);slowSteps=slowSteps+(decision==1);overrideSteps=overrideSteps+override;
        % Traffic responds to leaders and crossing pedestrians. No teleports.
        npcA=zeros(size(npcV));
        egoSpec=makeSpec(0,'ego','car',[4.5,1.8,1.5],.9,[.1,.8,.7],1,2.5,3);
        candidate=[states;pose(1:2),pose(3),v*cos(pose(3)),v*sin(pose(3)),0,egoS];
        candidateSpecs=[specs,egoSpec];
        for i=1:config.vehicle_count
            dir=specs(i).direction;want=specs(i).nominal_speed_mps*(1+.1*sin(.08*t+phase(i)));
            own=states(i,:);forward=dir*[cos(roadAngle(npcS(i))),sin(roadAngle(npcS(i)))];
            want=traffic_npc_speed_target(i,candidate,candidateSpecs,config,npcS(i),npcD(i),npcV(i),want);
            % Predict crossing conflicts; current-distance following alone reacts late.
            for j=config.vehicle_count+(1:np)
                if norm(states(j,1:2)-own(1:2))>20 || states(j,6)==0,continue;end
                for tau=.25:.25:3
                    predictedOwn=own(1:3);predictedOwn(1:2)=own(1:2)+npcV(i)*tau*forward;
                    predictedPed=states(j,1:2)+tau*states(j,4:5);
                    if pedestrianGap(predictedPed,predictedOwn,specs(i),.5)<=0,want=0;break;end
                end
            end
            npcA(i)=max(-3,min(1.1,1.8*(want-npcV(i))));
            if want<=.05 && npcV(i)>.02
                % Finish a stop in finite time instead of creeping towards the
                % pedestrian under exponentially decaying proportional braking.
                npcA(i)=-min(3,max(.8,3*npcV(i)));
            end
        end
        for sub=1:round(config.control_dt_s/config.integration_dt_s)
            fineTime=t+sub*config.integration_dt_s;
            beforeEgoState=[pose,v*cos(pose(3)),v*sin(pose(3)),0,egoS];
            nextV=max(0,v+acceleration*config.integration_dt_s);distance=(v+nextV)/2*config.integration_dt_s;
            kappa=tan(delta)/config.wheelbase_m;turn=distance*kappa;
            if abs(kappa)<1e-10,pose(1:2)=pose(1:2)+distance*[cos(pose(3)),sin(pose(3))];
            else,pose(1:2)=pose(1:2)+[(sin(pose(3)+turn)-sin(pose(3)))/kappa,(-cos(pose(3)+turn)+cos(pose(3)))/kappa];end
            pose(3)=atan2(sin(pose(3)+turn),cos(pose(3)+turn));v=nextV;
            beforeNpcStates=states;beforeNpcS=npcS;beforeNpcD=npcD;
            for i=1:config.vehicle_count
                [~,~,curve]=indian_loop_pose(npcS(i));
                newV=max(0,npcV(i)+npcA(i)*config.integration_dt_s);
                npcS(i)=mod(npcS(i)+specs(i).direction*(npcV(i)+newV)/2*config.integration_dt_s/max(.5,1-npcD(i)*curve),loopLength);
                npcV(i)=newV;
                desiredOffset=specs(i).base_offset_m+amplitude(i)*sin(.12*fineTime+phase(i));
                if config.npc_parked_clearance_enabled
                    % Plan clearance before braking next to a parked body. A
                    % stopped car cannot rely on continued lateral weave to
                    % escape a false permanent following constraint.
                    for parkedIndex=1:3
                        along=mod(specs(i).direction*(parkS(parkedIndex)-npcS(i))+loopLength/2,loopLength)-loopLength/2;
                        if along>=-8 && along<24 && specs(i).direction*parkD(parkedIndex)>0
                            parkedSpec=specs(config.vehicle_count+np+parkedIndex);
                            safeOffset=abs(parkD(parkedIndex))-(specs(i).width_m+parkedSpec.width_m)/2-config.npc_parked_lateral_clearance_m;
                            desiredOffset=specs(i).direction*min(abs(desiredOffset),max(.8,safeOffset));
                        end
                    end
                end
                lateralStep=.10*min(1,newV)*config.integration_dt_s;
                npcD(i)=npcD(i)+max(-lateralStep,min(lateralStep,desiredOffset-npcD(i)));
            end
            states=actorStates(fineTime);
            guardBefore=[beforeNpcStates;beforeEgoState];
            guardProposed=[states;pose(1:2),pose(3),v*cos(pose(3)),v*sin(pose(3)),0,egoS];
            [~,held,blockers]=traffic_npc_step_guard(guardBefore,guardProposed,candidateSpecs,config.vehicle_count);
            for i=1:config.vehicle_count
                if held(i)
                    requestedStep=mod(npcS(i)-beforeNpcS(i)+loopLength/2,loopLength)-loopLength/2;
                    npcInterventions(end+1)=traffic_npc_intervention(fineTime,i,blockers(i),candidateSpecs, ...
                        npcV(i),requestedStep,npcD(i)-beforeNpcD(i)); %#ok<AGROW>
                    npcS(i)=beforeNpcS(i);npcD(i)=beforeNpcD(i);npcV(i)=0;
                    if ~previousNpcHeld(i)
                        events(end+1)=struct('time_s',fineTime,'actor_id',i,'event','npc_safety_constraint_started','s_m',npcS(i),'offset_m',npcD(i)); %#ok<AGROW>
                    end
                elseif previousNpcHeld(i)
                    events(end+1)=struct('time_s',fineTime,'actor_id',i,'event','npc_safety_constraint_released','s_m',npcS(i),'offset_m',npcD(i)); %#ok<AGROW>
                end
            end
            previousNpcHeld=held';states=actorStates(fineTime);
            carIds=[1:config.vehicle_count,config.vehicle_count+np+(1:3)];
            vehicleStates=[states(carIds,:);pose(1:2),pose(3),v*cos(pose(3)),v*sin(pose(3)),0,egoS];
            vehicleSpecs=[specs(carIds),egoSpec];
            for i=1:np
                beforeS=pedS(i);beforeD=pedD(i);beforeMode=pedMode(i);
                beforeDetourTarget=pedDetourTarget(i);detourReached=false;
                beforePed=indian_loop_pose(pedS(i),pedD(i));
                if pedMode(i)==0
                    pedS(i)=mod(pedS(i)+pedDir(i)*pedSpeed(i)*config.integration_dt_s/max(.5,1-pedD(i)*roadCurve(pedS(i))),loopLength);
                    if fineTime>=nextCross(i) && roadCurve(pedS(i))==0
                        pedMode(i)=1;crossTarget(i)=-pedD(i);
                    end
                else
                    if fineTime>=pauseUntil(i)
                        movement=sign(crossTarget(i)-pedD(i))*pedSpeed(i)*config.integration_dt_s;
                        proposedD=pedD(i)+movement;
                        proposedPoint=indian_loop_pose(pedS(i),proposedD);
                        blocked=0;
                        for j=1:size(vehicleStates,1)
                            if norm(proposedPoint-vehicleStates(j,1:2))<8 && pedestrianGap(proposedPoint,vehicleStates(j,1:3),vehicleSpecs(j),.35)<=0
                                blocked=j;break;
                            end
                        end
                        if blocked>0
                            pedBlockedSteps=pedBlockedSteps+1;
                            % Keep one cluster-clearing target rather than
                            % alternating between whichever corner is nearest.
                            if isnan(pedDetourTarget(i)) && norm(vehicleStates(blocked,4:5))<.25
                                [planned,found]=traffic_pedestrian_detour_target(pedS(i),pedD(i),crossTarget(i),vehicleStates,vehicleSpecs,loopLength);
                                if found
                                    pedDetourTarget(i)=planned;pedDetours=pedDetours+1;pedDetouring(i)=true;
                                    events(end+1)=struct('time_s',fineTime,'actor_id',config.vehicle_count+i,'event','persistent_detour_planned','s_m',planned,'offset_m',pedD(i)); %#ok<AGROW>
                                end
                            end
                        elseif abs(crossTarget(i)-pedD(i))<=abs(movement)
                            pedD(i)=crossTarget(i);pedMode(i)=0;
                            pedDetourTarget(i)=nan;pedDetouring(i)=false;
                        elseif isnan(pedDetourTarget(i)),pedD(i)=proposedD;pedDetouring(i)=false;end
                        if isfinite(pedDetourTarget(i))
                            remaining=mod(pedDetourTarget(i)-pedS(i)+loopLength/2,loopLength)-loopLength/2;
                            ds=sign(remaining)*min(abs(remaining),pedSpeed(i)*config.integration_dt_s);
                            proposedS=mod(pedS(i)+ds,loopLength);sidePoint=indian_loop_pose(proposedS,pedD(i));free=true;
                            for j=1:size(vehicleStates,1)
                                if norm(sidePoint-vehicleStates(j,1:2))<8 && pedestrianGap(sidePoint,vehicleStates(j,1:3),vehicleSpecs(j),.3)<=0,free=false;break;end
                            end
                            if free
                                pedS(i)=proposedS;
                                if abs(remaining)<=abs(ds)+1e-8
                                    detourReached=true;
                                end
                            end
                        end
                    end
                end
                afterPed=indian_loop_pose(pedS(i),pedD(i));
                otherPed=indian_loop_pose(pedS(setdiff(1:np,i))',pedD(setdiff(1:np,i))');
                if ~isempty(otherPed) && any(vecnorm(otherPed-afterPed,2,2)<.65)
                    % Local collision avoidance uses physical steps, never relocations.
                    % A walker yields and sidesteps within the footpath; a crossing
                    % pedestrian can sidestep along the road while checking vehicles.
                    pedSeparationWaits=pedSeparationWaits+1;
                    pedS(i)=beforeS;pedD(i)=beforeD;pedMode(i)=beforeMode;
                    ds=pedSpeed(i)*config.integration_dt_s;
                    candidates=[beforeS,beforeD+ds;beforeS,beforeD-ds;beforeS+ds,beforeD;beforeS-ds,beforeD];
                    bestScore=-inf;choice=0;oldClear=min(vecnorm(otherPed-beforePed,2,2));
                    for candidateIndex=1:size(candidates,1)
                        candidate=candidates(candidateIndex,:);
                        if beforeMode==0 && (abs(candidate(2))<5.9 || abs(candidate(2))>7.7),continue;end
                        point=indian_loop_pose(candidate(1),candidate(2));
                        clear=min(vecnorm(otherPed-point,2,2));
                        if clear<.61 || clear<oldClear-.001,continue;end
                        free=true;
                        for j=1:size(vehicleStates,1)
                            if norm(point-vehicleStates(j,1:2))<8 && pedestrianGap(point,vehicleStates(j,1:3),vehicleSpecs(j),.3)<=0,free=false;break;end
                        end
                        score=clear+.02*(candidateIndex==1);
                        if config.directional_pedestrian_passing
                            score=traffic_pedestrian_step_score(candidate,beforeS,beforeD,beforeMode,pedDir(i),crossTarget(i),ds,clear);
                        end
                        if free && score>bestScore,bestScore=score;choice=candidateIndex;end
                    end
                    if choice>0,pedS(i)=mod(candidates(choice,1),loopLength);pedD(i)=candidates(choice,2);end
                    afterPed=indian_loop_pose(pedS(i),pedD(i));
                end
                if detourReached
                    remaining=mod(pedDetourTarget(i)-pedS(i)+loopLength/2,loopLength)-loopLength/2;
                    if abs(remaining)<1e-8
                        pedDetourTarget(i)=nan;pedDetouring(i)=false;
                        events(end+1)=struct('time_s',fineTime,'actor_id',config.vehicle_count+i,'event','persistent_detour_reached','s_m',pedS(i),'offset_m',pedD(i)); %#ok<AGROW>
                    end
                elseif beforeMode==1 && pedMode(i)==1 && isfinite(beforeDetourTarget) && isnan(pedDetourTarget(i))
                    % A pedestrian-separation rollback can revert completion.
                    pedDetourTarget(i)=beforeDetourTarget;pedDetouring(i)=true;
                end
                % Log transitions only after the physical step is accepted;
                % separation may have reverted a proposed start or completion.
                if beforeMode==0 && pedMode(i)==1
                    crossCount=crossCount+1;
                    events(end+1)=struct('time_s',fineTime,'actor_id',config.vehicle_count+i,'event','crossing_started','s_m',pedS(i),'offset_m',pedD(i)); %#ok<AGROW>
                elseif beforeMode==1 && pedMode(i)==0
                    nextCross(i)=fineTime+18+55*rand;completedCrossCount=completedCrossCount+1;
                    events(end+1)=struct('time_s',fineTime,'actor_id',config.vehicle_count+i,'event','crossing_completed','s_m',pedS(i),'offset_m',pedD(i)); %#ok<AGROW>
                end
                pedVelocity(i,:)=(afterPed-beforePed)/config.integration_dt_s;
            end
            states=actorStates(fineTime);
            [gap,collision]=egoClearance(pose,states,specs,0);
            road=roadClearance(pose);minGap=min(minGap,gap);minRoad=min(minRoad,road);bodyChecks=bodyChecks+1;
            if collision
                egoCollisions=egoCollisions+1;
                events(end+1)=struct('time_s',fineTime,'actor_id',0,'event','ego_collision_stop_run','s_m',egoS,'offset_m',0); %#ok<AGROW>
                break;
            end
        end
        previousAcceleration=(v-oldControlSpeed)/(fineTime-t);
        [gap,~]=egoClearance(pose,states,specs,0);
        egoLog(step+1,:)=[fineTime,pose,v,delta,acceleration,target,progress,decision,double(override),gap,maneuver.mode];
        actorLog(step+1,:,:)=reshape(states,[1,nActors,7]);modes(step+1)=decision;overrides(step+1)=override;
        if mod(step,100)==0
            writeJSON(fullfile(outputDir,'progress.json'),struct('time_s',fineTime,'wall_seconds',toc(started), ...
                'ego_progress_m',progress,'ego_collisions',egoCollisions,'overtakes_completed',maneuver.completed));
        end
        if egoCollisions>0 || road<0,break;end
    end
    egoLog=egoLog(1:step+1,:);actorLog=actorLog(1:step+1,:,:);modes=modes(1:step+1);overrides=overrides(1:step+1);
    % Legacy in-run non-ego audit samples every 0.5 s. The independent saved
    % trajectory audit checks every recorded 0.1 s frame and retains incidents.
    for frame=1:5:size(actorLog,1)
        a=squeeze(actorLog(frame,:,:));
        for i=1:nActors
            for j=i+1:nActors
                if norm(a(i,1:2)-a(j,1:2))>8,continue;end
                if pairOverlap(a(i,:),specs(i),a(j,:),specs(j)),actorCollisionSamples=actorCollisionSamples+1;end
            end
        end
    end
    result=struct('status','completed','seed',seed,'simulation_time_s',egoLog(end,1), ...
        'run_wall_seconds',toc(started),'ego_collision_samples',egoCollisions, ...
        'non_ego_overlap_samples_at_0_5s',actorCollisionSamples,'road_departure',minRoad<0, ...
        'minimum_ego_actor_clearance_lower_bound_m',minGap,'minimum_sampled_road_clearance_m',minRoad, ...
        'footprint_checks_at_0_02s',bodyChecks,'ego_progress_m',progress,'completed_laps',floor(progress/loopLength), ...
        'stop_decision_seconds',stopSteps*.1,'slow_decision_seconds',slowSteps*.1,'braking_override_seconds',overrideSteps*.1, ...
        'crossings_started',crossCount,'crossings_completed',completedCrossCount, ...
        'pedestrian_stopped_vehicle_detours',pedDetours,'pedestrian_blocked_person_seconds',pedBlockedSteps*.02, ...
        'pedestrian_separation_wait_person_seconds',pedSeparationWaits*.02, ...
        'overtakes_started',maneuver.started,'overtakes_completed',maneuver.completed, ...
        'last_overtake_rejection',maneuver.last_rejection, ...
        'npc_safety_constraint_step_holds',numel(npcInterventions), ...
        'npc_safety_constraint_vehicle_seconds',numel(npcInterventions)*config.integration_dt_s, ...
        'maximum_ego_speed_mps',max(egoLog(:,5)),'vehicle_count',config.vehicle_count, ...
        'pedestrian_count',np,'parked_vehicle_count',3,'rl_used',false,'learned_mask_used',false, ...
        'final_driving_evaluation',false,'last_behavior_reason',reason, ...
        'limits','Synthetic illustrative actors; prescribed NPC lateral motion; CV predictions; sampled collision checks; fixed loop route; no calibrated perception, RL, traffic signals or junction routing.');
    if egoCollisions>0 || minRoad<0,result.status='failed_safety_development_run';end
    writetable(array2table(egoLog,'VariableNames',{'time_s','rear_x_m','rear_y_m','heading_rad','speed_mps', ...
        'steering_rad','acceleration_mps2','target_speed_mps','progress_m','decision_code','braking_override','clearance_lower_bound_m','maneuver_code'}),fullfile(outputDir,'ego_trace.csv'));
    writeJSON(fullfile(outputDir,'events.json'),events);
    writeJSON(fullfile(outputDir,'decisions.json'),decisionLog(1:step));
    writeJSON(fullfile(outputDir,'npc_safety_interventions.json'),npcInterventions);
    writeJSON(fullfile(outputDir,'result.json'),result);
    replay=struct('schema_version','0.1','config',config,'actors',specs,'route',round(route,5), ...
        'ego',round(egoLog,6),'actor_states',round(actorLog,6),'events',events,'result',result, ...
        'actor_state_columns',{{'x_m','y_m','heading_rad','vx_mps','vy_mps','pedestrian_mode_0_walk_1_cross','route_s_m'}}, ...
        'decision_codes',struct('PROCEED',0,'SLOW',1,'STOP',2));
    writeJSON(fullfile(outputDir,'replay.json'),replay);
    save(fullfile(outputDir,'world.mat'),'config','specs','route','egoLog','actorLog','events','result');
    disp(jsonencode(result,'PrettyPrint',true));
catch err
    writeJSON(fullfile(outputDir,'failure.json'),struct('status','failed','error',err.message));
    save(fullfile(outputDir,'failure_state.mat'));
    rethrow(err);
end

    function states=actorStates(time)
        states=zeros(nActors,7);
        for ii=1:config.vehicle_count
            [p,h]=indian_loop_pose(npcS(ii),npcD(ii));h=h+(specs(ii).direction<0)*pi;h=atan2(sin(h),cos(h));
            vel=npcV(ii)*[cos(h),sin(h)];
            states(ii,:)=[p,h,vel,0,npcS(ii)];
        end
        for ii=1:np
            id=config.vehicle_count+ii;[p,h]=indian_loop_pose(pedS(ii),pedD(ii));
            if time>0,vel=pedVelocity(ii,:);
            elseif pedMode(ii)==0,vel=pedDir(ii)*pedSpeed(ii)*[cos(h),sin(h)];
            elseif time<pauseUntil(ii),vel=[0,0];
            else,vel=sign(crossTarget(ii)-pedD(ii))*pedSpeed(ii)*[-sin(h),cos(h)];end
            yaw=atan2(vel(2),vel(1));states(id,:)=[p,yaw,vel,pedMode(ii),pedS(ii)];
        end
        for ii=1:3
            id=config.vehicle_count+np+ii;[p,h]=indian_loop_pose(parkS(ii),parkD(ii));
            states(id,:)=[p,h,0,0,0,parkS(ii)];
        end
    end

    function [mode,emergency,ttc,why,blocking]=behavior(p,v,s,actors)
        mode=0;emergency=false;ttc=inf;why='route_clear';blocking=0;
        proximity=hypot(actors(:,1)-p(1),actors(:,2)-p(2));near=find(proximity<25);
        probeSpeed=max(v,config.slow_speed_mps);
        localOffset=traffic_maneuver_offset(progress,maneuver);
        [anchor,baseHeading,curve]=indian_loop_pose(s,localOffset);
        correction=p(1:2)-anchor;
        for tau=.2:.2:config.prediction_horizon_s
            futureS=s+probeSpeed*tau/max(.5,1-config.ego_lane_offset_m*curve);
            distance=futureS-s;futureProgress=progress+distance;
            futureOffset=traffic_maneuver_offset(futureProgress,maneuver);
            [futureXY,futureH]=indian_loop_pose(futureS,futureOffset);
            if maneuver.mode>0
                derivative=(traffic_maneuver_offset(futureProgress+.1,maneuver)-traffic_maneuver_offset(futureProgress-.1,maneuver))/.2;
                futureH=futureH+atan(derivative);
            end
            predicted=[futureXY+correction,futureH];
            for jj=near'
                actorFuture=actors(jj,:);actorFuture(1:2)=actorFuture(1:2)+tau*actorFuture(4:5);
                [~,risk]=egoClearance(predicted,actorFuture,specs(jj),config.safety_margin_m);
                if risk && tau<ttc,ttc=tau;blocking=jj;end
            end
            if isfinite(ttc),break;end
        end
        if ttc<=config.stop_ttc_s,mode=2;why='predicted_route_conflict';
        elseif isfinite(ttc),mode=1;why='conflict_in_lookahead';end
        % Independent short-range braking corridor, evaluated from current state.
        direction=[cos(p(3)),sin(p(3))];left=[-direction(2),direction(1)];
        stopping=v*v/(2*config.max_deceleration_mps2)+v*.2+1.0;
        for jj=near'
            rel=actors(jj,1:2)-p(1:2);along=dot(rel,direction);side=abs(dot(rel,left));
            halfLength=specs(jj).length_m/2;
            if strcmp(specs(jj).kind,'pedestrian'),halfLength=.25;end
            crossing=abs(dot(actors(jj,4:5),left));
            if along>-.5 && along<3.6+halfLength+stopping && ...
                    side<.9+specs(jj).width_m/2+config.safety_margin_m+min(.8,crossing*.4) && v>.05
                emergency=true;mode=2;why='independent_braking_corridor';blocking=jj;break;
            end
        end
    end
end

function value=roadCurve(s)
[~,~,value]=indian_loop_pose(s);
end
function value=roadAngle(s)
[~,value]=indian_loop_pose(s);
end
function spec=makeSpec(id,name,kind,dims,rear,color,direction,offset,speed)
spec=struct('id',id,'name',name,'kind',kind,'length_m',dims(1),'width_m',dims(2), ...
    'rear_overhang_m',rear,'height_m',dims(3),'color',color,'direction',direction, ...
    'base_offset_m',offset,'nominal_speed_mps',speed);
end
function [gap,hit]=egoClearance(pose,states,specs,margin)
ego=struct('length_m',4.5,'width_m',1.8,'rear_overhang_m',.9,'kind','car');
gap=inf;hit=false;
for j=1:size(states,1)
    actor=specs(j);state=states(j,:);
    if norm(state(1:2)-pose(1:2))>15,continue;end
    if strcmp(actor.kind,'pedestrian')
        u=[cos(pose(3)),sin(pose(3))];n=[-u(2),u(1)];rel=state(1:2)-pose(1:2);
        q=[dot(rel,u),dot(rel,n)];nearest=[max(-.9-margin,min(3.6+margin,q(1))),max(-.9-margin,min(.9+margin,q(2)))];
        distance=norm(q-nearest)-.25;
    else,distance=obbGap(pose,ego,state(1:3),actor,margin);end
    gap=min(gap,distance);hit=hit || distance<=0;
end
end
function gap=obbGap(a,as,b,bs,margin)
au=[cos(a(3)),sin(a(3))];av=[-au(2),au(1)];bu=[cos(b(3)),sin(b(3))];bv=[-bu(2),bu(1)];
ac=a(1:2)+(as.length_m/2-as.rear_overhang_m)*au;
bc=b(1:2)+(bs.length_m/2-bs.rear_overhang_m)*bu;
axis=[au;av;bu;bv];d=bc-ac;
g=abs(axis*d')-(as.length_m/2+margin)*abs(axis*au')-(as.width_m/2+margin)*abs(axis*av') ...
    -bs.length_m/2*abs(axis*bu')-bs.width_m/2*abs(axis*bv');
gap=max(g); % Positive separating-axis projection gap is a distance lower bound.
end
function hit=pairOverlap(a,as,b,bs)
hit=traffic_body_gap(a,as,b,bs,0)<=0;
end
function gap=pedestrianGap(point,pose,spec,margin)
u=[cos(pose(3)),sin(pose(3))];n=[-u(2),u(1)];relative=point-pose(1:2);
q=[dot(relative,u),dot(relative,n)];
nearest=[max(-spec.rear_overhang_m-margin,min(spec.length_m-spec.rear_overhang_m+margin,q(1))), ...
    max(-spec.width_m/2-margin,min(spec.width_m/2+margin,q(2)))];
gap=norm(q-nearest)-.25;
end
function clear=roadClearance(pose)
% Samples all four footprint edges; ring signed-distance geometry is separate.
body=[-.9,-.9;3.6,-.9;3.6,.9;-.9,.9];sample=[];
for k=1:4,other=mod(k,4)+1;sample=[sample;(1-(0:.2:1)')*body(k,:)+(0:.2:1)'*body(other,:)];end %#ok<AGROW>
u=pose(3);points=sample*[cos(u),sin(u);-sin(u),cos(u)]+pose(1:2);
q=abs(points-[80,60])-[42,22];signed=hypot(max(q(:,1),0),max(q(:,2),0))+min(max(q,[],2),0)-18;
clear=min(5.5-abs(signed));
end
function writeJSON(path,data)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0,'Could not write JSON.');clean=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(data));
end
