function result = run_static_tracking_reference(referenceDir, outputDir)
%RUN_STATIC_TRACKING_REFERENCE One static closed-loop Stanley/PI development run.
% Uses declared ground-truth geometry and a saved Hybrid A* route. No RL,
% learned camera map, moving actors, AEB or vehicle tyre/slip dynamics.
arguments
    referenceDir (1,:) char = ''
    outputDir (1,:) char = ''
end
projectDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if isempty(referenceDir), referenceDir=fullfile(projectDir,'runs','static_reference_20261003'); end
if isempty(outputDir), outputDir=fullfile(projectDir,'runs','static_tracking_20261003'); end
assert(~exist(outputDir,'dir'),'Choose a new output directory; preserve earlier evidence.');
data = load(fullfile(referenceDir,'reference_scene.mat'),'config','poses');
scene = data.config; reference = data.poses;
mkdir(outputDir);
config.kind = 'ground_truth_closed_loop_static_development_reference';
config.controller_dt_s = .1;
config.integration_dt_s = .01;
config.maximum_simulation_time_s = 60;
config.cruise_speed_mps = 3;
config.position_gain = 2.5;
config.speed_kp = 1.5;
config.speed_ki = .15;
config.speed_feedforward = 'Derivative of the distance-based stopping-speed profile, plus PI feedback.';
config.maximum_acceleration_mps2 = 1.5;
config.maximum_deceleration_mps2 = 3;
config.goal_profile_deceleration_mps2 = 1.0;
config.goal_profile_stop_before_m = .12;
config.maximum_steering_rad = atan(scene.vehicle.wheelbase_m/scene.min_turning_radius_m);
config.maximum_steering_rate_radps = deg2rad(45);
config.goal_position_tolerance_m = .5;
config.goal_heading_tolerance_rad = deg2rad(10);
config.goal_speed_tolerance_mps = .1;
config.scene = scene;
config.reference_directory = referenceDir;
config.pose_reference = 'rear_axle_center';
config.stanley_reference = 'Desired front axle path derived from rear axle poses; tangent headings in degrees at the MATLAB API only.';
config.units = struct('position','metres','heading','radians','time','seconds','speed','metres_per_second');
config.note = 'Development gains selected before this run. A single static reference is not a general driving or safety evaluation.';
writeJSON(fullfile(outputDir,'tracking_config.json'),config);
result.status = 'running'; result.kind=config.kind;
result.rl_used=false;result.learned_mask_used=false;result.moving_actors_used=false;result.aeb_implemented=false;
try
    assert(exist('lateralControllerStanley','file')==2,'Automated Driving Toolbox Stanley function is unavailable.');
    % Rear-axle kinematic bicycle, exact constant-curvature arc per substep.
    pose=scene.start_pose; speed=0; steering=0; integral=0;
    s=[0;cumsum(hypot(diff(reference(:,1)),diff(reference(:,2))))];
    frontReference=reference(:,1:2)+scene.vehicle.wheelbase_m*[cos(reference(:,3)),sin(reference(:,3))];
    frontHeading=atan2(gradient(frontReference(:,2)),gradient(frontReference(:,1)));
    steps=ceil(config.maximum_simulation_time_s/config.controller_dt_s);
    history=nan(steps+1,13); fineHistory=nan(steps*10+1,7);
    [roadClear,obstacleClear]=clearance(pose,scene);
    history(1,:)=[0,pose,speed,steering,0,0,0,0,roadClear,obstacleClear,0];
    fineHistory(1,:)=[0,pose,speed,roadClear,obstacleClear];fineCount=1;
    completed=false; started=tic;
    for step=1:steps
        front=pose(1:2)+scene.vehicle.wheelbase_m*[cos(pose(3)),sin(pose(3))];
        [~,frontIndex]=min(sum((frontReference-front).^2,2));
        refPose=[frontReference(frontIndex,:),rad2deg(frontHeading(frontIndex))];
        currentPose=pose;currentPose(3)=rad2deg(currentPose(3));
        desiredSteering=deg2rad(lateralControllerStanley(refPose,currentPose,speed, ...
            'PositionGain',config.position_gain,'Wheelbase',scene.vehicle.wheelbase_m, ...
            'MaxSteeringAngle',rad2deg(config.maximum_steering_rad),'Direction',1));
        maximumChange=config.maximum_steering_rate_radps*config.controller_dt_s;
        steering=steering+max(-maximumChange,min(maximumChange,desiredSteering-steering));
        [rearDistance,rearIndex]=min(sum((reference(:,1:2)-pose(1:2)).^2,2));
        remaining=max(0,s(end)-s(rearIndex)-dot(pose(1:2)-reference(rearIndex,1:2), ...
            [cos(reference(rearIndex,3)),sin(reference(rearIndex,3))]));
        targetSpeed=min(config.cruise_speed_mps,sqrt(2*config.goal_profile_deceleration_mps2*max(0,remaining-config.goal_profile_stop_before_m)));
        brakingProfile=targetSpeed<config.cruise_speed_mps;
        feedforward=0;
        if brakingProfile
            feedforward=-config.goal_profile_deceleration_mps2*speed/max(targetSpeed,.2);
        end
        speedError=targetSpeed-speed;
        tentativeIntegral=max(-5,min(5,integral+speedError*config.controller_dt_s));
        command=feedforward+config.speed_kp*speedError+config.speed_ki*tentativeIntegral;
        % Conditional integration prevents PI windup at either acceleration limit.
        if (command <= config.maximum_acceleration_mps2 && command >= -config.maximum_deceleration_mps2) || ...
                (command > config.maximum_acceleration_mps2 && speedError < 0) || ...
                (command < -config.maximum_deceleration_mps2 && speedError > 0)
            integral=tentativeIntegral;
        end
        acceleration=max(-config.maximum_deceleration_mps2,min(config.maximum_acceleration_mps2,feedforward+config.speed_kp*speedError+config.speed_ki*integral));
        substeps=round(config.controller_dt_s/config.integration_dt_s);
        for substep=1:substeps
            nextSpeed=max(0,speed+acceleration*config.integration_dt_s);
            travelled=(speed+nextSpeed)/2*config.integration_dt_s;
            curvature=tan(steering)/scene.vehicle.wheelbase_m;
            headingChange=travelled*curvature;
            if abs(curvature)<1e-10
                pose(1:2)=pose(1:2)+travelled*[cos(pose(3)),sin(pose(3))];
            else
                pose(1:2)=pose(1:2)+[(sin(pose(3)+headingChange)-sin(pose(3)))/curvature, ...
                    (-cos(pose(3)+headingChange)+cos(pose(3)))/curvature];
            end
            pose(3)=wrapAngle(pose(3)+headingChange);speed=nextSpeed;
            [roadClear,obstacleClear]=clearance(pose,scene);
            fineCount=fineCount+1;
            time=(step-1)*config.controller_dt_s+substep*config.integration_dt_s;
            fineHistory(fineCount,:)=[time,pose,speed,roadClear,obstacleClear];
            assert(roadClear>=scene.safety_margin_m && obstacleClear>=scene.safety_margin_m, ...
                'The actual vehicle footprint violated the requested margin; stop this reference and inspect the trace.');
        end
        headingError=wrapAngle(reference(rearIndex,3)-pose(3));
        history(step+1,:)=[step*config.controller_dt_s,pose,speed,steering,acceleration,targetSpeed,sqrt(rearDistance), ...
            headingError,roadClear,obstacleClear,remaining];
        if norm(pose(1:2)-scene.goal_pose(1:2))<=config.goal_position_tolerance_m && ...
                abs(wrapAngle(pose(3)-scene.goal_pose(3)))<=config.goal_heading_tolerance_rad && speed<=config.goal_speed_tolerance_mps
            completed=true;break;
        end
    end
    history=history(1:step+1,:);fineHistory=fineHistory(1:fineCount,:);
    % Measure rear axle deviation at the stored post-step poses, separately.
    nearestDistance=zeros(size(history,1),1);
    for k=1:size(history,1)
        nearestDistance(k)=sqrt(min(sum((reference(:,1:2)-history(k,2:3)).^2,2)));
    end
    history(:,9)=nearestDistance;
    writetable(array2table(history,'VariableNames',{'time_s','rear_axle_x_m','rear_axle_y_m','heading_rad', ...
        'speed_mps','steering_rad','acceleration_mps2','target_speed_mps','rear_path_distance_m', ...
        'heading_error_rad','road_clearance_m','obstacle_clearance_m','remaining_reference_m'}),fullfile(outputDir,'tracking_trace.csv'));
    writetable(array2table(fineHistory,'VariableNames',{'time_s','rear_axle_x_m','rear_axle_y_m','heading_rad', ...
        'speed_mps','road_clearance_m','obstacle_clearance_m'}),fullfile(outputDir,'integration_trace.csv'));
    result.status='completed';result.goal_reached=completed;result.timeout=~completed;
    result.simulation_time_s=history(end,1);result.run_wall_seconds=toc(started);
    result.goal_position_error_m=norm(pose(1:2)-scene.goal_pose(1:2));
    result.goal_heading_error_rad=abs(wrapAngle(pose(3)-scene.goal_pose(3)));
    result.final_speed_mps=speed;
    result.maximum_speed_mps=max(fineHistory(:,5));
    result.rear_path_distance_rms_m=sqrt(mean(nearestDistance.^2));
    result.rear_path_distance_max_m=max(nearestDistance);
    result.minimum_road_footprint_clearance_m=min(fineHistory(:,6));
    result.minimum_obstacle_footprint_clearance_m=min(fineHistory(:,7));
    result.maximum_absolute_steering_rad=max(abs(history(:,6)));
    result.maximum_absolute_steering_rate_radps=max(abs(diff(history(:,6)))/config.controller_dt_s);
    result.maximum_acceleration_mps2=max(history(:,7));result.maximum_deceleration_mps2=-min(history(:,7));
    result.footprint_checks=fineCount;result.controller='MATLAB lateralControllerStanley and bounded PI speed controller';
    result.collision_check_scope='Actual oriented body checked every 0.01 simulation seconds; no continuous-time safety proof or predictive AEB.';
    result.closed_loop_reference_verified=completed;
    result.final_driving_evaluation=false;
    writeJSON(fullfile(outputDir,'tracking_result.json'),result);
    replay=struct('schema_version','0.1-draft','kind','closed_loop_static_tracking_replay','config',scene, ...
        'poses',fineHistory(:,2:4),'time_s',fineHistory(:,1),'speed_mps',fineHistory(:,5), ...
        'controller_used',true,'rl_used',false,'learned_mask_used',false);
    writeJSON(fullfile(outputDir,'tracking_replay.json'),replay);
    save(fullfile(outputDir,'tracking_scene.mat'),'config','reference','history','fineHistory','result');
    disp(jsonencode(result,'PrettyPrint',true));
    assert(completed,'Static tracking reference timed out; preserve and inspect its saved results.');
catch exception
    result.status='failed';result.error=exception.message;
    writeJSON(fullfile(outputDir,'tracking_result.json'),result);
    if exist('history','var'), save(fullfile(outputDir,'failure_state.mat'),'config','reference','history','fineHistory','pose','speed','steering'); end
    rethrow(exception);
end
end

function [roadDistance, obstacleDistance]=clearance(pose,scene)
v=scene.vehicle;
body=[-v.rear_overhang_m,-v.width_m/2;v.front_of_rear_axle_m,-v.width_m/2; ...
    v.front_of_rear_axle_m,v.width_m/2;-v.rear_overhang_m,v.width_m/2];
angle=pose(3);corners=body*[cos(angle),sin(angle);-sin(angle),cos(angle)]+pose(1:2);
r=scene.road_bounds_xyxy_m;o=scene.obstacle_bounds_xyxy_m;
roadDistance=min([corners(:,1)-r(1);r(3)-corners(:,1);corners(:,2)-r(2);r(4)-corners(:,2)]);
obstacle=[o(1:2);o(3),o(2);o(3:4);o(1),o(4)];
axes=[1,0;0,1;cos(angle),sin(angle);-sin(angle),cos(angle)];
disjoint=false;
for k=1:4
    a=corners*axes(k,:)';b=obstacle*axes(k,:)';
    disjoint=disjoint || max(a)<min(b) || max(b)<min(a);
end
if ~disjoint,obstacleDistance=0;return;end
obstacleDistance=inf;
for k=1:4
    next=mod(k,4)+1;
    for j=1:4
        obstacleDistance=min(obstacleDistance,pointSegment(corners(j,:),obstacle(k,:),obstacle(next,:)));
        obstacleDistance=min(obstacleDistance,pointSegment(obstacle(j,:),corners(k,:),corners(next,:)));
    end
end
end

function distance=pointSegment(point,a,b)
v=b-a;fraction=max(0,min(1,dot(point-a,v)/dot(v,v)));
distance=norm(point-(a+fraction*v));
end

function angle=wrapAngle(angle)
angle=atan2(sin(angle),cos(angle));
end

function writeJSON(path,data)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0,'Could not write JSON.');
closer=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(data,'PrettyPrint',true));
end
