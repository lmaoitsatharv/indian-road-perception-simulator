function [maneuver,note]=traffic_overtake_plan(pose,s,progress,v,actors,specs,config,maneuver,time)
% Ground-truth, rule-based local overtake planner. No RL or future RNG access.
note='';
if maneuver.mode>0
    q=progress-maneuver.start_progress;
    if q>=maneuver.total_length
        [lane,h]=indian_loop_pose(s,2.5);u=[cos(h),sin(h)];n=[-u(2),u(1)];
        aligned=abs(dot(pose(1:2)-lane,n))<.4 && abs(atan2(sin(pose(3)-h),cos(pose(3)-h)))<deg2rad(6);
        lead=actors(maneuver.lead_id,:);sp=specs(maneuver.lead_id);
        leadFront=lead(1:2)+(sp.length_m-sp.rear_overhang_m)*[cos(lead(3)),sin(lead(3))];
        rear=pose(1:2)-.9*[cos(pose(3)),sin(pose(3))];
        passed=dot(rear-leadFront,u)>2;
        returnClear=true;egoSpec=struct('kind','car','length_m',4.5,'width_m',1.8,'rear_overhang_m',.9);
        for j=1:numel(specs)
            if traffic_body_gap(pose,egoSpec,actors(j,1:3),specs(j),.45)<=0
                returnClear=false;break
            end
        end
        if aligned && passed && returnClear
            maneuver.mode=0;maneuver.completed=maneuver.completed+1;maneuver.cooldown=time+10;
            note='overtake_completed';
        else
            maneuver.mode=3;maneuver.last_rejection='completion_wait_for_actual_lane_and_body_clearance';
        end
    elseif q>=maneuver.return_start,maneuver.mode=3;
    elseif q>=maneuver.out_length,maneuver.mode=2;
    else,maneuver.mode=1;end
    return
end
if ~config.overtaking_enabled || v<.8 || time<maneuver.cooldown,return;end
% Four straight sections; reserve eight metres before the next curve.
arc=pi*18/2;starts=[0,84+arc,128+2*arc,212+3*arc];lengths=[84,44,84,44];
segment=find(s>=starts & s<starts+lengths,1);
if isempty(segment),maneuver.last_rejection='curve';return;end
available=starts(segment)+lengths(segment)-s-8;
[anchor,h]=indian_loop_pose(s,2.5);u=[cos(h),sin(h)];n=[-u(2),u(1)];
lead=0;nearest=inf;
for j=1:config.vehicle_count
    ahead=dot(actors(j,1:2)-pose(1:2),u);side=abs(dot(actors(j,1:2)-anchor,n));
    speed=dot(actors(j,4:5),u);
    if specs(j).direction==1 && ahead>6 && ahead<23 && side<1.4 && speed<config.ego_cruise_mps-.6 && ahead<nearest
        nearest=ahead;lead=j;
    end
end
if lead==0,return;end
leadSpeed=norm(actors(lead,4:5));cruise=config.ego_cruise_mps;
outLength=16;
returnStart=max(outLength+8,(nearest+specs(lead).length_m+5)*cruise/max(.6,cruise-leadSpeed)+5);
totalLength=returnStart+outLength;
if totalLength>available,maneuver.last_rejection='insufficient_straight_distance';return;end
horizon=totalLength/cruise+max(0,cruise-v)/1.5+3;
% Whole-manoeuvre oncoming clearance, including space to return. This checks
% route distance, so an oncoming actor around a nearby bend is not ignored.
for j=1:config.vehicle_count
    ahead=mod(actors(j,7)-s,config.loop_length_m);
    if specs(j).direction<0 && ahead<totalLength+norm(actors(j,4:5))*horizon+12
        maneuver.last_rejection='oncoming_traffic';return
    end
end
% Do not initiate around a pedestrian in or approaching the passing corridor.
for j=config.vehicle_count+(1:config.pedestrian_count)
    ahead=dot(actors(j,1:2)-pose(1:2),u);side=abs(dot(actors(j,1:2)-pose(1:2),n));
    if ahead>-5 && ahead<totalLength+10 && side<10
        maneuver.last_rejection='pedestrian_near_manoeuvre';return
    end
end
candidate=maneuver;candidate.mode=1;candidate.start_progress=progress;
candidate.out_length=outLength;candidate.return_start=returnStart;
candidate.total_length=totalLength;candidate.pass_offset=-1.0;
candidate.lead_id=lead;candidate.start_time=time;
% Predict the complete profile before committing, with time-growing margin.
for distance=0:.75:totalLength
    tau=travelTime(distance,v,cruise,1.5);
    d=traffic_maneuver_offset(progress+distance,candidate);
    before=traffic_maneuver_offset(progress+distance-.1,candidate);
    after=traffic_maneuver_offset(progress+distance+.1,candidate);
    yaw=h+atan2(after-before,.2);point=indian_loop_pose(s+distance,d);
    ego=[point,yaw];
    if abs(d)+.9*abs(cos(yaw-h))+3.6*abs(sin(yaw-h))>5.0
        maneuver.last_rejection='road_footprint';return
    end
    for j=1:numel(specs)
        if strcmp(specs(j).kind,'pedestrian'),continue;end
        other=actors(j,1:3);other(1:2)=other(1:2)+tau*actors(j,4:5);
        margin=.5+min(.6,tau*.025);
        if rectOverlap(ego,[4.5,1.8,.9],other,[specs(j).length_m,specs(j).width_m,specs(j).rear_overhang_m],margin)
            maneuver.last_rejection='predicted_vehicle_or_return_conflict';return
        end
    end
end
maneuver=candidate;maneuver.started=maneuver.started+1;maneuver.last_rejection='';
note='overtake_started';
end
function time=travelTime(distance,speed,cruise,acceleration)
accelerateTime=max(0,(cruise-speed)/acceleration);
accelerateDistance=(speed+cruise)/2*accelerateTime;
if distance<=accelerateDistance,time=(-speed+sqrt(speed^2+2*acceleration*distance))/acceleration;
else,time=accelerateTime+(distance-accelerateDistance)/cruise;end
end
function hit=rectOverlap(a,as,b,bs,margin)
au=[cos(a(3)),sin(a(3))];av=[-au(2),au(1)];bu=[cos(b(3)),sin(b(3))];bv=[-bu(2),bu(1)];
ac=a(1:2)+(as(1)/2-as(3))*au;bc=b(1:2)+(bs(1)/2-bs(3))*bu;
axes=[au;av;bu;bv];
gap=abs(axes*(bc-ac)')-(as(1)/2+margin)*abs(axes*au')-(as(2)/2+margin)*abs(axes*av') ...
    -bs(1)/2*abs(axes*bu')-bs(2)/2*abs(axes*bv');
hit=max(gap)<=0;
end
