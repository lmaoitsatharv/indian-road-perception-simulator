function [target,leadId,gap]=traffic_follow_speed(s,progress,v,actors,specs,config,maneuver)
% Route-relative following for the synthetic loop, using current states only.
% This limits the nominal target; collision prediction and braking still run.
target=config.ego_cruise_mps;leadId=0;gap=inf;
if maneuver.mode>0,return;end
egoOffset=traffic_maneuver_offset(progress,maneuver);
for j=1:config.vehicle_count
    if specs(j).direction~=1,continue;end
    ahead=mod(actors(j,7)-s,config.loop_length_m);
    if ahead<=0 || ahead>35,continue;end
    [center,h]=indian_loop_pose(actors(j,7));
    normal=[-sin(h),cos(h)];offset=dot(actors(j,1:2)-center,normal);
    if abs(offset-egoOffset)>(1.8+specs(j).width_m)/2+.55,continue;end
    % Integrate lane distance through curves instead of using the current
    % heading projection, which distorts the gap near a bend.
    probes=linspace(s,s+ahead,max(3,ceil(ahead/.5)));
    xy=indian_loop_pose(probes,egoOffset);
    bodyGap=sum(vecnorm(diff(xy),2,2))-3.6-specs(j).rear_overhang_m;
    desiredGap=config.follow_standstill_gap_m+config.follow_headway_s*v;
    leadSpeed=max(0,dot(actors(j,4:5),[cos(h),sin(h)]));
    wanted=max(0,leadSpeed+config.follow_gap_gain*(bodyGap-desiredGap));
    if wanted<target,target=wanted;leadId=j;gap=bodyGap;end
end
end
