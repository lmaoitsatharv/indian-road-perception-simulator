function [targetS,found]=traffic_pedestrian_detour_target(s,offset,crossTarget,vehicles,specs,loopLength)
% Choose one persistent route around the full stopped-vehicle cluster.
% Both the approach and eventual crossing must clear every stopped body.
% Rechoosing the nearest corner every 0.02 s previously oscillated between
% the front of a stopped car and a neighbouring parked vehicle indefinitely.
targetS=nan;found=false;[origin,h]=indian_loop_pose(s,offset);u=[cos(h),sin(h)];
stopped=find(vecnorm(vehicles(:,4:5),2,2)<.25 & vecnorm(vehicles(:,1:2)-origin,2,2)<16);
if isempty(stopped),return;end
candidates=[];
for j=stopped'
    vehicle=vehicles(j,:);sp=specs(j);cu=[cos(vehicle(3)),sin(vehicle(3))];cn=[-cu(2),cu(1)];
    center=vehicle(1:2)+(sp.length_m/2-sp.rear_overhang_m)*cu;
    relative=dot(center-origin,u);
    extent=sp.length_m/2*abs(dot(cu,u))+sp.width_m/2*abs(dot(cn,u));
    candidates=[candidates,relative-extent-.85,relative+extent+.85]; %#ok<AGROW>
end
[~,order]=sort(abs(candidates));
for candidateIndex=order
    delta=candidates(candidateIndex);
    if abs(delta)<.1 || abs(delta)>12,continue;end
    endpoint=s+delta;
    approach=indian_loop_pose(linspace(s,endpoint,max(2,ceil(abs(delta)/.15)+1)),offset);
    crossing=indian_loop_pose(endpoint,linspace(offset,crossTarget,max(2,ceil(abs(crossTarget-offset)/.15)+1)));
    points=[approach;crossing];free=true;
    for j=stopped'
        for k=1:size(points,1)
            pedestrian=struct('kind','pedestrian','width_m',.5);
            if traffic_body_gap([points(k,:),0],pedestrian,vehicles(j,1:3),specs(j),.35)<=0
                free=false;break
            end
        end
        if ~free,break;end
    end
    if free,targetS=mod(endpoint,loopLength);found=true;return;end
end
end
