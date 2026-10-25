function [wanted,blocker,gap]=traffic_npc_speed_target(index,states,specs,config,s,offset,speed,nominal)
% Full-body following along the curved route; parked actors/peds do not move.
% The former rear-point/half-length projection could brake after body contact.
wanted=nominal;blocker=0;gap=inf;own=specs(index);direction=own.direction;
lookahead=18;distance=(0:.4:lookahead)';
[xy,h]=indian_loop_pose(s+direction*distance,offset);
h=h+(direction<0)*pi;
path=[xy,atan2(sin(h),cos(h))];
laneDistances=[0;cumsum(vecnorm(diff(xy),2,2))];
for j=1:numel(specs)
    if j==index || norm(states(j,1:2)-states(index,1:2))>lookahead+8,continue;end
    other=specs(j);
    % Only actors in the forward route corridor constrain nominal following.
    % Snapshot geometry is conservative; the independent step guard also runs.
    first=find(pathGaps(path,own,states(j,1:3),other,.55)<=0,1);
    if isempty(first),continue;end
    bodyGap=max(0,laneDistances(max(1,first-1)));
    sameDirection=other.direction==direction && ~strcmp(other.kind,'parked_car');
    otherSpeed=0;
    if sameDirection,otherSpeed=max(0,norm(states(j,4:5)));end
    desired=1.1+speed*.8;
    target=max(0,otherSpeed+.65*(bodyGap-desired));
    if target<wanted,wanted=target;blocker=j;gap=bodyGap;end
end
end
function gaps=pathGaps(path,own,otherPose,other,margin)
% Vectorized full-body checks keep repeated traffic trials practical.
au=[cos(path(:,3)),sin(path(:,3))];av=[-au(:,2),au(:,1)];
if strcmp(other.kind,'pedestrian')
    relative=otherPose(1:2)-path(:,1:2);
    q=[sum(relative.*au,2),sum(relative.*av,2)];
    nearest=[max(-own.rear_overhang_m-margin,min(own.length_m-own.rear_overhang_m+margin,q(:,1))), ...
        max(-own.width_m/2-margin,min(own.width_m/2+margin,q(:,2)))];
    gaps=vecnorm(q-nearest,2,2)-other.width_m/2;return
end
bu=[cos(otherPose(3)),sin(otherPose(3))];bv=[-bu(2),bu(1)];
ac=path(:,1:2)+(own.length_m/2-own.rear_overhang_m)*au;
bc=otherPose(1:2)+(other.length_m/2-other.rear_overhang_m)*bu;
d=bc-ac;al=own.length_m/2+margin;aw=own.width_m/2+margin;
bl=other.length_m/2;bw=other.width_m/2;
g1=abs(sum(au.*d,2))-al-bl*abs(au*bu')-bw*abs(au*bv');
g2=abs(sum(av.*d,2))-aw-bl*abs(av*bu')-bw*abs(av*bv');
g3=abs(d*bu')-al*abs(au*bu')-aw*abs(av*bu')-bl;
g4=abs(d*bv')-al*abs(au*bv')-aw*abs(av*bv')-bw;
gaps=max([g1,g2,g3,g4],[],2);
end
