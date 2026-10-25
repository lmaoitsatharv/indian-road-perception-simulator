function gap=traffic_body_gap(a,as,b,bs,margin)
%TRAFFIC_BODY_GAP Body clearance for rear-axle vehicles and pedestrian discs.
% Rectangle gaps are separating-axis lower bounds, not Euclidean distances.
if nargin<5,margin=0;end
ap=strcmp(as.kind,'pedestrian');bp=strcmp(bs.kind,'pedestrian');
if ap && bp
    gap=norm(a(1:2)-b(1:2))-(as.width_m+bs.width_m)/2-margin;return
elseif ap
    gap=discGap(a(1:2),as.width_m/2,b,bs,margin);return
elseif bp
    gap=discGap(b(1:2),bs.width_m/2,a,as,margin);return
end
au=[cos(a(3)),sin(a(3))];av=[-au(2),au(1)];
bu=[cos(b(3)),sin(b(3))];bv=[-bu(2),bu(1)];
ac=a(1:2)+(as.length_m/2-as.rear_overhang_m)*au;
bc=b(1:2)+(bs.length_m/2-bs.rear_overhang_m)*bu;
axes=[au;av;bu;bv];
g=abs(axes*(bc-ac)')-(as.length_m/2+margin)*abs(axes*au') ...
    -(as.width_m/2+margin)*abs(axes*av') ...
    -bs.length_m/2*abs(axes*bu')-bs.width_m/2*abs(axes*bv');
gap=max(g);
end
function gap=discGap(point,radius,pose,spec,margin)
u=[cos(pose(3)),sin(pose(3))];n=[-u(2),u(1)];rel=point-pose(1:2);
q=[dot(rel,u),dot(rel,n)];
nearest=[max(-spec.rear_overhang_m-margin,min(spec.length_m-spec.rear_overhang_m+margin,q(1))), ...
    max(-spec.width_m/2-margin,min(spec.width_m/2+margin,q(2)))];
gap=norm(q-nearest)-radius;
end
