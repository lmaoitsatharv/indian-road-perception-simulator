function score=traffic_pedestrian_step_score(candidate,beforeS,beforeD,mode,walkDirection,crossTarget,stepLength,clearance)
% Prefer progress, then passing to the pedestrian's right. Opposite walkers
% therefore choose opposite side steps instead of both reversing indefinitely.
ds=(candidate(1)-beforeS)/stepLength;dd=(candidate(2)-beforeD)/stepLength;
if mode==1
    direction=sign(crossTarget-beforeD);
    forward=direction*dd;right=direction*ds;
else
    forward=walkDirection*ds;right=-walkDirection*dd;
end
score=2*forward+.6*right+.05*min(clearance,1);
end
