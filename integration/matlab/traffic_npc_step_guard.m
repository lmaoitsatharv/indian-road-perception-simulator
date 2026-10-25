function [accepted,held,blocker]=traffic_npc_step_guard(before,proposed,specs,npcCount)
% Explicit synthetic safety constraint on prospective NPC motion.
% Holds complete proposed steps, without moving actors out of existing contact.
% Every hold is exposed by the caller as an intervention; it is not proof that
% the behavior policy itself would avoid collisions. Check middle/end poses.
accepted=proposed;held=false(npcCount,1);blocker=zeros(npcCount,1);
for pass=1:npcCount+1
    changed=false;
    for i=1:npcCount
        if held(i),continue;end
        for j=1:numel(specs)
            if i==j || norm(before(i,1:2)-before(j,1:2))>9,continue;end
            risk=false;
            for fraction=[.5,1]
                a=interpolate(before(i,:),accepted(i,:),fraction);
                b=interpolate(before(j,:),accepted(j,:),fraction);
                if traffic_body_gap(a,specs(i),b,specs(j),.08)<=0,risk=true;break;end
            end
            if risk
                held(i)=true;blocker(i)=j;accepted(i,:)=before(i,:);accepted(i,4:5)=0;
                changed=true;break
            end
        end
    end
    if ~changed,break;end
end
end
function p=interpolate(before,after,fraction)
p=before+fraction*(after-before);
p(3)=before(3)+fraction*atan2(sin(after(3)-before(3)),cos(after(3)-before(3)));
end
