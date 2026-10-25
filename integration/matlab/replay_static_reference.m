function replay_static_reference(runDir, speedFactor)
%REPLAY_STATIC_REFERENCE Visualize the planned car poses at 1x, 4x, etc.
% This directly replays a geometric path. It is not controller or RL execution.
arguments
    runDir (1,:) char = ''
    speedFactor (1,1) double {mustBePositive} = 1
end
projectDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if isempty(runDir), runDir = fullfile(projectDir, 'runs', 'static_reference_20261003'); end
data = load(fullfile(runDir, 'reference_scene.mat'), 'config', 'map', 'poses');
config = data.config; poses = data.poses;
fig = figure('Color', 'w', 'Name', 'Ground-truth geometric car replay');
ax = axes(fig); show(data.map, 'Parent', ax); hold(ax,'on');
set(ax,'Color','w','XColor','k','YColor','k');
plot(ax,poses(:,1),poses(:,2),'b-','LineWidth',1.8);
body = [-.9,-.9;3.6,-.9;3.6,.9;-.9,.9];
car = patch(ax,nan(4,1),nan(4,1),[.1,.7,.5],'EdgeColor','k');
nominalSpeed = 3;  % Playback parameter, not a measured controlled speed.
elapsedTime = 0;
axis(ax,'equal'); xlim(ax,data.map.XWorldLimits); ylim(ax,data.map.YWorldLimits);
indices = unique([1:4:size(poses,1), size(poses,1)]);
for index = 1:numel(indices)
    k = indices(index);
    if ~isgraphics(fig), return; end
    angle = poses(k,3);
    corners = body*[cos(angle),sin(angle);-sin(angle),cos(angle)] + poses(k,1:2);
    set(car,'XData',corners(:,1),'YData',corners(:,2));
    title(ax,{sprintf('Geometric car replay | %.1fx | nominal %.1f km/h | t = %.1f s',speedFactor,nominalSpeed*3.6,elapsedTime), ...
        'Ground-truth map; RL and controller not active'},'Color','k');
    drawnow;
    if index == numel(indices), break; end
    next = indices(index+1);
    stepDistance = sum(hypot(diff(poses(k:next,1)),diff(poses(k:next,2))));
    stepTime = stepDistance/nominalSpeed;
    pause(stepTime/speedFactor);
    elapsedTime = elapsedTime+stepTime;
end
end
