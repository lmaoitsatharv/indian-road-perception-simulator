function export_reference_preview(runDir)
%EXPORT_REFERENCE_PREVIEW Read saved poses; do not rerun the planner.
data = load(fullfile(runDir,'reference_scene.mat'),'config','poses');
config = data.config; poses = data.poses;
target = fullfile(runDir,'static_reference_readable.png');
assert(~isfile(target),'Preserve the previous preview.');
fig = figure('Visible','off','Color','w','Position',[100,100,1400,720]);
closer = onCleanup(@() close(fig));
ax = axes(fig); hold(ax,'on');
set(ax,'Color',[.87,.92,.88],'XColor','k','YColor','k');
r = config.road_bounds_xyxy_m; o = config.obstacle_bounds_xyxy_m;
rectangle(ax,'Position',[r(1),r(2),r(3)-r(1),r(4)-r(2)],'FaceColor',[.92,.94,.96],'EdgeColor',[.25,.3,.35],'LineWidth',1.5);
rectangle(ax,'Position',[o(1),o(2),o(3)-o(1),o(4)-o(2)],'FaceColor',[1,.72,.32],'EdgeColor',[.65,.35,.06],'LineWidth',1.5);
pathLine = plot(ax,poses(:,1),poses(:,2),'-','Color',[.1,.4,.8],'LineWidth',2.5);
body = [-.9,-.9;3.6,-.9;3.6,.9;-.9,.9];
for k = unique(round(linspace(1,size(poses,1),10)))
    angle=poses(k,3);
    corners=body*[cos(angle),sin(angle);-sin(angle),cos(angle)]+poses(k,1:2);
    vehiclePatch = patch(ax,corners(:,1),corners(:,2),[.1,.7,.5],'FaceAlpha',.18,'EdgeColor',[.0,.4,.25]);
end
scatter(ax,config.start_pose(1),config.start_pose(2),80,[.1,.6,.2],'filled');
scatter(ax,config.goal_pose(1),config.goal_pose(2),80,[.9,.4,.1],'filled');
text(ax,8,17,'Start','Color','k','HorizontalAlignment','center');
text(ax,52,17,'Goal','Color','k','HorizontalAlignment','center');
text(ax,30,18,'Parked obstacle','Color','k','HorizontalAlignment','center');
title(ax,{'Hybrid A* static development reference', ...
    'Saved rear axle path and full vehicle footprints | Ground-truth geometry'},'Color','k');
xlabel(ax,'World x (metres)','Color','k');ylabel(ax,'World y (metres)','Color','k');
axis(ax,'equal');xlim(ax,[0,60]);ylim(ax,[0,30]);grid(ax,'on');
legend(ax,[pathLine,vehiclePatch],{'Rear axle path','Vehicle footprint samples'},'Location','southoutside','TextColor','k','Color','w');
exportgraphics(fig,target,'Resolution',150);
end
