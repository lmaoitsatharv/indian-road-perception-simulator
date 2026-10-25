function verify_indian_3d
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'integration','matlab'));
out=fullfile(root,'runs','traffic_overnight_20261004');
fig=replay_indian_traffic_3d(fullfile(root,'runs','indian_traffic_20261004_seed42_02'));
cleanup=onCleanup(@()closePresent(fig));
timerHandle=getappdata(fig,'ReplayTimer');ax=getappdata(fig,'MainAxes');
camera=findobj(fig,'Tag','Camera');slider=findobj(fig,'Tag','Seek');play=findobj(fig,'Tag','Play');
rate=findobj(fig,'Tag','Rate');reset=findobj(fig,'Tag','Reset');
assert(strcmp(ax.Projection,'perspective'));assert(isvalid(timerHandle));
slider.Value=15;feval(slider.Callback,slider,[]);drawnow;
followPosition=campos(ax);exportgraphics(fig,fullfile(out,'follow_camera.png'),'Resolution',120);
camera.Value=2;feval(camera.Callback,camera,[]);drawnow;
driverPosition=campos(ax);assert(norm(driverPosition-followPosition)>5);
assert(abs(driverPosition(3)-1.35)<1e-9);assert(strcmp(getappdata(fig,'EgoTransform').Visible,'off'));
exportgraphics(fig,fullfile(out,'driver_camera.png'),'Resolution',120);
camera.Value=3;feval(camera.Callback,camera,[]);drawnow;
assert(campos(ax)*[0;0;1]>100);exportgraphics(fig,fullfile(out,'overview_camera.png'),'Resolution',120);
camera.Value=1;feval(camera.Callback,camera,[]);
feval(play.Callback,play,[]);pause(1);drawnow;assert(slider.Value>15.2);
feval(play.Callback,play,[]);paused=slider.Value;pause(.15);assert(slider.Value==paused);
rate.Value=3;feval(rate.Callback,rate,[]);feval(play.Callback,play,[]);pause(.6);drawnow;
assert(slider.Value>paused+4);feval(play.Callback,play,[]);
feval(reset.Callback,reset,[]);assert(slider.Value==0);
close(fig);assert(~isvalid(timerHandle));
result=struct('status','passed','release',version('-release'),'all_three_cameras',true, ...
    'play_pause_seek_reset',true,'speed_16x',true,'owned_timer_deleted',true);
fid=fopen(fullfile(out,'viewer_verification.json'),'w','n','UTF-8');fprintf(fid,'%s\n',jsonencode(result));fclose(fid);
disp(jsonencode(result));
end
function closePresent(fig)
if isgraphics(fig),close(fig);end
end
