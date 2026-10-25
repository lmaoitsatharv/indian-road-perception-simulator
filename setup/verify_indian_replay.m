function verify_indian_replay
% Exercise the actual MATLAB replay callbacks after the name-resolution fix.
projectDir=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectDir,'integration','matlab'));
outputDir=fullfile(projectDir,'runs','indian_traffic_20261004_seed42_02');
fig=replay_indian_traffic_world(outputDir);
cleanup=onCleanup(@()closeIfPresent(fig));
replayTimer=getappdata(fig,'ReplayTimer');
assert(isvalid(replayTimer) && strcmp(replayTimer.Running,'on'),'Replay timer did not start.');
play=findobj(fig,'Style','pushbutton','String','Play');
reset=findobj(fig,'Style','pushbutton','String','Reset');
slider=findobj(fig,'Style','slider');
rate=findobj(fig,'Style','popupmenu');
assert(numel(play)==1 && numel(reset)==1 && numel(slider)==1 && numel(rate)==1);
feval(play.Callback,play,[]);pause(.7);drawnow;
assert(slider.Value>.2 && strcmp(play.String,'Pause'),'Playback did not advance.');
feval(play.Callback,play,[]);pausedTime=slider.Value;pause(.15);
assert(abs(slider.Value-pausedTime)<.01,'Paused replay advanced.');
rate.Value=3;feval(rate.Callback,rate,[]);
slider.Value=20;feval(slider.Callback,slider,[]);
feval(play.Callback,play,[]);pause(.5);drawnow;
fastTime=slider.Value;assert(fastTime>23,'16x replay did not advance at the selected speed.');
feval(play.Callback,play,[]);
slider.Value=240;feval(slider.Callback,slider,[]);drawnow;
assert(abs(slider.Value-240)<1e-8,'Seek did not render selected simulation time.');
feval(reset.Callback,reset,[]);assert(slider.Value==0 && strcmp(play.String,'Play'));
close(fig);assert(~isvalid(replayTimer),'Closing the viewer left its timer active.');
result=struct('status','passed','timer_started',true,'play_pause_verified',true, ...
    'speed_16x_verified',true,'seek_240s_verified',true,'reset_verified',true, ...
    'timer_deleted_on_close',true,'simulation_rerun',false,'matlab_release',version('-release'));
fid=fopen(fullfile(outputDir,'replay_verification.json'),'w','n','UTF-8');
assert(fid>=0);fprintf(fid,'%s\n',jsonencode(result));fclose(fid);
disp(jsonencode(result,'PrettyPrint',true));
end
function closeIfPresent(fig)
if isgraphics(fig),close(fig);end
end
