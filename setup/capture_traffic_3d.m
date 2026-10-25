function capture_traffic_3d(runDir,outputDir,time)
% Export scene views from an explicitly selected saved rollout.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'integration','matlab'));
assert(~isfolder(outputDir),'Preserve existing scene exports.');mkdir(outputDir);
fig=replay_indian_traffic_3d(runDir);cleanup=onCleanup(@()close(fig)); %#ok<NASGU>
seek=findobj(fig,'Tag','Seek');seek.Value=min(seek.Max,time);feval(seek.Callback,seek,[]);
camera=findobj(fig,'Tag','Camera');names={'follow','driver','overview'};
for choice=1:3
    camera.Value=choice;feval(camera.Callback,camera,[]);drawnow;
    exportgraphics(getappdata(fig,'MainAxes'),fullfile(outputDir,[names{choice},'.png']),'Resolution',150);
end
fid=fopen(fullfile(outputDir,'source.json'),'w','n','UTF-8');
fprintf(fid,'%s\n',jsonencode(struct('run_directory',runDir,'time_s',time,'release',version('-release'), ...
    'limits','Native MATLAB scene exports; UI controls are not included.')));fclose(fid);
copyfile(fullfile(root,'integration','matlab','replay_indian_traffic_3d.m'),outputDir);
end
