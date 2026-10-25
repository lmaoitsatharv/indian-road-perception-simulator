function result=verify_viewer_review_20261013(outputDir)
%VERIFY_VIEWER_REVIEW_20261013 Exercise real MATLAB replay controls and inputs.
% Uses saved simulation data plus isolated fixtures. Generates no traffic trial.
arguments
    outputDir (1,:) char
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'integration','matlab'));
assert(~isfolder(outputDir),'Choose a new viewer verification output directory.');
mkdir(outputDir);
visibility=get(groot,'DefaultFigureVisible');set(groot,'DefaultFigureVisible','off');
restore=onCleanup(@()set(groot,'DefaultFigureVisible',visibility)); %#ok<NASGU>
source=fullfile(root,'runs','traffic_overnight_20261004','mixed_seed42_v6');
data=load_traffic_replay(source);
assert(size(data.egoLog,1)>100 && numel(data.specs)>1);
unrelated=timer('Name','TeslaReviewUnrelatedTimer');
unrelatedCleanup=onCleanup(@()deleteTimer(unrelated)); %#ok<NASGU>
checks={};

% Real saved recording, including the dated viewer regression.
fig=replay_indian_traffic_3d(source,'AutoPlay',true);
figCleanup=onCleanup(@()closePresent(fig));
pause(.6);drawnow;
assert(getappdata(fig,'DisplayedTime')>.15,'Manual replay did not advance.');
assert(getappdata(fig,'IsPlaying') && ~isappdata(fig,'LastReplayError'));
play=findobj(fig,'Tag','Play');invoke(play);paused=getappdata(fig,'DisplayedTime');
pause(.15);assert(getappdata(fig,'DisplayedTime')==paused);
timerHandle=getappdata(fig,'ReplayTimer');assert(strcmp(timerHandle.Running,'off'));
camera=findobj(fig,'Tag','Camera');ax=getappdata(fig,'MainAxes');
camera.Value=2;invoke(camera);assert(abs(campos(ax)*[0;0;1]-1.35)<1e-9);
assert(strcmp(getappdata(fig,'EgoTransform').Visible,'off'));
camera.Value=3;invoke(camera);assert(campos(ax)*[0;0;1]>100);
camera.Value=1;invoke(camera);assert(strcmp(getappdata(fig,'EgoTransform').Visible,'on'));
slider=findobj(fig,'Tag','Seek');slider.Value=15;invoke(slider);
assert(getappdata(fig,'DisplayedTime')==15);
rate=findobj(fig,'Tag','Rate');rate.Value=3;invoke(rate);invoke(play);
pause(.4);invoke(play);assert(getappdata(fig,'DisplayedTime')>19);
reset=findobj(fig,'Tag','Reset');invoke(reset);
assert(getappdata(fig,'DisplayedTime')==0 && ~getappdata(fig,'IsPlaying'));
close(fig);assert(~isvalid(timerHandle) && isvalid(unrelated));clear figCleanup;
checks{end+1}='saved_v6_autoplay_pause_seek_reset_three_cameras_and_16x';

% One actor used to be transposed by squeeze(), breaking both renderers.
fixture=data;fixture.config.vehicle_count=1;fixture.config.pedestrian_count=0;
fixture.specs=data.specs(1);fixture.egoLog=data.egoLog(1:3,:);
fixture.egoLog(:,1)=[0;.5;1];fixture.actorLog=data.actorLog(1:3,1,:);
singleDir=fullfile(outputDir,'single_actor');mkdir(singleDir);saveScene(singleDir,fixture);
for renderer={@replay_indian_traffic_3d,@replay_indian_traffic_world}
    fig=renderer{1}(singleDir);figCleanup=onCleanup(@()closePresent(fig));
    timerHandle=getappdata(fig,'ReplayTimer');assert(strcmp(timerHandle.Running,'off'));
    slider=findobj(fig,'Tag','Seek');repeat=findobj(fig,'Tag','Repeat');
    rate=findobj(fig,'Tag','Rate');play=findobj(fig,'Tag','Play');
    repeat.Value=0;rate.Value=3;invoke(rate);slider.Value=.9;invoke(slider);invoke(play);
    pause(.3);drawnow;
    assert(getappdata(fig,'DisplayedTime')==1 && ~getappdata(fig,'IsPlaying'));
    assert(strcmp(timerHandle.Running,'off') && strcmp(play.String,'Play'));
    invoke(play);assert(getappdata(fig,'DisplayedTime')<1,'Play at the end must restart.');invoke(play);
    repeat.Value=1;slider.Value=.99;invoke(slider);invoke(play);pause(.15);drawnow;
    assert(getappdata(fig,'IsPlaying'));
    assert(getappdata(fig,'DisplayedTime')>=0 && getappdata(fig,'DisplayedTime')<1);
    invoke(play);assert(~isappdata(fig,'LastReplayError'));
    % Direct deletion bypasses CloseRequestFcn but must still delete its timer.
    delete(fig);assert(~isvalid(timerHandle) && isvalid(unrelated));clear figCleanup;
end
checks{end+1}='single_actor_both_renderers_end_restart_repeat_and_direct_delete';

% Malformed logs fail before a figure/timer is created.
invalid=fixture;invalid.egoLog(:,1)=[0;0;1];
expectInvalidScene('duplicate_timestamps',invalid);
invalid=fixture;invalid.egoLog(2,2)=NaN;expectInvalidScene('nan_pose',invalid);
invalid=fixture;invalid.actorLog=invalid.actorLog(1:2,:,:);expectInvalidScene('actor_time_mismatch',invalid);
invalid=fixture;invalid.egoLog(:,10)=3;expectInvalidScene('invalid_decision',invalid);
invalid=fixture;invalid.egoLog=invalid.egoLog(1,:);invalid.actorLog=invalid.actorLog(1,:,:);
expectInvalidScene('zero_duration',invalid);
invalid=fixture;invalid.specs.width_m=-1;expectInvalidScene('negative_body_width',invalid);
checks{end+1}='malformed_logs_rejected_before_creating_viewers';

frameSource=fullfile(root,'runs','inference_cli_example_20261003','001_Image_9');
frame=read_camera_prediction(frameSource);
assert(~frame.planning_use_valid && isempty(frame.metric_planning_map));
assert(isequal(frame.road_mask,imread(fullfile(frameSource,'road_mask_01.png'))==1));
cameraDir=fullfile(outputDir,'camera_fixture');mkdir(cameraDir);
copyfile(fullfile(frameSource,'road_mask_01.png'),cameraDir);
metadata=jsondecode(fileread(fullfile(frameSource,'frame.json')));
invalid=metadata;invalid.planning_use_valid=true;expectInvalidCamera('planning_true',invalid,'Tesla:CameraPlanningClaim');
invalid=metadata;invalid.metric_projection_available=true;expectInvalidCamera('metric_true',invalid,'Tesla:CameraPlanningClaim');
invalid=metadata;invalid.planning_map=struct('occupied',false);expectInvalidCamera('planning_map',invalid,'Tesla:CameraPlanningClaim');
invalid=metadata;invalid.mask_file='../road_mask_01.png';expectInvalidCamera('path_traversal',invalid,'Tesla:CameraFrameFormat');
invalid=metadata;invalid.threshold=[.5,.5];expectInvalidCamera('threshold_array',invalid,'Tesla:CameraFrameFormat');
invalid=metadata;invalid.image_shape_hwc=[512,512,1];expectInvalidCamera('shape',invalid,'Tesla:CameraFrameFormat');
invalid=rmfield(metadata,'planning_use_valid');expectInvalidCamera('missing_flag',invalid,'Tesla:CameraFrameFormat');
writeJSON(fullfile(cameraDir,'frame.json'),metadata);
imwrite(uint8(255*ones(512)),fullfile(cameraDir,'road_mask_01.png'));
expectError(@()read_camera_prediction(cameraDir),'Tesla:CameraMaskFormat');
checks{end+1}='binary_camera_handoff_preserved_invalid_metadata_and_display_masks_rejected';

result=struct('status','passed','matlab_release',version('-release'), ...
    'checks',{checks},'traffic_trials_generated',false,'reserved_test_images_read',false, ...
    'unrelated_timer_preserved',isvalid(unrelated),'source_recording',source);
writeJSON(fullfile(outputDir,'viewer_review_verification.json'),result);
disp(jsonencode(result,'PrettyPrint',true));

    function expectInvalidScene(name,bad)
        dir=fullfile(outputDir,name);mkdir(dir);saveScene(dir,bad);
        count=numel(findall(groot,'Type','figure'));
        expectError(@()replay_indian_traffic_3d(dir),'Tesla:ReplayFormat');
        assert(numel(findall(groot,'Type','figure'))==count);
    end
    function expectInvalidCamera(name,bad,id)
        writeJSON(fullfile(cameraDir,'frame.json'),bad);
        expectError(@()read_camera_prediction(cameraDir),id);
        checks{end+1}=['camera_rejected_',name];
    end
end
function invoke(control)
feval(control.Callback,control,[]);
end
function saveScene(dir,data)
save(fullfile(dir,'world.mat'),'-struct','data');
end
function expectError(callback,id)
try
    callback();
catch err
    assert(strcmp(err.identifier,id),'Expected %s, got %s: %s',id,err.identifier,err.message);
    return
end
error('Expected %s to be raised.',id);
end
function writeJSON(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
function closePresent(fig)
if isgraphics(fig),close(fig);end
end
function deleteTimer(handle)
if isvalid(handle),stop(handle);delete(handle);end
end
