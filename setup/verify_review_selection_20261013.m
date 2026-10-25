function result=verify_review_selection_20261013(outputDir)
%VERIFY_REVIEW_SELECTION_20261013 Check promoted recordings and baseline access.
% Opens saved viewers only, and closes only figures created by this verifier.
arguments
    outputDir (1,:) char
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'integration','matlab'));
assert(~isfolder(outputDir),'Choose a new selection verification output directory.');
mkdir(outputDir);
visibility=get(groot,'DefaultFigureVisible');set(groot,'DefaultFigureVisible','off');
restore=onCleanup(@()set(groot,'DefaultFigureVisible',visibility)); %#ok<NASGU>
ownedFigures=gobjects(0);ownedTimers={};
% The cleanup callback must not depend on a nested workspace being dismantled.
% A handle container keeps the current owned figures available on exception exit.
cleanupState=containers.Map('KeyType','char','ValueType','any');
cleanupState('figures')=ownedFigures;
cleanup=onCleanup(@()closeTrackedFigures(cleanupState));
legacySelection=fullfile(root,'runs','traffic_overnight_20261004','selected_replay.json');
legacyBefore=fileread(legacySelection);
selectionPath=fullfile(root,'runs','review_20261013','selected_replay.json');
assert(isfile(selectionPath),'The new reviewed selection must exist.');
reviewed=jsondecode(fileread(selectionPath));
assert(startsWith(strrep(reviewed.run_directory,'\','/'),'runs/'), ...
    'Use a portable project-relative reviewed recording path.');
expectedMixed=fullfile(root,'runs','review_20261013','traffic_verification_v8_02','mixed_seed42_v8');
expectedOvertake=fullfile(root,'runs','review_20261013','traffic_verification_v8_02','overtake_clear_v8');
expectedBaseline=fullfile(root,'runs','traffic_overnight_20261004','mixed_seed42_v6');
unrelated=figure('Visible','off','Tag','UnrelatedReviewFigure');
ownedFigures(end+1)=unrelated;
cleanupState('figures')=ownedFigures;

fig=openScene();assert(samePath(getappdata(fig,'RunDirectory'),expectedMixed));
pause(.4);drawnow;assert(getappdata(fig,'IsPlaying') && getappdata(fig,'DisplayedTime')>.1);
pauseViewer(fig);assert(~isappdata(fig,'LastReplayError'));
seek=findobj(fig,'Tag','Seek');seek.Value=130;feval(seek.Callback,seek,[]);
ax=getappdata(fig,'MainAxes');
exportgraphics(ax,fullfile(outputDir,'review_selected_follow.png'),'Resolution',140);
camera=findobj(fig,'Tag','Camera');camera.Value=2;feval(camera.Callback,camera,[]);
exportgraphics(ax,fullfile(outputDir,'review_selected_driver.png'),'Resolution',140);
camera.Value=3;feval(camera.Callback,camera,[]);
exportgraphics(ax,fullfile(outputDir,'review_selected_overview.png'),'Resolution',140);
% Bad selection is rejected before destroying a working recording.
try
    launch_indian_traffic_3d(string(fullfile(outputDir,'missing_recording')));
    error('Expected missing recording to be rejected.');
catch err
    assert(strcmp(err.identifier,'Tesla:ReplayMissing'));
end
assert(isgraphics(fig) && isgraphics(unrelated));

fig=openScene('mixed');assert(samePath(getappdata(fig,'RunDirectory'),expectedMixed));pauseViewer(fig);
fig=openScene('pedestrians');assert(samePath(getappdata(fig,'RunDirectory'),expectedMixed));pauseViewer(fig);
fig=openScene('baseline');assert(samePath(getappdata(fig,'RunDirectory'),expectedBaseline));pauseViewer(fig);
fig=openScene('overtaking');assert(samePath(getappdata(fig,'RunDirectory'),expectedOvertake));pauseViewer(fig);
assert(isgraphics(unrelated),'Launcher closed an unrelated figure.');
assert(strcmp(fileread(legacySelection),legacyBefore),'The historical selection was changed.');
closeOwned();
clear cleanup; % Run the guard while its handle container is still alive.
assert(all(cellfun(@(t)~isvalid(t),ownedTimers)),'A verifier-owned replay timer was leaked.');
result=struct('status','passed','matlab_release',version('-release'), ...
    'default_mixed_and_pedestrians',expectedMixed,'baseline',expectedBaseline, ...
    'overtaking',expectedOvertake,'project_relative_selection',true, ...
    'invalid_scene_preserved_working_viewer',true,'unrelated_figure_preserved',true, ...
    'historical_selection_unchanged',true,'owned_timers_deleted',true, ...
    'traffic_trials_generated',false,'reserved_test_images_read',false);
fid=fopen(fullfile(outputDir,'selection_verification.json'),'w','n','UTF-8');assert(fid>=0);
fileCleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(result,'PrettyPrint',true));
disp(jsonencode(result,'PrettyPrint',true));

    function fig=openScene(varargin)
        fig=launch_indian_traffic_3d(varargin{:});
        ownedFigures(end+1)=fig;
        cleanupState('figures')=ownedFigures;
        ownedTimers{end+1}=getappdata(fig,'ReplayTimer');
    end
    function closeOwned()
        for k=1:numel(ownedFigures)
            if isgraphics(ownedFigures(k)),close(ownedFigures(k));end
        end
    end
end
function pauseViewer(fig)
if getappdata(fig,'IsPlaying')
    play=findobj(fig,'Tag','Play');feval(play.Callback,play,[]);
end
end
function equal=samePath(a,b)
[okA,infoA]=fileattrib(a);[okB,infoB]=fileattrib(b);
equal=okA && okB && strcmpi(infoA.Name,infoB.Name);
end
function closeTrackedFigures(state)
figures=state('figures');
for k=1:numel(figures)
    if isgraphics(figures(k)),close(figures(k));end
end
end
