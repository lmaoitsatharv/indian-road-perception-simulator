function fig=launch_indian_traffic_3d(scene)
%LAUNCH_INDIAN_TRAFFIC_3D Open a verified 3D replay in this MATLAB desktop.
arguments
    scene (1,1) string = "selected"
end
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
folder=fullfile(root,'runs','traffic_overnight_20261004');
reviewSelection=fullfile(root,'runs','review_20261013','selected_replay.json');
if any(scene==["selected","mixed","pedestrians","overtaking"])
    selectionFile=fullfile(folder,'selected_replay.json');
    if isfile(reviewSelection),selectionFile=reviewSelection;end
    selected=jsondecode(fileread(selectionFile));
    field='run_directory';if scene=="overtaking",field='overtaking_demo';end
    assert(isstruct(selected) && isscalar(selected) && isfield(selected,field) && ...
        ischar(selected.(field)) && isrow(selected.(field)) && ~isempty(selected.(field)), ...
        'Tesla:ReplaySelection','Selected replay must identify one saved run directory.');
    runDir=resolveSelection(root,folder,selected.(field));
elseif scene=="baseline"
    runDir=fullfile(folder,'mixed_seed42_v6');
else
    runDir=char(scene);
end
% Validate before closing an existing working viewer. The aliases deliberately
% select an explicit reviewed recording, not whichever implementation is newest.
load_traffic_replay(char(runDir));
old=findall(groot,'Type','figure','Tag','TeslaTraffic3D');
for k=1:numel(old),close(old(k));end
fig=replay_indian_traffic_3d(char(runDir),'AutoPlay',true);
end
function path=resolveSelection(root,legacyFolder,path)
% New selection files use project-relative paths. Older snapshots use absolute
% paths, so a moved project can resolve the saved /runs/ suffix in its new root.
absolute=~isempty(regexp(path,'^([A-Za-z]:[\\/]|[\\/])','once'));
if ~absolute,path=fullfile(root,path);end
if ~isfile(fullfile(path,'world.mat'))
    portable=strrep(path,'\','/');marker=strfind(portable,'/runs/');
    if ~isempty(marker)
        candidate=fullfile(root,portable(marker(end)+1:end));
        if isfile(fullfile(candidate,'world.mat')),path=candidate;return;end
    end
    [~,savedFolder]=fileparts(path);candidate=fullfile(legacyFolder,savedFolder);
    if isfile(fullfile(candidate,'world.mat')),path=candidate;end
end
end
