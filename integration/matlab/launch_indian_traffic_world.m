function runDir=launch_indian_traffic_world(seed,duration)
%LAUNCH_INDIAN_TRAFFIC_WORLD Generate new traffic and open the replay controls.
% Each new run is preserved. Replaying an existing run does not regenerate it.
arguments
    seed (1,1) double {mustBeInteger,mustBeNonnegative}=42
    duration (1,1) double {mustBePositive}=300
end
assert(isfinite(duration),'Tesla:TrafficDuration','Simulation duration must be finite.');
projectDir=fileparts(fileparts(fileparts(mfilename('fullpath'))));
[~,uniqueSuffix]=fileparts(tempname(fullfile(projectDir,'runs')));
runDir=fullfile(projectDir,'runs',['indian_traffic_',datestr(now,'yyyymmdd_HHMMSS'),'_seed',num2str(seed),'_',uniqueSuffix]);
run_indian_traffic_world(runDir,seed,duration);
replay_indian_traffic_world(runDir);
end
