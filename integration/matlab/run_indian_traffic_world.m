function result=run_indian_traffic_world(outputDir,seed,duration)
%RUN_INDIAN_TRAFFIC_WORLD Compatibility entry point for reviewed mixed traffic.
% All actor dynamics live in run_indian_traffic_trial; do not maintain a second
% simulator here. Earlier run source snapshots preserve their original behavior.
arguments
    outputDir (1,:) char = ''
    seed (1,1) double {mustBeInteger,mustBeNonnegative}=42
    duration (1,1) double {mustBePositive}=300
end
result=run_indian_traffic_trial(outputDir,seed,duration, ...
    'IndependentDirections',true,'ParkedClearance',true, ...
    'SmoothFollowing',true,'PedestrianPassing',true,'Overtaking',false);
end
