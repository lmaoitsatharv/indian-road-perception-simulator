% Read-only capability inventory for this project's next integration stage.
% This does not install products, change licenses or run a training job.
scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(fileparts(scriptDir));
outDir = fullfile(projectDir, 'integration', 'runtime');
if ~exist(outDir, 'dir'), mkdir(outDir); end
inventory.status = 'completed';
inventory.matlab_version = version;
inventory.release = version('-release');
inventory.matlab_root = matlabroot;
inventory.checked_at_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', "yyyy-MM-dd'T'HH:mm:ssXXX"));
products = ver;
inventory.installed_products = products;
features = {'MATLAB', 'Simulink', 'Navigation_Toolbox', 'Automated_Driving_Toolbox', ...
    'Sensor_Fusion_and_Tracking', 'Reinforcement_Learning_Toolbox', 'Stateflow'};
inventory.license_tests = struct;
for k = 1:numel(features)
    try
        inventory.license_tests.(features{k}) = logical(license('test', features{k}));
    catch exception
        inventory.license_tests.(features{k}) = false;
        inventory.license_test_errors.(features{k}) = exception.message;
    end
end
functions = {'plannerHybridAStar', 'binaryOccupancyMap', 'validatorOccupancyMap', ...
    'stateSpaceSE2', 'drivingScenario', 'rlDQNAgent', 'rlFunctionEnv', 'jsondecode'};
inventory.function_paths = struct;
for k = 1:numel(functions)
    inventory.function_paths.(functions{k}) = which(functions{k});
end
inventory.note = 'Installed functions and license tests are capability checks; actual execution still needs verification.';
inventory.constructor_smoke = struct;
try
    drivingObject = drivingScenario;
    inventory.constructor_smoke.drivingScenario = true;
catch exception
    inventory.constructor_smoke.drivingScenario = false;
    inventory.constructor_errors.drivingScenario = exception.message;
end
try
    trackingObject = trackingKF;
    inventory.constructor_smoke.trackingKF = true;
catch exception
    inventory.constructor_smoke.trackingKF = false;
    inventory.constructor_errors.trackingKF = exception.message;
end
outFile = fullfile(outDir, 'matlab_environment.json');
fid = fopen(outFile, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'Could not create project capability inventory.');
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(inventory, 'PrettyPrint', true));
disp(['Saved MATLAB capability inventory: ', outFile]);
