function result = run_static_reference(outputDir)
%RUN_STATIC_REFERENCE One declared ground-truth Hybrid A* development scene.
% Road geometry is synthetic. No camera mask, RL, moving actors or controller.
% Vehicle poses refer to the rear axle center; x/y are metres, heading radians.
arguments
    outputDir (1,:) char = ''
end
projectDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if isempty(outputDir)
    outputDir = fullfile(projectDir, 'runs', 'static_reference_20261003');
end
assert(~exist(outputDir, 'dir'), 'Choose a new output directory to preserve previous evidence.');
mkdir(outputDir);
result.status = 'running';
result.kind = 'ground_truth_static_development_reference';
result.learned_mask_used = false;
result.rl_used = false;
result.controller_used = false;
result.timestamp_source = 'simulation_time';
try
    rng(42, 'twister');
    config.schema_version = '0.1-draft';
    config.seed = 42;
    config.world_frame = 'reference_world_x_forward_y_left';
    config.units = struct('position', 'metres', 'heading', 'radians', 'time', 'seconds');
    config.pose_reference = 'rear_axle_center';
    config.map_size_m = [60, 30];
    config.cells_per_metre = 5;
    config.road_bounds_xyxy_m = [2, 4, 58, 26];
    config.obstacle_bounds_xyxy_m = [27.75, 14, 32.25, 16];
    config.start_pose = [8, 15, 0];
    config.goal_pose = [52, 15, 0];
    config.vehicle = struct('length_m', 4.5, 'width_m', 1.8, ...
        'wheelbase_m', 2.7, 'rear_overhang_m', .9, 'front_of_rear_axle_m', 3.6);
    config.safety_margin_m = .25;
    config.map_discretization_extra_m = sqrt(2)/(2*config.cells_per_metre);
    % Conservative enclosing circle centered at the rear axle, including margin.
    config.footprint_radius_m = hypot(config.vehicle.front_of_rear_axle_m, config.vehicle.width_m/2);
    config.inflation_radius_m = config.footprint_radius_m + config.safety_margin_m + config.map_discretization_extra_m;
    config.min_turning_radius_m = 6;
    config.motion_primitive_length_m = 1;
    config.path_spacing_m = .05;
    config.validation_spacing_m = .05;
    config.motion_direction = 'forward';
    config.search_mode = 'greedy';
    config.max_nodes = 50000;
    config.note = 'Wide development corridor permits conservative circle inflation. This is one static scene, not final driving evaluation.';
    writeJSON(fullfile(outputDir, 'scene_config.json'), config);

    map = binaryOccupancyMap(config.map_size_m(1), config.map_size_m(2), config.cells_per_metre);
    [x, y] = meshgrid((.5:map.GridSize(2)-.5)/map.Resolution, (.5:map.GridSize(1)-.5)/map.Resolution);
    road = config.road_bounds_xyxy_m;
    obstacle = config.obstacle_bounds_xyxy_m;
    blocked = x < road(1) | x > road(3) | y < road(2) | y > road(4) | ...
        (x >= obstacle(1) & x <= obstacle(3) & y >= obstacle(2) & y <= obstacle(4));
    setOccupancy(map, [x(:), y(:)], blocked(:));
    planningMap = copy(map);
    inflate(planningMap, config.inflation_radius_m);
    space = stateSpaceSE2;
    space.StateBounds = [map.XWorldLimits; map.YWorldLimits; [-pi, pi]];
    validator = validatorOccupancyMap(space);
    validator.Map = planningMap;
    validator.ValidationDistance = config.validation_spacing_m;
    assert(all(isStateValid(validator, [config.start_pose; config.goal_pose])), 'Start/goal invalid in the inflated map.');
    planner = plannerHybridAStar(validator, MinTurningRadius=config.min_turning_radius_m, ...
        MotionPrimitiveLength=config.motion_primitive_length_m, InterpolationDistance=config.path_spacing_m, ...
        MotionDirection=config.motion_direction, MaxNumNodes=config.max_nodes, MaxNumPathStates=10000);
    started = tic;
    [path, directions, info] = plan(planner, config.start_pose, config.goal_pose, SearchMode=config.search_mode);
    elapsed = toc(started);
    assert(info.IsPathFound && ~isempty(directions), 'Hybrid A* did not find a path.');
    poses = path.States;
    assert(all(directions == 1), 'The reference scene requires forward motion.');
    assert(all(isStateValid(validator, poses)), 'A planned pose entered inflated occupied space.');

    % Independently check the actual oriented vehicle rectangle at every output pose.
    body = [-.9, -.9; 3.6, -.9; 3.6, .9; -.9, .9];
    obsCorners = [obstacle(1:2); obstacle(3),obstacle(2); obstacle(3:4); obstacle(1),obstacle(4)];
    minimumRoadClearance = inf;
    minimumObstacleClearance = inf;
    for k = 1:size(poses, 1)
        angle = poses(k,3);
        rotation = [cos(angle), -sin(angle); sin(angle), cos(angle)];
        corners = body*rotation' + poses(k,1:2);
        roadClearance = min([corners(:,1)-road(1); road(3)-corners(:,1); ...
                            corners(:,2)-road(2); road(4)-corners(:,2)]);
        obstacleClearance = rectangleDistance(corners, obsCorners, angle);
        minimumRoadClearance = min(minimumRoadClearance, roadClearance);
        minimumObstacleClearance = min(minimumObstacleClearance, obstacleClearance);
        assert(roadClearance >= config.safety_margin_m, 'Vehicle footprint left the road safety margin.');
        assert(obstacleClearance >= config.safety_margin_m, 'Vehicle footprint violated the obstacle safety margin.');
    end
    distance = hypot(diff(poses(:,1)), diff(poses(:,2)));
    headingChange = atan2(sin(diff(poses(:,3))), cos(diff(poses(:,3))));
    nonzero = distance > 1e-8;
    curvature = abs(headingChange(nonzero))./distance(nonzero);
    assert(max(curvature) <= 1/config.min_turning_radius_m + 1e-3, 'Sampled turning radius check failed.');
    cumulative = [0; cumsum(distance)];
    trajectory = table(cumulative, poses(:,1), poses(:,2), poses(:,3), directions, ...
        'VariableNames', {'distance_m','rear_axle_x_m','rear_axle_y_m','heading_rad','direction'});
    writetable(trajectory, fullfile(outputDir, 'reference_path.csv'));
    replay = struct('schema_version', '0.1-draft', 'kind', 'geometric_path_replay', ...
        'config', config, 'poses', poses, 'distance_m', cumulative, ...
        'nominal_replay_speed_mps', 3, 'controller_used', false, 'rl_used', false, 'learned_mask_used', false);
    writeJSON(fullfile(outputDir, 'replay.json'), replay);

    result.status = 'completed';
    result.matlab_release = version('-release');
    result.path_found = logical(info.IsPathFound);
    result.search_info = info;
    result.plan_wall_seconds = elapsed;
    result.path_length_m = cumulative(end);
    result.path_poses = size(poses,1);
    result.minimum_road_footprint_clearance_m = minimumRoadClearance;
    result.minimum_obstacle_footprint_clearance_m = minimumObstacleClearance;
    result.maximum_sampled_curvature_per_m = max(curvature);
    result.minimum_sampled_turning_radius_m = 1/max(curvature);
    result.start_pose_error_m = norm(poses(1,1:2)-config.start_pose(1:2));
    result.goal_pose_error_m = norm(poses(end,1:2)-config.goal_pose(1:2));
    result.checked_every_exported_pose = true;
    result.collision_check_scope = 'Planner samples motions at 0.05m. Independent rectangle clearance checked at exported poses spaced about 0.05m; not a continuous-time driving proof.';
    result.closed_loop_driving_verified = false;
    result.final_simulation_evaluation = false;
    result.config = config;
    writeJSON(fullfile(outputDir, 'result.json'), result);

    figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100,100,1400,650]);
    closer = onCleanup(@() close(figureHandle));
    axesHandle = axes(figureHandle);
    show(map, 'Parent', axesHandle);
    set(axesHandle, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    hold(axesHandle, 'on');
    plot(axesHandle, poses(:,1), poses(:,2), '-', 'Color', [.05,.4,.75], 'LineWidth', 2.3);
    for index = unique(round(linspace(1,size(poses,1),10)))
        angle = poses(index,3);
        corners = body*[cos(angle),sin(angle);-sin(angle),cos(angle)] + poses(index,1:2);
        patch(axesHandle, corners(:,1), corners(:,2), [.1,.7,.5], 'FaceAlpha', .25, 'EdgeColor', [.0,.35,.25]);
    end
    scatter(axesHandle, config.start_pose(1), config.start_pose(2), 90, [.1,.6,.2], 'filled');
    scatter(axesHandle, config.goal_pose(1), config.goal_pose(2), 90, [.9,.4,.1], 'filled');
    text(axesHandle, config.start_pose(1), config.start_pose(2)-2, 'Start', 'HorizontalAlignment', 'center', 'Color', 'w');
    text(axesHandle, config.goal_pose(1), config.goal_pose(2)-2, 'Goal', 'HorizontalAlignment', 'center', 'Color', 'w');
    title(axesHandle, {'Ground-truth static Hybrid A* reference', ...
        'Rear axle poses, metre map and full rectangle checks | No U-Net map, RL or controller'}, 'Color', 'k');
    xlabel(axesHandle, 'World x (metres)', 'Color', 'k'); ylabel(axesHandle, 'World y (metres)', 'Color', 'k');
    axis(axesHandle, 'equal'); xlim(axesHandle, map.XWorldLimits); ylim(axesHandle, map.YWorldLimits);
    exportgraphics(figureHandle, fullfile(outputDir, 'static_reference.png'), 'Resolution', 150);
    save(fullfile(outputDir, 'reference_scene.mat'), 'config', 'map', 'planningMap', 'poses', 'directions', 'result');
    disp(jsonencode(rmfield(result, 'config'), 'PrettyPrint', true));
catch exception
    result.status = 'failed'; result.error = exception.message;
    writeJSON(fullfile(outputDir, 'result.json'), result);
    rethrow(exception);
end
end

function distance = rectangleDistance(first, second, angle)
axes = [1,0; 0,1; cos(angle),sin(angle); -sin(angle),cos(angle)];
disjoint = false;
for k = 1:4
    a = first*axes(k,:)'; b = second*axes(k,:)';
    disjoint = disjoint || max(a) < min(b) || max(b) < min(a);
end
if ~disjoint, distance = 0; return; end
distance = inf;
for k = 1:4
    next = mod(k,4)+1;
    for j = 1:4
        distance = min(distance, pointSegmentDistance(first(j,:),second(k,:),second(next,:)));
        distance = min(distance, pointSegmentDistance(second(j,:),first(k,:),first(next,:)));
    end
end
end

function distance = pointSegmentDistance(point, a, b)
vector = b-a;
fraction = max(0,min(1,dot(point-a,vector)/dot(vector,vector)));
distance = norm(point-(a+fraction*vector));
end

function writeJSON(path, value)
fid = fopen(path, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'Could not write output JSON.');
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end
