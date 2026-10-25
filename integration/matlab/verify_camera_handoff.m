function result = verify_camera_handoff(frameDir, outputPath)
%VERIFY_CAMERA_HANDOFF Actual MATLAB read of a Python prediction artifact.
frame = read_camera_prediction(frameDir);
source = imread(fullfile(frameDir,'road_mask_01.png'));
assert(isequal(frame.road_mask,source==1),'MATLAB changed the binary labels.');
assert(~frame.planning_use_valid && isempty(frame.metric_planning_map),'Image pixels must not become an uncalibrated planning map.');
assert(strcmp(frame.metadata.checkpoint_sha256,'93a8faf3d43b77f0ce74d5cdbaf94124728d371c3b68b2a3386f9f72fc106725'),'Wrong locked checkpoint.');
result.status = 'passed';
result.matlab_release = version('-release');
result.frame_directory = frameDir;
result.shape_hw = size(source);
result.original_mask_values = unique(source(:))';
result.positive_pixels = nnz(frame.road_mask);
result.binary_mask_exactly_preserved = true;
result.metric_planning_map_created = false;
result.test_image_contents_read = false;
fid = fopen(outputPath,'w','n','UTF-8');
assert(fid >= 0,'Could not write verification output.');
closer = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(result,'PrettyPrint',true));
disp(jsonencode(result,'PrettyPrint',true));
end
