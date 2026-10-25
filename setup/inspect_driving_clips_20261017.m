function inspect_driving_clips_20261017
% Verify downloaded video metadata and decode sample frames in MATLAB.
root=fileparts(fileparts(mfilename('fullpath')));
folder=fullfile(root,'media','real_driving_20261017');
manifest=jsondecode(fileread(fullfile(folder,'clips.json')));
output=fullfile(folder,'matlab_video_verification.json');
assert(~isfile(output),'Video verification already exists; preserve it.');
previews=fullfile(folder,'previews');assert(~isfolder(previews));mkdir(previews);
records=cell(numel(manifest.clips),1);
for k=1:numel(manifest.clips)
    entry=manifest.clips(k);reader=VideoReader(fullfile(folder,entry.file));
    assert(reader.Duration>0 && reader.Width>0 && reader.Height>0);
    times=[0,reader.Duration/2,max(0,reader.Duration-1)];names=cell(1,3);
    for j=1:3
        reader.CurrentTime=times(j);assert(hasFrame(reader));frame=readFrame(reader);
        assert(isa(frame,'uint8') && size(frame,3)==3);
        names{j}=sprintf('%s_frame%d.jpg',entry.clip_id,j);
        imwrite(frame,fullfile(previews,names{j}),'Quality',88);
    end
    records{k}=struct('file',entry.file,'clip_id',entry.clip_id,'duration_s',reader.Duration, ...
        'width',reader.Width,'height',reader.Height,'fps',reader.FrameRate, ...
        'video_format',reader.VideoFormat,'decoded_sample_times_s',times, ...
        'decoded_sample_images',{names},'matlab_readable',true);
end
result=struct('status','passed','matlab_release',version('-release'), ...
    'verification','VideoReader metadata and three decoded sample frames per clip; not a complete frame audit', ...
    'clips',{records},'model_inference_run',false);
fid=fopen(output,'w','n','UTF-8');assert(fid>=0);closer=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(result));clear closer
disp(jsonencode(result));
end
