function check_matlab_review_20261013(outputDir)
% Save Code Analyzer findings without executing project source files.
root=fileparts(fileparts(mfilename('fullpath')));
assert(~isfolder(outputDir),'Choose a new analyzer output folder.');mkdir(outputDir);
paths=[dir(fullfile(root,'integration','matlab','*.m'));dir(fullfile(root,'setup','*.m'))];
records=cell(numel(paths),1);parseFailures={};
for k=1:numel(paths)
    source=fullfile(paths(k).folder,paths(k).name);
    findings=checkcode(source,'-id');
    records{k}=struct('source',source,'findings',findings);
    if ~isempty(findings) && any(~cellfun('isempty',regexpi({findings.message},'parse error|syntax error')))
        parseFailures{end+1}=source; %#ok<AGROW>
    end
end
result=struct('status','completed','matlab_release',version('-release'), ...
    'files_checked',numel(paths),'parse_failure_files',{parseFailures},'records',{records}, ...
    'note','Static analyzer warnings are retained; dynamic regression results are separate.');
fid=fopen(fullfile(outputDir,'analysis.json'),'w','n','UTF-8');assert(fid>=0);
closer=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(result));clear closer
assert(isempty(parseFailures),'MATLAB Code Analyzer reported parsing failures.');
disp(struct('files_checked',numel(paths),'parse_failures',numel(parseFailures)));
end
