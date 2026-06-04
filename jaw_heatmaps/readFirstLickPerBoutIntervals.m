function [intervals, info] = readFirstLickPerBoutIntervals(behFile)
% readFirstLickPerBoutIntervals  First tongue-area lick in each bout (group ID).
% Uses Tongue_area_interval_detection_group_intervals_Interval Overlap Assign ID.

info = struct('nTotal', 0, 'nKept', 0, 'nBouts', 0);

T = readtable(behFile, 'VariableNamingRule', 'preserve');
v = T.Properties.VariableNames;

startCol = find(contains(v, 'Interval Start') & contains(v, 'interval_detection') & ...
    ~contains(v, 'group_intervals'), 1);
endCol = find(contains(v, 'Interval End') & contains(v, 'interval_detection') & ...
    ~contains(v, 'group_intervals'), 1);
boutCol = find(contains(v, 'group_intervals') & contains(v, 'Assign ID'), 1);

if isempty(startCol)
    startCol = find(strcmp(v, 'Tongue_area_interval_detection_Interval Start'), 1);
end
if isempty(endCol)
    endCol = find(strcmp(v, 'Tongue_area_interval_detection_Interval End'), 1);
end
if isempty(boutCol)
    boutCol = find(strcmp(v, 'Tongue_area_interval_detection_group_intervals_Interval Overlap Assign ID'), 1);
end
if isempty(startCol) || isempty(endCol) || isempty(boutCol)
    error('Behavior CSV missing lick or bout columns: %s', behFile);
end

starts = double(T{:, startCol});
ends = double(T{:, endCol});
boutIds = double(T{:, boutCol});

valid = isfinite(starts) & isfinite(ends) & ends >= starts & isfinite(boutIds);
info.nTotal = sum(valid);

keepRow = false(size(starts));
boutList = boutIds(valid);
boutList = boutList(:);
info.nBouts = numel(unique(boutList));

for u = unique(boutList)'
    idx = find(valid & boutIds == u);
    [~, im] = min(starts(idx));
    keepRow(idx(im)) = true;
end

info.nKept = sum(keepRow);
intervals = [starts(keepRow), ends(keepRow)];
end
