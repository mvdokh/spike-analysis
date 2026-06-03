function irt_telc_jaw_lick_position_heatmap
% irt_telc_jaw_lick_position_heatmap
% Gaussian density heatmaps of jaw keypoints during lick intervals
% (from bottom_behavior / side_behavior CSVs). One SVG with rows per animal
% (IRt_TeLC08, 09, 11): column 1 = Pre, then each post session individually.
% Bottom and side views are separate rows per animal.
%
% Run: irt_telc_jaw_lick_position_heatmap

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

jawCsvPaths = {
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Pre\IRt_TeLC08_pre_2026_03_31__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Pre\IRt_TeLC08_pre_2026_03_31_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_05\IRt_TeLC08_post_2026_04_05__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_05\IRt_TeLC08_post_2026_04_05_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_06\IRt_TeLC08_post_2026_04_06__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_06\IRt_TeLC08_post_2026_04_06_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_07\IRt_TeLC08_post_2026_04_07__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_07\IRt_TeLC08_post_2026_04_07_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_08\IRt_TeLC08_post_2026_04_08__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_08\IRt_TeLC08_post_2026_04_08_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_09\IRt_TeLC08_post_2026_04_09__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_09\IRt_TeLC08_post_2026_04_09_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_10\IRt_TeLC08_post_2026_04_10__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_10\IRt_TeLC08_post_2026_04_10_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Pre\IRt_TeLC09_pre_2026_04_01__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Pre\IRt_TeLC09_pre_2026_04_01_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_06\IRt_TeLC09_post_2026_04_06__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_06\IRt_TeLC09_post_2026_04_06_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_07\IRt_TeLC09_post_2026_04_07real__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_07\IRt_TeLC09_post_2026_04_07real_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_08\IRt_TeLC09_post_2026_04_08__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_08\IRt_TeLC09_post_2026_04_08_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_09\IRt_TeLC09_post_2026_04_09__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_09\IRt_TeLC09_post_2026_04_09_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Pre\IRt_TeLC11_pre_2026_03_30__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Pre\IRt_TeLC11_pre_2026_03_30_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_05\IRt_TeLC11_post_2026_04_05__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_05\IRt_TeLC11_post_2026_04_05_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_06\IRt_TeLC11_post_2026_04_06__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_06\IRt_TeLC11_post_2026_04_06_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_07\IRt_TeLC11_post_2026_04_07__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_07\IRt_TeLC11_post_2026_04_07_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_08\IRt_TeLC11_post_2026_04_08__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_08\IRt_TeLC11_post_2026_04_08_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_09\IRt_TeLC11_post_2026_04_09__jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_09\IRt_TeLC11_post_2026_04_09_1_jaw.csv'
    };

ANIMAL_IDS = {'IRt_TeLC08', 'IRt_TeLC09', 'IRt_TeLC11'};
VIEWS = {'bottom', 'side'};

PROB_MIN = 0;
GRID_SIZE = 256;
GAUSS_TRUNC_BINS = 5;
GAUSS_SIGMA_BINS = GAUSS_TRUNC_BINS / 3;
USE_LOG_SCALE = true;
AXIS_MIN = 0;
AXIS_MAX = 256;

thisDir = fileparts(mfilename('fullpath'));
OUTPUT_DIR = thisDir;
outfileBase = fullfile(OUTPUT_DIR, 'irt_telc_jaw_lick_all_positions');

%% Build per-session panel lists: Pre first, then each post session (sorted by date)
nAnimals = numel(ANIMAL_IDS);
nViews = numel(VIEWS);
sessionLists = cell(nAnimals, nViews);
nColsMax = 0;

for a = 1:nAnimals
    for v = 1:nViews
        sessionLists{a, v} = listJawSessions(jawCsvPaths, ANIMAL_IDS{a}, VIEWS{v});
        nColsMax = max(nColsMax, numel(sessionLists{a, v}));
    end
end

Hcell = cell(nViews, nAnimals, nColsMax);
metaCell = cell(nViews, nAnimals, nColsMax);

for v = 1:nViews
    viewTag = VIEWS{v};
    for a = 1:nAnimals
        sessions = sessionLists{a, v};
        for p = 1:nColsMax
            if p > numel(sessions)
                Hcell{v, a, p} = zeros(GRID_SIZE, GRID_SIZE);
                metaCell{v, a, p} = struct( ...
                    'label', '', 'view', viewTag, 'nPoints', 0, 'path', '');
                continue
            end

            jawFile = sessions{p};
            [xv, yv, nPts] = lickJawPointsForSession(jawFile, viewTag, PROB_MIN);
            if nPts < 1
                Hcell{v, a, p} = zeros(GRID_SIZE, GRID_SIZE);
            else
                Hcell{v, a, p} = gaussianSplat2d(xv, yv, AXIS_MIN, AXIS_MAX, ...
                    AXIS_MIN, AXIS_MAX, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS);
            end
            metaCell{v, a, p} = struct( ...
                'label', sessionPanelLabel(jawFile), ...
                'view', viewTag, ...
                'nPoints', nPts, ...
                'path', jawFile);
        end
    end
end

%% Shared color scale
if USE_LOG_SCALE
    Zcell = cellfun(@(H) log1p(double(H)), Hcell, 'UniformOutput', false);
    cbLabel = 'log(1 + count)';
else
    Zcell = cellfun(@double, Hcell, 'UniformOutput', false);
    cbLabel = 'count';
end
zMax = max(cellfun(@(Z) max(Z(:)), Zcell(:)));
zMax = max(zMax, eps);

%% Figure layout: per animal, bottom row then side row; columns Pre | post sessions
nrow = nAnimals * nViews;
ncol = nColsMax;
panelW = 180;
panelH = 180;
figW = min(140 + panelW * ncol + 90, 3200);
figH = min(140 + panelH * nrow, 2200);

fig = figure('Name', 'IRt TeLC jaw positions during licks', 'NumberTitle', 'off', ...
    'Color', 'w', 'Position', [40 40 figW figH]);

tl = tiledlayout(fig, nrow, ncol, 'Padding', 'compact', 'TileSpacing', 'compact');
title(tl, 'IRt TeLC — jaw position density during licks (Pre + individual post sessions)', ...
    'FontWeight', 'bold', 'FontSize', 14, 'Color', 'k');

tileIdx = 1;
for a = 1:nAnimals
    for v = 1:nViews
        viewLabel = viewTagLabel(VIEWS{v});
        nSess = numel(sessionLists{a, v});
        for p = 1:nColsMax
            ax = nexttile(tl, tileIdx);
            tileIdx = tileIdx + 1;

            meta = metaCell{v, a, p};
            if p > nSess
                axis(ax, 'off');
                continue
            end

            imagesc(ax, Zcell{v, a, p});
            axis(ax, 'image');
            pbaspect(ax, [1 1 1]);
            set(ax, 'YDir', 'reverse', 'XColor', 'k', 'YColor', 'k', 'FontSize', 8);
            colormap(ax, parula(256));
            caxis(ax, [0 zMax]);
            xticks(ax, []);
            yticks(ax, []);

            if p == 1
                ylabel(ax, sprintf('%s\n%s', ANIMAL_IDS{a}, viewLabel), ...
                    'FontWeight', 'bold', 'FontSize', 9, 'Color', 'k');
            end

            title(ax, {meta.label, sprintf('%d pts', meta.nPoints)}, ...
                'FontSize', 7, 'Color', 'k', 'Interpreter', 'none');
        end
    end
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Color = 'k';
cb.Label.String = cbLabel;
cb.Label.Interpreter = 'none';
cb.Label.Color = 'k';

outPath = [outfileBase '.svg'];
try
    exportgraphics(fig, outPath, 'ContentType', 'vector');
catch
    print(fig, outPath, '-dsvg', '-painters');
end
fprintf('Saved: %s\n', outPath);

for a = 1:nAnimals
    for v = 1:nViews
        for p = 1:numel(sessionLists{a, v})
            meta = metaCell{v, a, p};
            fprintf('%s | %s | %s: %d points\n', ...
                ANIMAL_IDS{a}, meta.view, meta.label, meta.nPoints);
        end
    end
end
end


function label = viewTagLabel(viewTag)
switch lower(viewTag)
    case 'bottom'
        label = 'Bottom view';
    case 'side'
        label = 'Side view';
    otherwise
        label = viewTag;
end
end


function sessions = listJawSessions(jawCsvPaths, animalTag, viewTag)
sessions = {};
sortKeys = [];

for k = 1:numel(jawCsvPaths)
    jawFile = jawCsvPaths{k};
    if ~isfile(jawFile)
        continue
    end
    [animal, cond, view] = parseJawCsvMeta(jawFile);
    if ~strcmp(animal, animalTag) || ~strcmp(view, viewTag)
        continue
    end
    sessions{end+1, 1} = jawFile; %#ok<AGROW>
    if strcmp(cond, 'pre')
        sortKeys(end+1, 1) = 0; %#ok<AGROW>
    else
        sortKeys(end+1, 1) = sessionSortKey(jawFile); %#ok<AGROW>
    end
end

if isempty(sessions)
    return
end
[~, ord] = sort(sortKeys);
sessions = sessions(ord);
end


function key = sessionSortKey(jawFile)
[~, baseName, ~] = fileparts(jawFile);
tok = regexp(lower(baseName), '(\d{4})_(\d{2})_(\d{2})', 'tokens', 'once');
if isempty(tok)
    key = inf;
else
    key = str2double(tok{1}) * 10000 + str2double(tok{2}) * 100 + str2double(tok{3});
end
end


function label = sessionPanelLabel(jawFile)
[~, baseName, ~] = fileparts(jawFile);
baseLower = lower(baseName);
if contains(baseLower, '_pre_') || contains(baseLower, '_pre')
    label = 'Pre';
    return
end
tok = regexp(baseLower, 'post_(\d{4}[_-]\d{2}[_-]\d{2}[a-z]*)', 'tokens', 'once');
if ~isempty(tok)
    label = ['Post ' strrep(tok{1}, '_', '-')];
    return
end
label = baseName;
end


function [xv, yv, nPoints] = lickJawPointsForSession(jawFile, viewTag, probMin)
sessionDir = fileparts(jawFile);
behFile = findBehaviorCsv(sessionDir, viewTag);
if isempty(behFile)
    warning('No %s behavior CSV in: %s', viewTag, sessionDir);
    xv = [];
    yv = [];
    nPoints = 0;
    return
end

[xv, yv] = jawPointsDuringLicks(jawFile, behFile, probMin);
nPoints = numel(xv);
if nPoints < 1
    warning('No lick jaw points: %s', jawFile);
end
end


function [animal, cond, view] = parseJawCsvMeta(jawFile)
[~, baseName, ~] = fileparts(jawFile);
baseLower = lower(baseName);

tok = regexp(baseLower, 'irt_telc(\d+)', 'tokens', 'once');
if isempty(tok)
    animal = '';
else
    animal = sprintf('IRt_TeLC%s', tok{1});
end

if contains(baseLower, '_pre_') || contains(baseLower, '_pre')
    cond = 'pre';
elseif contains(baseLower, '_post_') || contains(baseLower, '_post')
    cond = 'post';
else
    cond = '';
end

if endsWith(baseLower, '__jaw') || contains(baseLower, '__jaw')
    view = 'bottom';
elseif contains(baseLower, '_1_jaw')
    view = 'side';
else
    view = '';
end
end


function behFile = findBehaviorCsv(sessionDir, viewTag)
behFile = '';
switch lower(viewTag)
    case 'bottom'
        pattern = '*bottom_behavior*.csv';
    case 'side'
        pattern = '*side_behavior*.csv';
    otherwise
        return
end
listing = dir(fullfile(sessionDir, pattern));
if isempty(listing)
    return
end
[~, ord] = sort({listing.name});
behFile = fullfile(sessionDir, listing(ord(1)).name);
end


function [xv, yv] = jawPointsDuringLicks(jawFile, behFile, probMin)
tbl = readJawCsv(jawFile);
intervals = readLickIntervals(behFile);

if isempty(intervals)
    xv = [];
    yv = [];
    return
end

frm = tbl.Frame;
if isempty(probMin) || (~isscalar(probMin)) || probMin <= 0
    keepProb = true(size(frm));
else
    keepProb = tbl.Probability >= probMin;
end

lickMask = false(size(frm));
for i = 1:size(intervals, 1)
    s = intervals(i, 1);
    e = intervals(i, 2);
    lickMask = lickMask | (frm >= s & frm <= e);
end

keep = keepProb & lickMask;
xv = tbl.X(keep);
yv = tbl.Y(keep);
end


function intervals = readLickIntervals(behFile)
T = readtable(behFile, 'VariableNamingRule', 'preserve');
v = T.Properties.VariableNames;
startCol = find(contains(v, 'Interval Start') & contains(v, 'interval_detection'), 1);
endCol = find(contains(v, 'Interval End') & contains(v, 'interval_detection'), 1);
if isempty(startCol)
    startCol = find(strcmp(v, 'Tongue_area_interval_detection_Interval Start'), 1);
end
if isempty(endCol)
    endCol = find(strcmp(v, 'Tongue_area_interval_detection_Interval End'), 1);
end
if isempty(startCol) || isempty(endCol)
    error('Behavior CSV missing lick interval columns: %s', behFile);
end
starts = double(T{:, startCol});
ends = double(T{:, endCol});
valid = isfinite(starts) & isfinite(ends) & ends >= starts;
intervals = [starts(valid), ends(valid)];
end


function tbl = readJawCsv(csvFile)
T = readtable(csvFile, 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
v = T.Properties.VariableNames;
vl = lower(strtrim(v));
fc = @(s) strcmp(vl, lower(s));

fi = fc('frame');
xi = fc('x');
yi = fc('y');
pi = fc('probability');
if ~(any(fi) && any(xi) && any(yi))
    error('Jaw CSV must include Frame, X, and Y columns: %s', csvFile);
end

F = double(T{:, fi});
X = double(T{:, xi});
Y = double(T{:, yi});
if any(pi)
    Pr = double(T{:, pi});
else
    Pr = ones(size(F));
end
tbl = table(F, X, Y, Pr, 'VariableNames', {'Frame', 'X', 'Y', 'Probability'});
end


function H = gaussianSplat2d(xv, yv, xMin, xMax, yMin, yMax, n, sigma, truncR)
H = zeros(n, n);
if isempty(xv)
    return
end

xv = double(xv(:));
yv = double(yv(:));

dx = max(double(xMax) - double(xMin), eps);
dy = max(double(yMax) - double(yMin), eps);

truncR = max(1, round(double(truncR)));
sigma = max(double(sigma), eps);
truncRsq = truncR^2;

xCol = (xv - double(xMin)) ./ dx .* double(n);
yRow = (yv - double(yMin)) ./ dy .* double(n);

for kk = 1:numel(xv)
    cx = xCol(kk);
    cy = yRow(kk);
    jLo = max(1, ceil(cx - truncR - 1));
    jHi = min(n, floor(cx + truncR + 1));
    iLo = max(1, ceil(cy - truncR - 1));
    iHi = min(n, floor(cy + truncR + 1));
    jj = jLo:jHi;
    ii = iLo:iHi;
    [JG, IG] = meshgrid(jj, ii);
    dxc = JG - 0.5 - cx;
    dyr = IG - 0.5 - cy;
    dc2 = dxc.^2 + dyr.^2;
    mask = dc2 <= truncRsq;
    kern = zeros(size(dc2));
    kern(mask) = exp(-dc2(mask) ./ (2 * sigma^2));
    H(ii, jj) = H(ii, jj) + kern;
end
end
