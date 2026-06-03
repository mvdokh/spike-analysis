function irt_telc_jaw_lick_trajectory_phase_by_session
% irt_telc_jaw_lick_trajectory_phase_by_session
% Intra-lick jaw trajectories in image pixel coordinates, colored by phase
% (0=start, 1=end of lick) from behavior CSV lick intervals.
% One SVG: rows = animal x view (bottom/side), columns = Pre then each post
% session individually. Style matches lick_trajectory_phase_density_overlay_by_animal.m
%
% Run: irt_telc_jaw_lick_trajectory_phase_by_session

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
MIN_LICK_FRAMES = 2;
AXIS_MIN = 0;
AXIS_MAX = 256;
AXIS_PAD = 8;
DRAW_SEGMENT_LINES = true;
DRAW_SCATTER = true;
LINE_WIDTH = 1.15;
MARKER_SIZE = 14;
MAX_LICKS_PLOT = inf;

thisDir = fileparts(mfilename('fullpath'));
OUTPUT_DIR = thisDir;
outfileBase = fullfile(OUTPUT_DIR, 'irt_telc_jaw_lick_trajectory_phase_by_session');

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

cmapPhase = phaseColormap256();

nrow = nAnimals * nViews;
ncol = nColsMax;
panelW = 200;
panelH = 200;
figW = min(160 + panelW * ncol + 100, 3400);
figH = min(160 + panelH * nrow, 2400);

fig = figure('Name', 'IRt TeLC jaw lick trajectories by session', 'NumberTitle', 'off', ...
    'Color', 'w', 'Position', [40 40 figW figH]);

tl = tiledlayout(fig, nrow, ncol, 'Padding', 'compact', 'TileSpacing', 'compact');
title(tl, 'IRt TeLC — intra-lick jaw trajectories by session (Pre + individual post)', ...
    'FontWeight', 'bold', 'FontSize', 14, 'Color', 'k');
colormap(fig, cmapPhase);

tileIdx = 1;
for a = 1:nAnimals
    for v = 1:nViews
        viewTag = VIEWS{v};
        viewLabel = viewTagLabel(viewTag);
        sessions = sessionLists{a, v};
        nSess = numel(sessions);

        for p = 1:nColsMax
            ax = nexttile(tl, tileIdx);
            tileIdx = tileIdx + 1;

            if p > nSess
                axis(ax, 'off');
                continue
            end

            jawFile = sessions{p};
            sessLabel = sessionPanelLabel(jawFile);
            [~, ~, condTag] = parseJawCsvMeta(jawFile);

            nLicks = drawSessionTrajectories(ax, jawFile, viewTag, PROB_MIN, ...
                MIN_LICK_FRAMES, cmapPhase, DRAW_SEGMENT_LINES, DRAW_SCATTER, ...
                LINE_WIDTH, MARKER_SIZE, MAX_LICKS_PLOT, AXIS_MIN, AXIS_MAX, AXIS_PAD);

            setupTrajectoryAxes(ax, AXIS_MIN, AXIS_MAX, cmapPhase);

            if p == 1
                ylabel(ax, sprintf('%s\n%s', ANIMAL_IDS{a}, viewLabel), ...
                    'FontWeight', 'bold', 'FontSize', 9, 'Color', 'k');
            end

            titleColor = condTitleColor(condTag);
            title(ax, {sessLabel, sprintf('%d licks', nLicks)}, ...
                'FontSize', 8, 'Color', titleColor, 'Interpreter', 'none');
        end
    end
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Color = 'k';
cb.Label.String = 'Intra-lick phase (0=start, 1=end)';
cb.Label.Interpreter = 'none';
cb.Label.Color = 'k';

outPath = [outfileBase '.svg'];
try
    exportgraphics(fig, outPath, 'ContentType', 'vector');
catch
    print(fig, outPath, '-dsvg', '-painters');
end
fprintf('Saved: %s\n', outPath);
end


function setupTrajectoryAxes(ax, axisMin, axisMax, cmapPhase)
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
axis(ax, 'equal');
axis(ax, 'square');
set(ax, 'YDir', 'reverse');
xlim(ax, [axisMin axisMax]);
ylim(ax, [axisMin axisMax]);
xticks(ax, [axisMin axisMax/2 axisMax]);
yticks(ax, [axisMin axisMax/2 axisMax]);
colormap(ax, cmapPhase);
caxis(ax, [0 1]);
end


function rgb = condTitleColor(condTag)
switch lower(condTag)
    case 'pre'
        rgb = [0.05 0.35 0.75];
    case 'post'
        rgb = [0.75 0.15 0.05];
    otherwise
        rgb = 'k';
end
end


function nLicks = drawSessionTrajectories(ax, jawFile, viewTag, probMin, minLickFrames, ...
    cmapPhase, drawLines, drawScatter, lineW, markerSize, maxLicksPlot, axisMin, axisMax, axisPad)

nLicks = 0;
[lickX, lickY, lickPhase] = sessionJawLickTrajectories(jawFile, viewTag, probMin, minLickFrames);
if isempty(lickX)
    return
end

nPlot = numel(lickX);
if isfinite(maxLicksPlot) && nPlot > maxLicksPlot
    pick = round(linspace(1, nPlot, maxLicksPlot));
    lickX = lickX(pick);
    lickY = lickY(pick);
    lickPhase = lickPhase(pick);
    nPlot = numel(lickX);
end

hold(ax, 'on');
for j = 1:nPlot
    xs = lickX{j};
    ys = lickY{j};
    ph = lickPhase{j};
    if isempty(xs)
        continue
    end
    nLicks = nLicks + 1;

    if drawLines && numel(xs) > 1
        pmid = (ph(1:end-1) + ph(2:end)) / 2;
        lineColors = rgbFromPhase(pmid, cmapPhase);
        for ii = 1:(numel(xs) - 1)
            plot(ax, xs(ii:ii + 1), ys(ii:ii + 1), '-', ...
                'Color', lineColors(ii, :), 'LineWidth', lineW, ...
                'HandleVisibility', 'off');
        end
    end

    if drawScatter
        scatter(ax, xs, ys, markerSize, ph, 'filled', ...
            'MarkerFaceAlpha', 0.9, 'MarkerEdgeColor', 'none', ...
            'HandleVisibility', 'off');
    end
end
hold(ax, 'off');

if nLicks > 0
    allX = cell2mat(lickX(:));
    allY = cell2mat(lickY(:));
    xLo = max(axisMin, min(allX) - axisPad);
    xHi = min(axisMax, max(allX) + axisPad);
    yLo = max(axisMin, min(allY) - axisPad);
    yHi = min(axisMax, max(allY) + axisPad);
    if xHi > xLo && yHi > yLo
        xlim(ax, [xLo xHi]);
        ylim(ax, [yLo yHi]);
    end
end
end


function [lickX, lickY, lickPhase] = sessionJawLickTrajectories(jawFile, viewTag, probMin, minLickFrames)
lickX = {};
lickY = {};
lickPhase = {};

sessionDir = fileparts(jawFile);
behFile = findBehaviorCsv(sessionDir, viewTag);
if isempty(behFile)
    warning('No %s behavior CSV in: %s', viewTag, sessionDir);
    return
end

tbl = readJawCsv(jawFile);
intervals = readLickIntervals(behFile);
if isempty(intervals)
    return
end

frm = tbl.Frame;
xv = tbl.X;
yv = tbl.Y;
if isempty(probMin) || (~isscalar(probMin)) || probMin <= 0
    keepProb = true(size(frm));
else
    keepProb = tbl.Probability >= probMin;
end

for i = 1:size(intervals, 1)
    s = intervals(i, 1);
    e = intervals(i, 2);
    inLick = (frm >= s) & (frm <= e) & keepProb;
    if ~any(inLick)
        continue
    end

    fi = frm(inLick);
    xi = xv(inLick);
    yi = yv(inLick);
    [~, ord] = sort(fi);
    xs = xi(ord);
    ys = yi(ord);
    L = numel(xs);
    if L < minLickFrames
        continue
    end

    if L == 1
        ph = 0.5;
    else
        ph = ((0:(L - 1))' / (L - 1));
    end

    lickX{end + 1} = xs; %#ok<AGROW>
    lickY{end + 1} = ys; %#ok<AGROW>
    lickPhase{end + 1} = ph; %#ok<AGROW>
end
end


function cmap = phaseColormap256()
try
    cmap = turbo(256);
catch
    cmap = jet(256);
end
end


function rgb = rgbFromPhase(phase, cmap)
phase = phase(:);
phase = max(0, min(1, phase));
idx = round(phase * (size(cmap, 1) - 1)) + 1;
rgb = cmap(idx, :);
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
