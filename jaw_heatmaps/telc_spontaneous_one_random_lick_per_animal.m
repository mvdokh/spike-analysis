function telc_spontaneous_one_random_lick_per_animal
% telc_spontaneous_one_random_lick_per_animal
% IRt_TeLC Pre side view: N_RANDOM_LICKS random spontaneous licks per animal.
% No trajectory or probability filter. PCHIP-smoothed phase-colored line only.
%
% Run: telc_spontaneous_one_random_lick_per_animal

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

jawCsvPaths = telc_pre_side_jaw_paths();

PROB_MIN = 0;
MIN_LICK_FRAMES = 2;
LICK_FRAME_PAD = 10;     % extra frames before/after behavior lick Start/End
RANDOM_SEED = [];
N_RANDOM_LICKS = 5;            % random licks drawn per animal (without replacement)

SMOOTH_N_POINTS = 128;
PLOT_HALF = 50;            % 100x100 centered on jaw rest
LINE_WIDTH = 2.0;
SAVE_SVG = true;

%% =======================================================================

thisDir = fileparts(mfilename('fullpath'));
outRoot = fullfile(thisDir, 'telc_spontaneous_tip_trajectories');
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

if ~isempty(RANDOM_SEED)
    rng(RANDOM_SEED);
end

fprintf('\n=== IRt_TeLC spontaneous : %d random licks per animal ===\n', N_RANDOM_LICKS);

picks = collectRandomTelcLickPerAnimal(jawCsvPaths, PROB_MIN, MIN_LICK_FRAMES, ...
    LICK_FRAME_PAD, N_RANDOM_LICKS);

if isempty(picks)
    error('No spontaneous licks found (check jaw/behavior paths under IRt_TeLC##/IRt_TeLC##_Pre).');
end

cmapPhase = phaseColormap256();
nAnimals = numel(picks);
nCol = nAnimals;
nRow = 1;

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 280 * nCol + 120 420]);
tl = tiledlayout(fig, nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');
colormap(fig, cmapPhase);
title(tl, sprintf('IRt_TeLC Pre — jaw trajectory during spontaneous lick (%d random licks per animal, smoothed)', N_RANDOM_LICKS), ...
    'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 13, 'Color', 'k');

for i = 1:nAnimals
    ax = nexttile(tl);
    setupCenteredJawAxes(ax, PLOT_HALF, cmapPhase);
    hold(ax, 'on');

    nLicks = numel(picks(i).licks);
    for L = 1:nLicks
        xs = picks(i).licks(L).x;
        ys = picks(i).licks(L).y;
        [xsS, ysS, phS] = smoothLickTrajectory(xs, ys, SMOOTH_N_POINTS);
        draw_phase_line(ax, xsS, ysS, phS, LINE_WIDTH, [], PLOT_HALF);
        fprintf('  %s [%d/%d] : %s (%d raw frames -> %d smooth)\n', ...
            picks(i).animal, L, nLicks, picks(i).licks(L).sessionBase, ...
            numel(xs), SMOOTH_N_POINTS);
    end
    drawJawRestMarker(ax);

    title(ax, {picks(i).animal, 'Jaw trajectory during spontaneous lick', ...
        sprintf('%d random licks', nLicks)}, ...
        'Interpreter', 'none', 'FontSize', 9, 'Color', 'k');
    hold(ax, 'off');
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Color = 'k';
cb.Label.String = 'Intra-lick phase (0=start, 1=end)';
cb.Label.Interpreter = 'none';

if SAVE_SVG
    outPath = fullfile(outRoot, 'telc_spontaneous_one_random_lick_per_animal.svg');
    try
        exportgraphics(fig, outPath, 'ContentType', 'vector');
    catch %#ok<CTCH>
        print(fig, outPath, '-dsvg', '-painters');
    end
    fprintf('  saved: telc_spontaneous_one_random_lick_per_animal.svg\n');
end
close(fig);

fprintf('\nDone. Output: %s\n', outRoot);
end


%% =======================================================================
%% Random selection
%% =======================================================================

function picks = collectRandomTelcLickPerAnimal(jawPaths, probMin, minLickFrames, framePad, nLicks)

if nargin < 5 || isempty(nLicks)
    nLicks = 5;
end
nLicks = max(1, round(nLicks));

pool = struct('animal', {}, 'sessionBase', {}, 'x', {}, 'y', {});

for k = 1:numel(jawPaths)
    jawFile = jawPaths{k};
    if ~isfile(jawFile)
        continue
    end
    meta = parseTelcJawMeta(jawFile);
    if ~meta.isPre || ~meta.isSide || isempty(meta.animal)
        continue
    end

    behFile = findTelcSideBehaviorCsv(fileparts(jawFile));
    if isempty(behFile)
        continue
    end

    intervals = readSpontaneousLickIntervals(behFile);
    if isempty(intervals)
        continue
    end

    [lx, ly, ~] = extractJawLickTrajectories(jawFile, intervals, probMin, minLickFrames, framePad, false);
    [jx, jy] = jawSessionRestXY(jawFile, probMin);
    [lx, ly] = centerLickCells(lx, ly, jx, jy);
    for j = 1:numel(lx)
        pi = numel(pool) + 1;
        pool(pi).animal = meta.animal;
        pool(pi).sessionBase = meta.base;
        pool(pi).x = lx{j};
        pool(pi).y = ly{j};
    end
end

if isempty(pool)
    picks = repmat(struct('animal', '', 'licks', []), 0, 1);
    return
end

animals = unique({pool.animal}, 'stable');
picks = repmat(struct('animal', '', 'licks', []), numel(animals), 1);

for a = 1:numel(animals)
    tag = animals{a};
    idx = find(strcmp({pool.animal}, tag));
    nPick = min(nLicks, numel(idx));
    pickIdx = idx(randperm(numel(idx), nPick));
    picks(a).animal = tag;
    picks(a).licks = pool(pickIdx);
end
end


function [xOut, yOut, phOut] = smoothLickTrajectory(x, y, nOut)
x = x(:);
y = y(:);
n = numel(x);
if n < 2
    xOut = x;
    yOut = y;
    phOut = 0.5 * ones(size(x));
    return
end
if nargin < 3 || isempty(nOut) || nOut < n
    nOut = max(n, 64);
end
t = linspace(0, 1, n)';
tq = linspace(0, 1, nOut)';
xOut = interp1(t, x, tq, 'pchip');
yOut = interp1(t, y, tq, 'pchip');
phOut = tq;
end


%% =======================================================================
%% File / metadata helpers
%% =======================================================================

function meta = parseTelcJawMeta(jawFile)
[~, baseName, ~] = fileparts(jawFile);
meta.base = regexprep(baseName, '_jaw$', '');
baseLower = lower(baseName);

meta.isSide = contains(baseLower, '_1_jaw') || endsWith(baseLower, '_1_jaw');
meta.isPre = contains(baseLower, '_pre_') || contains(baseLower, '_pre') || ...
    contains(lower(fileparts(jawFile)), '_pre');

tok = regexp(baseLower, 'irt_telc(\d+)', 'tokens', 'once');
if isempty(tok)
    meta.animal = '';
else
    meta.animal = sprintf('IRt_TeLC%s', tok{1});
end
end


function behFile = findTelcSideBehaviorCsv(sessionDir)
behFile = '';
listing = dir(fullfile(sessionDir, '*side_behavior*.csv'));
if isempty(listing)
    return
end
[~, ord] = sort({listing.name});
behFile = fullfile(sessionDir, listing(ord(1)).name);
end


function tbl = readJawCsv(csvFile)
T = readtable(csvFile, 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
v = T.Properties.VariableNames;
vl = lower(strtrim(v));
fc = @(s) strcmp(vl, lower(s));

fi = fc('frame');
xi = fc('x');
yi = fc('y');
pidx = fc('probability');
if ~(any(fi) && any(xi) && any(yi))
    error('Jaw CSV must include Frame, X, and Y columns: %s', csvFile);
end

F = double(T{:, fi});
X = double(T{:, xi});
Y = double(T{:, yi});
if any(pidx)
    Pr = double(T{:, pidx});
else
    Pr = ones(size(F));
end
tbl = table(F, X, Y, Pr, 'VariableNames', {'Frame', 'X', 'Y', 'Probability'});
end


function cmap = phaseColormap256()
try
    cmap = turbo(256);
catch %#ok<CTCH>
    cmap = jet(256);
end
end
