function telc_spontaneous_random20_similar_lick_per_animal
% telc_spontaneous_random20_similar_lick_per_animal
% IRt_TeLC Pre: per animal, randomly sample N_RANDOM_POOL first-in-bout licks,
% then keep N_SIMILAR_LICKS with the most similar jaw shape within that subset.
% Reduces bias from comparing all ~300 bouts in the full session pool.
%
% Run: telc_spontaneous_random20_similar_lick_per_animal

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

jawCsvPaths = telc_pre_side_jaw_paths();

PROB_MIN = 0;
MIN_LICK_FRAMES = 2;
LICK_FRAME_PAD = 10;
N_RANDOM_POOL = 20;            % random first-in-bout licks sampled per animal
N_SIMILAR_LICKS = 5;           % most similar among the random pool
SIMILARITY_N_POINTS = 64;
RANDOM_SEED = [];              % e.g. 42 for reproducible pool draws; [] = shuffle

SMOOTH_N_POINTS = 256;
SMOOTH_MOVAVG_WIN = 7;
PLOT_HALF = 50;
LINE_WIDTH = 2.0;
SAVE_SVG = true;

%% =======================================================================

thisDir = fileparts(mfilename('fullpath'));
outRoot = fullfile(thisDir, 'telc_spontaneous_tip_trajectories');
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

if isempty(RANDOM_SEED)
    rng('shuffle');
else
    rng(RANDOM_SEED);
end

fprintf(['\n=== IRt_TeLC spontaneous : %d random bout licks -> %d most similar ' ...
    'per animal ===\n'], N_RANDOM_POOL, N_SIMILAR_LICKS);

picks = collectRandom20SimilarTelcLickPerAnimal(jawCsvPaths, PROB_MIN, MIN_LICK_FRAMES, ...
    LICK_FRAME_PAD, N_RANDOM_POOL, N_SIMILAR_LICKS, SIMILARITY_N_POINTS);

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
title(tl, sprintf(['IRt_TeLC Pre — spontaneous lick (%d random bout licks, ' ...
    '%d most similar)'], N_RANDOM_POOL, N_SIMILAR_LICKS), ...
    'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 13, 'Color', 'k');

for i = 1:nAnimals
    ax = nexttile(tl);
    setupCenteredJawAxes(ax, PLOT_HALF, cmapPhase);
    hold(ax, 'on');

    nLicks = numel(picks(i).licks);
    pathLens = zeros(nLicks, 1);
    maxExc = zeros(nLicks, 1);
    for L = 1:nLicks
        xs = picks(i).licks(L).x;
        ys = picks(i).licks(L).y;
        [xsS, ysS, phS] = smoothLickTrajectory(xs, ys, SMOOTH_N_POINTS, SMOOTH_MOVAVG_WIN);
        pathLens(L) = trajectoryPathLength(xsS, ysS);
        maxExc(L) = trajectoryMaxExcursionFromStart(xsS, ysS);
        draw_phase_line(ax, xsS, ysS, phS, LINE_WIDTH, [], PLOT_HALF);
        fprintf('  %s [%d/%d] : %s (pool n=%d, %d frames, arc %.1f px, max-from-start %.1f px)\n', ...
            picks(i).animal, L, nLicks, picks(i).licks(L).sessionBase, picks(i).nPoolSampled, ...
            numel(xs), pathLens(L), maxExc(L));
    end
    drawJawRestMarker(ax);
    annotatePathLengthStats(ax, pathLens, maxExc);
    if nLicks > 1
        fprintf(['  %s arc length: %.1f +/- %.1f px; max distance from start: ' ...
            '%.1f +/- %.1f px (n=%d)\n'], picks(i).animal, mean(pathLens), ...
            std(pathLens, 0), mean(maxExc), std(maxExc, 0), nLicks);
    else
        fprintf('  %s arc length: %.1f px; max distance from start: %.1f px\n', ...
            picks(i).animal, pathLens(1), maxExc(1));
    end

    title(ax, {picks(i).animal, 'Jaw trajectory during spontaneous lick', ...
        sprintf('%d most similar of %d random bout licks', nLicks, picks(i).nPoolSampled)}, ...
        'Interpreter', 'none', 'FontSize', 9, 'Color', 'k');
    hold(ax, 'off');
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Color = 'k';
cb.Label.String = 'Intra-lick phase (0=start, 1=end)';
cb.Label.Interpreter = 'none';

if SAVE_SVG
    outPath = fullfile(outRoot, 'telc_spontaneous_random20_similar_lick_per_animal.svg');
    try
        exportgraphics(fig, outPath, 'ContentType', 'vector');
    catch %#ok<CTCH>
        print(fig, outPath, '-dsvg', '-painters');
    end
    fprintf('  saved: telc_spontaneous_random20_similar_lick_per_animal.svg\n');
end
close(fig);

fprintf('\nDone. Output: %s\n', outRoot);
end


%% =======================================================================
%% Random pool, then similar-shape selection
%% =======================================================================

function picks = collectRandom20SimilarTelcLickPerAnimal(jawPaths, probMin, minLickFrames, ...
    framePad, nRandomPool, nSimilar, nResample)

if nargin < 5 || isempty(nRandomPool)
    nRandomPool = 20;
end
if nargin < 6 || isempty(nSimilar)
    nSimilar = 5;
end
if nargin < 7 || isempty(nResample)
    nResample = 64;
end
nRandomPool = max(1, round(nRandomPool));
nSimilar = max(1, round(nSimilar));

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

    intervals = readFirstLickPerBoutIntervals(behFile);
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
    picks = repmat(struct('animal', '', 'nPoolSampled', 0, 'licks', []), 0, 1);
    return
end

animals = unique({pool.animal}, 'stable');
picks = repmat(struct('animal', '', 'nPoolSampled', 0, 'licks', []), numel(animals), 1);

for a = 1:numel(animals)
    tag = animals{a};
    idx = find(strcmp({pool.animal}, tag));
    nAll = numel(idx);
    nDraw = min(nRandomPool, nAll);
    randIdx = idx(randperm(nAll, nDraw));
    sub = pool(randIdx);
    nPick = min(nSimilar, nDraw);
    localPick = selectMostSimilarLickIndices({sub.x}, {sub.y}, nPick, nResample);
    picks(a).animal = tag;
    picks(a).nPoolSampled = nDraw;
    picks(a).licks = sub(localPick);
    fprintf('  %s : %d bout licks in pool, sampled %d at random\n', tag, nAll, nDraw);
end
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


function cmap = phaseColormap256()
try
    cmap = turbo(256);
catch %#ok<CTCH>
    cmap = jet(256);
end
end
