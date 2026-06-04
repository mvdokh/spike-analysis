function jaw_trajectory_shape_clusters
% jaw_trajectory_shape_clusters
% Cluster jaw-tip licks per animal using the same shape distance as the
% similar-lick overlay scripts (lickShapeDistanceMatrix). Saves summary and
% per-animal figures showing cluster size distributions and trajectory groups.
%
% Run: jaw_trajectory_shape_clusters

close all

%% CONFIG
SIMILARITY_N_POINTS = 64;
MAX_CLUSTERS = 8;              % max k tested via silhouette
PROB_MIN = 0;
MIN_LICK_FRAMES = 2;
LICK_FRAME_PAD = 10;
PLOT_HALF = 50;
DISPLAY_SMOOTH_N = 128;
DISPLAY_MOVAVG_WIN = 0;        % 0 = no extra smooth on cluster plots
LINE_WIDTH = 1.2;
SAVE_SVG = true;

thisDir = fileparts(mfilename('fullpath'));
outRoot = fullfile(thisDir, 'shape_cluster_figures');
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

%% Experiment definitions (same pools as similar-lick scripts)
jawCsvPaths_IRt = {
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0425\IRt_BiPoles_01_2025_0425_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0514\IRt_BiPoles_01_2025_0514_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0515\IRt_BiPoles_01_2025_0515_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0516\IRt_BiPoles_01_2025_0516_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0425\IRt_BiPoles_02_2025_0425_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0514\IRt_BiPoles_02_2025_0514_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0515\IRt_BiPoles_02_2025_0515_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0516\IRt_BiPoles_02_2025_0516_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_03\2025_0425\IRt_BiPoles_03_2025_0425_side_view_jaw.csv'
    };

jawCsvPaths_PCRt = {
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1206\PCRt_BiPoles_02_2024_1206_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1218\PCRt_BiPoles_02_2024_1218_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1223\PCRt_BiPoles_02_2024_1223_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0401\PCRt_BiPoles_07_2025_0401_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0403\PCRt_BiPoles_07_2025_0403_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0401\PCRt_BiPoles_08_2025_0401_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0403\PCRt_BiPoles_08_2025_0403_side_view_jaw.csv'
    };

experiments = struct( ...
    'tag',        {'IRt_BiPoles', 'PCRt_BiPoles', 'IRt_TeLC_Pre'}, ...
    'poolType',   {'bipoles', 'bipoles', 'telc'}, ...
    'paths',      {jawCsvPaths_IRt, jawCsvPaths_PCRt, {}});

for ex = 1:numel(experiments)
    expTag = experiments(ex).tag;
    fprintf('\n=== Shape clusters: %s ===\n', expTag);

    if strcmp(experiments(ex).poolType, 'telc')
        pool = buildTelcLickPool(PROB_MIN, MIN_LICK_FRAMES, LICK_FRAME_PAD);
    else
        pool = buildBipolesLickPool(experiments(ex).paths, PROB_MIN, MIN_LICK_FRAMES, LICK_FRAME_PAD);
    end

    if isempty(pool)
        warning('No licks in pool for %s.', expTag);
        continue
    end

    animals = unique({pool.animal}, 'stable');
    nAnimals = numel(animals);
    clusterSummary = repmat(struct('animal', '', 'nLicks', 0, 'nClusters', 0, ...
        'sizes', [], 'labels', [], 'meanSil', NaN, 'method', ''), nAnimals, 1);

    for a = 1:nAnimals
        tag = animals{a};
        idx = find(strcmp({pool.animal}, tag));
        sub = pool(idx);
        nL = numel(idx);

        D = lickShapeDistanceMatrix({sub.x}, {sub.y}, SIMILARITY_N_POINTS);
        [labels, cInfo] = clusterLicksByShapeDistance(D, MAX_CLUSTERS);
        uLabels = unique(labels);
        sizes = arrayfun(@(u) sum(labels == u), uLabels);

        [sizes, ord] = sort(sizes, 'descend');
        uLabels = uLabels(ord);

        clusterSummary(a).animal = tag;
        clusterSummary(a).nLicks = nL;
        clusterSummary(a).nClusters = numel(uLabels);
        clusterSummary(a).sizes = sizes;
        clusterSummary(a).labels = labels;
        clusterSummary(a).meanSil = cInfo.meanSilhouette;
        clusterSummary(a).method = cInfo.method;

        fprintf('  %s : %d licks -> %d clusters, sizes %s (sil=%.3f, %s)\n', ...
            tag, nL, numel(uLabels), mat2str(sizes), cInfo.meanSilhouette, cInfo.method);

        figAnimal = plotAnimalClusterDetail(sub, labels, uLabels, D, cInfo, tag, expTag, ...
            PLOT_HALF, DISPLAY_SMOOTH_N, DISPLAY_MOVAVG_WIN, LINE_WIDTH);
        if SAVE_SVG
            outName = sprintf('%s_%s_shape_clusters.svg', expTag, tag);
            saveFigureSvg(figAnimal, fullfile(outRoot, outName));
            fprintf('    saved %s\n', outName);
        end
        close(figAnimal);
    end

    figSum = plotExperimentClusterSummary(clusterSummary, expTag);
    if SAVE_SVG
        outSum = sprintf('%s_shape_cluster_summary.svg', expTag);
        saveFigureSvg(figSum, fullfile(outRoot, outSum));
        fprintf('  saved summary: %s\n', outSum);
    end
    close(figSum);
end

fprintf('\nDone. Figures: %s\n', outRoot);
end


%% =======================================================================
%% Pool builders (same lick rules as similar-lick scripts)
%% =======================================================================

function pool = buildBipolesLickPool(paths, probMin, minLickFrames, framePad)
pool = struct('animal', {}, 'sessionBase', {}, 'x', {}, 'y', {});
for k = 1:numel(paths)
    jawFile = paths{k};
    if ~isfile(jawFile)
        continue
    end
    meta = parseBipolesJawMeta(jawFile);
    if ~strcmp(meta.view, 'side') || isempty(meta.animal)
        continue
    end
    behFile = findBehaviorCsv(fileparts(jawFile), 'side');
    if isempty(behFile)
        continue
    end
    intervals = readLickIntervalsByLaser(behFile, 'on', true);
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
end


function pool = buildTelcLickPool(probMin, minLickFrames, framePad)
pool = struct('animal', {}, 'sessionBase', {}, 'x', {}, 'y', {});
jawPaths = telc_pre_side_jaw_paths();
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
end


%% =======================================================================
%% Figures
%% =======================================================================

function fig = plotAnimalClusterDetail(sub, labels, uLabels, D, cInfo, animalTag, expTag, ...
    plotHalf, smoothN, movAvgWin, lineW)

nC = numel(uLabels);
nRow = 2;
nCol = max(3, min(nC + 2, 6));
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [40 40 min(200 + 220 * nCol, 1600) 520]);
tl = tiledlayout(fig, nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');
title(tl, sprintf('%s | %s — jaw shape clusters (n=%d licks, k=%d)', ...
    expTag, animalTag, numel(labels), nC), 'Interpreter', 'none', 'FontWeight', 'bold');

cmapCluster = lines(max(nC, 7));

axAll = nexttile(tl, [1 2]);
setupClusterJawAxes(axAll, plotHalf);
hold(axAll, 'on');
for i = 1:numel(sub)
    c = find(uLabels == labels(i), 1);
    plotClusterLickTrace(axAll, sub(i).x, sub(i).y, cmapCluster(c, :), lineW, smoothN, movAvgWin, plotHalf);
end
drawJawRestMarker(axAll);
title(axAll, 'All licks (color = cluster)', 'Interpreter', 'none', 'FontSize', 9);
hold(axAll, 'off');

axBar = nexttile(tl);
counts = arrayfun(@(u) sum(labels == u), uLabels);
bar(axBar, counts, 'FaceColor', 'flat', 'CData', cmapCluster(1:nC, :));
set(axBar, 'XTick', 1:nC, 'XTickLabel', arrayfun(@(k) sprintf('C%d', k), 1:nC, 'UniformOutput', false));
ylabel(axBar, 'Licks');
title(axBar, 'Cluster sizes', 'FontSize', 9);
grid(axBar, 'on');

axDen = nexttile(tl);
if ~isempty(cInfo.linkageZ) && exist('dendrogram', 'file') == 2
    if numel(labels) <= 40
        lbl = arrayfun(@(k) sprintf('%d', k), 1:numel(labels), 'UniformOutput', false);
    else
        lbl = repmat({''}, 1, numel(labels));
    end
    dendrogram(cInfo.linkageZ, 0, 'Parent', axDen, 'Labels', lbl, 'ColorThreshold', 'default');
    title(axDen, 'Shape distance dendrogram', 'FontSize', 9);
else
    imagesc(axDen, D);
    axis(axDen, 'square');
    colorbar(axDen);
    title(axDen, 'Distance matrix', 'FontSize', 9);
end

for c = 1:min(nC, nCol - 1)
    axC = nexttile(tl);
    setupClusterJawAxes(axC, plotHalf);
    hold(axC, 'on');
    u = uLabels(c);
    for i = 1:numel(sub)
        if labels(i) ~= u
            continue
        end
        plotClusterLickTrace(axC, sub(i).x, sub(i).y, cmapCluster(c, :), lineW, smoothN, movAvgWin, plotHalf);
    end
    drawJawRestMarker(axC);
    title(axC, sprintf('C%d (n=%d)', c, counts(c)), 'FontSize', 9, 'Color', cmapCluster(c, :));
    hold(axC, 'off');
end
end


function fig = plotExperimentClusterSummary(clusterSummary, expTag)
nA = numel(clusterSummary);
maxC = max(arrayfun(@(s) numel(s.sizes), clusterSummary));
if isempty(maxC) || maxC < 1
    maxC = 1;
end

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 900 520]);
tl = tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(tl, sprintf('%s — shape cluster distribution across animals', expTag), ...
    'Interpreter', 'none', 'FontWeight', 'bold');

axK = nexttile(tl);
nClusters = [clusterSummary.nClusters];
bar(axK, nClusters, 'FaceColor', [0.35 0.55 0.85]);
set(axK, 'XTick', 1:nA, 'XTickLabel', {clusterSummary.animal}, 'XTickLabelRotation', 30);
ylabel(axK, 'Number of clusters');
title(axK, 'Clusters per animal', 'FontSize', 10);
grid(axK, 'on');

axStack = nexttile(tl);
fracMat = zeros(nA, maxC);
for a = 1:nA
    s = clusterSummary(a).sizes;
    fracMat(a, 1:numel(s)) = s / sum(s);
end
bar(axStack, fracMat, 'stacked');
set(axStack, 'XTick', 1:nA, 'XTickLabel', {clusterSummary.animal}, 'XTickLabelRotation', 30);
ylabel(axStack, 'Fraction of licks');
legend(axStack, arrayfun(@(k) sprintf('Cluster %d', k), 1:maxC, 'UniformOutput', false), ...
    'Location', 'eastoutside', 'FontSize', 8);
title(axStack, 'Cluster size mix (largest first)', 'FontSize', 10);

axHist = nexttile(tl);
allSizes = [];
for a = 1:nA
    allSizes = [allSizes; clusterSummary(a).sizes(:)]; %#ok<AGROW>
end
histogram(axHist, allSizes, 'BinMethod', 'integers', 'FaceColor', [0.85 0.45 0.35]);
xlabel(axHist, 'Licks per cluster');
ylabel(axHist, 'Count');
title(axHist, 'Pooled cluster sizes (all animals)', 'FontSize', 10);
grid(axHist, 'on');

axTxt = nexttile(tl);
axis(axTxt, 'off');
lines = cell(nA + 1, 1);
lines{1} = sprintf('%-12s  n   k   sizes', 'Animal');
for a = 1:nA
    s = clusterSummary(a);
    lines{a + 1} = sprintf('%-12s %3d %2d   %s', s.animal, s.nLicks, s.nClusters, mat2str(s.sizes));
end
text(axTxt, 0, 1, strjoin(lines, newline), 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontName', 'Consolas', 'FontSize', 9, 'Interpreter', 'none');
title(axTxt, 'Summary table', 'FontSize', 10);
end


function plotClusterLickTrace(ax, x, y, col, lineW, smoothN, movAvgWin, plotHalf)
% One polyline per lick (solid cluster color). Clip only when needed.
[xs, ys, ~] = smoothLickTrajectory(x, y, smoothN, movAvgWin);
inWin = xs >= -plotHalf & xs <= plotHalf & ys >= -plotHalf & ys <= plotHalf;
if all(inWin)
    plot(ax, xs, ys, '-', 'Color', col, 'LineWidth', lineW, 'HandleVisibility', 'off');
else
    drawSolidClippedLine(ax, xs, ys, col, lineW, plotHalf);
end
end


function saveFigureSvg(fig, outPath)
try
    exportgraphics(fig, outPath, 'ContentType', 'vector');
catch %#ok<CTCH>
    print(fig, outPath, '-dsvg', '-painters');
end
end


%% =======================================================================
%% BiPoles helpers (minimal copy from bipoles_jaw_tip_one_random_lick_per_animal)
%% =======================================================================

function meta = parseBipolesJawMeta(jawFile)
[~, baseName, ~] = fileparts(jawFile);
meta.base = regexprep(baseName, '_jaw$', '');
baseLower = lower(baseName);
tok = regexp(baseLower, '(irt|pcrt)_bipoles_(\d+)_(\d{4}_\d{4})_(bottom|side)_view', 'tokens', 'once');
if isempty(tok)
    meta.animal = '';
    meta.view = '';
    return
end
grp = 'IRt';
if strcmp(tok{1}, 'pcrt')
    grp = 'PCRt';
end
meta.animal = sprintf('%s_%s', grp, tok{2});
meta.view = tok{4};
end


function behFile = findBehaviorCsv(sessionDir, viewTag)
behFile = '';
switch lower(viewTag)
    case 'bottom'
        pattern = '*bottom_view_behavior*.csv';
    case 'side'
        pattern = '*side_view_behavior*.csv';
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


function intervals = readLickIntervalsByLaser(behFile, laserMode, firstPerLaser)
T = readtable(behFile, 'VariableNamingRule', 'preserve');
v = T.Properties.VariableNames;
startCol = find(contains(v, 'Interval Start') & contains(v, 'interval_detection'), 1);
endCol = find(contains(v, 'Interval End') & contains(v, 'interval_detection'), 1);
if isempty(startCol) || isempty(endCol)
    error('Behavior CSV missing lick interval columns: %s', behFile);
end
starts = double(T{:, startCol});
ends = double(T{:, endCol});
valid = isfinite(starts) & isfinite(ends) & ends >= starts;
laserCol = find(contains(v, 'laser') & contains(v, 'Assign ID'), 1);
switch lower(laserMode)
    case 'on'
        if isempty(laserCol)
            keepLaser = true(size(starts));
        else
            keepLaser = double(T{:, laserCol}) >= 0;
        end
    otherwise
        keepLaser = true(size(starts));
end
sel = valid & keepLaser;
s = starts(sel);
e = ends(sel);
if firstPerLaser && strcmpi(laserMode, 'on') && ~isempty(laserCol)
    lidSel = double(T{sel, laserCol});
    keepRow = false(numel(s), 1);
    for u = unique(lidSel(lidSel >= 0))'
        idx = find(lidSel == u);
        [~, im] = min(s(idx));
        keepRow(idx(im)) = true;
    end
    s = s(keepRow);
    e = e(keepRow);
end
intervals = [s(:), e(:)];
end


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
