function bipoles_jaw_tip_one_random_lick_per_animal
% bipoles_jaw_tip_one_random_lick_per_animal
% Side-view BiPoles: N_RANDOM_LICKS randomly chosen laser-ON licks per animal.
% No trajectory or probability filtering. Trajectories are smoothed (PCHIP
% resampling) and drawn as a single phase-colored line (no scatter).
%
% Run: bipoles_jaw_tip_one_random_lick_per_animal

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

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

LASER_MODE = 'on';
FIRST_LICK_PER_LASER = true;   % pool = first lick per laser pulse per session
PROB_MIN = 0;                  % no probability filter
MIN_LICK_FRAMES = 2;
LICK_FRAME_PAD = 10;           % extra frames before/after behavior lick Start/End
RANDOM_SEED = [];              % set e.g. 42 for reproducible picks; [] = new draw each run
N_RANDOM_LICKS = 5;            % random licks drawn per animal (without replacement)

SMOOTH_N_POINTS = 128;         % resampled points along smoothed curve
PLOT_HALF = 50;                % 100x100 axes centered on jaw rest (+/-50 px)
LINE_WIDTH = 2.0;
SAVE_SVG = true;

%% =======================================================================

thisDir = fileparts(mfilename('fullpath'));
outRoot = fullfile(thisDir, 'bipoles_jaw_tip_trajectories');
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

if ~isempty(RANDOM_SEED)
    rng(RANDOM_SEED);
end

experiments = struct( ...
    'tag',   {'IRt_BiPoles', 'PCRt_BiPoles'}, ...
    'paths', {jawCsvPaths_IRt, jawCsvPaths_PCRt});

cmapPhase = phaseColormap256();

for e = 1:numel(experiments)
    expTag = experiments(e).tag;
    paths = experiments(e).paths;

    fprintf('\n=== %s : %d random licks per animal ===\n', expTag, N_RANDOM_LICKS);

    picks = collectRandomLickPerAnimal(paths, LASER_MODE, FIRST_LICK_PER_LASER, ...
        PROB_MIN, MIN_LICK_FRAMES, LICK_FRAME_PAD, N_RANDOM_LICKS);

    if isempty(picks)
        warning('No licks found for %s.', expTag);
        continue
    end

    nAnimals = numel(picks);
    nCol = min(nAnimals, 5);
    nRow = ceil(nAnimals / nCol);

    figW = min(160 + 280 * nCol + 100, 2200);
    figH = min(140 + 260 * nRow, 1200);
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 figW figH]);
    tl = tiledlayout(fig, nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');
    colormap(fig, cmapPhase);
    title(tl, sprintf('%s — jaw trajectory during laser-ON lick (%d random licks per animal, smoothed)', ...
        expTag, N_RANDOM_LICKS), ...
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

        title(ax, {picks(i).animal, 'Jaw trajectory during laser-ON lick', ...
            sprintf('%d random licks', nLicks)}, ...
            'Interpreter', 'none', 'FontSize', 9, 'Color', 'k');
        if mod(i, nCol) == 1
            ylabel(ax, picks(i).animal, 'FontWeight', 'bold', 'FontSize', 10, ...
                'Interpreter', 'none', 'Color', 'k');
        end
        hold(ax, 'off');
    end

    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Color = 'k';
    cb.Label.String = 'Intra-lick phase (0=start, 1=end)';
    cb.Label.Interpreter = 'none';

    if SAVE_SVG
        outName = sprintf('bipoles_jaw_tip_one_random_lick_per_animal_%s.svg', expTag);
        outPath = fullfile(outRoot, outName);
        try
            exportgraphics(fig, outPath, 'ContentType', 'vector');
        catch %#ok<CTCH>
            print(fig, outPath, '-dsvg', '-painters');
        end
        fprintf('  saved: %s\n', outName);
    end
    close(fig);
end

fprintf('\nDone. Output root: %s\n', outRoot);
end


%% =======================================================================
%% Random selection
%% =======================================================================

function picks = collectRandomLickPerAnimal(paths, laserMode, firstPerLaser, probMin, minLickFrames, framePad, nLicks)
% Pool all laser-ON licks (per session) by animal; draw nLicks at random each.

if nargin < 7 || isempty(nLicks)
    nLicks = 5;
end
nLicks = max(1, round(nLicks));

pool = struct('animal', {}, 'dateLabel', {}, 'sessionBase', {}, 'x', {}, 'y', {});

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
    intervals = readLickIntervalsByLaser(behFile, laserMode, firstPerLaser);
    if isempty(intervals)
        continue
    end

    [lx, ly, ~] = extractJawLickTrajectories(jawFile, intervals, probMin, minLickFrames, framePad, false);
    [jx, jy] = jawSessionRestXY(jawFile, probMin);
    [lx, ly] = centerLickCells(lx, ly, jx, jy);
    for j = 1:numel(lx)
        pi = numel(pool) + 1;
        pool(pi).animal = meta.animal;
        pool(pi).dateLabel = formatSessionDate(meta.date);
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
% PCHIP interpolation along intra-lick phase (uniform in time order).
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


function intervals = readLickIntervalsByLaser(behFile, laserMode, firstPerLaser)
if nargin < 3
    firstPerLaser = true;
end

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

laserCol = find(contains(v, 'laser') & contains(v, 'Assign ID'), 1);
switch lower(laserMode)
    case 'on'
        if isempty(laserCol)
            keepLaser = true(size(starts));
        else
            lid = double(T{:, laserCol});
            keepLaser = lid >= 0;
        end
    case 'off'
        if isempty(laserCol)
            keepLaser = false(size(starts));
        else
            lid = double(T{:, laserCol});
            keepLaser = lid < 0;
        end
    otherwise
        keepLaser = true(size(starts));
end

sel = valid & keepLaser;
s = starts(sel);
e = ends(sel);
if firstPerLaser && strcmpi(laserMode, 'on') && ~isempty(laserCol)
    lidSel = double(T{sel, laserCol});
    s = s(:);
    e = e(:);
    lidSel = lidSel(:);
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


%% =======================================================================
%% File / metadata helpers
%% =======================================================================

function meta = parseBipolesJawMeta(jawFile)
[~, baseName, ~] = fileparts(jawFile);
meta.base = regexprep(baseName, '_jaw$', '');

baseLower = lower(baseName);
tok = regexp(baseLower, '(irt|pcrt)_bipoles_(\d+)_(\d{4}_\d{4})_(bottom|side)_view', ...
    'tokens', 'once');
if isempty(tok)
    meta.group = '';
    meta.animal = '';
    meta.date = '';
    meta.view = '';
    return
end

if strcmp(tok{1}, 'irt')
    grp = 'IRt';
else
    grp = 'PCRt';
end
meta.group = sprintf('%s_BiPoles', grp);
meta.animal = sprintf('%s_%s', grp, tok{2});
meta.date = tok{3};
meta.view = tok{4};
end


function lbl = formatSessionDate(dstr)
tok = regexp(dstr, '(\d{4})_(\d{2})(\d{2})', 'tokens', 'once');
if isempty(tok)
    lbl = dstr;
else
    lbl = sprintf('%s-%s-%s', tok{1}, tok{2}, tok{3});
end
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
