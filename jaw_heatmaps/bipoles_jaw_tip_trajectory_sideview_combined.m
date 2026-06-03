function bipoles_jaw_tip_trajectory_sideview_combined
% bipoles_jaw_tip_trajectory_sideview_combined
% Combined SIDE-VIEW intra-lick jaw-tip trajectories for the opto-activation
% experiments, laid out as a grid: one row per animal, one column per
% session. Trajectories are in image pixel coordinates, colored by intra-lick
% phase, and only laser-ON licks are shown. Two figures are produced: one for
% IRt_BiPoles and one for PCRt_BiPoles.
%
% Outlier rejection: the distribution of step distances between consecutive
% tracked points is examined per experiment. If it is heavy-tailed (a few
% "super-high" jumps exist, detected with a robust MAD threshold), the
% offending spike points are removed before plotting. Tight distributions
% (no wide tail) are left untouched.
%
% Run: bipoles_jaw_tip_trajectory_sideview_combined

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
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0112\IRt_BiPoles_09_2026_0112_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0113\IRt_BiPoles_09_2026_0113_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0116\IRt_BiPoles_09_2026_0116_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0112\IRt_BiPoles_10_2026_0112_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0113\IRt_BiPoles_10_2026_0113_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0116\IRt_BiPoles_10_2026_0116_side_view_jaw.csv'
    };

jawCsvPaths_PCRt = {
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1206\PCRt_BiPoles_02_2024_1206_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1218\PCRt_BiPoles_02_2024_1218_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1223\PCRt_BiPoles_02_2024_1223_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0401\PCRt_BiPoles_07_2025_0401_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0403\PCRt_BiPoles_07_2025_0403_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0401\PCRt_BiPoles_08_2025_0401_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0403\PCRt_BiPoles_08_2025_0403_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0514\PCRt_BiPoles_09_2025_0514_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0515\PCRt_BiPoles_09_2025_0515_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0516\PCRt_BiPoles_09_2025_0516_side_view_jaw.csv'
    };

LASER_MODE = 'on';       % 'on' | 'off' | 'all'  (see readLickIntervalsByLaser)
PROB_MIN = 0;            % minimum keypoint probability (0 = no filter)
MIN_LICK_FRAMES = 2;     % skip licks shorter than this many tracked frames
AXIS_MIN = 0;            % pixel-space axis bounds (full frame, not zoomed)
AXIS_MAX = 256;
DRAW_SEGMENT_LINES = true;
DRAW_SCATTER = true;
LINE_WIDTH = 1.0;
MARKER_SIZE = 8;

% --- Outlier rejection on consecutive-point step distances ---------------
OUTLIER_REMOVAL = true;  % master switch
OUTLIER_MAD_K = 5;       % default: threshold = median + K * (1.4826*MAD)
OUTLIER_MAD_K_IRT_09_10 = 10;  % looser (higher) threshold for IRt_09 / IRt_10
OUTLIER_MAX_ITERS = 3;   % repeat spike removal up to this many passes

%% =======================================================================

thisDir = fileparts(mfilename('fullpath'));
outRoot = fullfile(thisDir, 'bipoles_jaw_tip_trajectories');
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

experiments = struct( ...
    'tag',   {'IRt_BiPoles', 'PCRt_BiPoles'}, ...
    'paths', {jawCsvPaths_IRt, jawCsvPaths_PCRt});

cmapPhase = phaseColormap256();

for e = 1:numel(experiments)
    expTag = experiments(e).tag;
    paths = experiments(e).paths;

    fprintf('\n=== %s : side view, by animal x session ===\n', expTag);

    %% Collect one cell per (animal, session) side-view file
    cells = struct('animal', {}, 'dateKey', {}, 'dateLabel', {}, ...
        'base', {}, 'lickX', {}, 'lickY', {}, 'lickPhase', {});

    for k = 1:numel(paths)
        jawFile = paths{k};
        if ~isfile(jawFile)
            warning('Missing jaw CSV, skipping: %s', jawFile);
            continue
        end
        meta = parseBipolesJawMeta(jawFile);
        if ~strcmp(meta.view, 'side')
            continue
        end

        behFile = findBehaviorCsv(fileparts(jawFile), 'side');
        if isempty(behFile)
            fprintf('  skip (no side behavior CSV): %s\n', meta.base);
            continue
        end
        intervals = readLickIntervalsByLaser(behFile, LASER_MODE);
        if isempty(intervals)
            fprintf('  skip (no laser-%s licks): %s\n', LASER_MODE, meta.base);
            continue
        end

        [sx, sy, sp] = jawLickTrajectories(jawFile, intervals, PROB_MIN, MIN_LICK_FRAMES);
        if isempty(sx)
            continue
        end

        ci = numel(cells) + 1;
        cells(ci).animal = meta.animal;
        cells(ci).dateKey = dateSortKey(meta.date);
        cells(ci).dateLabel = formatSessionDate(meta.date);
        cells(ci).base = meta.base;
        cells(ci).lickX = sx;
        cells(ci).lickY = sy;
        cells(ci).lickPhase = sp;
        fprintf('  + %s : %d laser-%s licks\n', meta.base, numel(sx), LASER_MODE);
    end

    if isempty(cells)
        warning('No side-view licks found for %s; no figure written.', expTag);
        continue
    end

    %% Outlier rejection (per animal; looser threshold for IRt_09 / IRt_10)
    nPtsRemoved = 0;
    if OUTLIER_REMOVAL
        animalsOut = unique({cells.animal});
        nLicksDropped = 0;
        for a = 1:numel(animalsOut)
            animalTag = animalsOut{a};
            idxs = find(strcmp({cells.animal}, animalTag));
            steps = [];
            for ii = 1:numel(idxs)
                steps = [steps; poolStepDistances(cells(idxs(ii)).lickX, cells(idxs(ii)).lickY)]; %#ok<AGROW>
            end
            madK = outlierMadKForAnimal(animalTag, OUTLIER_MAD_K, OUTLIER_MAD_K_IRT_09_10);
            [T, isLongTail, info] = robustStepThreshold(steps, madK);
            fprintf(['  %s: n=%d steps, median=%.2f, scaledMAD=%.2f, max=%.2f, ' ...
                'K=%g, threshold=%.2f\n'], ...
                animalTag, info.n, info.median, info.scaledMad, info.max, madK, T);
            if ~isLongTail
                fprintf('    no wide tail -> keeping all points\n');
                continue
            end
            fprintf('    long tail: %d/%d steps (%.2f%%) exceed threshold -> removing spikes\n', ...
                info.nAbove, info.n, 100 * info.nAbove / max(info.n, 1));
            for ii = 1:numel(idxs)
                i = idxs(ii);
                [cells(i).lickX, cells(i).lickY, cells(i).lickPhase, nRem, nDrop] = ...
                    cleanLicks(cells(i).lickX, cells(i).lickY, cells(i).lickPhase, ...
                    T, OUTLIER_MAX_ITERS, MIN_LICK_FRAMES);
                nPtsRemoved = nPtsRemoved + nRem;
                nLicksDropped = nLicksDropped + nDrop;
            end
        end
        fprintf('  total: removed %d outlier points; dropped %d licks below %d frames\n', ...
            nPtsRemoved, nLicksDropped, MIN_LICK_FRAMES);
    end

    %% Build the animal x session grid
    animals = unique({cells.animal});      % unique() returns sorted (zero-padded ids order correctly)
    nRows = numel(animals);
    animalIdx = cell(nRows, 1);
    nCols = 0;
    for r = 1:nRows
        idxs = find(strcmp({cells.animal}, animals{r}));
        [~, ord] = sort([cells(idxs).dateKey]);
        animalIdx{r} = idxs(ord);
        nCols = max(nCols, numel(idxs));
    end

    panelW = 230;
    panelH = 230;
    figW = min(170 + panelW * nCols + 120, 2600);
    figH = min(120 + panelH * nRows, 2200);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 figW figH]);
    tl = tiledlayout(fig, nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');
    colormap(fig, cmapPhase);
    title(tl, sprintf('%s  —  side-view jaw-tip trajectories (laser-%s)   [rows = animal, cols = session]%s', ...
        expTag, upper(LASER_MODE), outlierTitleSuffix(OUTLIER_REMOVAL, nPtsRemoved)), ...
        'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 13, 'Color', 'k');

    for r = 1:nRows
        idxs = animalIdx{r};
        for c = 1:nCols
            ax = nexttile(tl, (r - 1) * nCols + c);
            if c > numel(idxs)
                axis(ax, 'off');
                continue
            end
            cc = cells(idxs(c));

            drawTile(ax, cc.lickX, cc.lickY, cc.lickPhase, cmapPhase, ...
                DRAW_SEGMENT_LINES, DRAW_SCATTER, LINE_WIDTH, MARKER_SIZE);
            setupTileAxes(ax, AXIS_MIN, AXIS_MAX, cmapPhase);

            if c == 1
                ylabel(ax, animals{r}, 'FontWeight', 'bold', 'FontSize', 10, ...
                    'Color', 'k', 'Interpreter', 'none');
            end
            title(ax, {cc.dateLabel, sprintf('%d licks', numel(cc.lickX))}, ...
                'FontSize', 8, 'Color', 'k', 'Interpreter', 'none');
        end
    end

    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Color = 'k';
    cb.Label.String = 'Intra-lick phase (0=start, 1=end)';
    cb.Label.Interpreter = 'none';

    outName = sprintf('bipoles_jaw_tip_trajectory_sideview_combined_%s.svg', expTag);
    outPath = fullfile(outRoot, outName);
    try
        exportgraphics(fig, outPath, 'ContentType', 'vector');
    catch %#ok<CTCH>
        print(fig, outPath, '-dsvg', '-painters');
    end
    close(fig);
    fprintf('  saved %dx%d grid: %s\n', nRows, nCols, outName);
end

fprintf('\nDone. Output root: %s\n', outRoot);
end


%% =======================================================================
%% Rendering
%% =======================================================================

function drawTile(ax, lickX, lickY, lickPhase, cmapPhase, drawLines, drawScatter, lineW, markerSize)
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
hold(ax, 'on');
colormap(ax, cmapPhase);
caxis(ax, [0 1]);

scatX = [];
scatY = [];
scatPh = [];

for j = 1:numel(lickX)
    xs = lickX{j};
    ys = lickY{j};
    ph = lickPhase{j};
    if isempty(xs)
        continue
    end
    if drawLines && numel(xs) > 1
        drawPhaseLine(ax, xs, ys, ph, lineW);
    end
    if drawScatter
        scatX = [scatX; xs(:)];   %#ok<AGROW>
        scatY = [scatY; ys(:)];   %#ok<AGROW>
        scatPh = [scatPh; ph(:)]; %#ok<AGROW>
    end
end

if drawScatter && ~isempty(scatX)
    scatter(ax, scatX, scatY, markerSize, scatPh, 'filled', ...
        'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'none');
end
hold(ax, 'off');
end


function setupTileAxes(ax, axisMin, axisMax, cmapPhase)
axis(ax, 'equal');
axis(ax, 'square');
set(ax, 'YDir', 'reverse');
xlim(ax, [axisMin axisMax]);
ylim(ax, [axisMin axisMax]);
xticks(ax, [axisMin (axisMin + axisMax) / 2 axisMax]);
yticks(ax, [axisMin (axisMin + axisMax) / 2 axisMax]);
colormap(ax, cmapPhase);
caxis(ax, [0 1]);
end


function drawPhaseLine(ax, x, y, ph, lineW)
x = x(:)';
y = y(:)';
ph = ph(:)';
surface(ax, [x; x], [y; y], zeros(2, numel(x)), [ph; ph], ...
    'FaceColor', 'none', 'EdgeColor', 'interp', 'LineWidth', lineW, ...
    'HandleVisibility', 'off');
end


%% =======================================================================
%% Outlier rejection
%% =======================================================================

function K = outlierMadKForAnimal(animalTag, defaultK, relaxedK)
% IRt_09 and IRt_10 use a higher MAD multiplier (looser spike filter).
if any(strcmp(animalTag, {'IRt_09', 'IRt_10'}))
    K = relaxedK;
else
    K = defaultK;
end
end


function steps = poolStepDistances(lickX, lickY)
% All Euclidean distances between consecutive tracked points, pooled.
steps = [];
for j = 1:numel(lickX)
    xs = lickX{j};
    ys = lickY{j};
    if numel(xs) > 1
        steps = [steps; hypot(diff(xs(:)), diff(ys(:)))]; %#ok<AGROW>
    end
end
end


function [T, isLongTail, info] = robustStepThreshold(steps, K)
% Robust (median + K*scaledMAD) threshold on step distances. The
% distribution is declared heavy-tailed only if some steps exceed T, so a
% tight distribution yields no removals.
d = steps(:);
d = d(isfinite(d) & d >= 0);
info = struct('n', numel(d), 'median', NaN, 'scaledMad', NaN, ...
    'max', NaN, 'T', Inf, 'nAbove', 0);
if isempty(d)
    T = Inf;
    isLongTail = false;
    return
end
med = median(d);
madv = median(abs(d - med));
scaledMad = 1.4826 * madv;
if scaledMad <= 0 || ~isfinite(scaledMad)
    scaledMad = std(d);
end
if scaledMad <= 0 || ~isfinite(scaledMad)
    T = Inf;
    isLongTail = false;
else
    T = med + K * scaledMad;
    isLongTail = any(d > T);
end
info.median = med;
info.scaledMad = scaledMad;
info.max = max(d);
info.T = T;
info.nAbove = sum(d > T);
end


function [lx, ly, lp, nRem, nDrop] = cleanLicks(lx, ly, lp, T, maxIters, minFrames)
% Remove spike points from each lick and drop licks that fall below the
% minimum frame count after cleaning.
nRem = 0;
keepLick = true(numel(lx), 1);
for j = 1:numel(lx)
    keepPts = spikeKeepMask(lx{j}, ly{j}, T, maxIters);
    r = sum(~keepPts);
    if r > 0
        nRem = nRem + r;
        lx{j} = lx{j}(keepPts);
        ly{j} = ly{j}(keepPts);
        lp{j} = lp{j}(keepPts);
    end
    if numel(lx{j}) < minFrames
        keepLick(j) = false;
    end
end
nDrop = sum(~keepLick);
lx = lx(keepLick);
ly = ly(keepLick);
lp = lp(keepLick);
end


function keep = spikeKeepMask(xs, ys, T, maxIters)
% Iteratively flag spike points: an interior point whose distance to BOTH
% neighbors exceeds T, or an endpoint whose single step exceeds T. Removing
% them eliminates spurious jumps while preserving normal points.
xs = xs(:);
ys = ys(:);
n = numel(xs);
keep = true(n, 1);
if n < 3
    return
end
for it = 1:maxIters %#ok<NASGU>
    idx = find(keep);
    m = numel(idx);
    if m < 3
        break
    end
    xx = xs(idx);
    yy = ys(idx);
    d = hypot(diff(xx), diff(yy)); % length m-1, d(i) = dist(i -> i+1)
    out = false(m, 1);
    if d(1) > T
        out(1) = true;
    end
    if d(end) > T
        out(end) = true;
    end
    for p = 2:(m - 1)
        if d(p - 1) > T && d(p) > T
            out(p) = true;
        end
    end
    if ~any(out)
        break
    end
    keep(idx(out)) = false;
end
end


function suffix = outlierTitleSuffix(removalOn, nPtsRemoved)
if removalOn && nPtsRemoved > 0
    suffix = sprintf('  ( %d outlier pts removed )', nPtsRemoved);
elseif removalOn
    suffix = '  ( no outliers removed )';
else
    suffix = '';
end
end


%% =======================================================================
%% Trajectory extraction
%% =======================================================================

function [lickX, lickY, lickPhase] = jawLickTrajectories(jawFile, intervals, probMin, minLickFrames)
lickX = {};
lickY = {};
lickPhase = {};

tbl = readJawCsv(jawFile);
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

    ph = ((0:(L - 1))' / (L - 1));

    lickX{end + 1} = xs;       %#ok<AGROW>
    lickY{end + 1} = ys;       %#ok<AGROW>
    lickPhase{end + 1} = ph;   %#ok<AGROW>
end
end


function intervals = readLickIntervalsByLaser(behFile, laserMode)
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
            warning('No laser Assign ID column; treating all licks as laser-ON: %s', behFile);
            keepLaser = true(size(starts));
        else
            lid = double(T{:, laserCol});
            keepLaser = lid >= 0;
        end
    case 'off'
        if isempty(laserCol)
            warning('No laser Assign ID column; cannot identify laser-OFF licks: %s', behFile);
            keepLaser = false(size(starts));
        else
            lid = double(T{:, laserCol});
            keepLaser = lid < 0;
        end
    otherwise % 'all'
        keepLaser = true(size(starts));
end

sel = valid & keepLaser;
intervals = [starts(sel), ends(sel)];
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
    meta.viewLabel = '';
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
meta.viewLabel = sprintf('%s view', tok{4});
end


function key = dateSortKey(dstr)
tok = regexp(dstr, '(\d{4})_(\d{2})(\d{2})', 'tokens', 'once');
if isempty(tok)
    key = inf;
else
    key = str2double([tok{1} tok{2} tok{3}]);
end
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
