function bipoles_jaw_tip_trajectory_sideview_combined_prob080
% bipoles_jaw_tip_trajectory_sideview_combined_prob080
% Same layout as bipoles_jaw_tip_trajectory_sideview_combined, but quality
% control uses only the jaw CSV Probability column (model confidence).
% Frames with Probability < PROB_MIN are excluded; no jump/hotspot filter.
%
% Run: bipoles_jaw_tip_trajectory_sideview_combined_prob080

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
FIRST_LICK_PER_LASER = true;  % one trajectory per laser pulse (first lick only)
PROB_MIN = 0.80;         % minimum jaw-tip Probability (model confidence)
MIN_LICK_FRAMES = 2;     % skip licks shorter than this many high-prob frames
AXIS_MIN = 0;            % pixel-space axis bounds (full frame, not zoomed)
AXIS_MAX = 256;
DRAW_SEGMENT_LINES = true;
DRAW_SCATTER = true;
LINE_WIDTH = 1.0;
MARKER_SIZE = 4;

%% =======================================================================

thisDir = fileparts(mfilename('fullpath'));
outRoot = fullfile(thisDir, 'bipoles_jaw_tip_trajectories_prob080');
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
        'base', {}, 'lickX', {}, 'lickY', {}, 'lickPhase', {}, 'lickFrame', {});

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
        intervals = readLickIntervalsByLaser(behFile, LASER_MODE, FIRST_LICK_PER_LASER);
        if isempty(intervals)
            fprintf('  skip (no laser-%s licks): %s\n', LASER_MODE, meta.base);
            continue
        end

        [sx, sy, sp, sf] = jawLickTrajectories(jawFile, intervals, PROB_MIN, MIN_LICK_FRAMES);
        if isempty(sx)
            continue
        end
        fprintf('  + %s : %d laser-%s licks (prob >= %.2f)\n', ...
            meta.base, numel(sx), LASER_MODE, PROB_MIN);

        ci = numel(cells) + 1;
        cells(ci).animal = meta.animal;
        cells(ci).dateKey = dateSortKey(meta.date);
        cells(ci).dateLabel = formatSessionDate(meta.date);
        cells(ci).base = meta.base;
        cells(ci).lickX = sx;
        cells(ci).lickY = sy;
        cells(ci).lickPhase = sp;
        cells(ci).lickFrame = sf;
    end

    if isempty(cells)
        warning('No side-view licks found for %s; no figure written.', expTag);
        continue
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
    title(tl, sprintf('%s  —  side view, prob >= %.2f (laser-%s)   [rows = animal, cols = session]', ...
        expTag, PROB_MIN, upper(LASER_MODE)), ...
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

            drawTile(ax, cc.lickX, cc.lickY, cc.lickPhase, cc.lickFrame, cmapPhase, ...
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

    outName = sprintf('bipoles_jaw_tip_trajectory_sideview_combined_prob080_%s.svg', expTag);
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

function drawTile(ax, lickX, lickY, lickPhase, lickFrame, cmapPhase, drawLines, drawScatter, lineW, markerSize)
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
    fr = lickFrame{j};
    if isempty(xs)
        continue
    end
    if drawLines && numel(xs) > 1
        draw_phase_line_frame_gaps(ax, xs, ys, ph, fr, lineW);
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


%% =======================================================================
%% Trajectory extraction
%% =======================================================================

function draw_phase_line_frame_gaps(ax, x, y, ph, fr, lineW)
% Break polylines where frame index skips (low-prob frames omitted).
fr = fr(:)';
if numel(fr) < 2
    return
end
gapAfter = find(diff(fr) > 1);
starts = [1, gapAfter + 1];
ends = [gapAfter, numel(fr)];
for k = 1:numel(starts)
    i0 = starts(k);
    i1 = ends(k);
    if i1 > i0
        draw_phase_line(ax, x(i0:i1), y(i0:i1), ph(i0:i1), lineW);
    end
end
end


function [lickX, lickY, lickPhase, lickFrame] = jawLickTrajectories(jawFile, intervals, probMin, minLickFrames)
lickX = {};
lickY = {};
lickPhase = {};
lickFrame = {};

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
    fs = fi(ord);
    L = numel(xs);
    if L < minLickFrames
        continue
    end

    if L == 1
        ph = 0.5;
    else
        ph = ((0:(L - 1))' / (L - 1));
    end

    lickX{end + 1} = xs;       %#ok<AGROW>
    lickY{end + 1} = ys;       %#ok<AGROW>
    lickPhase{end + 1} = ph;   %#ok<AGROW>
    lickFrame{end + 1} = fs;   %#ok<AGROW>
end
end


function intervals = readLickIntervalsByLaser(behFile, laserMode, firstPerLaser)
% When firstPerLaser is true and laserMode is 'on', keep only the first lick
% (minimum Interval Start) per laser_Interval Overlap Assign ID >= 0.
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
