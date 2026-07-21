function bipoles_jaw_tip_trajectory_by_session
% bipoles_jaw_tip_trajectory_by_session
% Intra-lick JAW-TIP trajectories for the opto-activation experiments
% (IRt_BiPoles and PCRt_BiPoles), plotted in image pixel coordinates and
% colored by intra-lick phase (0 = lick start, 1 = lick end).
%
% By default only LASER-ON licks are plotted, and within each laser pulse
% only the FIRST lick (earliest Interval Start per laser_Interval Overlap
% Assign ID) is kept. One SVG is written per session per
% view (bottom / side) into:
%   jaw_heatmaps/bipoles_jaw_tip_trajectories/<experiment>/
%
% Lick intervals come from the matching *_<view>_view_behavior_*.csv in the
% same folder as each jaw CSV. Style matches
% irt_telc_jaw_lick_trajectory_phase_by_session.m.
%
% Run: bipoles_jaw_tip_trajectory_by_session

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

jawCsvPaths_IRt = {
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0425\IRt_BiPoles_01_2025_0425_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0425\IRt_BiPoles_01_2025_0425_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0514\IRt_BiPoles_01_2025_0514_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0514\IRt_BiPoles_01_2025_0514_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0515\IRt_BiPoles_01_2025_0515_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0515\IRt_BiPoles_01_2025_0515_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0516\IRt_BiPoles_01_2025_0516_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0516\IRt_BiPoles_01_2025_0516_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0425\IRt_BiPoles_02_2025_0425_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0425\IRt_BiPoles_02_2025_0425_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0514\IRt_BiPoles_02_2025_0514_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0514\IRt_BiPoles_02_2025_0514_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0515\IRt_BiPoles_02_2025_0515_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0515\IRt_BiPoles_02_2025_0515_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0516\IRt_BiPoles_02_2025_0516_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0516\IRt_BiPoles_02_2025_0516_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_03\2025_0425\IRt_BiPoles_03_2025_0425_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_03\2025_0425\IRt_BiPoles_03_2025_0425_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0112\IRt_BiPoles_09_2026_0112_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0112\IRt_BiPoles_09_2026_0112_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0113\IRt_BiPoles_09_2026_0113_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0113\IRt_BiPoles_09_2026_0113_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0116\IRt_BiPoles_09_2026_0116_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0116\IRt_BiPoles_09_2026_0116_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0112\IRt_BiPoles_10_2026_0112_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0112\IRt_BiPoles_10_2026_0112_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0113\IRt_BiPoles_10_2026_0113_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0113\IRt_BiPoles_10_2026_0113_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0116\IRt_BiPoles_10_2026_0116_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0116\IRt_BiPoles_10_2026_0116_bottom_view_jaw.csv'
    };

jawCsvPaths_PCRt = {
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1206\PCRt_BiPoles_02_2024_1206_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1206\PCRt_BiPoles_02_2024_1206_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1218\PCRt_BiPoles_02_2024_1218_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1218\PCRt_BiPoles_02_2024_1218_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1223\PCRt_BiPoles_02_2024_1223_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1223\PCRt_BiPoles_02_2024_1223_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0401\PCRt_BiPoles_07_2025_0401_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0401\PCRt_BiPoles_07_2025_0401_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0403\PCRt_BiPoles_07_2025_0403_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0403\PCRt_BiPoles_07_2025_0403_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0321\PCRt_BiPoles_08_2025_0321_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0321\PCRt_BiPoles_08_2025_0321_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0326\PCRt_BiPoles_08_2025_0326_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0326\PCRt_BiPoles_08_2025_0326_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0401\PCRt_BiPoles_08_2025_0401_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0401\PCRt_BiPoles_08_2025_0401_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0403\PCRt_BiPoles_08_2025_0403_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0403\PCRt_BiPoles_08_2025_0403_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0514\PCRt_BiPoles_09_2025_0514_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0514\PCRt_BiPoles_09_2025_0514_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0515\PCRt_BiPoles_09_2025_0515_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0515\PCRt_BiPoles_09_2025_0515_bottom_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0516\PCRt_BiPoles_09_2025_0516_side_view_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0516\PCRt_BiPoles_09_2025_0516_bottom_view_jaw.csv'
    };

% Which licks to plot, based on the laser overlap-assign ID column:
%   'on'  -> opto-evoked licks (laser Assign ID >= 0)   [default]
%   'off' -> spontaneous licks (laser Assign ID  < 0)
%   'all' -> every detected lick
LASER_MODE = 'on';
FIRST_LICK_PER_LASER = true;  % one trajectory per laser pulse (first lick only)

PROB_MIN = 0;            % minimum keypoint probability (0 = no filter)
MIN_LICK_FRAMES = 2;     % skip licks shorter than this many tracked frames
LICK_FRAME_PAD = 10;     % extra frames before/after behavior lick Start/End
TRAJECTORY_FILTER = true;        % remove outlier points (large jumps, singleton coords)
TRAJECTORY_FILTER_MODE = 'points';  % 'points' = trim licks; 'lick' = drop whole lick
TRAJECTORY_STEP_MAD_K = 5;       % step threshold = median + K * (1.4826 * MAD)
TRAJECTORY_STEP_HARD_MAX = 20;   % also flag steps > this (px); catches inflated robust T
TRAJECTORY_HOTSPOT_MIN_COUNT = 20;   % session coord count: drop if any adj step > T
TRAJECTORY_HOTSPOT_PURGE_COUNT = 50; % drop all points at coords this common in session
TRAJECTORY_LINE_BREAK_MAX = 20;  % do not draw lines across gaps larger than this (px)
TRAJECTORY_FILTER_SINGLETON = true;
PLOT_HALF = 50;            % 100x100 axes centered on jaw rest (+/-50 px)
DRAW_SEGMENT_LINES = true;
DRAW_SCATTER = true;
LINE_WIDTH = 1.4;
MARKER_SIZE = 8;
OUTPUT_FMT = 'svg';

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

nWritten = 0;
nSkipped = 0;

for e = 1:numel(experiments)
    expTag = experiments(e).tag;
    paths = experiments(e).paths;
    outDir = fullfile(outRoot, expTag);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    fprintf('\n=== %s (%d session-view files) ===\n', expTag, numel(paths));

    for k = 1:numel(paths)
        jawFile = paths{k};
        if ~isfile(jawFile)
            warning('Missing jaw CSV, skipping: %s', jawFile);
            nSkipped = nSkipped + 1;
            continue
        end

        meta = parseBipolesJawMeta(jawFile);

        behFile = findBehaviorCsv(fileparts(jawFile), meta.view);
        if isempty(behFile)
            fprintf('  skip (no %s behavior CSV): %s\n', meta.view, meta.base);
            nSkipped = nSkipped + 1;
            continue
        end

        intervals = readLickIntervalsByLaser(behFile, LASER_MODE, FIRST_LICK_PER_LASER);
        if isempty(intervals)
            fprintf('  skip (no laser-%s licks): %s\n', LASER_MODE, meta.base);
            nSkipped = nSkipped + 1;
            continue
        end

        [lickX, lickY, lickPhase] = extractJawLickTrajectories(jawFile, intervals, PROB_MIN, ...
            MIN_LICK_FRAMES, LICK_FRAME_PAD, false);
        if isempty(lickX)
            fprintf('  skip (no jaw points within laser-%s licks): %s\n', LASER_MODE, meta.base);
            nSkipped = nSkipped + 1;
            continue
        end

        if TRAJECTORY_FILTER
            [lickX, lickY, lickPhase, fInfo] = filter_lick_trajectories(lickX, lickY, lickPhase, ...
                MIN_LICK_FRAMES, TRAJECTORY_STEP_MAD_K, TRAJECTORY_FILTER_SINGLETON, ...
                TRAJECTORY_FILTER_MODE, TRAJECTORY_STEP_HARD_MAX, ...
                TRAJECTORY_HOTSPOT_MIN_COUNT, TRAJECTORY_HOTSPOT_PURGE_COUNT);
            fprintf(['    trajectory filter (%s): %d licks, removed %d pts, ' ...
                'dropped %d short licks, T=%.2f\n'], fInfo.filterMode, fInfo.nKept, ...
                fInfo.nPtsRemoved, fInfo.nDropShort, fInfo.stepThreshold);
            if isempty(lickX)
                fprintf('  skip (no licks after trajectory filter): %s\n', meta.base);
                nSkipped = nSkipped + 1;
                continue
            end
        end

        [jx, jy] = jawSessionRestXY(jawFile, PROB_MIN);
        [lickX, lickY] = centerLickCells(lickX, lickY, jx, jy);

        outName = sprintf('%s_laser%s_jawtip_traj.%s', meta.base, upper(LASER_MODE), OUTPUT_FMT);
        outPath = fullfile(outDir, outName);

        renderAndSave(outPath, lickX, lickY, lickPhase, meta, cmapPhase, ...
            DRAW_SEGMENT_LINES, DRAW_SCATTER, LINE_WIDTH, MARKER_SIZE, ...
            PLOT_HALF, LASER_MODE, TRAJECTORY_LINE_BREAK_MAX);

        nWritten = nWritten + 1;
        fprintf('  saved (%d licks): %s\n', numel(lickX), outName);
    end
end

fprintf('\nDone. %d images written, %d skipped.\nOutput root: %s\n', ...
    nWritten, nSkipped, outRoot);
end


%% =======================================================================
%% Rendering
%% =======================================================================

function renderAndSave(outPath, lickX, lickY, lickPhase, meta, cmapPhase, ...
    drawLines, drawScatter, lineW, markerSize, plotHalf, laserMode, lineBreakMax)

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 620 560]);
ax = axes(fig); %#ok<LAXES>
hold(ax, 'on');
setupCenteredJawAxes(ax, plotHalf, cmapPhase);

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
        draw_phase_line(ax, xs, ys, ph, lineW, lineBreakMax, plotHalf);
    end
    if drawScatter
        scatX = [scatX; xs(:)];   %#ok<AGROW>
        scatY = [scatY; ys(:)];   %#ok<AGROW>
        scatPh = [scatPh; ph(:)]; %#ok<AGROW>
    end
end

if drawScatter && ~isempty(scatX)
    [scatX, scatY, scatPh] = filterScatterToSquare(scatX, scatY, scatPh, plotHalf);
    if ~isempty(scatX)
        scatter(ax, scatX, scatY, markerSize, scatPh, 'filled', ...
            'MarkerFaceAlpha', 0.9, 'MarkerEdgeColor', 'none');
    end
end

drawJawRestMarker(ax);
xlabel(ax, 'X relative to jaw rest (pixels)', 'Interpreter', 'none');
ylabel(ax, 'Y relative to jaw rest (pixels)', 'Interpreter', 'none');

title(ax, {meta.base, 'Jaw trajectory during laser-ON lick', ...
    sprintf('%d licks', numel(lickX))}, ...
    'Interpreter', 'none', 'FontSize', 10, 'Color', 'k');

cb = colorbar(ax);
cb.Label.String = 'Intra-lick phase (0=start, 1=end)';
cb.Label.Interpreter = 'none';
cb.Color = 'k';

hold(ax, 'off');

try
    exportgraphics(fig, outPath, 'ContentType', 'vector');
catch %#ok<CTCH>
    print(fig, outPath, '-dsvg', '-painters');
end
close(fig);
end


function lbl = laserModeLabel(laserMode)
switch lower(laserMode)
    case 'on'
        lbl = 'laser-ON';
    case 'off'
        lbl = 'laser-OFF';
    otherwise
        lbl = 'all';
end
end


function intervals = readLickIntervalsByLaser(behFile, laserMode, firstPerLaser)
% Read lick intervals from a BiPoles behavior CSV, filtered by laser state.
% When firstPerLaser is true and laserMode is 'on', keep only the first lick
% (minimum Interval Start) for each laser_Interval Overlap Assign ID >= 0.
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
