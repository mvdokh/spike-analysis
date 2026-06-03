function telc_spontaneous_tip_trajectory_by_session_prob080
% telc_spontaneous_tip_trajectory_by_session_prob080
% Same as telc_spontaneous_tip_trajectory_by_session, but jaw-tip quality
% uses only the jaw CSV Probability column (model confidence >= PROB_MIN).
% Stereotyped-lick selection from behavior CSV is unchanged. No jump filter.
%
% Run: telc_spontaneous_tip_trajectory_by_session_prob080

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

jawCsvPaths = {
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Pre\IRt_TeLC08_pre_2026_03_31_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Pre\IRt_TeLC09_pre_2026_04_01_1_jaw.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Pre\IRt_TeLC11_pre_2026_03_30_1_jaw.csv'
    };

PROB_MIN = 0.80;         % minimum jaw-tip Probability (model confidence)
MIN_LICK_FRAMES = 2;
AXIS_MIN = 0;
AXIS_MAX = 256;
DRAW_SEGMENT_LINES = true;
DRAW_SCATTER = true;
LINE_WIDTH = 1.4;
MARKER_SIZE = 8;

% Stereotyped-lick filters (MAD-based, per behavior CSV session)
DURATION_MAD_K = 3;   % keep licks with duration within median +/- K*scaledMAD
AREA_MAD_K = 3;       % drop licks with Tongue_area_Interval Max far below median

OUTPUT_FMT = 'svg';

%% =======================================================================

thisDir = fileparts(mfilename('fullpath'));
outRoot = fullfile(thisDir, 'telc_spontaneous_tip_trajectories_prob080');
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

cmapPhase = phaseColormap256();
nWritten = 0;
nSkipped = 0;

fprintf('\n=== IRt_TeLC spontaneous (Pre, side view) ===\n');

for k = 1:numel(jawCsvPaths)
    jawFile = jawCsvPaths{k};
    if ~isfile(jawFile)
        warning('Missing jaw CSV, skipping: %s', jawFile);
        nSkipped = nSkipped + 1;
        continue
    end

    meta = parseTelcJawMeta(jawFile);
    if ~meta.isPre || ~meta.isSide
        fprintf('  skip (not Pre side view): %s\n', meta.base);
        nSkipped = nSkipped + 1;
        continue
    end

    behFile = findTelcSideBehaviorCsv(fileparts(jawFile));
    if isempty(behFile)
        fprintf('  skip (no side behavior CSV): %s\n', meta.base);
        nSkipped = nSkipped + 1;
        continue
    end

    [intervals, filtInfo] = readStereotypedSpontaneousLicks(behFile, DURATION_MAD_K, AREA_MAD_K);
    fprintf('  %s: %d/%d licks kept (duration [%g,%g] frames, area max >= %g)\n', ...
        meta.base, filtInfo.nKept, filtInfo.nTotal, ...
        filtInfo.durLo, filtInfo.durHi, filtInfo.areaLo);

    if isempty(intervals)
        fprintf('  skip (no stereotyped licks): %s\n', meta.base);
        nSkipped = nSkipped + 1;
        continue
    end

    [lickX, lickY, lickPhase, lickFrame] = jawLickTrajectories(jawFile, intervals, PROB_MIN, MIN_LICK_FRAMES);
    if isempty(lickX)
        fprintf('  skip (no jaw points with prob >= %.2f in intervals): %s\n', PROB_MIN, meta.base);
        nSkipped = nSkipped + 1;
        continue
    end

    outName = sprintf('%s_spontaneous_jawtip_traj_prob080.%s', meta.base, OUTPUT_FMT);
    outPath = fullfile(outRoot, outName);

    renderAndSave(outPath, lickX, lickY, lickPhase, lickFrame, meta, cmapPhase, ...
        DRAW_SEGMENT_LINES, DRAW_SCATTER, LINE_WIDTH, MARKER_SIZE, AXIS_MIN, AXIS_MAX, PROB_MIN);

    nWritten = nWritten + 1;
    fprintf('  saved (%d licks): %s\n', numel(lickX), outName);
end

fprintf('\nDone. %d images written, %d skipped.\nOutput: %s\n', nWritten, nSkipped, outRoot);
end


%% =======================================================================
%% Rendering
%% =======================================================================

function renderAndSave(outPath, lickX, lickY, lickPhase, lickFrame, meta, cmapPhase, ...
    drawLines, drawScatter, lineW, markerSize, axisMin, axisMax, probMin)

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 620 560]);
ax = axes(fig); %#ok<LAXES>
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
        'MarkerFaceAlpha', 0.9, 'MarkerEdgeColor', 'none');
end

axis(ax, 'equal');
axis(ax, 'square');
set(ax, 'YDir', 'reverse');
xlim(ax, [axisMin axisMax]);
ylim(ax, [axisMin axisMax]);
xticks(ax, [axisMin (axisMin + axisMax) / 2 axisMax]);
yticks(ax, [axisMin (axisMin + axisMax) / 2 axisMax]);
xlabel(ax, 'X (pixels)', 'Interpreter', 'none');
ylabel(ax, 'Y (pixels)', 'Interpreter', 'none');

title(ax, {meta.base, sprintf('%d licks (prob >= %.2f)', numel(lickX), probMin)}, ...
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


function draw_phase_line_frame_gaps(ax, x, y, ph, fr, lineW)
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


%% =======================================================================
%% Lick selection (stereotyped spontaneous)
%% =======================================================================

function [intervals, info] = readStereotypedSpontaneousLicks(behFile, durationMadK, areaMadK)
% Keep licks whose duration is near the session median (robust band) and
% whose Tongue_area_Interval Max is not anomalously small.
info = struct('nTotal', 0, 'nKept', 0, 'durLo', NaN, 'durHi', NaN, 'areaLo', NaN);

T = readtable(behFile, 'VariableNamingRule', 'preserve');
v = T.Properties.VariableNames;

startCol = find(contains(v, 'Interval Start') & contains(v, 'interval_detection'), 1);
endCol = find(contains(v, 'Interval End') & contains(v, 'interval_detection'), 1);
durCol = find(contains(v, 'Interval Duration') & contains(v, 'interval_detection'), 1);
areaCol = find(contains(v, 'Interval Max'), 1);
if isempty(areaCol)
    areaCol = find(contains(v, 'Tongue_area'), 1);
end

if isempty(startCol) || isempty(endCol)
    error('Behavior CSV missing lick interval columns: %s', behFile);
end

starts = double(T{:, startCol});
ends = double(T{:, endCol});
valid = isfinite(starts) & isfinite(ends) & ends >= starts;
info.nTotal = sum(valid);

if any(durCol)
    dur = double(T{:, durCol});
else
    dur = ends - starts + 1;
end
if any(areaCol)
    areaMax = double(T{:, areaCol});
else
    areaMax = nan(size(starts));
end

dur = dur(valid);
starts = starts(valid);
ends = ends(valid);
areaMax = areaMax(valid);

[keepDur, durLo, durHi] = robustCentralMask(dur, durationMadK);
[keepArea, areaLo] = robustLowerMask(areaMax, areaMadK);

keep = keepDur & keepArea;
info.nKept = sum(keep);
info.durLo = durLo;
info.durHi = durHi;
info.areaLo = areaLo;

intervals = [starts(keep), ends(keep)];
end


function [keep, lo, hi] = robustCentralMask(x, K)
% Keep values within median +/- K * (1.4826 * MAD).
x = x(:);
x = x(isfinite(x));
keep = false(size(x));
lo = NaN;
hi = NaN;
if isempty(x)
    return
end
med = median(x);
madv = median(abs(x - med));
smad = 1.4826 * madv;
if smad <= 0 || ~isfinite(smad)
    smad = std(x);
end
if smad <= 0 || ~isfinite(smad)
    keep(:) = true;
    lo = min(x);
    hi = max(x);
    return
end
lo = med - K * smad;
hi = med + K * smad;
keep = (x >= lo) & (x <= hi);
end


function [keep, lo] = robustLowerMask(x, K)
% Drop values far below the session median (super-small tongue area).
x = x(:);
x = x(isfinite(x));
keep = false(size(x));
lo = NaN;
if isempty(x)
    return
end
med = median(x);
madv = median(abs(x - med));
smad = 1.4826 * madv;
if smad <= 0 || ~isfinite(smad)
    smad = std(x);
end
if smad <= 0 || ~isfinite(smad)
    keep(:) = true;
    lo = min(x);
    return
end
lo = med - K * smad;
keep = x >= lo;
end


%% =======================================================================
%% Trajectory extraction
%% =======================================================================

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


%% =======================================================================
%% File / metadata helpers
%% =======================================================================

function meta = parseTelcJawMeta(jawFile)
[~, baseName, ~] = fileparts(jawFile);
meta.base = regexprep(baseName, '_jaw$', '');
baseLower = lower(baseName);

meta.isSide = contains(baseLower, '_1_jaw') || endsWith(baseLower, '_1_jaw');
meta.isPre = contains(baseLower, '_pre_') || contains(baseLower, '_pre');

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
