function telc_spontaneous_tip_trajectory_by_session_prob080
% telc_spontaneous_tip_trajectory_by_session_prob080
% Same as telc_spontaneous_tip_trajectory_by_session, but jaw-tip quality
% uses only the jaw CSV Probability column (model confidence >= PROB_MIN).
% All spontaneous licks from behavior CSV. No jump filter.
%
% Run: telc_spontaneous_tip_trajectory_by_session_prob080

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

jawCsvPaths = telc_pre_side_jaw_paths();

PROB_MIN = 0.80;         % minimum jaw-tip Probability (model confidence)
MIN_LICK_FRAMES = 2;
LICK_FRAME_PAD = 10;     % extra frames before/after behavior lick Start/End
PLOT_HALF = 50;          % 100x100 centered on jaw rest
DRAW_SEGMENT_LINES = true;
DRAW_SCATTER = true;
LINE_WIDTH = 1.4;
MARKER_SIZE = 8;

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

    [intervals, filtInfo] = readSpontaneousLickIntervals(behFile);
    fprintf('  %s: %d spontaneous licks from behavior CSV\n', meta.base, filtInfo.nKept);

    if isempty(intervals)
        fprintf('  skip (no spontaneous licks): %s\n', meta.base);
        nSkipped = nSkipped + 1;
        continue
    end

    [lickX, lickY, lickPhase, lickFrame] = extractJawLickTrajectories(jawFile, intervals, PROB_MIN, ...
        MIN_LICK_FRAMES, LICK_FRAME_PAD, true);
    if isempty(lickX)
        fprintf('  skip (no jaw points with prob >= %.2f in intervals): %s\n', PROB_MIN, meta.base);
        nSkipped = nSkipped + 1;
        continue
    end

    [jx, jy] = jawSessionRestXY(jawFile, PROB_MIN);
    [lickX, lickY] = centerLickCells(lickX, lickY, jx, jy);

    outName = sprintf('%s_spontaneous_jawtip_traj_prob080.%s', meta.base, OUTPUT_FMT);
    outPath = fullfile(outRoot, outName);

    renderAndSave(outPath, lickX, lickY, lickPhase, lickFrame, meta, cmapPhase, ...
        DRAW_SEGMENT_LINES, DRAW_SCATTER, LINE_WIDTH, MARKER_SIZE, PLOT_HALF, PROB_MIN);

    nWritten = nWritten + 1;
    fprintf('  saved (%d licks): %s\n', numel(lickX), outName);
end

fprintf('\nDone. %d images written, %d skipped.\nOutput: %s\n', nWritten, nSkipped, outRoot);
end


%% =======================================================================
%% Rendering
%% =======================================================================

function renderAndSave(outPath, lickX, lickY, lickPhase, lickFrame, meta, cmapPhase, ...
    drawLines, drawScatter, lineW, markerSize, plotHalf, probMin)

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
    fr = lickFrame{j};
    if isempty(xs)
        continue
    end
    if drawLines && numel(xs) > 1
        draw_phase_line_frame_gaps(ax, xs, ys, ph, fr, lineW, plotHalf);
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

title(ax, {meta.base, 'Jaw trajectory during spontaneous lick', ...
    sprintf('%d licks (prob >= %.2f)', numel(lickX), probMin)}, ...
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
