function telc_spontaneous_tip_trajectory_by_session
% telc_spontaneous_tip_trajectory_by_session
% Intra-lick jaw-tip trajectories during spontaneous licking for IRt_TeLC
% Pre sessions, side view only. All licks from side_behavior CSV interval
% columns. One SVG per animal/session.
%
% Run: telc_spontaneous_tip_trajectory_by_session

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

% IRt_TeLC08/09/11 each have IRt_TeLC##_Pre and IRt_TeLC##_Post subfolders.
jawCsvPaths = telc_pre_side_jaw_paths();

PROB_MIN = 0;
MIN_LICK_FRAMES = 2;
LICK_FRAME_PAD = 10;     % extra frames before/after behavior lick Start/End
TRAJECTORY_FILTER = true;        % drop whole licks with singleton coords or big jumps
TRAJECTORY_STEP_MAD_K = 5;
TRAJECTORY_FILTER_SINGLETON = true;
PLOT_HALF = 50;            % 100x100 centered on jaw rest
DRAW_SEGMENT_LINES = true;
DRAW_SCATTER = true;
LINE_WIDTH = 1.4;
MARKER_SIZE = 8;

OUTPUT_FMT = 'svg';

%% =======================================================================

thisDir = fileparts(mfilename('fullpath'));
outRoot = fullfile(thisDir, 'telc_spontaneous_tip_trajectories');
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

    [intervals, filtInfo] = readFirstLickPerBoutIntervals(behFile);
    fprintf('  %s: %d first-in-bout licks (%d total licks, %d bouts)\n', ...
        meta.base, filtInfo.nKept, filtInfo.nTotal, filtInfo.nBouts);

    if isempty(intervals)
        fprintf('  skip (no spontaneous licks): %s\n', meta.base);
        nSkipped = nSkipped + 1;
        continue
    end

    [lickX, lickY, lickPhase] = extractJawLickTrajectories(jawFile, intervals, PROB_MIN, ...
        MIN_LICK_FRAMES, LICK_FRAME_PAD, false);
    if isempty(lickX)
        fprintf('  skip (no jaw points in intervals): %s\n', meta.base);
        nSkipped = nSkipped + 1;
        continue
    end

    if TRAJECTORY_FILTER
        [lickX, lickY, lickPhase, fInfo] = filter_lick_trajectories(lickX, lickY, lickPhase, ...
            MIN_LICK_FRAMES, TRAJECTORY_STEP_MAD_K, TRAJECTORY_FILTER_SINGLETON, 'lick');
        fprintf(['    trajectory filter: kept %d/%d licks (singleton -%d, jump -%d, ' ...
            'T=%.2f)\n'], fInfo.nKept, fInfo.nIn, fInfo.nDropSingleton, fInfo.nDropJump, ...
            fInfo.stepThreshold);
        if isempty(lickX)
            fprintf('  skip (no licks after trajectory filter): %s\n', meta.base);
            nSkipped = nSkipped + 1;
            continue
        end
    end

    [jx, jy] = jawSessionRestXY(jawFile, PROB_MIN);
    [lickX, lickY] = centerLickCells(lickX, lickY, jx, jy);

    outName = sprintf('%s_spontaneous_jawtip_traj.%s', meta.base, OUTPUT_FMT);
    outPath = fullfile(outRoot, outName);

    renderAndSave(outPath, lickX, lickY, lickPhase, meta, cmapPhase, ...
        DRAW_SEGMENT_LINES, DRAW_SCATTER, LINE_WIDTH, MARKER_SIZE, PLOT_HALF);

    nWritten = nWritten + 1;
    fprintf('  saved (%d licks): %s\n', numel(lickX), outName);
end

fprintf('\nDone. %d images written, %d skipped.\nOutput: %s\n', nWritten, nSkipped, outRoot);
end


%% =======================================================================
%% Rendering
%% =======================================================================

function renderAndSave(outPath, lickX, lickY, lickPhase, meta, cmapPhase, ...
    drawLines, drawScatter, lineW, markerSize, plotHalf)

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
        draw_phase_line(ax, xs, ys, ph, lineW, [], plotHalf);
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
    sprintf('%d first-in-bout licks', numel(lickX))}, ...
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
