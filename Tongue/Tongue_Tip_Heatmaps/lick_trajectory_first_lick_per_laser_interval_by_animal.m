function lick_trajectory_first_lick_per_laser_interval_by_animal
% lick_trajectory_first_lick_per_laser_interval_by_animal
%
% For each session directory listed in CONFIG, reads the behavior CSV next to
% keypoints.csv. Filename may have any prefix; it must end with BEHAVIOR_FILE_SUFFIX
% (e.g. *bottom_view_behavior_100_3.csv).
%
% Laser intervals: columns laser_Interval Overlap Assign ID, Start, End (repeated on each
% lick row). Each unique laser ID (excluding -1) gets a frame window [min(Start), max(End)]
% across rows for that ID.
%
% First lick per interval: among tongue-tip keypoint segments (same gap rules as other
% Tongue_Tip_Heatmaps scripts), the segment with the earliest start frame that overlaps
% the laser Assign [Start, End] window.
%
% One figure per cohort (PCRt / IRt): subplot per animal; all first-lick trajectories
% from all sessions for that animal are overlaid (phase colormap, same style as
% lick_trajectory_phase_density_overlay_by_animal).
%
% Run: lick_trajectory_first_lick_per_laser_interval_by_animal

close all

%% CONFIG (paths mirror lick_trajectory_phase_density_overlay_by_animal)

csvPaths_PCRt = {
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1206\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1218\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1223\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0401\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0403\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0401\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0403\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0514\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0515\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0516\keypoints.csv'
    };

csvPaths_IRt = {
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0425\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0514\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0515\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0516\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0425\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0514\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0515\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0516\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_03\2025_0425\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0113\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0116\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0112\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0113\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0116\keypoints.csv'
    'C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0112\keypoints.csv'
    };

% Basename suffix (not necessarily the whole filename). Any CSV in the session
% folder whose name ends with this string is used (e.g. ABC_bottom_view_behavior_100_3.csv).
BEHAVIOR_FILE_SUFFIX = 'bottom_view_behavior_100_3.csv';

PROB_MIN = 0;
GAP_FRAMES = 8;
MIN_LICK_FRAMES = 5;

REL_EXTENT_HALF = 128;
DRAW_SEGMENT_LINES = true;
LINE_WIDTH = 1.0;
SCATTER_SIZE = 14;

OUTPUT_DIR = '';
SAVE_SVG = true;

thisDir = fileparts(mfilename('fullpath'));
if isempty(OUTPUT_DIR)
    OUTPUT_DIR = thisDir;
end

xMin = -REL_EXTENT_HALF;
xMax = REL_EXTENT_HALF;
yMin = -REL_EXTENT_HALF;
yMax = REL_EXTENT_HALF;

plotOneCohort(csvPaths_PCRt, 'PCRt_BiPoles', 'PCRt', BEHAVIOR_FILE_SUFFIX, ...
    PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, xMin, xMax, yMin, yMax, ...
    DRAW_SEGMENT_LINES, LINE_WIDTH, SCATTER_SIZE, OUTPUT_DIR, SAVE_SVG);

plotOneCohort(csvPaths_IRt, 'IRt_BiPoles', 'IRt', BEHAVIOR_FILE_SUFFIX, ...
    PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, xMin, xMax, yMin, yMax, ...
    DRAW_SEGMENT_LINES, LINE_WIDTH, SCATTER_SIZE, OUTPUT_DIR, SAVE_SVG);

end

%% ------------------------------------------------------------------------

function plotOneCohort(csvList, groupTag, shortTag, behaviorFileSuffix, probMin, gapFrames, minLickFrames, ...
    xMin, xMax, yMin, yMax, drawLines, lineW, scatterSz, outDir, saveSvg)

    animalGroups = groupCsvFilesByAnimal(csvList);
    if isempty(animalGroups)
        warning('No valid CSV files found for %s.', groupTag);
        return
    end

    cmapPhase = phaseColormap256();
    nAnimals = numel(animalGroups);
    nCol = 2;
    nRow = ceil(nAnimals / nCol);

    figW = min(260 + 360 * nCol, 2200);
    figH = min(220 + 280 * nRow, 1800);
    fig = figure('Name', sprintf('First lick / laser interval — %s', groupTag), ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [80 80 figW figH]);

    tl = tiledlayout(fig, nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, sprintf('First keypoint lick per laser window ([Assign Start/End]) — %s', groupTag), ...
        'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 14, 'Color', 'k');

    colormap(fig, cmapPhase);

    for i = 1:nAnimals
        ax = nexttile(tl);
        set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
        axis(ax, 'equal');
        axis(ax, 'square');
        set(ax, 'YDir', 'reverse');
        caxis(ax, [0 1]);
        hold(ax, 'on');

        animalLabel = animalGroups(i).label;
        csvFiles = animalGroups(i).files;
        nPlotted = 0;
        nSessionsUsed = 0;

        for k = 1:numel(csvFiles)
            csvFile = csvFiles{k};
            if ~isfile(csvFile)
                continue
            end
            sessionDir = fileparts(csvFile);
            bPath = resolveBehaviorCsvPath(sessionDir, behaviorFileSuffix);
            if isempty(bPath)
                warning('Missing behavior CSV (*%s) in: %s', behaviorFileSuffix, sessionDir);
                continue
            end

            laserIntervals = extractLaserIntervalsFromBehavior(bPath);
            if isempty(laserIntervals)
                continue
            end

            [lickX, lickY, lickPhase, f0, f1] = sessionSortedTrajectoryPointsWithFrames(csvFile, probMin, gapFrames, minLickFrames);
            if isempty(lickX)
                continue
            end

            nSessionsUsed = nSessionsUsed + 1;

            for r = 1:numel(laserIntervals)
                iv = laserIntervals(r);
                jLick = firstKeypointLickOverlappingLaserWindow(f0, f1, iv.lStart, iv.lEnd);
                if isempty(jLick)
                    continue
                end

                xs = lickX{jLick};
                ys = lickY{jLick};
                ph = lickPhase{jLick};
                if isempty(xs)
                    continue
                end

                if drawLines && numel(xs) > 1
                    pmid = (ph(1:end-1) + ph(2:end)) / 2;
                    lineColors = rgbFromPhase(pmid, cmapPhase);
                    for ii = 1:(numel(xs) - 1)
                        plot(ax, xs(ii:ii + 1), ys(ii:ii + 1), '-', ...
                            'Color', lineColors(ii, :), 'LineWidth', lineW);
                    end
                end

                scatter(ax, xs, ys, scatterSz, ph, 'filled', ...
                    'MarkerFaceAlpha', 0.88, 'MarkerEdgeColor', 'none');
                nPlotted = nPlotted + 1;
            end
        end

        plot(ax, 0, 0, 'ws', 'MarkerSize', 14, 'LineWidth', 2.6, 'MarkerFaceColor', 'none');
        plot(ax, 0, 0, 'ks', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'none');
        plot(ax, 0, 0, 'w+', 'MarkerSize', 18, 'LineWidth', 2.8);
        plot(ax, 0, 0, 'k+', 'MarkerSize', 16, 'LineWidth', 2.2);

        xlim(ax, [xMin xMax]);
        ylim(ax, [yMin yMax]);
        xticks(ax, [xMin 0 xMax]);
        yticks(ax, [yMin 0 yMax]);

        title(ax, sprintf('%s (%d sess, %d trajs)', animalLabel, nSessionsUsed, nPlotted), ...
            'Interpreter', 'none', 'Color', 'k', 'FontSize', 10);
        xlabel(ax, 'X relative to jaw (pixels)', 'Interpreter', 'none');
        ylabel(ax, 'Y relative to jaw (pixels)', 'Interpreter', 'none');
        hold(ax, 'off');
    end

    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Color = 'k';
    cb.Label.String = 'Intra-lick phase (0=start, 1=end)';
    cb.Label.Interpreter = 'none';

    if saveSvg
        outPath = fullfile(outDir, sprintf('lick_trajectory_first_per_laser_interval_by_animal_%s', shortTag));
        try
            exportgraphics(fig, [outPath '.svg'], 'ContentType', 'vector');
        catch %#ok<*CTCH>
            print(fig, [outPath '.svg'], '-dsvg', '-painters');
        end
        fprintf('Wrote %s.svg\n', outPath);
    end
end

function bPath = resolveBehaviorCsvPath(sessionDir, fileSuffix)
% Pick behavior CSV: exact basename match if present, else any file in sessionDir
% whose name ends with fileSuffix (wildcard *suffix).
    bPath = '';
    if nargin < 2 || isempty(sessionDir) || isempty(fileSuffix)
        return
    end
    fileSuffix = strtrim(fileSuffix);
    if isempty(fileSuffix)
        return
    end

    cand = fullfile(sessionDir, fileSuffix);
    if isfile(cand)
        bPath = cand;
        return
    end

    if fileSuffix(1) == '*'
        pat = fileSuffix;
    else
        pat = ['*' fileSuffix];
    end

    L = dir(fullfile(sessionDir, pat));
    if isempty(L)
        return
    end
    L = L(~[L.isdir]);
    if isempty(L)
        return
    end

    names = sort({L.name});
    bPath = fullfile(sessionDir, names{1});
    if numel(names) > 1
        warning('Multiple behavior CSVs match "%s" in %s; using %s', pat, sessionDir, names{1});
    end
end

%% Behavior: laser windows from Assign ID / Start / End; keypoint picks first overlapping lick

function intervals = extractLaserIntervalsFromBehavior(behaviorPath)

    T = readBehaviorCsv(behaviorPath);
    if isempty(T) || height(T) < 1
        intervals = repmat(struct('laserId', NaN, 'lStart', NaN, 'lEnd', NaN), 0, 1);
        return
    end

    [iId, iLS, iLE] = laserOverlapAssignColumnIndices(T);
    if isempty(iId) || isempty(iLS) || isempty(iLE)
        warning('Could not resolve laser overlap assign columns (ID, Start, End) in %s', behaviorPath);
        return
    end

    laserId = double(T{:, iId});
    assignStart = double(T{:, iLS});
    assignEnd = double(T{:, iLE});

    okRow = isfinite(laserId) & isfinite(assignStart) & isfinite(assignEnd) & laserId ~= -1;
    if ~any(okRow)
        return
    end

    uniq = unique(laserId(okRow));
    n = numel(uniq);
    if n < 1
        intervals = repmat(struct('laserId', NaN, 'lStart', NaN, 'lEnd', NaN), 0, 1);
        return
    end

    intervals = repmat(struct('laserId', NaN, 'lStart', NaN, 'lEnd', NaN), n, 1);

    for ii = 1:n
        u = uniq(ii);
        m = okRow & laserId == u;
        % Rows for this interval repeat the same laser window; span all listed values.
        intervals(ii).laserId = u;
        intervals(ii).lStart = min(assignStart(m));
        intervals(ii).lEnd = max(assignEnd(m));
    end
end


function [iId, iLS, iLE] = laserOverlapAssignColumnIndices(T)

    v = T.Properties.VariableNames;
    nv = numel(v);
    iId = [];
    iLS = [];
    iLE = [];

    for k = 1:nv
        s = lower(strrep(v{k}, ' ', ''));
        if ~contains(s, 'laser') || ~contains(s, 'overlap') || ~contains(s, 'assign')
            continue
        end
        if contains(s, 'start')
            iLS = k;
        elseif contains(s, 'end')
            iLE = k;
        elseif contains(s, 'id')
            iId = k;
        end
    end
end


% Among keypoint segments [f0,f1], return the one with the smallest segment start that
% overlaps the laser assign window [lStart, lEnd]. Empty if none overlap.
function j = firstKeypointLickOverlappingLaserWindow(f0, f1, lStart, lEnd)

    j = [];
    n = numel(f0);
    if n < 1 || isempty(lStart) || isempty(lEnd)
        return
    end

    lo = min(double(lStart), double(lEnd));
    hi = max(double(lStart), double(lEnd));

    bestF0 = inf;
    for jj = 1:n
        fj0 = min(double(f0(jj)), double(f1(jj)));
        fj1 = max(double(f0(jj)), double(f1(jj)));
        a = max(lo, fj0);
        b = min(hi, fj1);
        ov = max(0, b - a + 1);
        if ov > 0 && fj0 < bestF0
            bestF0 = fj0;
            j = jj;
        end
    end
end


function T = readBehaviorCsv(path)

    try
        opts = detectImportOptions(path, 'VariableNamingRule', 'preserve');
        T = readtable(path, opts);
    catch %#ok<*CTCH>
        try
            T = readtable(path, 'VariableNamingRule', 'preserve');
        catch
            T = readtable(path);
        end
    end
end


%% Keypoint segmentation (+ frame span per lick)

function [lickX, lickY, lickPhase, f0, f1] = sessionSortedTrajectoryPointsWithFrames(csvFile, probMin, gapFrames, minLickFrames)

    lickX = {};
    lickY = {};
    lickPhase = {};
    f0 = [];
    f1 = [];

    tbl = readKeypointsCsv(csvFile);
    frm = tbl.Frame;
    xv = tbl.X;
    yv = tbl.Y;
    if isempty(probMin) || (~isscalar(probMin)) || probMin <= 0
        keep = true(size(xv));
    else
        keep = tbl.Probability >= probMin;
    end
    frm = frm(keep);
    xv = xv(keep);
    yv = yv(keep);
    if numel(xv) < 1
        return
    end

    [jx, jy] = jawMeanFromBottomView(csvFile, probMin);
    if isnan(jx) || isnan(jy)
        return
    end

    xr = xv(:) - jx;
    yr = yv(:) - jy;

    [fs, ord] = sort(double(frm(:)));
    xo = xr(ord);
    yo = yr(ord);
    nf = numel(fs);

    bk = find(diff(fs) > gapFrames);
    starts = [1; bk + 1];
    ends = [bk; nf];

    for j = 1:numel(starts)
        s = starts(j);
        e = ends(j);
        L = e - s + 1;
        if L < minLickFrames
            continue
        end
        xs = xo(s:e);
        ys = yo(s:e);
        fr = fs(s:e);
        if L == 1
            ph = 0.5;
        else
            ph = ((0:(L - 1))' / (L - 1));
        end

        lickX{end + 1} = xs; %#ok<AGROW>
        lickY{end + 1} = ys; %#ok<AGROW>
        lickPhase{end + 1} = ph; %#ok<AGROW>
        f0(end + 1, 1) = fr(1); %#ok<AGROW>
        f1(end + 1, 1) = fr(end); %#ok<AGROW>
    end
end

%% Shared helpers (same as lick_trajectory_phase_density_overlay_by_animal)

function groups = groupCsvFilesByAnimal(csvList)
    groups = struct('label', {}, 'files', {});
    labelToIndex = containers.Map('KeyType', 'char', 'ValueType', 'double');

    for k = 1:numel(csvList)
        csvFile = csvList{k};
        if ~isfile(csvFile)
            continue
        end

        animalLabel = animalLabelFromCsv(csvFile);
        if isempty(animalLabel)
            continue
        end

        if isKey(labelToIndex, animalLabel)
            idx = labelToIndex(animalLabel);
            groups(idx).files{end + 1} = csvFile; %#ok<AGROW>
        else
            idx = numel(groups) + 1;
            labelToIndex(animalLabel) = idx;
            groups(idx).label = animalLabel;
            groups(idx).files = {csvFile};
        end
    end
end


function label = animalLabelFromCsv(csvFile)
    token = regexp(csvFile, '(PCRt|IRt)_[0-9]+', 'match', 'once');
    if isempty(token)
        label = '';
    else
        label = token;
    end
end


function cmap = phaseColormap256()
    try
        cmap = turbo(256);
    catch %#ok<*CTCH>
        cmap = jet(256);
    end
end


function rgb = rgbFromPhase(phase, cmap)
    phase = phase(:);
    phase = max(0, min(1, phase));
    idx = round(phase * (size(cmap, 1) - 1)) + 1;
    rgb = cmap(idx, :);
end


function [jx, jy] = jawMeanFromBottomView(csvFile, probMin)
    jx = NaN;
    jy = NaN;
    sessionDir = fileparts(csvFile);
    jawFiles = dir(fullfile(sessionDir, '*_bottom_view_jaw.csv'));
    if isempty(jawFiles)
        return
    end
    names = sort({jawFiles.name});
    jawFile = fullfile(sessionDir, names{1});

    jawTbl = readKeypointsCsv(jawFile);
    if height(jawTbl) < 1
        return
    end

    if isempty(probMin) || (~isscalar(probMin)) || probMin <= 0
        keep = true(height(jawTbl), 1);
    else
        keep = jawTbl.Probability >= probMin;
        if ~any(keep)
            return
        end
    end

    jx = mean(jawTbl.X(keep), 'omitnan');
    jy = mean(jawTbl.Y(keep), 'omitnan');
end


function tbl = readKeypointsCsv(csvFile)
    T = readtable(csvFile);
    v = T.Properties.VariableNames;
    vl = lower(v);
    fc = @(s) strcmp(vl, lower(s));

    fi = fc('frame');
    xi = fc('x');
    yi = fc('y');
    pi = fc('probability');
    if ~(any(fi) && any(xi) && any(yi))
        error('CSV must include Frame, X, and Y columns: %s', csvFile);
    end
    if sum(fi) > 1 || sum(xi) > 1 || sum(yi) > 1
        error('Ambiguous duplicate column names: %s', csvFile);
    end

    F = double(T{:, fi});
    X = double(T{:, xi});
    Y = double(T{:, yi});
    if any(pi)
        Pr = double(T{:, pi});
    else
        Pr = ones(size(F));
    end

    tbl = table(F, X, Y, Pr, 'VariableNames', {'Frame', 'X', 'Y', 'Probability'});
end
