function lick_trajectory_phase_density_overlay_by_animal_start_to_max
% lick_trajectory_phase_density_overlay_by_animal_start_to_max
%
% Jaw-centered tongue trajectories per animal (one subplot per animal), like
% lick_trajectory_phase_density_overlay_by_animal, but each lick is drawn only
% from its first frame to the frame of maximum protrusion. Protrusion is the
% Euclidean distance from the jaw-centered origin per lick; the peak index is
% chosen within that lick. Phase is linear 0=start of path, 1=max protrusion
% (open trajectory, no return phase).
%
% Edit csv lists below. Run: lick_trajectory_phase_density_overlay_by_animal_start_to_max

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

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

PROB_MIN = 0;
GAP_FRAMES = 8;
MIN_LICK_FRAMES = 5;

REL_EXTENT_HALF = 128;
DRAW_SEGMENT_LINES = true;
LINE_WIDTH = 1.15;

OUTPUT_DIR = '';
SAVE_SVG = true;

%% =======================================================================

thisDir = fileparts(mfilename('fullpath'));
if isempty(OUTPUT_DIR)
    OUTPUT_DIR = thisDir;
end

xMin = -REL_EXTENT_HALF;
xMax = REL_EXTENT_HALF;
yMin = -REL_EXTENT_HALF;
yMax = REL_EXTENT_HALF;

plotOneExperimentByAnimal(csvPaths_PCRt, 'PCRt_BiPoles', 'PCRt', PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, ...
    xMin, xMax, yMin, yMax, DRAW_SEGMENT_LINES, LINE_WIDTH, OUTPUT_DIR, SAVE_SVG);

plotOneExperimentByAnimal(csvPaths_IRt, 'IRt_BiPoles', 'IRt', PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, ...
    xMin, xMax, yMin, yMax, DRAW_SEGMENT_LINES, LINE_WIDTH, OUTPUT_DIR, SAVE_SVG);

end


function plotOneExperimentByAnimal(csvList, groupTag, shortTag, probMin, gapFrames, minLickFrames, ...
    xMin, xMax, yMin, yMax, drawLines, lineW, outDir, saveSvg)

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
    fig = figure('Name', sprintf('Lick trajectory start to max protrusion by animal (%s)', groupTag), ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [80 80 figW figH]);

    tl = tiledlayout(fig, nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, sprintf('Start to max protrusion by animal (%s)', groupTag), ...
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
        nSessions = 0;

        for k = 1:numel(csvFiles)
            csvFile = csvFiles{k};
            if ~isfile(csvFile)
                continue
            end

            [lickX, lickY, ~] = sessionSortedTrajectoryPoints(csvFile, probMin, gapFrames, minLickFrames);
            if isempty(lickX)
                continue
            end
            nSessions = nSessions + 1;

            for j = 1:numel(lickX)
                [xs, ys, ph] = trimStartToMaxProtrusion(lickX{j}, lickY{j});
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

                scatter(ax, xs, ys, 16, ph, 'filled', ...
                    'MarkerFaceAlpha', 0.95, 'MarkerEdgeColor', 'none');
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

        title(ax, sprintf('%s (%d sessions)', animalLabel, nSessions), ...
            'Interpreter', 'none', 'Color', 'k', 'FontSize', 10);
        xlabel(ax, 'X relative to jaw (pixels)', 'Interpreter', 'none');
        ylabel(ax, 'Y relative to jaw (pixels)', 'Interpreter', 'none');
        hold(ax, 'off');
    end

    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Color = 'k';
    cb.Label.String = 'Phase along protrusion (0=start, 1=max protrusion)';
    cb.Label.Interpreter = 'none';

    if saveSvg
        outPath = fullfile(outDir, sprintf('lick_trajectory_phase_start_to_max_by_animal_%s', shortTag));
        try
            exportgraphics(fig, [outPath '.svg'], 'ContentType', 'vector');
        catch %#ok<*CTCH>
            print(fig, [outPath '.svg'], '-dsvg', '-painters');
        end
        fprintf('Wrote %s.svg\n', outPath);
    end
end


function [xs, ys, ph] = trimStartToMaxProtrusion(xs, ys)
% Keep frames 1:k where k is the first index attaining max ||(x,y)|| (jaw-centered).
    xs = xs(:);
    ys = ys(:);
    if isempty(xs)
        ph = [];
        return
    end
    r = hypot(xs, ys);
    [~, k] = max(r);
    xs = xs(1:k);
    ys = ys(1:k);
    L = numel(xs);
    if L == 1
        ph = 0;
    else
        ph = ((0:(L - 1))' / (L - 1));
    end
end


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


function [lickX, lickY, lickPhase] = sessionSortedTrajectoryPoints(csvFile, probMin, gapFrames, minLickFrames)
    lickX = {};
    lickY = {};
    lickPhase = {};

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
        if L == 1
            ph = 0.5;
        else
            ph = ((0:(L - 1))' / (L - 1));
        end

        lickX{end + 1} = xs; %#ok<AGROW>
        lickY{end + 1} = ys; %#ok<AGROW>
        lickPhase{end + 1} = ph; %#ok<AGROW>
    end
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
