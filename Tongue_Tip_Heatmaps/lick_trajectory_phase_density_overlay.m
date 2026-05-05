function lick_trajectory_phase_density_overlay
% lick_trajectory_phase_density_overlay
%
% Jaw-centered pooled tongue trajectories only (no density background). Segments are drawn as
% colored line pieces; color is intra-lick phase (0 = first frame of lick, 1 = last frame).
%
% Licks: sort keypoints by Frame; start a new lick where consecutive frame gap > GAP_FRAMES.
% Only licks with at least MIN_LICK_FRAMES keypoint rows are kept.
%
% Edit csv lists below. Run: lick_trajectory_phase_density_overlay

close all

%% =======================================================================
%% CONFIG (match jaw_centered_pooled_heatmap.m)
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

% Minimum number of keypoint rows (frames with detections) per lick segment.
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

plotOneGroup(csvPaths_PCRt, 'PCRt_BiPoles', 'PCRt', PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, ...
    xMin, xMax, yMin, yMax, DRAW_SEGMENT_LINES, LINE_WIDTH, OUTPUT_DIR, SAVE_SVG);

plotOneGroup(csvPaths_IRt, 'IRt_BiPoles', 'IRt', PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, ...
    xMin, xMax, yMin, yMax, DRAW_SEGMENT_LINES, LINE_WIDTH, OUTPUT_DIR, SAVE_SVG);

end


function plotOneGroup(csvList, groupTag, shortTag, probMin, gapFrames, minLickFrames, ...
    xMin, xMax, yMin, yMax, drawLines, lineW, outDir, saveSvg)
    nSessions = 0;

    cmapPhase = phaseColormap256();

    fig = figure('Name', sprintf('Lick trajectory phase (%s)', groupTag), ...
        'NumberTitle', 'off', 'Color', 'w', ...
        'Position', [80 80 660 580]);
    ax = axes(fig);
    set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    axis(ax, [xMin xMax yMin yMax]);
    axis(ax, 'image');
    set(ax, 'YDir', 'reverse');
    colormap(ax, cmapPhase);
    caxis(ax, [0 1]);

    xlabel(ax, 'X relative to jaw (pixels)', 'Interpreter', 'none');
    ylabel(ax, 'Y relative to jaw (pixels)', 'Interpreter', 'none');
    hold(ax, 'on');

    cx0 = 0;
    cy0 = 0;

    for k = 1:numel(csvList)
        csvFile = csvList{k};
        if ~isfile(csvFile)
            continue
        end

        [cx, cy, phase, bridge] = sessionSortedTrajectoryPoints(csvFile, probMin, gapFrames, minLickFrames);
        if isempty(cx)
            continue
        end
        nSessions = nSessions + 1;

        if drawLines && ~isempty(bridge)
            pmid = (phase(1:end-1) + phase(2:end)) / 2;
            rgbLines = rgbFromPhase(pmid, cmapPhase);
            for ii = 1:numel(bridge)
                if bridge(ii)
                    plot(ax, [cx(ii), cx(ii + 1)], [cy(ii), cy(ii + 1)], ...
                        '-', 'Color', rgbLines(ii, :), 'LineWidth', lineW);
                end
            end
        end
    end

    title(ax, sprintf(['Intra-lick trajectories only (%d sessions) — %s'], ...
        nSessions, groupTag), 'Interpreter', 'none', 'FontSize', 11, 'Color', 'k');

    % Jaw reference on top (+ and box only).
    plot(ax, cx0, cy0, 'ws', 'MarkerSize', 14, 'LineWidth', 2.6, 'MarkerFaceColor', 'none');
    plot(ax, cx0, cy0, 'ks', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'none');
    plot(ax, cx0, cy0, 'w+', 'MarkerSize', 18, 'LineWidth', 2.8);
    plot(ax, cx0, cy0, 'k+', 'MarkerSize', 16, 'LineWidth', 2.2);

    hold(ax, 'off');

    cb = colorbar(ax);
    cb.Color = 'k';
    cb.Label.String = 'Intra-lick phase (0=start, 1=end)';
    cb.Label.Interpreter = 'none';

    if saveSvg
        outPath = fullfile(outDir, sprintf('lick_trajectory_phase_only_%s', shortTag));
        try
            exportgraphics(fig, [outPath '.svg'], 'ContentType', 'vector');
        catch %#ok<*CTCH>
            print(fig, [outPath '.svg'], '-dsvg', '-painters');
        end
        fprintf('Wrote %s.svg\n', outPath);
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


function [cx, cy, phase, bridge] = sessionSortedTrajectoryPoints(csvFile, probMin, gapFrames, minLickFrames)

    cx = [];
    cy = [];
    phase = [];
    bridge = [];

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
        cx = [cx; xs]; %#ok<AGROW>
        cy = [cy; ys]; %#ok<AGROW>
        phase = [phase; ph]; %#ok<AGROW>
        if L > 1
            bridge = [bridge; true(L - 1, 1)]; %#ok<AGROW>
        end
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
