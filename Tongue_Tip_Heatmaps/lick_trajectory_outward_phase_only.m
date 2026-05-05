function lick_trajectory_outward_phase_only
% lick_trajectory_outward_phase_only
%
% Jaw-centered pooled tongue trajectories only (no density background). Segments are drawn as
% colored line pieces; color is outward intra-lick phase (0 = lick start, 1 = max protrusion).
% Each lick is truncated to the segment from lick start up to the first frame at max distance
% from the jaw.
%
% Licks: sort keypoints by Frame; start a new lick where consecutive frame gap > GAP_FRAMES.
% Only licks with at least MIN_LICK_FRAMES keypoint rows (before truncation) are kept.
%
% Edit csv lists below. Run: lick_trajectory_outward_phase_only

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

% Minimum number of keypoint rows per lick segment (full lick before outward truncation).
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

    fig = figure('Name', sprintf('Outward lick trajectory phase (%s)', groupTag), ...
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

    for k = 1:numel(csvList)
        csvFile = csvList{k};
        if ~isfile(csvFile)
            continue
        end

        [cx, cy, phase, bridge] = sessionOutwardTrajectoryPoints(csvFile, probMin, gapFrames, minLickFrames);
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

    title(ax, sprintf(['Outward trajectories only (%d sessions) — %s'], ...
        nSessions, groupTag), 'Interpreter', 'none', 'FontSize', 11, 'Color', 'k');

    % Jaw reference on top (+ and box only).
    plot(ax, 0, 0, 'ws', 'MarkerSize', 14, 'LineWidth', 2.6, 'MarkerFaceColor', 'none');
    plot(ax, 0, 0, 'ks', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'none');
    plot(ax, 0, 0, 'w+', 'MarkerSize', 18, 'LineWidth', 2.8);
    plot(ax, 0, 0, 'k+', 'MarkerSize', 16, 'LineWidth', 2.2);

    hold(ax, 'off');

    cb = colorbar(ax);
    cb.Color = 'k';
    cb.Label.String = 'Outward phase (0=lick start, 1=max protrusion)';
    cb.Label.Interpreter = 'none';

    if saveSvg
        outPath = fullfile(outDir, sprintf('lick_trajectory_outward_phase_only_%s', shortTag));
        try
            exportgraphics(fig, [outPath '.svg'], 'ContentType', 'vector');
        catch %#ok<*CTCH>
            print(fig, [outPath '.svg'], '-dsvg', '-painters');
        end
        fprintf('Wrote %s.svg\n', outPath);
    end
end


function [cx, cy, phase, bridge] = sessionOutwardTrajectoryPoints(csvFile, probMin, gapFrames, minLickFrames)
    cx = [];
    cy = [];
    phase = [];
    bridge = [];

    [xo, yo, starts, ends] = sortedJawCenteredLicks(csvFile, probMin, gapFrames);
    if isempty(xo)
        return
    end

    for j = 1:numel(starts)
        s = starts(j);
        e = ends(j);
        Lfull = e - s + 1;
        if Lfull < minLickFrames
            continue
        end
        x = xo(s:e);
        y = yo(s:e);
        [xOut, yOut, pOut] = outwardSegmentFromLick(x, y);
        if numel(xOut) < 2
            continue
        end
        L = numel(xOut);
        cx = [cx; xOut(:)]; %#ok<AGROW>
        cy = [cy; yOut(:)]; %#ok<AGROW>
        phase = [phase; pOut(:)]; %#ok<AGROW>
        bridge = [bridge; true(L - 1, 1)]; %#ok<AGROW>
    end
end


function [xOut, yOut, pOut] = outwardSegmentFromLick(x, y)
    d2 = x(:).^2 + y(:).^2;
    [~, imax] = max(d2); % first max index gives outward component start->max
    xOut = x(1:imax);
    yOut = y(1:imax);
    L = numel(xOut);
    if L <= 1
        pOut = 1;
    else
        pOut = ((0:(L - 1))' / (L - 1));
    end
end


function [xo, yo, starts, ends] = sortedJawCenteredLicks(csvFile, probMin, gapFrames)
    xo = [];
    yo = [];
    starts = [];
    ends = [];

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
    breaks = find(diff(fs) > gapFrames);
    starts = [1; breaks + 1];
    ends = [breaks; nf];
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
