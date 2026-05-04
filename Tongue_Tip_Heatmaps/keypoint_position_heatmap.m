function keypoint_position_heatmap
% keypoint_position_heatmap
% Build 256×256 density heatmaps (MATLAB parula). X/Y axes use a fixed pixel range (see CONFIG:
% AXIS_MIN, AXIS_MAX; default 0-256). Binning ignores keypoint extent. Each detection splats as a
% truncated 2-D Gaussian (support clipped at GAUSS_TRUNC_BINS in bin units).
%
% Figure 1: all keypoints. Figure 2: max-extension points only (per bout).
%
% CSV columns: Frame, X, Y, Probability (header row recommended).
%
% Bout rule: sort by Frame; a new bout starts where gap exceeds GAP_FRAMES.
% Maximum extension within a bout: farthest-from-onset vs farthest-from-offset by Euclidean
% distance; keep whichever distance is larger (ties use onset-based).

close all

%% =======================================================================
%% CONFIG — edit csvPaths below for batch runs
%% =======================================================================

csvPaths = {
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1206\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1218\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_02\2024_1223\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0401\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_07\2025_0403\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0321\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0326\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0401\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_08\2025_0403\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0514\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0515\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\PCRt_BiPoles\PCRt_09\2025_0516\keypoints.csv"

    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0425\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0514\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0515\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_01\2025_0516\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0425\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0514\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0515\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_02\2025_0516\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_03\2025_0425\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0113\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0116\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_09\2026_0112\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0113\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0116\keypoints.csv"
    "C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\IRt_10\2026_0112\keypoints.csv"


};

GAP_FRAMES = 8;

PROB_MIN = 0;

GRID_SIZE = 256;

% On the heatmap grid, splat Gaussian exp(-r^2 / (2*sigma^2)), only bins whose centers lie
% within truncR Euclidean distance (bin units ~= heatmap pixels).
GAUSS_TRUNC_BINS = 5;
GAUSS_SIGMA_BINS = GAUSS_TRUNC_BINS / 3;

USE_LOG_DISPLAY = true;

% Fixed pixel frame for binning and axes (always 256×256 coordinate range, independent of data).
AXIS_MIN = 0;
AXIS_MAX = 256;

% Save vector figures next to keypoints.csv (same folder as side_view.svg / bottom_view.svg).
SAVE_SVG = true;

%% =======================================================================

for k = 1:numel(csvPaths)
    csvFile = csvPaths{k};
    if ~isfile(csvFile)
        warning('Skipping (file not found): %s', csvFile);
        continue
    end

    tbl = readKeypointsCsv(csvFile);

    frm = tbl.Frame;
    xv = tbl.X;
    yv = tbl.Y;
    if isempty(PROB_MIN) || (~isscalar(PROB_MIN)) || PROB_MIN <= 0
        keep = true(size(frm));
    else
        keep = tbl.Probability >= PROB_MIN;
    end
    frm = frm(keep); xv = xv(keep); yv = yv(keep);

    if numel(xv) < 1
        warning('No keypoints kept (missing data or PROB_MIN too high?): %s', csvFile);
        continue
    end

    xMin = AXIS_MIN;
    xMax = AXIS_MAX;
    yMin = AXIS_MIN;
    yMax = AXIS_MAX;

    H_all = gaussianSplat2d(xv, yv, xMin, xMax, yMin, yMax, GRID_SIZE, ...
        GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS);

    [extX, extY] = maxExtensionPointsPerBout(frm, xv, yv, GAP_FRAMES);
    H_ext = gaussianSplat2d(extX, extY, xMin, xMax, yMin, yMax, GRID_SIZE, ...
        GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS);

    outDir = fileparts(csvFile);
    outStem = outputStemFromCsv(csvFile);

    f1 = figure('Name', sprintf('All keypoints — %s', outStem), ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [80 420 560 520]);
    plotHeatmap(H_all, xMin, xMax, yMin, yMax, USE_LOG_DISPLAY);
    title(sprintf('All positions — %s', outStem), 'Interpreter', 'none', 'Color', 'k');
    if SAVE_SVG
        exportHeatmapSvg(f1, fullfile(outDir, sprintf('%s_heatmap_all.svg', outStem)));
    end

    f2 = figure('Name', sprintf('Max extension only — %s', outStem), ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [660 420 560 520]);
    plotHeatmap(H_ext, xMin, xMax, yMin, yMax, USE_LOG_DISPLAY);
    title(sprintf('Max-extension points only (%d bouts) — %s', ...
        numel(extX), outStem), 'Interpreter', 'none', 'Color', 'k');
    if SAVE_SVG
        exportHeatmapSvg(f2, fullfile(outDir, sprintf('%s_heatmap_max_extension_only.svg', outStem)));
    end
end
end


function outStem = outputStemFromCsv(csvFile)
    parentDir = fileparts(csvFile);
    parts = strsplit(parentDir, filesep);
    parts = parts(~cellfun(@isempty, parts));
    if numel(parts) >= 3
        outStem = strjoin(parts(end-2:end), '_');
    else
        [~, outStem, ~] = fileparts(csvFile);
    end
    outStem = regexprep(outStem, '[^A-Za-z0-9_-]', '_');
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


function H = gaussianSplat2d(xv, yv, xMin, xMax, yMin, yMax, n, sigma, truncR)
    H = zeros(n, n);
    if isempty(xv)
        return
    end

    xv = double(xv(:));
    yv = double(yv(:));

    dx = max(double(xMax) - double(xMin), eps);
    dy = max(double(yMax) - double(yMin), eps);

    truncR = max(1, round(double(truncR)));
    sigma = max(double(sigma), eps);
    truncRsq = truncR^2;

    % Continuous coords in bin space spanning [0, n]; bin (row i, col j) center at (j-0.5, i-0.5).
    xCol = (xv - double(xMin)) ./ dx .* double(n);
    yRow = (yv - double(yMin)) ./ dy .* double(n);

    for kk = 1:numel(xv)
        cx = xCol(kk);
        cy = yRow(kk);
        jLo = max(1, ceil(cx - truncR - 1));
        jHi = min(n, floor(cx + truncR + 1));
        iLo = max(1, ceil(cy - truncR - 1));
        iHi = min(n, floor(cy + truncR + 1));
        jj = jLo:jHi;
        ii = iLo:iHi;
        [JG, IG] = meshgrid(jj, ii);
        dxc = JG - 0.5 - cx;
        dyr = IG - 0.5 - cy;
        dc2 = dxc.^2 + dyr.^2;
        mask = dc2 <= truncRsq;
        kern = zeros(size(dc2));
        kern(mask) = exp(-dc2(mask) ./ (2 * sigma^2));
        H(ii, jj) = H(ii, jj) + kern;
    end
end


function [extX, extY] = maxExtensionPointsPerBout(frame, xv, yv, gapFrames)
    [fs, ord] = sort(frame(:));
    xo = xv(ord);
    yo = yv(ord);
    nf = numel(fs);
    if nf == 0
        extX = [];
        extY = [];
        return
    end
    breaks = find(diff(fs) > gapFrames);
    starts = [1; breaks + 1];
    ends = [breaks; nf];
    nB = numel(starts);
    extX = zeros(nB, 1);
    extY = zeros(nB, 1);
    for b = 1:nB
        s = starts(b);
        e = ends(b);
        xs = xo(s:e);
        ys = yo(s:e);
        pFirst = [xs(1); ys(1)];
        pLast = [xs(end); ys(end)];
        dFirst = hypot(xs - pFirst(1), ys - pFirst(2));
        dLast = hypot(xs - pLast(1), ys - pLast(2));
        [mFirst, iF] = max(dFirst);
        [mLast, iL] = max(dLast);
        if mFirst >= mLast
            pick = iF;
        else
            pick = iL;
        end
        extX(b) = xs(pick);
        extY(b) = ys(pick);
    end
end


function exportHeatmapSvg(fig, filepath)
    try
        exportgraphics(fig, filepath, 'ContentType', 'vector');
    catch %#ok<*CTCH>
        print(fig, filepath, '-dsvg', '-painters');
    end
end


function plotHeatmap(H, xMin, xMax, yMin, yMax, useLog)
    ax = axes('Parent', gcf);
    set(ax, 'XColor', 'k', 'YColor', 'k');
    if useLog
        Z = log1p(double(H));
        cbLbl = 'log(1 + count)';
    else
        Z = double(H);
        cbLbl = 'count';
    end
    nt = size(Z, 1);
    imagesc(ax, Z);
    axis(ax, 'image');
    % Image / video coords: y increases downward (row 1 = small Y at top).
    set(ax, 'YDir', 'reverse');
    xticks(ax, [0.5, nt / 2 + 0.5, nt + 0.5]);
    xticklabels(ax, {sprintf('%.2f', xMin), sprintf('%.2f', (xMin + xMax) / 2), ...
            sprintf('%.2f', xMax)});
    yticks(ax, [0.5, nt / 2 + 0.5, nt + 0.5]);
    yticklabels(ax, {sprintf('%.2f', yMin), sprintf('%.2f', (yMin + yMax) / 2), ...
            sprintf('%.2f', yMax)});
    xlabel(ax, 'X (pixels, 0-256)', 'Interpreter', 'none', 'Color', 'k');
    ylabel(ax, 'Y (pixels, 0-256)', 'Interpreter', 'none', 'Color', 'k');
    colormap(ax, parula(256));
    cb = colorbar(ax);
    cb.Color = 'k';
    cb.Label.String = cbLbl;
    cb.Label.Interpreter = 'none';
    cb.Label.Color = 'k';
end
