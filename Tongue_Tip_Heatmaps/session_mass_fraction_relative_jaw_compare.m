function session_mass_fraction_relative_jaw_compare
% session_mass_fraction_relative_jaw_compare
%
% Per session: build the same jaw-centered Gaussian density heatmap as jaw_centered_pooled_heatmap.m
% (single session only), then report what fraction of integrated heatmap mass falls on each side of
% the jaw reference at (0,0):
%   - Y: "above" jaw on the figure = smaller relative Y (upper half of imagesc) => y_rel < 0 by default
%   - X: "to one side" of the jaw vertical line => x_rel < 0 by default (left half)
%
% Edit csv lists below (match combined_keypoint_heatmaps_by_type.m / jaw_centered_pooled_heatmap.m).
% Outputs: figure comparing PCRt vs IRt, and a CSV table of per-session metrics.
%
% Run: session_mass_fraction_relative_jaw_compare

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

GRID_SIZE = 256;
GAUSS_TRUNC_BINS = 5;
GAUSS_SIGMA_BINS = GAUSS_TRUNC_BINS / 3;
REL_EXTENT_HALF = 128;

% Mass is summed over bins whose center falls in the region (same grid as heatmap).
% 'y_negative' = upper half of jaw-centered plot (y_rel < 0). Flip to 'y_positive' if you want y > 0.
Y_REGION = 'y_negative';
% 'x_negative' = left half (x_rel < 0). Use 'x_positive' for x > 0.
X_REGION = 'x_negative';

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

[tPCRt, sPCRt] = metricsForList(csvPaths_PCRt, 'PCRt', PROB_MIN, xMin, xMax, yMin, yMax, ...
    GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, Y_REGION, X_REGION);
[tIRt, sIRt] = metricsForList(csvPaths_IRt, 'IRt', PROB_MIN, xMin, xMax, yMin, yMax, ...
    GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, Y_REGION, X_REGION);

T = [tPCRt; tIRt];
outCsv = fullfile(OUTPUT_DIR, 'session_mass_fractions_relative_jaw.csv');
writetable(T, outCsv);
fprintf('Wrote %s\n', outCsv);

yLbl = labelForY(Y_REGION);
xLbl = labelForX(X_REGION);

fig = figure('Name', 'Session mass vs jaw (PCRt vs IRt)', 'NumberTitle', 'off', 'Color', 'w', ...
    'Position', [80 80 900 420]);
set(fig, 'InvertHardcopy', 'off');
sgtitle(fig, ...
    'Per-session heatmap mass percent (same splat as jaw_centered_pooled_heatmap; default y_rel<0, x_rel<0)', ...
    'FontSize', 10, 'Interpreter', 'none', 'Color', 'k');

tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
plotGroupViolin(ax1, sPCRt, sIRt, 'pctY', [yLbl, ' (%)'], ...
    {'PCRt', 'IRt'}, [0.2 0.45 0.7], [0.85 0.35 0.35]);

ax2 = nexttile;
plotGroupViolin(ax2, sPCRt, sIRt, 'pctX', [xLbl, ' (%)'], ...
    {'PCRt', 'IRt'}, [0.2 0.45 0.7], [0.85 0.35 0.35]);

if SAVE_SVG
    outFig = fullfile(OUTPUT_DIR, 'session_mass_fraction_compare_PCRt_vs_IRt.svg');
    try
        exportgraphics(fig, outFig, 'ContentType', 'vector');
    catch %#ok<*CTCH>
        print(fig, outFig, '-dsvg', '-painters');
    end
    fprintf('Wrote %s\n', outFig);
end
end


function [T, S] = metricsForList(csvList, groupName, probMin, xMin, xMax, yMin, yMax, ...
    gridN, sigma, truncR, yRegion, xRegion)

    n = numel(csvList);
    pctY = NaN(n, 1);
    pctX = NaN(n, 1);
    nPts = zeros(n, 1);
    sessionId = cell(n, 1);
    csvPath = cell(n, 1);

    valid = false(n, 1);
    for k = 1:n
        csvFile = csvList{k};
        csvPath{k} = csvFile;
        sessionId{k} = sessionLabelFromPath(csvFile);
        if ~isfile(csvFile)
            warning('Skipping (not found): %s', csvFile);
            continue
        end

        tbl = readKeypointsCsv(csvFile);
        xv = tbl.X;
        yv = tbl.Y;
        if isempty(probMin) || (~isscalar(probMin)) || probMin <= 0
            keep = true(size(xv));
        else
            keep = tbl.Probability >= probMin;
        end
        xv = xv(keep);
        yv = yv(keep);
        nPts(k) = numel(xv);
        if nPts(k) < 1
            continue
        end

        [jx, jy] = jawMeanFromBottomView(csvFile, probMin);
        if isnan(jx) || isnan(jy)
            warning('Skipping (no jaw): %s', csvFile);
            continue
        end

        xr = xv(:) - jx;
        yr = yv(:) - jy;

        H = gaussianSplat2d(xr, yr, xMin, xMax, yMin, yMax, gridN, sigma, truncR);
        tot = sum(H, 'all');
        if tot <= 0
            continue
        end

        [mY, mX] = massInRegions(H, xMin, xMax, yMin, yMax, gridN, yRegion, xRegion);
        pctY(k) = 100 * mY / tot;
        pctX(k) = 100 * mX / tot;
        valid(k) = true;
    end

    T = table(sessionId(valid), csvPath(valid), repmat({groupName}, sum(valid), 1), ...
        nPts(valid), pctY(valid), pctX(valid), ...
        'VariableNames', {'sessionId', 'csvPath', 'group', 'nKeypoints', 'pctHeatmapY', 'pctHeatmapX'});
    T.Properties.VariableDescriptions{'pctHeatmapY'} = sprintf('%% Gaussian mass: %s', labelForY(yRegion));
    T.Properties.VariableDescriptions{'pctHeatmapX'} = sprintf('%% Gaussian mass: %s', labelForX(xRegion));

    S.group = groupName;
    S.pctY = pctY(valid);
    S.pctX = pctX(valid);
    S.sessionId = sessionId(valid);
end


function lbl = sessionLabelFromPath(csvFile)
    sessionDir = fileparts(csvFile);
    [~, sess, ~] = fileparts(sessionDir);
    animalDir = fileparts(sessionDir);
    [~, animal, ~] = fileparts(animalDir);
    lbl = sprintf('%s / %s', animal, sess);
end


function [massYside, massXside] = massInRegions(H, xMin, xMax, yMin, yMax, n, yRegion, xRegion)
    dy = double(yMax) - double(yMin);
    dx = double(xMax) - double(xMin);
    yCenters = double(yMin) + ((double((1:n)') - 0.5) ./ double(n)) .* dy;
    xCenters = double(xMin) + ((double(1:n) - 0.5) ./ double(n)) .* dx;

    switch yRegion
        case 'y_negative'
            rowMask = yCenters < 0;
        case 'y_positive'
            rowMask = yCenters > 0;
        otherwise
            error('Unknown Y_REGION: %s', yRegion);
    end

    switch xRegion
        case 'x_negative'
            colMask = xCenters < 0;
        case 'x_positive'
            colMask = xCenters > 0;
        otherwise
            error('Unknown X_REGION: %s', xRegion);
    end

    rows = rowMask;
    cols = colMask;
    massYside = sum(H(rows, :), 'all');
    massXside = sum(H(:, cols), 'all');
end


function s = labelForY(yRegion)
    switch yRegion
        case 'y_negative'
            s = 'Heatmap mass with y_{rel} < 0 (above jaw on plot)';
        case 'y_positive'
            s = 'Heatmap mass with y_{rel} > 0 (below jaw on plot)';
    end
end


function s = labelForX(xRegion)
    switch xRegion
        case 'x_negative'
            s = 'Heatmap mass with x_{rel} < 0 (left of jaw)';
        case 'x_positive'
            s = 'Heatmap mass with x_{rel} > 0 (right of jaw)';
    end
end


function plotGroupViolin(ax, sA, sB, fieldY, yAxisLbl, groupNames, colA, colB)

    yA = sA.(fieldY)(:);
    yB = sB.(fieldY)(:);
    yA = yA(~isnan(yA));
    yB = yB(~isnan(yB));
    nA = numel(yA);
    nB = numel(yB);

    xPos = [1, 2];
    violinW = 0.42;

    set(ax, 'Color', 'w', 'Box', 'off', 'LineWidth', 1);
    set(ax, 'GridColor', [0.82 0.82 0.82], 'MinorGridColor', [0.9 0.9 0.9], ...
        'GridAlpha', 0.9, 'MinorGridAlpha', 0.5);
    ax.XAxis.TickLabelInterpreter = 'none';
    ax.YAxis.TickLabelInterpreter = 'tex';

    hold(ax, 'on');
    if nA > 0
        drawViolin(ax, xPos(1), yA, violinW, colA, colA * 0.55);
    end
    if nB > 0
        drawViolin(ax, xPos(2), yB, violinW, colB, colB * 0.55);
    end

    % Mean ± sample SD (vertical error bar); mean marked by horizontal crossbar.
    if nA > 0
        plotMeanSd(ax, xPos(1), yA);
    end
    if nB > 0
        plotMeanSd(ax, xPos(2), yB);
    end
    hold(ax, 'off');

    xlim(ax, [0.35 2.65]);
    set(ax, 'XTick', xPos, 'XTickLabel', groupNames);
    ylabel(ax, yAxisLbl, 'Interpreter', 'tex');
    grid(ax, 'on');
    set(ax, 'XColor', 'k', 'YColor', 'k');
    yline(ax, 50, ':', 'Color', [0.72 0.72 0.72]);
end


function drawViolin(ax, xc, yv, width, faceCol, edgeCol)

    if isempty(yv)
        return
    end

    if numel(yv) == 1
        ys = yv(1);
        dy = max(1, abs(ys) * 0.02 + 0.5);
        patch(ax, xc + [-1 -1 1 1] * width * 0.25, ys + [-dy dy dy -dy], faceCol, ...
            'FaceAlpha', 0.45, 'EdgeColor', edgeCol, 'LineWidth', 0.9);
        return
    end

    try
        [f, xi] = ksdensity(yv(:), 'NumPoints', min(512, max(64, numel(yv) * 16)));
    catch %#ok<*CTCH>
        % Fallback without Statistics Toolbox: histogram envelope
        nb = max(16, min(64, numel(yv)));
        ed = linspace(min(yv), max(yv), nb + 1);
        counts = histcounts(yv, ed);
        xi = (ed(1:end-1) + ed(2:end)) / 2;
        f = double(counts(:)) / max(double(counts(:)) + eps);
    end

    f = f(:) / max(f(:) + eps);
    xi = xi(:);
    xl = xc - width * f;
    xr = xc + width * f;
    xpoly = [xl.', fliplr(xr.')];
    ypoly = [xi.', fliplr(xi.')];
    patch(ax, xpoly, ypoly, faceCol, 'FaceAlpha', 0.42, 'EdgeColor', edgeCol, ...
        'LineWidth', 0.9, 'LineJoin', 'round');
end


function plotMeanSd(ax, xc, yv)

    mu = mean(yv, 'omitnan');
    sd = std(yv, 'omitnan');

    cap = min(0.28, 0.35 / 2);
    if numel(yv) > 1 && isfinite(sd) && sd > 0
        errorbar(ax, xc, mu, sd, 'k', 'LineStyle', 'none', 'LineWidth', 1.65, ...
            'CapSize', 10, 'Marker', 'none');
        plot(ax, [xc - cap, xc + cap], [mu, mu], 'k-', 'LineWidth', 2.3);
    else
        plot(ax, [xc - cap, xc + cap], [mu, mu], 'k-', 'LineWidth', 2.3);
        plot(ax, xc, mu, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
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
