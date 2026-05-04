function combined_keypoint_heatmaps_by_type
% combined_keypoint_heatmaps_by_type
% Builds 6 combined multi-panel figures (3 for PCRt, 3 for IRt):
%   1) All tongue-tip keypoints (Gaussian density, 0-256 px)
%   2) Max-extension point per lick (gap-separated segments; same rule as keypoint_position_heatmap)
%   3) First keypoint per lick (first XY in each gap-separated segment)
% Experiment names are shown on every panel (short label + full path stem).
%
% Edit csvPaths_PCRt and csvPaths_IRt below. Run: combined_keypoint_heatmaps_by_type

close all

%% =======================================================================
%% CONFIG
%% =======================================================================

% PCRt BiPoles — add one keypoints.csv path per session:
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

% IRt BiPoles — add one keypoints.csv path per session:
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

% New lick segment when sorted Frame gap exceeds this (same as keypoint_position_heatmap).
GAP_FRAMES = 8;

GRID_SIZE = 256;

GAUSS_TRUNC_BINS = 5;
GAUSS_SIGMA_BINS = GAUSS_TRUNC_BINS / 3;

USE_LOG_SCALE = true;

AXIS_MIN = 0;
AXIS_MAX = 256;

OVERLAY_JAW_MEAN = true;

% Leave empty to save next to this .m file; otherwise set a folder path.
OUTPUT_DIR = '';

SAVE_SVG = true;

%% =======================================================================

thisDir = fileparts(mfilename('fullpath'));
if isempty(OUTPUT_DIR)
    OUTPUT_DIR = thisDir;
end

xMin = AXIS_MIN;
xMax = AXIS_MAX;
yMin = AXIS_MIN;
yMax = AXIS_MAX;

buildOneCombinedFigure(csvPaths_PCRt, 'PCRt BiPoles - all tongue-tip keypoints', ...
    fullfile(OUTPUT_DIR, 'combined_PCRt_BiPoles_tongue_all'), ...
    'all', GAP_FRAMES, xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
    PROB_MIN, USE_LOG_SCALE, OVERLAY_JAW_MEAN, SAVE_SVG);

buildOneCombinedFigure(csvPaths_PCRt, 'PCRt BiPoles - max extension per lick', ...
    fullfile(OUTPUT_DIR, 'combined_PCRt_BiPoles_tongue_max_extension'), ...
    'maxExt', GAP_FRAMES, xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
    PROB_MIN, USE_LOG_SCALE, OVERLAY_JAW_MEAN, SAVE_SVG);

buildOneCombinedFigure(csvPaths_PCRt, 'PCRt BiPoles - first point per lick', ...
    fullfile(OUTPUT_DIR, 'combined_PCRt_BiPoles_tongue_first_lick'), ...
    'firstLick', GAP_FRAMES, xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
    PROB_MIN, USE_LOG_SCALE, OVERLAY_JAW_MEAN, SAVE_SVG);

buildOneCombinedFigure(csvPaths_IRt, 'IRt BiPoles - all tongue-tip keypoints', ...
    fullfile(OUTPUT_DIR, 'combined_IRt_BiPoles_tongue_all'), ...
    'all', GAP_FRAMES, xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
    PROB_MIN, USE_LOG_SCALE, OVERLAY_JAW_MEAN, SAVE_SVG);

buildOneCombinedFigure(csvPaths_IRt, 'IRt BiPoles - max extension per lick', ...
    fullfile(OUTPUT_DIR, 'combined_IRt_BiPoles_tongue_max_extension'), ...
    'maxExt', GAP_FRAMES, xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
    PROB_MIN, USE_LOG_SCALE, OVERLAY_JAW_MEAN, SAVE_SVG);

buildOneCombinedFigure(csvPaths_IRt, 'IRt BiPoles - first point per lick', ...
    fullfile(OUTPUT_DIR, 'combined_IRt_BiPoles_tongue_first_lick'), ...
    'firstLick', GAP_FRAMES, xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
    PROB_MIN, USE_LOG_SCALE, OVERLAY_JAW_MEAN, SAVE_SVG);
end


function buildOneCombinedFigure(csvList, sgTitleTxt, outfileBaseNoExt, heatmapMode, gapFrames, ...
    xMin, xMax, yMin, yMax, gridN, sigma, truncR, probMin, useLog, overlayJaw, saveSvg)

    % Collect valid sessions
    Hcell = cell(0, 1);
    csvUsed = cell(0, 1);
    for k = 1:numel(csvList)
        csvFile = csvList{k};
        if ~isfile(csvFile)
            warning('Skipping (file not found): %s', csvFile);
            continue
        end
        tbl = readKeypointsCsv(csvFile);
        frm = tbl.Frame;
        xv = tbl.X;
        yv = tbl.Y;
        if isempty(probMin) || (~isscalar(probMin)) || probMin <= 0
            keep = true(size(frm));
        else
            keep = tbl.Probability >= probMin;
        end
        frm = frm(keep);
        xv = xv(keep);
        yv = yv(keep);
        if numel(xv) < 1
            warning('No keypoints after filter: %s', csvFile);
            continue
        end

        switch lower(heatmapMode)
            case 'all'
                H = gaussianSplat2d(xv, yv, xMin, xMax, yMin, yMax, gridN, sigma, truncR);
            case 'maxext'
                [px, py] = maxExtensionPointsPerLickSegment(frm, xv, yv, gapFrames);
                H = gaussianSplat2d(px, py, xMin, xMax, yMin, yMax, gridN, sigma, truncR);
            case 'firstlick'
                [px, py] = firstPointPerLickSegment(frm, xv, yv, gapFrames);
                H = gaussianSplat2d(px, py, xMin, xMax, yMin, yMax, gridN, sigma, truncR);
            otherwise
                error('Unknown heatmapMode: %s', heatmapMode);
        end

        Hcell{end+1, 1} = H; %#ok<AGROW>
        csvUsed{end+1, 1} = csvFile; %#ok<AGROW>
    end

    n = numel(Hcell);
    if n < 1
        warning('No panels to plot for: %s', sgTitleTxt);
        return
    end

    if useLog
        Zcell = cellfun(@(H) log1p(double(H)), Hcell, 'UniformOutput', false);
        zMax = max(cellfun(@(Z) max(Z(:)), Zcell));
        cbLabel = 'log(1 + count)';
    else
        Zcell = cellfun(@double, Hcell, 'UniformOutput', false);
        zMax = max(cellfun(@(Z) max(Z(:)), Zcell));
        cbLabel = 'count';
    end
    zMax = max(zMax, eps);

    ncol = ceil(sqrt(n));
    nrow = ceil(n / ncol);

    figW = min(200 + 180 * ncol, 2000);
    figH = min(180 + 160 * nrow, 1200);
    fig = figure('Name', sgTitleTxt, 'NumberTitle', 'off', 'Color', 'w', ...
        'Position', [40 40 figW figH]);

    tl = tiledlayout(fig, nrow, ncol, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, sgTitleTxt, 'FontWeight', 'bold', 'FontSize', 14, 'Color', 'k');

    for i = 1:n
        nexttile(tl);
        ax = gca;
        imagesc(ax, Zcell{i});
        axis(ax, 'image');
        set(ax, 'YDir', 'reverse', 'XColor', 'k', 'YColor', 'k', 'FontSize', 8);
        colormap(ax, parula(256));
        caxis(ax, [0 zMax]);
        xticks(ax, []);
        yticks(ax, []);

        if overlayJaw
            [jx, jy] = jawReferenceFromSession(csvUsed{i}, probMin);
            overlayReferencePoint(ax, jx, jy, xMin, xMax, yMin, yMax, gridN);
        end

        shortLab = experimentShortLabelFromCsv(csvUsed{i});
        stemLab = outputStemFromCsv(csvUsed{i});
        title(ax, {shortLab, stemLab}, 'Interpreter', 'none', 'Color', 'k', 'FontSize', 8);
    end

    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Color = 'k';
    cb.Label.String = cbLabel;
    cb.Label.Interpreter = 'none';
    cb.Label.Color = 'k';

    if saveSvg
        outPath = [outfileBaseNoExt '.svg'];
        try
            exportgraphics(fig, outPath, 'ContentType', 'vector');
        catch %#ok<*CTCH>
            print(fig, outPath, '-dsvg', '-painters');
        end
    end
end


function shortLab = experimentShortLabelFromCsv(csvFile)
    parentDir = fileparts(csvFile);
    parts = strsplit(parentDir, filesep);
    parts = parts(~cellfun(@isempty, parts));
    if numel(parts) >= 2
        shortLab = sprintf('%s | %s', parts{end-1}, parts{end});
    else
        [~, shortLab, ~] = fileparts(parentDir);
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


function [jawX, jawY] = jawReferenceFromSession(csvFile, probMin)
    jawX = NaN;
    jawY = NaN;
    sessionDir = fileparts(csvFile);
    jawFiles = dir(fullfile(sessionDir, '*_bottom_view_jaw.csv'));
    if isempty(jawFiles)
        warning('No *_bottom_view_jaw.csv found in session folder: %s', sessionDir);
        return
    end
    names = sort({jawFiles.name});
    jawFile = fullfile(sessionDir, names{1});

    jawTbl = readKeypointsCsv(jawFile);
    if height(jawTbl) < 1
        warning('Jaw CSV has no data rows (empty): %s', jawFile);
        return
    end

    if isempty(probMin) || (~isscalar(probMin)) || probMin <= 0
        keep = true(height(jawTbl), 1);
    else
        keep = jawTbl.Probability >= probMin;
        if ~any(keep)
            warning(['No jaw rows left after Probability >= PROB_MIN ' ...
                '(PROB_MIN=%g): %s'], probMin, jawFile);
            return
        end
    end

    jawX = mean(jawTbl.X(keep), 'omitnan');
    jawY = mean(jawTbl.Y(keep), 'omitnan');
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


function [extX, extY] = maxExtensionPointsPerLickSegment(frame, xv, yv, gapFrames)
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
    nL = numel(starts);
    extX = zeros(nL, 1);
    extY = zeros(nL, 1);
    for b = 1:nL
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


function [fx, fy] = firstPointPerLickSegment(frame, xv, yv, gapFrames)
    [fs, ord] = sort(frame(:));
    xo = xv(ord);
    yo = yv(ord);
    nf = numel(fs);
    if nf == 0
        fx = [];
        fy = [];
        return
    end
    breaks = find(diff(fs) > gapFrames);
    starts = [1; breaks + 1];
    nL = numel(starts);
    fx = zeros(nL, 1);
    fy = zeros(nL, 1);
    for b = 1:nL
        s = starts(b);
        fx(b) = xo(s);
        fy(b) = yo(s);
    end
end


function overlayReferencePoint(ax, refX, refY, xMin, xMax, yMin, yMax, n)
    if isnan(refX) || isnan(refY)
        return
    end
    dx = max(xMax - xMin, eps);
    dy = max(yMax - yMin, eps);
    cx = (refX - xMin) ./ dx .* n + 0.5;
    cy = (refY - yMin) ./ dy .* n + 0.5;
    hold(ax, 'on');
    plot(ax, cx, cy, 'wo', 'MarkerSize', 6, 'LineWidth', 1.2);
    plot(ax, cx, cy, 'kx', 'MarkerSize', 7, 'LineWidth', 1);
    hold(ax, 'off');
end
