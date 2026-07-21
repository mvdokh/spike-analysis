function jaw_centered_pooled_heatmap
% jaw_centered_pooled_heatmap
% For each session keypoints.csv: subtract mean jaw position from *_bottom_view_jaw.csv, then pool
% all relative tongue-tip points across sessions and build ONE density heatmap per group (PCRt vs
% IRt). Axes are jaw-centered pixels: origin (0,0) is the jaw reference.
%
% Optional: POOL_MODE = 'byAnimal' builds one pooled heatmap per animal folder (PCRt_02, IRt_01, ...).
%
% Uses the same experiment lists as combined_keypoint_heatmaps_by_type.m (edit below).

close all

%% =======================================================================
%% CONFIG - match combined_keypoint_heatmaps_by_type.m lists
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

USE_LOG_DISPLAY = true;

% Jaw-centered axis: symmetric extent in pixels (tongue X/Y minus jaw mean).
REL_EXTENT_HALF = 128;

% 'group' = one pooled heatmap for all PCRt paths and one for all IRt paths.
% 'byAnimal' = separate pooled heatmap per animal folder (e.g. PCRt_02, IRt_01), still split by PCRt vs IRt file lists.
POOL_MODE = 'group';

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

switch lower(POOL_MODE)
    case 'group'
        runPooledGroup(csvPaths_PCRt, 'PCRt_BiPoles', 'PCRt', xMin, xMax, yMin, yMax, ...
            GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, PROB_MIN, USE_LOG_DISPLAY, OUTPUT_DIR, SAVE_SVG);
        runPooledGroup(csvPaths_IRt, 'IRt_BiPoles', 'IRt', xMin, xMax, yMin, yMax, ...
            GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, PROB_MIN, USE_LOG_DISPLAY, OUTPUT_DIR, SAVE_SVG);

    case 'byanimal'
        runPooledByAnimal(csvPaths_PCRt, 'PCRt', xMin, xMax, yMin, yMax, ...
            GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, PROB_MIN, USE_LOG_DISPLAY, OUTPUT_DIR, SAVE_SVG);
        runPooledByAnimal(csvPaths_IRt, 'IRt', xMin, xMax, yMin, yMax, ...
            GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, PROB_MIN, USE_LOG_DISPLAY, OUTPUT_DIR, SAVE_SVG);

    otherwise
        error('POOL_MODE must be ''group'' or ''byAnimal''.');
end
end


function runPooledGroup(csvList, groupTag, shortTag, xMin, xMax, yMin, yMax, ...
    gridN, sigma, truncR, probMin, useLog, outDir, saveSvg)

    [allX, allY, nSessions] = collectJawCenteredPoints(csvList, probMin);
    if isempty(allX)
        warning('No pooled points for group %s.', groupTag);
        return
    end

    H = gaussianSplat2d(allX, allY, xMin, xMax, yMin, yMax, gridN, sigma, truncR);

    ttl = sprintf('Pooled tongue tip (jaw-centered), %d sessions - %s', nSessions, groupTag);
    fname = fullfile(outDir, sprintf('pooled_%s_jaw_centered_all_sessions', shortTag));

    plotSinglePooledHeatmap(H, xMin, xMax, yMin, yMax, ttl, useLog, fname, saveSvg);
end


function runPooledByAnimal(csvList, modalityTag, xMin, xMax, yMin, yMax, ...
    gridN, sigma, truncR, probMin, useLog, outDir, saveSvg)

    ids = uniqueAnimalIds(csvList);
    if isempty(ids)
        warning('No paths for byAnimal pooling (%s).', modalityTag);
        return
    end

    for a = 1:numel(ids)
        aid = ids{a};
        mask = strcmp(animalFolderFromCsv(csvList), aid);
        subList = csvList(mask);

        [allX, allY, nSessions] = collectJawCenteredPoints(subList, probMin);
        if isempty(allX)
            warning('No pooled points for animal %s (%s).', aid, modalityTag);
            continue
        end

        H = gaussianSplat2d(allX, allY, xMin, xMax, yMin, yMax, gridN, sigma, truncR);

        ttl = sprintf('Pooled tongue tip (jaw-centered), %d sessions - %s (%s)', ...
            nSessions, aid, modalityTag);
        safeAid = regexprep(aid, '[^A-Za-z0-9_-]', '_');
        fname = fullfile(outDir, sprintf('pooled_%s_%s_jaw_centered', modalityTag, safeAid));

        plotSinglePooledHeatmap(H, xMin, xMax, yMin, yMax, ttl, useLog, fname, saveSvg);
    end
end


function plotSinglePooledHeatmap(H, xMin, xMax, yMin, yMax, titleTxt, useLog, outfileBase, saveSvg)

    fig = figure('Name', titleTxt, 'NumberTitle', 'off', 'Color', 'w', ...
        'Position', [80 80 620 560]);
    ax = axes(fig);
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
    set(ax, 'YDir', 'reverse');
    colormap(ax, parula(256));

    xticks(ax, [0.5, nt / 2 + 0.5, nt + 0.5]);
    xticklabels(ax, {sprintf('%.0f', xMin), '0', sprintf('%.0f', xMax)});
    yticks(ax, [0.5, nt / 2 + 0.5, nt + 0.5]);
    yticklabels(ax, {sprintf('%.0f', yMin), '0', sprintf('%.0f', yMax)});

    xlabel(ax, 'X relative to jaw (pixels)', 'Interpreter', 'none', 'Color', 'k');
    ylabel(ax, 'Y relative to jaw (pixels)', 'Interpreter', 'none', 'Color', 'k');
    title(ax, titleTxt, 'Interpreter', 'none', 'Color', 'k', 'FontSize', 11);

    hold(ax, 'on');
    cx0 = (0 - xMin) / (xMax - xMin) * nt + 0.5;
    cy0 = (0 - yMin) / (yMax - yMin) * nt + 0.5;
    plot(ax, cx0, cy0, 'ws', 'MarkerSize', 14, 'LineWidth', 2.6, 'MarkerFaceColor', 'none');
    plot(ax, cx0, cy0, 'ks', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'none');
    plot(ax, cx0, cy0, 'w+', 'MarkerSize', 18, 'LineWidth', 2.8);
    plot(ax, cx0, cy0, 'k+', 'MarkerSize', 16, 'LineWidth', 2.2);
    hold(ax, 'off');

    cb = colorbar(ax);
    cb.Color = 'k';
    cb.Label.String = cbLbl;
    cb.Label.Interpreter = 'none';
    cb.Label.Color = 'k';

    if saveSvg
        outPath = [outfileBase '.svg'];
        try
            exportgraphics(fig, outPath, 'ContentType', 'vector');
        catch %#ok<*CTCH>
            print(fig, outPath, '-dsvg', '-painters');
        end
    end
end


function [allX, allY, nSessions] = collectJawCenteredPoints(csvList, probMin)
    allX = [];
    allY = [];
    nSessions = 0;

    for k = 1:numel(csvList)
        csvFile = csvList{k};
        if ~isfile(csvFile)
            warning('Skipping (file not found): %s', csvFile);
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
        if numel(xv) < 1
            continue
        end

        [jx, jy] = jawMeanFromBottomView(csvFile, probMin);
        if isnan(jx) || isnan(jy)
            warning('Skipping session (no jaw reference): %s', csvFile);
            continue
        end

        allX = [allX; xv(:) - jx];
        allY = [allY; yv(:) - jy];
        nSessions = nSessions + 1;
    end
end


function id = animalFolderFromCsv(csvFile)
    sessionDir = fileparts(csvFile);
    animalDir = fileparts(sessionDir);
    [~, id, ~] = fileparts(animalDir);
end


function ids = uniqueAnimalIds(csvList)
    raw = cellfun(@animalFolderFromCsv, csvList, 'UniformOutput', false);
    ids = unique(raw);
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
