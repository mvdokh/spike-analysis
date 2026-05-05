function jaw_centered_pooled_heatmap_lick_normalized
% jaw_centered_pooled_heatmap_lick_normalized
% Jaw-centered pooled heatmaps like jaw_centered_pooled_heatmap.m, but each SESSION is normalized
% by its number of LICKS before pooling:
%
%   Licks = contiguous runs of tongue detections in Frame order, separated when the frame gap
%   exceeds GAP_FRAMES (same rule as other tongue scripts: gaps with no keypoint / tongue in mouth).
%
%   If a session has L licks, lick j has n_j keypoint rows, each row gets weight 100/(L * n_j).
%   So each lick contributes total mass 100/L and the whole session sums to 100 across all Gaussian
%   splats from that session (sessions still combine additively when pooled).
%
% Edit csv paths and POOL_MODE below. Run: jaw_centered_pooled_heatmap_lick_normalized

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

% New lick when sorted Frame gap exceeds this (no tongue keypoint for at least this many frames).
GAP_FRAMES = 8;

GRID_SIZE = 256;

GAUSS_TRUNC_BINS = 5;
GAUSS_SIGMA_BINS = GAUSS_TRUNC_BINS / 3;

USE_LOG_DISPLAY = true;

REL_EXTENT_HALF = 128;

POOL_MODE = 'group';

% Per-session total weight after lick normalization (divide by L*n_j, then scale by this).
LICK_WEIGHT_SCALE = 100;

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
        runPooledGroup(csvPaths_PCRt, 'PCRt_BiPoles', 'PCRt', GAP_FRAMES, ...
            xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
            PROB_MIN, USE_LOG_DISPLAY, OUTPUT_DIR, SAVE_SVG, LICK_WEIGHT_SCALE);
        runPooledGroup(csvPaths_IRt, 'IRt_BiPoles', 'IRt', GAP_FRAMES, ...
            xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
            PROB_MIN, USE_LOG_DISPLAY, OUTPUT_DIR, SAVE_SVG, LICK_WEIGHT_SCALE);

    case 'byanimal'
        runPooledByAnimal(csvPaths_PCRt, 'PCRt', GAP_FRAMES, ...
            xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
            PROB_MIN, USE_LOG_DISPLAY, OUTPUT_DIR, SAVE_SVG, LICK_WEIGHT_SCALE);
        runPooledByAnimal(csvPaths_IRt, 'IRt', GAP_FRAMES, ...
            xMin, xMax, yMin, yMax, GRID_SIZE, GAUSS_SIGMA_BINS, GAUSS_TRUNC_BINS, ...
            PROB_MIN, USE_LOG_DISPLAY, OUTPUT_DIR, SAVE_SVG, LICK_WEIGHT_SCALE);

    otherwise
        error('POOL_MODE must be ''group'' or ''byAnimal''.');
end
end


function runPooledGroup(csvList, groupTag, shortTag, gapFrames, ...
    xMin, xMax, yMin, yMax, gridN, sigma, truncR, probMin, useLog, outDir, saveSvg, lickWeightScale)

    [allX, allY, allW, nSessions] = collectJawCenteredWeighted(csvList, probMin, gapFrames, lickWeightScale);
    if isempty(allX)
        warning('No pooled points for group %s.', groupTag);
        return
    end

    H = gaussianSplat2dWeighted(allX, allY, allW, xMin, xMax, yMin, yMax, gridN, sigma, truncR);

    ttl = sprintf(['Pooled tongue (jaw-centered, lick-normalized per session), %d sessions - %s'], ...
        nSessions, groupTag);
    fname = fullfile(outDir, sprintf('pooled_%s_jaw_centered_lick_normalized', shortTag));

    plotSingleHeatmap(H, xMin, xMax, yMin, yMax, ttl, useLog, fname, saveSvg);
end


function runPooledByAnimal(csvList, modalityTag, gapFrames, ...
    xMin, xMax, yMin, yMax, gridN, sigma, truncR, probMin, useLog, outDir, saveSvg, lickWeightScale)

    ids = uniqueAnimalIds(csvList);
    if isempty(ids)
        warning('No paths for byAnimal pooling (%s).', modalityTag);
        return
    end

    for a = 1:numel(ids)
        aid = ids{a};
        mask = strcmp(animalFolderFromCsv(csvList), aid);
        subList = csvList(mask);

        [allX, allY, allW, nSessions] = collectJawCenteredWeighted(subList, probMin, gapFrames, lickWeightScale);
        if isempty(allX)
            warning('No pooled points for animal %s (%s).', aid, modalityTag);
            continue
        end

        H = gaussianSplat2dWeighted(allX, allY, allW, xMin, xMax, yMin, yMax, gridN, sigma, truncR);

        ttl = sprintf(['Pooled tongue (jaw-centered, lick-normalized), %d sessions - %s (%s)'], ...
            nSessions, aid, modalityTag);
        safeAid = regexprep(aid, '[^A-Za-z0-9_-]', '_');
        fname = fullfile(outDir, sprintf('pooled_%s_%s_jaw_centered_lick_normalized', modalityTag, safeAid));

        plotSingleHeatmap(H, xMin, xMax, yMin, yMax, ttl, useLog, fname, saveSvg);
    end
end


function plotSingleHeatmap(H, xMin, xMax, yMin, yMax, titleTxt, useLog, outfileBase, saveSvg)

    fig = figure('Name', titleTxt, 'NumberTitle', 'off', 'Color', 'w', ...
        'Position', [80 80 620 560]);
    ax = axes(fig);
    set(ax, 'XColor', 'k', 'YColor', 'k');

    if useLog
        Z = log1p(double(H));
        cbLbl = 'log(1 + weighted density)';
    else
        Z = double(H);
        cbLbl = 'weighted density';
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
    plot(ax, cx0, cy0, 'w+', 'MarkerSize', 12, 'LineWidth', 1.8);
    plot(ax, cx0, cy0, 'k+', 'MarkerSize', 10, 'LineWidth', 1);
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


function [allX, allY, allW, nSessions] = collectJawCenteredWeighted(csvList, probMin, gapFrames, lickWeightScale)
    allX = [];
    allY = [];
    allW = [];
    nSessions = 0;

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
            keep = true(size(xv));
        else
            keep = tbl.Probability >= probMin;
        end
        frm = frm(keep);
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

        xr = xv(:) - jx;
        yr = yv(:) - jy;

        [wx, wy, ww] = perSessionLickWeights(frm, xr, yr, gapFrames, lickWeightScale);
        if isempty(wx)
            continue
        end

        allX = [allX; wx];
        allY = [allY; wy];
        allW = [allW; ww];
        nSessions = nSessions + 1;
    end
end


function [xr, yr, w] = perSessionLickWeights(frm, xvRel, yvRel, gapFrames, lickWeightScale)
    [fs, ord] = sort(double(frm(:)));
    xo = xvRel(ord);
    yo = yvRel(ord);
    nf = numel(fs);
    if nf == 0
        xr = [];
        yr = [];
        w = [];
        return
    end

    breaks = find(diff(fs) > gapFrames);
    starts = [1; breaks + 1];
    ends = [breaks; nf];
    nLicks = numel(starts);

    xr = zeros(nf, 1);
    yr = zeros(nf, 1);
    w = zeros(nf, 1);

    for j = 1:nLicks
        s = starts(j);
        e = ends(j);
        nInLick = e - s + 1;
        wt = lickWeightScale / (nLicks * nInLick);
        idx = s:e;
        xr(idx) = xo(idx);
        yr(idx) = yo(idx);
        w(idx) = wt;
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


function H = gaussianSplat2dWeighted(xv, yv, wv, xMin, xMax, yMin, yMax, n, sigma, truncR)
    H = zeros(n, n);
    if isempty(xv)
        return
    end

    xv = double(xv(:));
    yv = double(yv(:));
    wv = double(wv(:));
    if numel(wv) ~= numel(xv)
        error('Weight vector length must match coordinates.');
    end

    dx = max(double(xMax) - double(xMin), eps);
    dy = max(double(yMax) - double(yMin), eps);

    truncR = max(1, round(double(truncR)));
    sigma = max(double(sigma), eps);
    truncRsq = truncR^2;

    xCol = (xv - double(xMin)) ./ dx .* double(n);
    yRow = (yv - double(yMin)) ./ dy .* double(n);

    for kk = 1:numel(xv)
        wt = wv(kk);
        if wt <= 0 || isnan(wt)
            continue
        end
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
        H(ii, jj) = H(ii, jj) + wt .* kern;
    end
end
