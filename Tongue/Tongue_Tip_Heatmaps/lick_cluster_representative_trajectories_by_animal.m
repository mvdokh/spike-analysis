function lick_cluster_representative_trajectories_by_animal
% lick_cluster_representative_trajectories_by_animal
%
% For each animal (PCRt and IRt cohorts), pools all licks from listed sessions, builds
% a per-lick feature vector (extent, duration, timing of peak protrusion, protrusion /
% retraction path speeds), z-scores features within animal, then clusters licks with
% k-means. Number of clusters k is chosen by maximizing the Calinski–Harabasz index
% over a configurable k range (pure MATLAB; no Statistics Toolbox required).
%
% Representative = mean of phase-resampled paths, then radial scaling so the curve’s
% max distance from the jaw matches an aggregate of per-lick max protrusion in that
% cluster (default: max lick in cluster × small gain). Use RESCALE_TARGET_R_STAT /
% REPRESENTATIVE_EXTENT_GAIN to tune draw length. k uses Calinski–Harabasz + CH_ACCEPT_RELATIVE.
%
% Edit csv lists below. Run: lick_cluster_representative_trajectories_by_animal

close all

%% =======================================================================
%% CONFIG (same CSV lists as other Tongue_Tip_Heatmaps scripts)
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
REP_LINE_WIDTH = 2.4;
% More points along each representative curve (smoother, clearer extent)
RESAMPLE_N = 72;

% 'phase' = uniform in lick index (same as intra-lick phase 0–1); preserves reach vs
% 'arclength' which averages heterogeneous path lengths and often looks short.
REPRESENTATIVE_RESAMPLE_MODE = 'phase';

% After averaging, scale the polyline from the jaw so max(||xy||) equals a statistic of
% per-lick max protrusion in the group (see RESCALE_TARGET_R_STAT), then multiply by
% REPRESENTATIVE_EXTENT_GAIN (visual “full reach”, compensates for averaging shrinkage).
RESCALE_REPRESENTATIVE_RADIAL = true;
% 'max' = longest lick in cluster; 'p97'/'p95'/'p90' robust highs; 'mean' conservative
RESCALE_TARGET_R_STAT = 'max';
REPRESENTATIVE_EXTENT_GAIN = 1.18;

% Pin polyline ends to mean first/last keypoint across licks (fixes “cut off” start/end).
PIN_REPRESENTATIVE_ENDPOINTS = true;

% Clustering
K_MIN = 2;
K_MAX = 6;
MIN_LICKS_TOTAL = 30;
MIN_LICKS_PER_CLUSTER = 8;
KMEANS_MAX_ITER = 80;
KMEANS_N_STARTS = 16;
RNG_SEED = 42;

% 'max_ch' = k with best Calinski–Harabasz (usually fewer clusters).
% 'largest_in_ch_band' = among k with CH >= CH_ACCEPT_RELATIVE*max(CH), pick largest k.
K_PICK_MODE = 'max_ch';
CH_ACCEPT_RELATIVE = 0.92;

% If true, choose k using CH + rule above; else use FIXED_K
USE_AUTO_K = true;
FIXED_K = 3;

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

runOneCohort(csvPaths_PCRt, 'PCRt_BiPoles', 'PCRt', PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, ...
    xMin, xMax, yMin, yMax, RESAMPLE_N, REP_LINE_WIDTH, REPRESENTATIVE_RESAMPLE_MODE, ...
    RESCALE_REPRESENTATIVE_RADIAL, RESCALE_TARGET_R_STAT, REPRESENTATIVE_EXTENT_GAIN, ...
    PIN_REPRESENTATIVE_ENDPOINTS, K_MIN, K_MAX, MIN_LICKS_TOTAL, MIN_LICKS_PER_CLUSTER, ...
    K_PICK_MODE, CH_ACCEPT_RELATIVE, KMEANS_MAX_ITER, KMEANS_N_STARTS, RNG_SEED, ...
    USE_AUTO_K, FIXED_K, OUTPUT_DIR, SAVE_SVG);

runOneCohort(csvPaths_IRt, 'IRt_BiPoles', 'IRt', PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, ...
    xMin, xMax, yMin, yMax, RESAMPLE_N, REP_LINE_WIDTH, REPRESENTATIVE_RESAMPLE_MODE, ...
    RESCALE_REPRESENTATIVE_RADIAL, RESCALE_TARGET_R_STAT, REPRESENTATIVE_EXTENT_GAIN, ...
    PIN_REPRESENTATIVE_ENDPOINTS, K_MIN, K_MAX, MIN_LICKS_TOTAL, MIN_LICKS_PER_CLUSTER, ...
    K_PICK_MODE, CH_ACCEPT_RELATIVE, KMEANS_MAX_ITER, KMEANS_N_STARTS, RNG_SEED, ...
    USE_AUTO_K, FIXED_K, OUTPUT_DIR, SAVE_SVG);

end


function runOneCohort(csvList, groupTag, shortTag, probMin, gapFrames, minLickFrames, ...
    xMin, xMax, yMin, yMax, resampleN, lineW, resMode, rescaleRadial, rescaleRStat, extentGain, ...
    pinEndpoints, kMin, kMax, minLicksTotal, minPerClust, kPickMode, chAcceptRel, ...
    kmMaxIter, kmNStarts, rngSeed, useAutoK, fixedK, outDir, saveSvg)

    animalGroups = groupCsvFilesByAnimal(csvList);
    if isempty(animalGroups)
        warning('No valid CSV files found for %s.', groupTag);
        return
    end

    rng(rngSeed);

    nAnimals = numel(animalGroups);
    nCol = 2;
    nRow = ceil(nAnimals / nCol);

    figW = min(280 + 380 * nCol, 2280);
    figH = min(260 + 300 * nRow, 1920);
    fig = figure('Name', sprintf('Cluster lick representatives (%s)', groupTag), ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [60 40 figW figH]);

    tl = tiledlayout(fig, nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, {
        sprintf('Representative trajectories by within-animal lick clusters (%s)', groupTag)
        'k = best CH (see K_PICK_MODE); curves phase-mean, radial scale, endpoints pinned to mean first/last keypoint'
        }, ...
        'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'k');

    for ai = 1:nAnimals
        ax = nexttile(tl);
        hold(ax, 'on');
        set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
        set(ax, 'YDir', 'reverse');
        axis(ax, 'equal');
        axis(ax, 'square');

        label = animalGroups(ai).label;
        files = animalGroups(ai).files;

        allX = {};
        allY = {};
        Feat = [];

        for kf = 1:numel(files)
            fp = files{kf};
            if ~isfile(fp)
                continue
            end
            [lickX, lickY] = sessionSortedTrajectoryPoints(fp, probMin, gapFrames, minLickFrames);
            if isempty(lickX)
                continue
            end
            for jj = 1:numel(lickX)
                xs = lickX{jj};
                ys = lickY{jj};
                if isempty(xs)
                    continue
                end
                fv = featureVectorFromLick(xs, ys);
                if isempty(fv)
                    continue
                end
                allX{end + 1} = xs(:); %#ok<AGROW>
                allY{end + 1} = ys(:);
                Feat(end + 1, :) = fv; %#ok<AGROW>
            end
        end

        nL = numel(allX);

        if nL < 2
            title(ax, sprintf('%s (no valid licks)', label), 'Interpreter', 'none', 'FontSize', 10);
            plotJawReference(ax);
        elseif nL < minLicksTotal || size(Feat, 1) < minLicksTotal
            [xm0, ym0] = meanResampledTrajectory(allX, allY, resampleN, resMode, rescaleRadial, rescaleRStat, extentGain, pinEndpoints);
            h0 = plot(ax, xm0, ym0, '-', 'Color', [0.15 0.15 0.18], 'LineWidth', lineW, ...
                'DisplayName', sprintf('pooled mean (n=%d, need n≥%d to cluster)', nL, minLicksTotal));
            plotJawReference(ax);
            legend(ax, h0, {sprintf('pooled mean (n=%d, need n≥%d to cluster)', nL, minLicksTotal)}, ...
                'Location', 'best', 'Interpreter', 'none', 'FontSize', 8, 'AutoUpdate', 'off');
            title(ax, sprintf('%s | pooled mean only', label), 'Interpreter', 'none', 'FontSize', 10);
        else
            Z = zscoreRows(Feat);
            Z(~isfinite(Z)) = 0;

            kMaxCand = min(kMax, floor(nL / max(1, minPerClust)));
            kMaxCand = max(kMaxCand, kMin);

            if useAutoK && kMaxCand >= kMin
                [kBest, idx, chVal] = pickKAndCluster(Z, kMin, kMaxCand, kmMaxIter, kmNStarts, chAcceptRel, kPickMode);
                titleStr = sprintf('%s | n=%d | k=%d (CH=%.1f)', label, nL, kBest, chVal);
            elseif ~useAutoK && fixedK >= 1 && nL >= fixedK * minPerClust
                kBest = min(fixedK, kMaxCand);
                idx = kmeansBestOfRestarts(Z, kBest, kmMaxIter, kmNStarts);
                titleStr = sprintf('%s | n=%d | k=%d (fixed)', label, nL, kBest);
            else
                kBest = 1;
                idx = ones(nL, 1);
                titleStr = sprintf('%s | n=%d | single mean (insufficient for k>=2)', label, nL);
            end

            clustersPresent = unique(idx(:)');
            nPlot = numel(clustersPresent);
            cmapClust = clusterColormap(max(nPlot, kBest));
            hCl = gobjects(0, 1);
            leg = {};

            for ii = 1:nPlot
                c = clustersPresent(ii);
                m = idx == c;
                if ~any(m)
                    continue
                end
                [xm, ym] = meanResampledTrajectory(allX(m), allY(m), resampleN, resMode, rescaleRadial, rescaleRStat, extentGain, pinEndpoints);
                xm = xm(:);
                ym = ym(:);
                if ~all(isfinite(xm)) || ~all(isfinite(ym))
                    continue
                end
                if (max(xm) - min(xm)) + (max(ym) - min(ym)) < 1e-9
                    continue
                end

                nColors = size(cmapClust, 1);
                col = cmapClust(mod(ii - 1, nColors) + 1, :);
                hCl(end + 1) = plot(ax, xm, ym, '-', 'Color', col, 'LineWidth', lineW); %#ok<AGROW>
                leg{end + 1} = sprintf('Cluster %d (n=%d)', c, sum(m)); %#ok<AGROW>
            end

            plotJawReference(ax);

            if ~isempty(hCl) && all(isgraphics(hCl(:)))
                legend(ax, hCl(:), leg, 'Location', 'best', 'Interpreter', 'none', 'FontSize', 8, ...
                    'AutoUpdate', 'off');
            end
            title(ax, titleStr, 'Interpreter', 'none', 'FontSize', 10);

            if isempty(hCl)
                text(ax, 0.05, 0.95, 'No finite cluster mean trajectories', 'Units', 'normalized', ...
                    'Interpreter', 'none', 'FontSize', 9, 'VerticalAlignment', 'top');
            end
        end

        xlim(ax, [xMin xMax]);
        ylim(ax, [yMin yMax]);
        xticks(ax, [xMin 0 xMax]);
        yticks(ax, [yMin 0 yMax]);
        xlabel(ax, 'X rel. jaw (px)', 'Interpreter', 'none');
        ylabel(ax, 'Y rel. jaw (px)', 'Interpreter', 'none');
        grid(ax, 'on');
        hold(ax, 'off');
    end

    if saveSvg
        outPath = fullfile(outDir, sprintf('lick_cluster_representatives_by_animal_%s', shortTag));
        svgPath = [outPath '.svg'];
        if isgraphics(fig)
            try
                exportgraphics(fig, svgPath, 'ContentType', 'vector');
            catch %#ok<*CTCH>
                try
                    print(fig, svgPath, '-dsvg', '-painters');
                catch
                    warning('SVG export failed for %s.', shortTag);
                end
            end
            if isfile(svgPath)
                fprintf('Wrote %s\n', svgPath);
            end
        end
    end
end


function fv = featureVectorFromLick(xs, ys)
% One row of features: max protrusion, log duration, normalized time to peak,
% mean path speed on protrusion / retraction segments (distance per frame).
    xs = xs(:);
    ys = ys(:);
    L = numel(xs);
    if L < 2
        fv = [];
        return
    end

    r = hypot(xs, ys);
    [maxR, kMax] = max(r);

    dx = diff(xs);
    dy = diff(ys);
    step = hypot(dx, dy);
    pathTot = sum(step);

    if kMax > 1
        pathProt = sum(step(1:kMax - 1));
        nStepProt = kMax - 1;
    else
        pathProt = 0;
        nStepProt = 0;
    end

    if kMax < L
        pathRetr = sum(step(kMax:end));
        nStepRetr = numel(step(kMax:end));
    else
        pathRetr = 0;
        nStepRetr = 0;
    end

    spdProt = pathProt / max(1, nStepProt);
    spdRetr = pathRetr / max(1, nStepRetr);

    tPeakFrac = (kMax - 1) / max(1, L - 1);

    fv = [
        log(1 + maxR)
        log(L)
        tPeakFrac
        log(1 + spdProt)
        log(1 + spdRetr)
        log(1 + pathTot)
        ];
end


function Z = zscoreRows(X)
    mu = mean(X, 1, 'omitnan');
    sg = std(X, 0, 1, 'omitnan');
    sg(sg < eps) = 1;
    Z = (X - mu) ./ sg;
end


function [kBest, idx, chScoreForPick] = pickKAndCluster(Z, kMin, kMaxCand, maxIter, nStarts, chAcceptRelative, kPickMode)

    if nargin < 6 || isempty(chAcceptRelative)
        chAcceptRelative = 0.92;
    end
    if nargin < 7 || isempty(kPickMode)
        kPickMode = 'max_ch';
    end
    modeStr = lower(strtrim(char(kPickMode)));

    kList = kMin:kMaxCand;
    chList = nan(size(kList));
    idxCell = cell(numel(kList), 1);

    for ii = 1:numel(kList)
        k = kList(ii);
        idxK = kmeansBestOfRestarts(Z, k, maxIter, nStarts);
        chList(ii) = calinskiHarabasz(Z, idxK, k);
        idxCell{ii} = idxK;
    end

    valid = find(isfinite(chList) & ~isnan(chList));
    if isempty(valid)
        idx = ones(size(Z, 1), 1);
        kBest = 1;
        chScoreForPick = 0;
        return
    end

    if strcmp(modeStr, 'largest_in_ch_band')
        mx = max(chList(valid));
        if ~isfinite(mx)
            idx = ones(size(Z, 1), 1);
            kBest = 1;
            chScoreForPick = 0;
            return
        end
        thr = chAcceptRelative * mx;
        ok = chList >= thr & isfinite(chList);
        if any(ok)
            pickIi = find(ok);
            kBest = max(kList(pickIi));
            jj = find(kList == kBest, 1);
            idx = idxCell{jj};
            chScoreForPick = chList(jj);
        else
            [chScoreForPick, rel] = max(chList(valid));
            jj = valid(rel);
            kBest = kList(jj);
            idx = idxCell{jj};
        end
    else
        % Default: k that maximizes Calinski–Harabasz; ties -> smallest k (fewer clusters)
        mx = max(chList(valid));
        if ~isfinite(mx)
            idx = ones(size(Z, 1), 1);
            kBest = 1;
            chScoreForPick = 0;
            return
        end
        tol = 1e-9 * max(1, abs(mx));
        cand = find(abs(chList - mx) <= tol & isfinite(chList));
        if isempty(cand)
            cand = valid;
        end
        [kBest, rel] = min(kList(cand));
        jj = cand(rel);
        chScoreForPick = chList(jj);
        idx = idxCell{jj};
    end
end


function ch = calinskiHarabasz(X, idx, k)
    n = size(X, 1);
    if k < 2 || n <= k
        ch = 0;
        return
    end

    mu = mean(X, 1);
    B = 0;
    W = 0;

    for c = 1:k
        m = idx == c;
        if ~any(m)
            continue
        end
        Xc = X(m, :);
        nc = sum(m);
        muc = mean(Xc, 1);
        B = B + nc * sum((muc - mu).^2);
        W = W + sum(sum((Xc - muc).^2));
    end

    if W < eps
        ch = inf;
        return
    end

    ch = (B / (k - 1)) / (W / (n - k));
end


function idx = kmeansBestOfRestarts(X, k, maxIter, nStarts)
    n = size(X, 1);
    p = size(X, 2);
    bestInertia = inf;
    idx = ones(n, 1);

    for t = 1:nStarts
        seed = t * 7919 + randi(1e6);
        idxT = kmeansLloyd(X, k, maxIter, seed);
        centr = zeros(k, p);
        counts = zeros(k, 1);
        for c = 1:k
            m = idxT == c;
            if any(m)
                centr(c, :) = mean(X(m, :), 1);
                counts(c) = sum(m);
            else
                centr(c, :) = NaN;
            end
        end
        inertia = 0;
        for i = 1:n
            ci = idxT(i);
            if ~isnan(centr(ci, 1))
                inertia = inertia + sum((X(i, :) - centr(ci, :)).^2);
            end
        end
        if inertia < bestInertia
            bestInertia = inertia;
            idx = idxT;
        end
    end
end


function idx = kmeansLloyd(X, k, maxIter, seed)
    rng(seed);
    [n, p] = size(X);
    if k >= n
        idx = (1:n)';
        return
    end

    pick = randperm(n, k);
    C = X(pick, :);

    idx = ones(n, 1);
    for it = 1:maxIter
        for i = 1:n
            best = inf;
            bi = 1;
            for j = 1:k
                d = sum((X(i, :) - C(j, :)).^2);
                if d < best
                    best = d;
                    bi = j;
                end
            end
            idx(i) = bi;
        end

        moved = false;
        for j = 1:k
            m = idx == j;
            if any(m)
                newC = mean(X(m, :), 1);
                if any(abs(newC - C(j, :)) > 1e-10 * (1 + abs(C(j, :))))
                    moved = true;
                end
                C(j, :) = newC;
            end
        end

        if ~moved && it > 1
            break
        end
    end
end


function [xm, ym] = meanResampledTrajectory(cellX, cellY, n, resMode, rescaleRadial, rescaleRStat, extentGain, pinEndpoints)
    if nargin < 4 || isempty(resMode)
        resMode = 'phase';
    end
    if nargin < 5 || isempty(rescaleRadial)
        rescaleRadial = true;
    end
    if nargin < 6 || isempty(rescaleRStat)
        rescaleRStat = 'max';
    end
    if nargin < 7 || isempty(extentGain)
        extentGain = 1;
    end
    if nargin < 8 || isempty(pinEndpoints)
        pinEndpoints = true;
    end

    S = numel(cellX);
    accX = zeros(n, 1);
    accY = zeros(n, 1);

    for i = 1:S
        xs = cellX{i}(:);
        ys = cellY{i}(:);
        if strcmpi(resMode, 'arclength')
            [xu, yu] = resampleArcLength(xs, ys, n);
        else
            [xu, yu] = resamplePhaseUniform(xs, ys, n);
        end
        accX = accX + xu;
        accY = accY + yu;
    end

    xm = accX / S;
    ym = accY / S;

    if rescaleRadial && S >= 1
        tgtMaxR = targetRadiusFromLicks(cellX, cellY, rescaleRStat) * extentGain;
        rm = hypot(xm, ym);
        rmx = max(rm);
        if rmx > 1e-9 && tgtMaxR > 1e-9
            fac = tgtMaxR / rmx;
            xm = xm * fac;
            ym = ym * fac;
        end
    end

    if pinEndpoints && S >= 1 && n >= 2
        sx = mean(cellfun(@(x) x(1), cellX));
        sy = mean(cellfun(@(y) y(1), cellY));
        ex = mean(cellfun(@(x) x(end), cellX));
        ey = mean(cellfun(@(y) y(end), cellY));
        xm(1) = sx;
        ym(1) = sy;
        xm(end) = ex;
        ym(end) = ey;
    end
end


function tgt = targetRadiusFromLicks(cellX, cellY, statTag)
    S = numel(cellX);
    mr = zeros(S, 1);
    for i = 1:S
        mr(i) = max(hypot(cellX{i}(:), cellY{i}(:)));
    end

    st = lower(strtrim(char(statTag)));
    switch st
        case 'mean'
            tgt = mean(mr);
        case 'median'
            tgt = median(mr);
        case 'max'
            tgt = max(mr);
        case 'p90'
            tgt = prctileSafe(mr, 90);
        case 'p95'
            tgt = prctileSafe(mr, 95);
        case 'p97'
            tgt = prctileSafe(mr, 97);
        otherwise
            tgt = max(mr);
    end
end


function p = prctileSafe(v, q)
    v = v(isfinite(v));
    if isempty(v)
        p = 0;
        return
    end
    vs = sort(v(:));
    n = numel(vs);
    if n == 1
        p = vs(1);
        return
    end
    pos = (q / 100) * (n - 1) + 1;
    lo = max(1, floor(pos));
    hi = min(n, ceil(pos));
    if lo == hi
        p = vs(lo);
    else
        w = pos - lo;
        p = (1 - w) * vs(lo) + w * vs(hi);
    end
end


function [xu, yu] = resamplePhaseUniform(xs, ys, n)
% Resample uniformly in sample index (intra-lick phase 0 = first frame, 1 = last).
    xs = xs(:);
    ys = ys(:);
    L = numel(xs);
    if L < 2
        xu = repmat(xs(1), n, 1);
        yu = repmat(ys(1), n, 1);
        return
    end

    s = ((0:(L - 1))' / max(1, L - 1));
    tq = linspace(0, 1, n)';
    tq(1) = 0;
    tq(end) = 1;

    nk = numel(s);
    monoBump = ((0:nk - 1)' * 1e-12);
    sU = s + monoBump;

    xu = interp1(sU, xs, tq, 'linear', 'extrap');
    yu = interp1(sU, ys, tq, 'linear', 'extrap');
    if any(~isfinite(xu)) || any(~isfinite(yu))
        xu = linspace(xs(1), xs(end), n)';
        yu = linspace(ys(1), ys(end), n)';
    end
end


function [xu, yu] = resampleArcLength(xs, ys, n)
    xs = xs(:);
    ys = ys(:);
    if numel(xs) < 2
        xu = repmat(xs(1), n, 1);
        yu = repmat(ys(1), n, 1);
        return
    end

    dx = diff(xs);
    dy = diff(ys);
    seg = hypot(dx, dy);
    cuml = [0; cumsum(seg)];
    tot = cuml(end);
    if tot < 1e-9
        xu = repmat(xs(1), n, 1);
        yu = repmat(ys(1), n, 1);
        return
    end

    % interp1 requires strictly increasing sample abscissae; zero-length steps
    % duplicate cumulative arc length (e.g. repeated keypoints). Perturb slightly.
    nk = numel(cuml);
    monoBump = ((0:nk - 1) .* ((tot + 1) * 1e-9))';
    cumlUnique = cuml + monoBump;

    tq = linspace(cumlUnique(1), cumlUnique(end), n);
    xu = interp1(cumlUnique, xs, tq, 'linear', 'extrap');
    yu = interp1(cumlUnique, ys, tq, 'linear', 'extrap');
    if any(~isfinite(xu)) || any(~isfinite(yu))
        xu = linspace(xs(1), xs(end), n)';
        yu = linspace(ys(1), ys(end), n)';
    end
end


function cmap = clusterColormap(n)
    n = max(2, round(n));
    try
        cmap = parula(n);
    catch %#ok<*CTCH>
        try
            cmap = hsv(n);
        catch %#ok<*CTCH>
            cmap = jet(n);
        end
    end
end


function plotJawReference(ax)
    plot(ax, 0, 0, 'ws', 'MarkerSize', 14, 'LineWidth', 2.2, 'MarkerFaceColor', 'none', ...
        'HandleVisibility', 'off');
    plot(ax, 0, 0, 'ks', 'MarkerSize', 11, 'LineWidth', 1.8, 'MarkerFaceColor', 'none', ...
        'HandleVisibility', 'off');
    plot(ax, 0, 0, 'w+', 'MarkerSize', 16, 'LineWidth', 2.4, 'HandleVisibility', 'off');
    plot(ax, 0, 0, 'k+', 'MarkerSize', 14, 'LineWidth', 1.9, 'HandleVisibility', 'off');
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

        lickX{end + 1} = xs;
        lickY{end + 1} = ys;
        lickPhase{end + 1} = ph;
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
