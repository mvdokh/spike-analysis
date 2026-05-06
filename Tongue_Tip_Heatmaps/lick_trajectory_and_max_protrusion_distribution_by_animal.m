function lick_trajectory_and_max_protrusion_distribution_by_animal
% lick_trajectory_and_max_protrusion_distribution_by_animal
%
% One 2-D figure per experiment; one subplot per animal. Full intra-lick
% trajectories are drawn in jaw-centered X/Y (same as lick_trajectory_phase_density_
% overlay_by_animal): segment colors and scatter use intra-lick phase.
%
% Peak-tip distribution is shown by (1) optional warm RGB underlay and (2) bold
% contour lines on top of trajectories from the same smoothed 2-D histogram (max-
% protrusion tip per lick). Contours use their own color — phase colorbar unchanged.
%
% Run: lick_trajectory_and_max_protrusion_distribution_by_animal

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

% Smoothed 2-D histogram of max-protrusion tip (X,Y)
PEAK_HIST_BINS = 36;
PEAK_DENSITY_SMOOTH_SIGMA_BINS = 0.55;
MIN_PEAKS_FOR_DENSITY = 12;

% Warm RGB underlay (optional); alpha uses gamma so mid-density stays visible
SHOW_PEAK_DENSITY_UNDERLAY = true;
PEAK_UNDERLAY_ALPHA_MAX = 0.82;
PEAK_UNDERLAY_ALPHA_GAMMA = 0.55;
PEAK_DENSITY_MAP_NAME = 'hot';

% Dense contour lines drawn after trajectories — main cue for peak spatial mass
SHOW_PEAK_DENSITY_CONTOURS = true;
PEAK_CONTOUR_NUM_LEVELS = 9;
PEAK_CONTOUR_LINE_WIDTH = 2.05;
PEAK_CONTOUR_COLOR = [0.42 0.02 0.22];
PEAK_CONTOUR_LOW_FRAC = 0.14;
PEAK_CONTOUR_HIGH_FRAC = 0.96;

% Optional small markers at each max-protrusion sample (can clutter)
SHOW_PEAK_MARKERS = false;
PEAK_MARKER_SIZE = 14;
PEAK_MARKER_FACE = [0.15 0.45 0.85];
PEAK_MARKER_EDGE = [0.05 0.05 0.05];
PEAK_MARKER_ALPHA = 0.55;

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

plotOneExperimentByAnimal2d(csvPaths_PCRt, 'PCRt_BiPoles', 'PCRt', PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, ...
    xMin, xMax, yMin, yMax, DRAW_SEGMENT_LINES, LINE_WIDTH, ...
    SHOW_PEAK_DENSITY_UNDERLAY, PEAK_HIST_BINS, PEAK_DENSITY_SMOOTH_SIGMA_BINS, MIN_PEAKS_FOR_DENSITY, ...
    PEAK_UNDERLAY_ALPHA_MAX, PEAK_UNDERLAY_ALPHA_GAMMA, PEAK_DENSITY_MAP_NAME, ...
    SHOW_PEAK_DENSITY_CONTOURS, PEAK_CONTOUR_NUM_LEVELS, PEAK_CONTOUR_LINE_WIDTH, PEAK_CONTOUR_COLOR, ...
    PEAK_CONTOUR_LOW_FRAC, PEAK_CONTOUR_HIGH_FRAC, SHOW_PEAK_MARKERS, PEAK_MARKER_SIZE, PEAK_MARKER_FACE, ...
    PEAK_MARKER_EDGE, PEAK_MARKER_ALPHA, OUTPUT_DIR, SAVE_SVG);

plotOneExperimentByAnimal2d(csvPaths_IRt, 'IRt_BiPoles', 'IRt', PROB_MIN, GAP_FRAMES, MIN_LICK_FRAMES, ...
    xMin, xMax, yMin, yMax, DRAW_SEGMENT_LINES, LINE_WIDTH, ...
    SHOW_PEAK_DENSITY_UNDERLAY, PEAK_HIST_BINS, PEAK_DENSITY_SMOOTH_SIGMA_BINS, MIN_PEAKS_FOR_DENSITY, ...
    PEAK_UNDERLAY_ALPHA_MAX, PEAK_UNDERLAY_ALPHA_GAMMA, PEAK_DENSITY_MAP_NAME, ...
    SHOW_PEAK_DENSITY_CONTOURS, PEAK_CONTOUR_NUM_LEVELS, PEAK_CONTOUR_LINE_WIDTH, PEAK_CONTOUR_COLOR, ...
    PEAK_CONTOUR_LOW_FRAC, PEAK_CONTOUR_HIGH_FRAC, SHOW_PEAK_MARKERS, PEAK_MARKER_SIZE, PEAK_MARKER_FACE, ...
    PEAK_MARKER_EDGE, PEAK_MARKER_ALPHA, OUTPUT_DIR, SAVE_SVG);

end


function plotOneExperimentByAnimal2d(csvList, groupTag, shortTag, probMin, gapFrames, minLickFrames, ...
    xMin, xMax, yMin, yMax, drawLines, lineW, ...
    showUnderlay, histBins, smoothSigma, minPeaksDensity, underlayAlphaMax, underlayAlphaGamma, densityMapName, ...
    showContours, nContourLv, contourLw, contourColor, contourLoFrac, contourHiFrac, ...
    showPeakMarkers, peakMrkSz, peakFace, peakEdge, peakAlpha, outDir, saveSvg)

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
    figH = min(260 + 280 * nRow, 1850);
    fig = figure('Name', sprintf('2-D trajectories + peak-tip density (%s)', groupTag), ...
        'NumberTitle', 'off', 'Color', 'w', 'Position', [80 60 figW figH]);

    tl = tiledlayout(fig, nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, {
        sprintf('Intra-lick trajectories + max-protrusion tip density (%s)', groupTag)
        'Dark red contours / warm underlay = pooled tip at max jaw distance per lick; trajectory colors = phase'
        }, ...
        'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 13, 'Color', 'k');

    colormap(fig, cmapPhase);

    for ai = 1:nAnimals
        ax = nexttile(tl);
        set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
        set(ax, 'YDir', 'reverse');
        caxis(ax, [0 1]);
        hold(ax, 'on');

        animalLabel = animalGroups(ai).label;
        csvFiles = animalGroups(ai).files;

        pkx = [];
        pky = [];
        for kk = 1:numel(csvFiles)
            csvF = csvFiles{kk};
            if ~isfile(csvF)
                continue
            end
            [lxp, lyp] = sessionSortedTrajectoryPoints(csvF, probMin, gapFrames, minLickFrames);
            if isempty(lxp)
                continue
            end
            for jq = 1:numel(lxp)
                [pxe, pye] = tongueTipMxProtrusionXY(lxp{jq}, lyp{jq});
                pkx(end + 1, 1) = pxe;
                pky(end + 1, 1) = pye;
            end
        end

        [Xm, Ym, Zs, densOk] = peakSmoothedDensityGrid(pkx, pky, xMin, xMax, yMin, yMax, histBins, smoothSigma);

        hi = [];
        if densOk && numel(pkx) >= minPeaksDensity
            hi = plotPeakDensityRgbUnderlay(ax, Xm, Ym, Zs, showUnderlay, ...
                underlayAlphaMax, underlayAlphaGamma, densityMapName);
            if ~isempty(hi)
                try
                    uistack(hi, 'bottom');
                catch %#ok<*CTCH>
                end
            end
        end

        nSessions = 0;
        nLicks = 0;

        for k = 1:numel(csvFiles)
            csvFile = csvFiles{k};
            if ~isfile(csvFile)
                continue
            end

            [lickX, lickY, lickPhase] = sessionSortedTrajectoryPoints(csvFile, probMin, gapFrames, minLickFrames);
            if isempty(lickX)
                continue
            end
            nSessions = nSessions + 1;

            for j = 1:numel(lickX)
                xs = lickX{j};
                ys = lickY{j};
                ph = lickPhase{j};
                if isempty(xs)
                    continue
                end
                nLicks = nLicks + 1;

                if drawLines && numel(xs) > 1
                    pmid = (ph(1:end - 1) + ph(2:end)) / 2;
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

        if densOk && showContours && numel(pkx) >= minPeaksDensity && nContourLv >= 2
            plotPeakDensityContourOverlay(ax, Xm, Ym, Zs, nContourLv, contourLw, contourColor, contourLoFrac, contourHiFrac);
        end

        if showPeakMarkers && ~isempty(pkx)
            scatter(ax, pkx, pky, peakMrkSz, peakFace, 'filled', ...
                'MarkerEdgeColor', peakEdge, 'MarkerFaceAlpha', peakAlpha, 'LineWidth', 0.35);
        end

        plot(ax, 0, 0, 'ws', 'MarkerSize', 14, 'LineWidth', 2.6, 'MarkerFaceColor', 'none');
        plot(ax, 0, 0, 'ks', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'none');
        plot(ax, 0, 0, 'w+', 'MarkerSize', 18, 'LineWidth', 2.8);
        plot(ax, 0, 0, 'k+', 'MarkerSize', 16, 'LineWidth', 2.2);

        xlim(ax, [xMin xMax]);
        ylim(ax, [yMin yMax]);
        xticks(ax, [xMin 0 xMax]);
        yticks(ax, [yMin 0 yMax]);
        axis(ax, 'equal');
        axis(ax, 'square');

        title(ax, sprintf('%s (%d sessions | %d licks | %d peaks)', animalLabel, nSessions, nLicks, numel(pkx)), ...
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
        outPath = fullfile(outDir, sprintf('lick_trajectory_peak_density_underlay_by_animal_%s', shortTag));
        svgPath = [outPath '.svg'];
        if isgraphics(fig)
            try
                exportgraphics(fig, svgPath, 'ContentType', 'vector');
            catch %#ok<*CTCH>
                try
                    print(fig, svgPath, '-dsvg', '-painters');
                catch %#ok<*CTCH>
                    warning('Could not export figure to SVG; save manually from the figure window.');
                end
            end
            if isfile(svgPath)
                fprintf('Wrote %s\n', svgPath);
            end
        end
    end
end


function [Xm, Ym, Zs, ok] = peakSmoothedDensityGrid(pkx, pky, xMin, xMax, yMin, yMax, nb, sigma)

    Xm = [];
    Ym = [];
    Zs = [];
    ok = false;
    if nb < 4 || isempty(pkx) || isempty(pky) || numel(pkx) ~= numel(pky)
        return
    end

    xEdges = linspace(xMin, xMax, nb + 1);
    yEdges = linspace(yMin, yMax, nb + 1);
    [Nbin, ~, ~] = histcounts2(double(pkx(:)), double(pky(:)), xEdges, yEdges);
    Zs = smooth2dBins(double(Nbin), sigma);
    if isempty(Zs) || max(Zs(:)) <= 0
        return
    end

    xc = xEdges(1:end - 1) + diff(xEdges) / 2;
    yc = yEdges(1:end - 1) + diff(yEdges) / 2;
    [Xm, Ym] = ndgrid(xc, yc);
    ok = true;
end


function hi = plotPeakDensityRgbUnderlay(ax, Xm, Ym, Zs, wantUnderlay, alphaMax, alphaGamma, mapName)

    hi = [];
    if ~wantUnderlay || isempty(Zs)
        return
    end

    mx = max(Zs(:));
    if mx <= 0
        return
    end

    H = double(Zs) ./ mx;
    Hy = H.';
    Ag = max(0, min(1, Hy));
    if ~(isempty(alphaGamma)) && isscalar(alphaGamma) && alphaGamma > 0 && alphaGamma ~= 1
        Ag = Ag.^alphaGamma;
    end

    try
        cmapDen = feval(mapName, 256);
    catch %#ok<*CTCH>
        cmapDen = parula(256);
    end
    qi = uint8(min(255, floor(Hy * 255))) + 1;
    rgb = ind2rgb(qi, cmapDen);

    xl = [min(Xm(:)) max(Xm(:))];
    yl = [min(Ym(:)) max(Ym(:))];
    hi = imagesc(ax, xl, yl, rgb);
    hi.AlphaData = alphaMax .* Ag;
    hi.PickableParts = 'none';

    axis(ax, 'on');
end


function plotPeakDensityContourOverlay(ax, Xm, Ym, Zs, nLevels, lw, lineColor, loFrac, hiFrac)

    if isempty(Zs) || nLevels < 2
        return
    end

    top = max(Zs(:));
    if top <= 0
        return
    end

    lo = max(0, loFrac) * top;
    hi = min(top, hiFrac * top);
    if hi <= lo
        return
    end

    lv = linspace(lo, hi, nLevels);
    % Plain contour call only — LineJoin / LineSmoothing are not portable across MATLAB versions.
    [~, hc] = contour(ax, Xm, Ym, Zs, lv);
    if isempty(hc)
        return
    end
    hList = hc;
    if ~isscalar(hc)
        hList = hc(:);
    end
    for ih = 1:numel(hList)
        hiCh = hList(ih);
        if ~(isgraphics(hiCh) && isvalid(hiCh))
            continue
        end
        try
            set(hiCh, 'LineWidth', lw, 'Color', lineColor);
        catch %#ok<*CTCH>
            try
                hiCh.LineWidth = lw;
                hiCh.Color = lineColor;
            catch %#ok<*CTCH>
            end
        end
    end
end


function [px, py] = tongueTipMxProtrusionXY(xs, ys)
    xs = xs(:);
    ys = ys(:);
    [~, k] = max(hypot(xs, ys));
    px = xs(k);
    py = ys(k);
end


function Zs = smooth2dBins(Z, sigmaBins)
    if isempty(Z)
        Zs = Z;
        return
    end
    if nargin < 2 || isempty(sigmaBins) || (~isscalar(sigmaBins)) || sigmaBins <= 0
        Zs = Z;
        return
    end
    r = max(3, ceil(3 * sigmaBins));
    k = (-r:r)';
    kern1 = exp(-(k.^2) / (2 * sigmaBins^2));
    kern1 = kern1 / sum(kern1);
    Zs = conv2(kern1, kern1(:)', Z, 'same');
    Zs(isnan(Zs)) = 0;
    Zs = max(0, Zs);
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
