function [lx, ly, lp, info] = filter_lick_trajectories(lx, ly, lp, minLickFrames, stepMadK, removeSingleton, filterMode, stepHardMax, hotspotMinCount, hotspotPurgeCount)
% filter_lick_trajectories
% Quality filter on jaw-tip trajectories within one plot/session scope.
%
% filterMode:
%   'lick'   - drop entire lick if any point would be removed (TeLC default)
%   'points' - remove outlier points only; drop lick only if too short after
%              (BiPoles default). Catches large jumps to frequent coordinates
%              (e.g. corner hotspots) via step-distance threshold T.
%
% Criteria use session-pooled step distances:
%   T = median + stepMadK * (1.4826 * MAD)  (Iglewicz-Hoaglin style)
% Optional stepHardMax: also flag steps with d > stepHardMax (px), so a
%   session-inflated robust T cannot miss obvious teleports.
% Plus optional singleton (X,Y) rule on rounded pixel coordinates.
% Multi-frame dwell at one rounded (X,Y) after a large jump is removed in
%   full (hotspot stutter runs), not only the first/last frame of the run.

if nargin < 6 || isempty(removeSingleton)
    removeSingleton = true;
end
if nargin < 7 || isempty(filterMode)
    filterMode = 'lick';
end
if nargin < 8
    stepHardMax = [];
end
if nargin < 9 || isempty(hotspotMinCount)
    hotspotMinCount = 0;
end
if nargin < 10 || isempty(hotspotPurgeCount)
    hotspotPurgeCount = Inf;
end

info = struct('nIn', 0, 'nKept', 0, 'nDropSingleton', 0, 'nDropJump', 0, ...
    'nDropShort', 0, 'nPtsRemoved', 0, 'stepThreshold', NaN, ...
    'stepHardMax', stepHardMax, 'hotspotMinCount', hotspotMinCount, ...
    'hotspotPurgeCount', hotspotPurgeCount, ...
    'stepFilterActive', false, 'filterMode', filterMode);

if isempty(lx)
    return
end
info.nIn = numel(lx);

cnt = [];
if removeSingleton
    cnt = buildCoordCountMap(lx, ly);
end

steps = poolStepDistances(lx, ly);
[T, isLongTail, ~] = robustStepThreshold(steps, stepMadK);
Tflag = effectiveStepThreshold(T, stepHardMax, steps);
info.stepThreshold = Tflag;
info.stepFilterActive = isLongTail || (~isempty(steps) && any(steps > Tflag));

switch lower(filterMode)
    case 'points'
        [lx, ly, lp, info] = filterPointsMode(lx, ly, lp, minLickFrames, ...
            cnt, Tflag, info.stepFilterActive, removeSingleton, info, ...
            hotspotMinCount, hotspotPurgeCount);
    otherwise % 'lick'
        [lx, ly, lp, info] = filterLickMode(lx, ly, lp, minLickFrames, ...
            cnt, Tflag, info.stepFilterActive, removeSingleton, info);
end
end


function [lx, ly, lp, info] = filterLickMode(lx, ly, lp, minLickFrames, cnt, T, isLongTail, removeSingleton, info)
keepLick = true(numel(lx), 1);
for j = 1:numel(lx)
    xs = lx{j}(:);
    ys = ly{j}(:);

    if removeSingleton && lickHasSingletonPoint(xs, ys, cnt)
        keepLick(j) = false;
        info.nDropSingleton = info.nDropSingleton + 1;
        continue
    end

    if isLongTail && lickHasAbnormalJump(xs, ys, T)
        keepLick(j) = false;
        info.nDropJump = info.nDropJump + 1;
    end
end

lx = lx(keepLick);
ly = ly(keepLick);
lp = lp(keepLick);
[lx, ly, lp, info] = dropShortLicks(lx, ly, lp, minLickFrames, info);
end


function [lx, ly, lp, info] = filterPointsMode(lx, ly, lp, minLickFrames, cnt, T, isLongTail, removeSingleton, info, hotspotMinCount, hotspotPurgeCount)
keepLick = true(numel(lx), 1);
for j = 1:numel(lx)
    xs = lx{j}(:);
    ys = ly{j}(:);
    ph = lp{j}(:);

    for pass = 1:4
        n = numel(xs);
        if n < 1
            break
        end
        removePt = false(n, 1);

        if removeSingleton && ~isempty(cnt)
            rx = round(xs);
            ry = round(ys);
            for p = 1:n
                if cnt(coordKey(rx(p), ry(p))) == 1
                    removePt(p) = true;
                end
            end
        end

        if ~isempty(cnt) && hotspotMinCount > 0
            removePt = removePt | frequentHotspotPointMask(xs, ys, T, cnt, ...
                hotspotMinCount, hotspotPurgeCount, isLongTail);
        end

        if isLongTail && n >= 2
            removePt = removePt | largeJumpPointMask(xs, ys, T);
            removePt = removePt | hotspotStutterRunMask(xs, ys, T);
            if n >= 3
                removePt = removePt | ~spikeKeepMask(xs, ys, T, 3);
            end
        end

        if ~any(removePt)
            break
        end
        info.nPtsRemoved = info.nPtsRemoved + sum(removePt);
        keepPt = ~removePt;
        xs = xs(keepPt);
        ys = ys(keepPt);
        ph = ph(keepPt);
    end

    lx{j} = xs;
    ly{j} = ys;
    lp{j} = ph;

    if numel(xs) < minLickFrames
        keepLick(j) = false;
        info.nDropShort = info.nDropShort + 1;
    end
end

lx = lx(keepLick);
ly = ly(keepLick);
lp = lp(keepLick);
info.nKept = numel(lx);
end


function Tflag = effectiveStepThreshold(T, hardMax, steps)
% Stricter of robust T and optional absolute pixel cap; never above hardMax.
Tflag = T;
if ~isempty(hardMax) && isscalar(hardMax) && isfinite(hardMax) && hardMax > 0
    Tflag = min(T, hardMax);
end
if isempty(steps)
    return
end
d = steps(isfinite(steps) & steps >= 0);
if isempty(d)
    return
end
% If robust T is so loose nothing exceeds it, still filter obvious teleports.
if ~isempty(hardMax) && isscalar(hardMax) && isfinite(hardMax) && hardMax > 0
    if ~any(d > Tflag) && any(d > hardMax)
        Tflag = hardMax;
    end
end
end


function removePt = frequentHotspotPointMask(xs, ys, T, cnt, minCount, purgeCount, isLongTail)
% Session-frequent pixels (corner piles): drop on any large adjacent step, or
% all points when the coordinate count exceeds purgeCount.
xs = xs(:);
ys = ys(:);
n = numel(xs);
removePt = false(n, 1);
if n < 1 || isempty(cnt)
    return
end
rx = round(xs);
ry = round(ys);
if n >= 2
    d = hypot(diff(xs), diff(ys));
else
    d = [];
end
for p = 1:n
    c = cnt(coordKey(rx(p), ry(p)));
    if c >= purgeCount
        removePt(p) = true;
        continue
    end
    if c < minCount || ~isLongTail
        continue
    end
    prevLarge = (p > 1) && (d(p - 1) > T);
    nextLarge = (p < n) && (d(p) > T);
    if prevLarge || nextLarge
        removePt(p) = true;
    end
end
end


function removePt = hotspotStutterRunMask(xs, ys, T)
% Remove every frame in a run of identical rounded (X,Y) when the run is
% reached or left by a large step (multi-frame corner glitches).
xs = xs(:);
ys = ys(:);
n = numel(xs);
removePt = false(n, 1);
if n < 2
    return
end
rx = round(xs);
ry = round(ys);
d = hypot(diff(xs), diff(ys));
p = 1;
while p <= n
    q = p;
    while q < n && rx(q + 1) == rx(p) && ry(q + 1) == ry(p)
        q = q + 1;
    end
    enterLarge = (p > 1) && (d(p - 1) > T);
    exitLarge = (q < n) && (d(q) > T);
    if enterLarge || exitLarge
        removePt(p:q) = true;
    end
    p = q + 1;
end
end


function removePt = largeJumpPointMask(xs, ys, T)
% Mark both endpoints of any step larger than T (catches jumps into hotspots).
xs = xs(:);
ys = ys(:);
n = numel(xs);
removePt = false(n, 1);
if n < 2
    return
end
d = hypot(diff(xs), diff(ys));
for i = 1:numel(d)
    if d(i) > T
        removePt(i) = true;
        removePt(i + 1) = true;
    end
end
end


function [lx, ly, lp, info] = dropShortLicks(lx, ly, lp, minLickFrames, info)
if isempty(lx)
    info.nKept = 0;
    return
end
keepLen = true(numel(lx), 1);
for j = 1:numel(lx)
    if numel(lx{j}) < minLickFrames
        keepLen(j) = false;
        info.nDropShort = info.nDropShort + 1;
    end
end
lx = lx(keepLen);
ly = ly(keepLen);
lp = lp(keepLen);
info.nKept = numel(lx);
end


function cnt = buildCoordCountMap(lx, ly)
cnt = containers.Map('KeyType', 'char', 'ValueType', 'double');
for j = 1:numel(lx)
    rx = round(lx{j}(:));
    ry = round(ly{j}(:));
    for p = 1:numel(rx)
        key = coordKey(rx(p), ry(p));
        if isKey(cnt, key)
            cnt(key) = cnt(key) + 1;
        else
            cnt(key) = 1;
        end
    end
end
end


function tf = lickHasSingletonPoint(xs, ys, cnt)
rx = round(xs(:));
ry = round(ys(:));
tf = false;
for p = 1:numel(rx)
    if cnt(coordKey(rx(p), ry(p))) == 1
        tf = true;
        return
    end
end
end


function tf = lickHasAbnormalJump(xs, ys, T)
xs = xs(:);
ys = ys(:);
if numel(xs) < 2
    tf = false;
    return
end
d = hypot(diff(xs), diff(ys));
if any(d > T)
    tf = true;
    return
end
if any(hotspotStutterRunMask(xs, ys, T))
    tf = true;
    return
end
keep = spikeKeepMask(xs, ys, T, 3);
tf = any(~keep);
end


function steps = poolStepDistances(lx, ly)
steps = [];
for j = 1:numel(lx)
    xs = lx{j};
    ys = ly{j};
    if numel(xs) > 1
        steps = [steps; hypot(diff(xs(:)), diff(ys(:)))]; %#ok<AGROW>
    end
end
end


function [T, isLongTail, info] = robustStepThreshold(steps, K)
d = steps(:);
d = d(isfinite(d) & d >= 0);
info = struct('n', numel(d), 'median', NaN, 'scaledMad', NaN, ...
    'max', NaN, 'T', Inf, 'nAbove', 0);
if isempty(d)
    T = Inf;
    isLongTail = false;
    return
end
med = median(d);
madv = median(abs(d - med));
scaledMad = 1.4826 * madv;
if scaledMad <= 0 || ~isfinite(scaledMad)
    scaledMad = std(d);
end
if scaledMad <= 0 || ~isfinite(scaledMad)
    T = Inf;
    isLongTail = false;
else
    T = med + K * scaledMad;
    isLongTail = any(d > T);
end
info.median = med;
info.scaledMad = scaledMad;
info.max = max(d);
info.T = T;
info.nAbove = sum(d > T);
end


function keep = spikeKeepMask(xs, ys, T, maxIters)
xs = xs(:);
ys = ys(:);
n = numel(xs);
keep = true(n, 1);
if n < 3
    return
end
for it = 1:maxIters %#ok<NASGU>
    idx = find(keep);
    m = numel(idx);
    if m < 3
        break
    end
    xx = xs(idx);
    yy = ys(idx);
    d = hypot(diff(xx), diff(yy));
    out = false(m, 1);
    if d(1) > T
        out(1) = true;
    end
    if d(end) > T
        out(end) = true;
    end
    for p = 2:(m - 1)
        if d(p - 1) > T && d(p) > T
            out(p) = true;
        end
    end
    if ~any(out)
        break
    end
    keep(idx(out)) = false;
end
end


function key = coordKey(x, y)
key = sprintf('%d,%d', x, y);
end
