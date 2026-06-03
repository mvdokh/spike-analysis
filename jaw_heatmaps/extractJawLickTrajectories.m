function [lickX, lickY, lickPhase, lickFrame] = extractJawLickTrajectories(jawFile, intervals, probMin, minLickFrames, framePad, returnFrames)
% extractJawLickTrajectories  Jaw-tip points per behavior lick interval.
% framePad: include this many frames before Start and after End (default 10).

if nargin < 5 || isempty(framePad)
    framePad = 10;
end
if nargin < 6
    returnFrames = false;
end

lickX = {};
lickY = {};
lickPhase = {};
lickFrame = {};

tbl = readJawTableShared(jawFile);
frm = tbl.Frame;
xv = tbl.X;
yv = tbl.Y;
if isempty(probMin) || (~isscalar(probMin)) || probMin <= 0
    keepProb = true(size(frm));
else
    keepProb = tbl.Probability >= probMin;
end

pad = max(0, round(framePad));

for i = 1:size(intervals, 1)
    s = intervals(i, 1) - pad;
    e = intervals(i, 2) + pad;
    inLick = (frm >= s) & (frm <= e) & keepProb;
    if ~any(inLick)
        continue
    end

    fi = frm(inLick);
    xi = xv(inLick);
    yi = yv(inLick);
    [~, ord] = sort(fi);
    xs = xi(ord);
    ys = yi(ord);
    fs = fi(ord);
    L = numel(xs);
    if L < minLickFrames
        continue
    end

    if L == 1
        ph = 0.5;
    else
        ph = ((0:(L - 1))' / (L - 1));
    end

    lickX{end + 1} = xs;       %#ok<AGROW>
    lickY{end + 1} = ys;       %#ok<AGROW>
    lickPhase{end + 1} = ph;   %#ok<AGROW>
    if returnFrames
        lickFrame{end + 1} = fs;   %#ok<AGROW>
    end
end
end


function tbl = readJawTableShared(csvFile)
T = readtable(csvFile, 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
v = T.Properties.VariableNames;
vl = lower(strtrim(v));
fc = @(s) strcmp(vl, lower(s));

fi = fc('frame');
xi = fc('x');
yi = fc('y');
pidx = fc('probability');
if ~(any(fi) && any(xi) && any(yi))
    error('Jaw CSV must include Frame, X, and Y columns: %s', csvFile);
end

F = double(T{:, fi});
X = double(T{:, xi});
Y = double(T{:, yi});
if any(pidx)
    Pr = double(T{:, pidx});
else
    Pr = ones(size(F));
end
tbl = table(F, X, Y, Pr, 'VariableNames', {'Frame', 'X', 'Y', 'Probability'});
end
