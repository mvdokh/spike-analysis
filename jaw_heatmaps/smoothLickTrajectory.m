function [xOut, yOut, phOut] = smoothLickTrajectory(x, y, nOut, movAvgWin)
% smoothLickTrajectory  PCHIP resample along intra-lick phase, optional moving average.
%
% movAvgWin: odd window >= 3 applies extra smoothing; 0 or [] skips.

x = x(:);
y = y(:);
n = numel(x);
if n < 2
    xOut = x;
    yOut = y;
    phOut = 0.5 * ones(size(x));
    return
end
if nargin < 3 || isempty(nOut)
    nOut = max(n, 64);
else
    nOut = max(2, round(nOut));
end
if nargin < 4
    movAvgWin = 0;
end

t = linspace(0, 1, n)';
tq = linspace(0, 1, nOut)';
xOut = interp1(t, x, tq, 'pchip');
yOut = interp1(t, y, tq, 'pchip');
phOut = tq;

if isempty(movAvgWin) || movAvgWin < 3
    return
end
w = max(3, round(movAvgWin));
if mod(w, 2) == 0
    w = w + 1;
end
ker = ones(w, 1) / w;
xOut = conv(xOut, ker, 'same');
yOut = conv(yOut, ker, 'same');
end
