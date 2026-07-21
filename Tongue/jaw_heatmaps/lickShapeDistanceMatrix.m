function [D, curves] = lickShapeDistanceMatrix(xCells, yCells, nResample)
% lickShapeDistanceMatrix  Pairwise shape distances (same metric as similar-lick pick).
%
% Each lick: PCHIP to nResample, translate to lick start, scale by max excursion.
% D(i,j) = mean pointwise Euclidean distance between normalized curves.

if nargin < 3 || isempty(nResample)
    nResample = 64;
end
nResample = max(8, round(nResample));

n = numel(xCells);
D = zeros(n, n);
curves = nan(nResample, 2, n);

if n < 1
    return
end

for i = 1:n
    [xs, ys, ~] = smoothLickTrajectory(xCells{i}, yCells{i}, nResample, 0);
    xy = [xs(:), ys(:)];
    xy(:, 1) = xy(:, 1) - xy(1, 1);
    xy(:, 2) = xy(:, 2) - xy(1, 2);
    ext = max(hypot(xy(:, 1), xy(:, 2)));
    if ext > 0
        xy = xy / ext;
    end
    curves(:, :, i) = xy;
end

for i = 1:n
    for j = (i + 1):n
        d = mean(vecnorm(curves(:, :, i) - curves(:, :, j), 2, 2), 'omitnan');
        D(i, j) = d;
        D(j, i) = d;
    end
end
end
