function pickIdx = selectMostSimilarLickIndices(xCells, yCells, nPick, nResample)
% selectMostSimilarLickIndices  Greedy cluster: nPick licks with most similar shape.
%
% Each lick is PCHIP-resampled, translated to start at origin, scale-normalized by
% max distance from start, then compared with mean pointwise Euclidean distance.
% Starts from the pool medoid and adds licks with lowest mean distance to the set.

if nargin < 4 || isempty(nResample)
    nResample = 64;
end
nPick = max(1, round(nPick));

n = numel(xCells);
pickIdx = (1:n)';
if n <= nPick
    return
end

D = lickShapeDistanceMatrix(xCells, yCells, nResample);

[~, seed] = min(sum(D, 2));
chosen = seed;
while numel(chosen) < nPick
    bestJ = 0;
    bestScore = inf;
    for j = 1:n
        if any(chosen == j)
            continue
        end
        score = mean(D(j, chosen));
        if score < bestScore
            bestScore = score;
            bestJ = j;
        end
    end
    chosen(end + 1) = bestJ; %#ok<AGROW>
end
pickIdx = chosen(:);
end
