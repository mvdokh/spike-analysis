function [labels, info] = clusterLicksByShapeDistance(D, maxK)
% clusterLicksByShapeDistance  Hierarchical clustering on shape distance matrix.
%
% Chooses k in 2..maxK by maximum mean silhouette on D.
% Falls back to threshold clustering if Statistics Toolbox unavailable.

info = struct('nLicks', size(D, 1), 'chosenK', 1, 'meanSilhouette', NaN, ...
    'method', '', 'linkageZ', []);

labels = ones(info.nLicks, 1);
n = info.nLicks;
if n < 1
    return
end
if n == 1
    info.method = 'single';
    return
end

maxK = max(2, round(maxK));
maxK = min(maxK, n - 1);

if maxK < 2
    info.method = 'single';
    return
end

if ~(exist('linkage', 'file') == 2 && exist('cluster', 'file') == 2)
    labels = thresholdCluster(D);
    info.chosenK = numel(unique(labels));
    info.method = 'threshold';
    return
end

Y = squareform(D, 'tovector');
Z = linkage(Y, 'average');
info.linkageZ = Z;

bestK = 2;
bestSil = -inf;
for k = 2:maxK
    idx = cluster(Z, 'maxclust', k);
    if numel(unique(idx)) < 2
        continue
    end
    s = silhouetteFromDiss(D, idx);
    m = mean(s, 'omitnan');
    if m > bestSil
        bestSil = m;
        bestK = k;
    end
end
labels = cluster(Z, 'maxclust', bestK);
info.chosenK = numel(unique(labels));
info.meanSilhouette = bestSil;
info.method = 'hclust_silhouette';
end


function labels = thresholdCluster(D)
% Single-linkage-style: merge components below 25th percentile distance.

n = size(D, 1);
dOff = D + eye(n) * inf;
thr = prctile(dOff(:), 25);
labels = (1:n)';
for i = 1:n
    for j = (i + 1):n
        if D(i, j) <= thr
            lj = labels(j);
            li = labels(i);
            labels(labels == lj) = li;
        end
    end
end
[~, ~, labels] = unique(labels);
end


function s = silhouetteFromDiss(D, idx)
% Mean silhouette per point from a precomputed dissimilarity matrix.

n = size(D, 1);
s = zeros(n, 1);
for i = 1:n
    ci = idx(i);
    inC = find(idx == ci);
    outC = find(idx ~= ci);
    if numel(inC) > 1
        a = mean(D(i, inC(inC ~= i)));
    else
        a = 0;
    end
    if isempty(outC)
        b = 0;
    else
        others = unique(idx(outC));
        bVals = zeros(numel(others), 1);
        for t = 1:numel(others)
            mask = idx == others(t);
            bVals(t) = mean(D(i, mask));
        end
        b = min(bVals);
    end
    denom = max(a, b);
    if denom > 0
        s(i) = (b - a) / denom;
    else
        s(i) = 0;
    end
end
end
