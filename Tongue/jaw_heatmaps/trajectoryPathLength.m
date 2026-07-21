function len = trajectoryPathLength(x, y)
% trajectoryPathLength  Arc length along a polyline (sum of segment lengths).

x = x(:);
y = y(:);
if numel(x) < 2
    len = 0;
    return
end
len = sum(hypot(diff(x), diff(y)));
end
