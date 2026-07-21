function d = trajectoryMaxExcursionFromStart(x, y)
% trajectoryMaxExcursionFromStart  Max distance from first point to any point on path.

x = x(:);
y = y(:);
if numel(x) < 1
    d = 0;
    return
end
d = max(hypot(x - x(1), y - y(1)));
end
