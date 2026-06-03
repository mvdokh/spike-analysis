function [sx, sy, sp] = filterScatterToSquare(x, y, ph, halfExtent)
% filterScatterToSquare  Keep only points inside [-h,h] for scatter overlays.

if nargin < 4 || isempty(halfExtent)
    sx = x;
    sy = y;
    sp = ph;
    return
end
m = abs(x(:)) <= halfExtent & abs(y(:)) <= halfExtent;
sx = x(m);
sy = y(m);
sp = ph(m);
end
