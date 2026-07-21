function [x0, y0, x1, y1, p0, p1, visible] = clipSegmentToSquare(xa, ya, pa, xb, yb, pb, h)
% clipSegmentToSquare  Liang-Barsky clip of segment to [-h,h] x [-h,h].
% Phase is linearly interpolated along the original segment.

x0 = NaN;
y0 = NaN;
x1 = NaN;
y1 = NaN;
p0 = NaN;
p1 = NaN;
visible = false;

if ~isfinite(h) || h <= 0
    return
end

dx = xb - xa;
dy = yb - ya;
dp = pb - pa;

% p(i)*t + q(i) >= 0 defines the inside of the clip box
p = [-dx, dx, -dy, dy];
q = [xa + h, h - xa, ya + h, h - ya];

t0 = 0;
t1 = 1;

for i = 1:4
    if abs(p(i)) < eps
        if q(i) < 0
            return
        end
    else
        t = q(i) / p(i);
        if p(i) < 0
            t0 = max(t0, t);
        else
            t1 = min(t1, t);
        end
        if t0 > t1
            return
        end
    end
end

x0 = xa + t0 * dx;
y0 = ya + t0 * dy;
x1 = xa + t1 * dx;
y1 = ya + t1 * dy;
p0 = pa + t0 * dp;
p1 = pa + t1 * dp;
visible = true;
end
