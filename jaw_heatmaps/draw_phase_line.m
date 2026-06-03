function draw_phase_line(ax, x, y, ph, lineW, maxSegGap)
% draw_phase_line  Phase-colored polyline; optional break at large gaps.
%
% When maxSegGap is set, segments between consecutive points farther apart
% than maxSegGap are not drawn (avoids lines across removed jump points).

x = x(:)';
y = y(:)';
ph = ph(:)';
if numel(x) < 2
    return
end

if nargin < 6 || isempty(maxSegGap) || ~isfinite(maxSegGap)
    drawPhaseChunk(ax, x, y, ph, lineW);
    return
end

d = hypot(diff(x), diff(y));
breakAfter = find(d > maxSegGap);
starts = [1, breakAfter + 1];
ends = [breakAfter, numel(x)];
for k = 1:numel(starts)
    i0 = starts(k);
    i1 = ends(k);
    if i1 > i0
        drawPhaseChunk(ax, x(i0:i1), y(i0:i1), ph(i0:i1), lineW);
    end
end
end


function drawPhaseChunk(ax, x, y, ph, lineW)
if numel(x) < 2
    return
end
surface(ax, [x; x], [y; y], zeros(2, numel(x)), [ph; ph], ...
    'FaceColor', 'none', 'EdgeColor', 'interp', 'LineWidth', lineW, ...
    'HandleVisibility', 'off');
end
