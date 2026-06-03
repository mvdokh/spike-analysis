function draw_phase_line(ax, x, y, ph, lineW, maxSegGap, clipHalf)
% draw_phase_line  Phase-colored polyline (turbo/jet via axes colormap).
%
% Colors each segment from midpoint phase (reliable in SVG export).
% Optional maxSegGap breaks the polyline across large spatial jumps.
% Optional clipHalf: clip segments to [-clipHalf, clipHalf] (e.g. 100x100 when clipHalf=50).

x = x(:)';
y = y(:)';
ph = ph(:)';
if numel(x) < 2
    return
end

useClip = nargin >= 7 && ~isempty(clipHalf) && isfinite(clipHalf) && clipHalf > 0;

if nargin < 5 || isempty(maxSegGap) || ~isfinite(maxSegGap)
    if useClip
        drawPhaseChunkClipped(ax, x, y, ph, lineW, clipHalf);
    else
        drawPhaseChunk(ax, x, y, ph, lineW);
    end
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
        if useClip
            drawPhaseChunkClipped(ax, x(i0:i1), y(i0:i1), ph(i0:i1), lineW, clipHalf);
        else
            drawPhaseChunk(ax, x(i0:i1), y(i0:i1), ph(i0:i1), lineW);
        end
    end
end
end


function drawPhaseChunkClipped(ax, x, y, ph, lineW, h)
cmap = getAxisColormap(ax);
for ii = 1:(numel(x) - 1)
    [x0, y0, x1, y1, p0, p1, vis] = clipSegmentToSquare( ...
        x(ii), y(ii), ph(ii), x(ii + 1), y(ii + 1), ph(ii + 1), h);
    if ~vis
        continue
    end
    pmid = (p0 + p1) / 2;
    plot(ax, [x0 x1], [y0 y1], '-', ...
        'Color', rgbFromPhase(pmid, cmap), 'LineWidth', lineW, 'HandleVisibility', 'off');
end
end


function drawPhaseChunk(ax, x, y, ph, lineW)
if numel(x) < 2
    return
end
cmap = getAxisColormap(ax);
pmid = (ph(1:end - 1) + ph(2:end)) / 2;
lineColors = rgbFromPhase(pmid, cmap);
for ii = 1:numel(pmid)
    plot(ax, x(ii:ii + 1), y(ii:ii + 1), '-', ...
        'Color', lineColors(ii, :), 'LineWidth', lineW, 'HandleVisibility', 'off');
end
end


function cmap = getAxisColormap(ax)
cmap = colormap(ax);
if isempty(cmap)
    try
        cmap = turbo(256);
    catch %#ok<CTCH>
        cmap = jet(256);
    end
end
end


function rgb = rgbFromPhase(phase, cmap)
phase = phase(:);
phase = max(0, min(1, phase));
idx = round(phase * (size(cmap, 1) - 1)) + 1;
rgb = cmap(idx, :);
end
