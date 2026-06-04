function drawSolidClippedLine(ax, x, y, rgb, lineW, clipHalf)
% drawSolidClippedLine  Single-color polyline clipped to [-clipHalf, clipHalf].

x = x(:)';
y = y(:)';
if numel(x) < 2
    return
end
if nargin < 6 || isempty(clipHalf) || ~isfinite(clipHalf) || clipHalf <= 0
    plot(ax, x, y, '-', 'Color', rgb, 'LineWidth', lineW, 'HandleVisibility', 'off');
    return
end

hold(ax, 'on');
for ii = 1:(numel(x) - 1)
    [x0, y0, x1, y1, ~, ~, vis] = clipSegmentToSquare( ...
        x(ii), y(ii), 0, x(ii + 1), y(ii + 1), 0, clipHalf);
    if ~vis
        continue
    end
    plot(ax, [x0 x1], [y0 y1], '-', ...
        'Color', rgb, 'LineWidth', lineW, 'HandleVisibility', 'off');
end
end
