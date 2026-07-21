function annotatePathLengthStats(ax, pathLengths, maxExcursions)
% annotatePathLengthStats  Arc-length and max excursion (start to farthest point).

pathLengths = pathLengths(:);
pathLengths = pathLengths(isfinite(pathLengths) & pathLengths >= 0);
if isempty(pathLengths)
    return
end

lines = {formatMeanSd('Path length (arc)', pathLengths)};

if nargin >= 3 && ~isempty(maxExcursions)
    maxExcursions = maxExcursions(:);
    maxExcursions = maxExcursions(isfinite(maxExcursions) & maxExcursions >= 0);
    if ~isempty(maxExcursions)
        lines{end + 1} = formatMeanSd('Max distance from start', maxExcursions); %#ok<AGROW>
    end
end

lbl = strjoin(lines, newline);

text(ax, 0.04, 0.96, lbl, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
    'FontSize', 8, 'Color', 'k', 'BackgroundColor', [1 1 1 0.8], ...
    'Margin', 2, 'Interpreter', 'tex', 'Clipping', 'off');
end


function lbl = formatMeanSd(prefix, vals)
if numel(vals) > 1
    lbl = sprintf('%s: %.1f \\pm %.1f px', prefix, mean(vals), std(vals, 0));
else
    lbl = sprintf('%s: %.1f px', prefix, vals(1));
end
end
