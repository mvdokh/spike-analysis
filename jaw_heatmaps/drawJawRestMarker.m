function drawJawRestMarker(ax)
% drawJawRestMarker  Small plus at jaw rest (origin after centering).

plot(ax, 0, 0, 'w+', 'MarkerSize', 8, 'LineWidth', 1.5, 'Color', 'w');
plot(ax, 0, 0, 'k+', 'MarkerSize', 7, 'LineWidth', 1.2);
end
