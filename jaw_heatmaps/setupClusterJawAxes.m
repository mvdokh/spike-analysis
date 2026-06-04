function setupClusterJawAxes(ax, halfExtent)
% setupClusterJawAxes  Centered square axes for cluster overlays (no phase colormap).

if nargin < 2 || isempty(halfExtent)
    halfExtent = 50;
end
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
axis(ax, 'equal');
axis(ax, 'square');
set(ax, 'YDir', 'reverse');
xlim(ax, [-halfExtent halfExtent]);
ylim(ax, [-halfExtent halfExtent]);
xticks(ax, [-halfExtent 0 halfExtent]);
yticks(ax, [-halfExtent 0 halfExtent]);
end
