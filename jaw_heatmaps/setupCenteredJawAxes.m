function setupCenteredJawAxes(ax, halfExtent, cmapPhase)
% setupCenteredJawAxes  Square plot window centered on jaw rest (default 100x100, +/-50 px).

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
colormap(ax, cmapPhase);
caxis(ax, [0 1]);
end
