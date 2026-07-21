function [lx, ly] = centerLickCells(lx, ly, jx, jy)
% centerLickCells  Subtract jaw rest position so rest is at (0, 0).

if isnan(jx) || isnan(jy)
    return
end
for j = 1:numel(lx)
    lx{j} = lx{j}(:) - jx;
    ly{j} = ly{j}(:) - jy;
end
end
