function draw_phase_line_frame_gaps(ax, x, y, ph, fr, lineW, clipHalf)
% draw_phase_line_frame_gaps  Phase line with breaks on frame gaps; optional box clip.

fr = fr(:)';
if numel(fr) < 2
    return
end
gapAfter = find(diff(fr) > 1);
starts = [1, gapAfter + 1];
ends = [gapAfter, numel(fr)];
for k = 1:numel(starts)
    i0 = starts(k);
    i1 = ends(k);
    if i1 > i0
        draw_phase_line(ax, x(i0:i1), y(i0:i1), ph(i0:i1), lineW, [], clipHalf);
    end
end
end
