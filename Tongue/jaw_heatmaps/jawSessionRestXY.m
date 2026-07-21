function [jx, jy] = jawSessionRestXY(jawCsvPath, probMin)
% jawSessionRestXY  Mean jaw (X,Y) over the session for plot centering.

jx = NaN;
jy = NaN;
if ~isfile(jawCsvPath)
    return
end

tbl = readJawTableLocal(jawCsvPath);
if height(tbl) < 1
    return
end

if nargin < 2 || isempty(probMin) || (~isscalar(probMin)) || probMin <= 0
    keep = true(height(tbl), 1);
else
    keep = tbl.Probability >= probMin;
    if ~any(keep)
        return
    end
end

jx = mean(tbl.X(keep), 'omitnan');
jy = mean(tbl.Y(keep), 'omitnan');
end


function tbl = readJawTableLocal(csvFile)
T = readtable(csvFile, 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
v = T.Properties.VariableNames;
vl = lower(strtrim(v));
fc = @(s) strcmp(vl, lower(s));

fi = fc('frame');
xi = fc('x');
yi = fc('y');
pidx = fc('probability');
if ~(any(fi) && any(xi) && any(yi))
    error('Jaw CSV must include Frame, X, and Y columns: %s', csvFile);
end

F = double(T{:, fi});
X = double(T{:, xi});
Y = double(T{:, yi});
if any(pidx)
    Pr = double(T{:, pidx});
else
    Pr = ones(size(F));
end
tbl = table(F, X, Y, Pr, 'VariableNames', {'Frame', 'X', 'Y', 'Probability'});
end
