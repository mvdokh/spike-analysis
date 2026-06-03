function paths = telc_pre_side_jaw_paths(animalRoot)
% telc_pre_side_jaw_paths  Side-view (*_1_jaw.csv) Pre sessions for TeLC animals.
%
% Layout: <animalRoot>/<IRt_TeLC08>/IRt_TeLC08_Pre/*_1_jaw.csv
% paths = telc_pre_side_jaw_paths() uses default Ina root.

if nargin < 1 || isempty(animalRoot)
    animalRoot = 'C:\Users\wanglab\Desktop\Ina\IRt_TeLC';
end

animals = {'IRt_TeLC08', 'IRt_TeLC09', 'IRt_TeLC11'};
paths = cell(0, 1);

for i = 1:numel(animals)
    preDir = fullfile(animalRoot, animals{i}, [animals{i} '_Pre']);
    if ~isfolder(preDir)
        warning('Missing Pre folder: %s', preDir);
        continue
    end
    listing = dir(fullfile(preDir, '*_1_jaw.csv'));
    listing = listing(~[listing.isdir]);
    if isempty(listing)
        warning('No side-view jaw CSV (*_1_jaw.csv) in: %s', preDir);
        continue
    end
    if numel(listing) > 1
        [~, ord] = sort({listing.name});
        listing = listing(ord(1));
        warning('Multiple *_1_jaw.csv in %s; using %s', preDir, listing(1).name);
    end
    paths{end + 1, 1} = fullfile(preDir, listing(1).name); %#ok<AGROW>
end
end
