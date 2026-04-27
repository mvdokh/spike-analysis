%% 
%% =========================================================================
%  TeLC Licking Behavior Analysis
%  Analyzes bottom + side camera CSVs for TeLC08, TeLC09, TeLC11
%
%  Metrics computed per session (bottom and side separately):
%    1. Lick rate            (licks / frame, normalised by video frame count)
%    2. Mean lick duration   (frames)
%    3. Mean interlick interval (ILI, frames between consecutive licks)
%    4. Bout rate            (bouts / frame, normalised)
%    5. Licks per bout       (mean across bouts)
%    6. Mean tongue area     (pixels^2)
%
%  Plots saved as high-DPI SVG to output_dir:
%    - Per-animal time-series  (Pre + each Post session, per view)
%    - Summary bar plots       (Pre vs Post mean+SEM across animals, per view)
%    - Recovery curves         (Post sessions normalised to Pre, all animals)
%
%  OUTPUT: C:\Users\wanglab\Desktop\Tongue-Whisker-Analysis\TeLC\output
% =========================================================================

clear; clc; close all;

%% -------------------------------------------------------------------------
%  OUTPUT DIRECTORY
%% -------------------------------------------------------------------------
output_dir = 'C:\Users\wanglab\Desktop\Tongue-Whisker-Analysis\TeLC\output';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% -------------------------------------------------------------------------
%  BUILD FILE TABLE  (animal | condition | session_label | view | csv_path)
%% -------------------------------------------------------------------------
F = build_file_table();

animal_ids    = {'08','09','11'};
views         = {'bottom','side'};
metric_fields = {'lick_rate','lick_dur','ILI','bout_rate','licks_per_bout','tongue_area'};
metric_labels = {'Lick Rate (licks/frame)', ...
                 'Lick Duration (frames)', ...
                 'Interlick Interval (frames)', ...
                 'Bout Rate (bouts/frame)', ...
                 'Licks per Bout', ...
                 'Mean Tongue Area (px^2)'};

colors.Pre  = [0.10 0.35 0.70];   % darker blue
colors.Post = [0.70 0.15 0.12];   % darker red
anim_colors = [0.00 0.45 0.74;    % TeLC08 (blue)
               0.85 0.33 0.10;    % TeLC09 (orange)
               0.49 0.18 0.56];   % TeLC11 (purple)
dark_col   = [0.08 0.08 0.08];

%% -------------------------------------------------------------------------
%  PARSE EVERY CSV AND COMPUTE METRICS
%% -------------------------------------------------------------------------
%  LOAD OR COMPUTE METRICS
%% -------------------------------------------------------------------------
metrics_csv = fullfile(output_dir,'TeLC_licking_metrics.csv');

if isfile(metrics_csv)
    % --- Fast path: metrics already computed, load directly ---
    fprintf('Metrics CSV found — loading from disk, skipping parse & MP4 read:\n    %s\n\n', metrics_csv);
    RT = readtable(metrics_csv, 'VariableNamingRule','preserve');
    % Ensure string columns are cell arrays of char (required for strcmp in plots)
    % 'animal' may be read as numeric (08->8, 09->9, 11->11) — zero-pad back to string
    for col = {'animal','condition','sess_lbl','view'}
        c = col{1};
        if ~ismember(c, RT.Properties.VariableNames), continue; end
        v = RT.(c);
        if iscell(v)
            % already cellstr, leave alone
        elseif isnumeric(v)
            RT.(c) = arrayfun(@(x) sprintf('%02d', x), v, 'UniformOutput', false);
        else
            RT.(c) = cellstr(v);
        end
    end
    % If a session has no licks (empty sheet/CSV), set per-lick metrics to 0
    % rather than NaN so all sessions are represented consistently in plots.
    if all(ismember({'n_licks','lick_dur','ILI','licks_per_bout','tongue_area'}, RT.Properties.VariableNames))
        zero_mask = RT.n_licks == 0;
        RT.lick_dur(zero_mask)       = 0;
        RT.ILI(zero_mask)            = 0;
        RT.licks_per_bout(zero_mask) = 0;
        RT.tongue_area(zero_mask)    = 0;
    end

else
    % --- Slow path: parse every CSV + read MP4 frame counts ---
    nFiles  = height(F);
    rec     = struct([]);   % growing struct array

    for f = 1:nFiles
        csv_path  = F.csv_path{f};
        animal    = F.animal{f};
        condition = F.condition{f};
        sess_lbl  = F.session_label{f};
        vw        = F.view{f};

        fprintf('[%s] [%s] [%s] [%s] ...', animal, condition, sess_lbl, vw);

        % -- Find companion MP4 in same directory --
        csv_dir  = fileparts(csv_path);
        mp4_list = dir(fullfile(csv_dir, '*.mp4'));
        if isempty(mp4_list)
            fprintf(' SKIP (no MP4)\n');
            continue
        end
        n_frames = get_frame_count(fullfile(csv_dir, mp4_list(1).name));
        if n_frames == 0
            fprintf(' SKIP (frame count = 0)\n');
            continue
        end

        % -- Read CSV --
        if ~isfile(csv_path)
            fprintf(' SKIP (CSV missing)\n');
            continue
        end
        T = readtable(csv_path, 'VariableNamingRule', 'preserve');

        cols      = T.Properties.VariableNames;
        idx_area  = find(contains(cols,'Max'),    1);
        idx_start = find(contains(cols,'Start'),  1);
        idx_end   = find(contains(cols,'End'),    1);
        idx_dur   = find(contains(cols,'Duration'),1);
        idx_bout  = find(contains(cols,'Assign'), 1);

        if any([isempty(idx_area),isempty(idx_start),isempty(idx_end),...
                isempty(idx_dur),isempty(idx_bout)])
            fprintf(' SKIP (unexpected columns)\n');
            continue
        end

        area    = T{:,idx_area};
        t_start = T{:,idx_start};
        t_end   = T{:,idx_end};
        dur     = T{:,idx_dur};
        bout_id = T{:,idx_bout};
        n_licks = height(T);

        % Lick rate
        lick_rate = n_licks / n_frames;

        % Mean lick duration
        mean_dur = mean(dur, 'omitnan');
        if isempty(dur) || isnan(mean_dur), mean_dur = 0; end

        % ILI: start(i+1) - end(i) for consecutive licks, positive only
        if n_licks > 1
            ili_vals = t_start(2:end) - t_end(1:end-1);
            ili_vals = ili_vals(ili_vals > 0);
            mean_ILI = mean(ili_vals, 'omitnan');
            if isempty(ili_vals) || isnan(mean_ILI), mean_ILI = 0; end
        elseif n_licks == 1
            mean_ILI = 0;
        else
            mean_ILI = 0;
        end

        % Bout metrics
        unique_bouts = unique(bout_id);
        n_bouts      = numel(unique_bouts);
        bout_rate    = n_bouts / n_frames;
        if n_bouts > 0
            lpb_vals  = arrayfun(@(b) sum(bout_id==b), unique_bouts);
            mean_lpb  = mean(lpb_vals, 'omitnan');
            if isempty(lpb_vals) || isnan(mean_lpb), mean_lpb = 0; end
        else
            mean_lpb  = 0;
        end

        % Tongue area
        mean_area = mean(area, 'omitnan');
        if isempty(area) || isnan(mean_area), mean_area = 0; end

        e.animal        = animal;
        e.condition     = condition;
        e.sess_lbl      = sess_lbl;
        e.view          = vw;
        e.n_frames      = n_frames;
        e.n_licks       = n_licks;
        e.lick_rate     = lick_rate;
        e.lick_dur      = mean_dur;
        e.ILI           = mean_ILI;
        e.bout_rate     = bout_rate;
        e.licks_per_bout= mean_lpb;
        e.tongue_area   = mean_area;

        if isempty(rec), rec = e; else, rec(end+1) = e; end %#ok<AGROW>
        fprintf(' done  (frames=%d, licks=%d)\n', n_frames, n_licks);
    end

    if isempty(rec)
        error('No data was loaded. Check that CSVs and MP4s are present.');
    end

    RT = struct2table(rec);
    writetable(RT, metrics_csv);
    fprintf('\nMetrics table saved: TeLC_licking_metrics.csv\n\n');
end

%% =========================================================================
%  PLOT 1 — PER-ANIMAL TIME-SERIES (Pre + all Post sessions)
%% =========================================================================
for vi = 1:numel(views)
    vw   = views{vi};
    vsub = RT(strcmp(RT.view,vw),:);

    for ai = 1:numel(animal_ids)
        anim = animal_ids{ai};
        asub = vsub(strcmp(vsub.animal,anim),:);
        if isempty(asub), continue; end

        pre_rows  = asub(strcmp(asub.condition,'Pre'), :);
        post_rows = sortrows(asub(strcmp(asub.condition,'Post'),:),'sess_lbl');
        all_rows  = [pre_rows; post_rows];
        n_sess    = height(all_rows);
        if n_sess == 0, continue; end

        x_vals   = 1:n_sess;
        x_labels = cell(n_sess,1);
        post_ctr = 0;
        for r = 1:n_sess
            if strcmp(all_rows.condition{r},'Pre')
                x_labels{r} = 'Pre';
            else
                post_ctr = post_ctr + 1;
                if post_ctr == 1
                    x_labels{r} = 'Post';
                else
                    x_labels{r} = sprintf('Post + %d', post_ctr-1);
                end
            end
        end
        pt_col   = zeros(n_sess,3);
        for r = 1:n_sess
            if strcmp(all_rows.condition{r},'Pre')
                pt_col(r,:) = colors.Pre;
            else
                pt_col(r,:) = colors.Post;
            end
        end

        fig = figure('Visible','off','Units','inches','Position',[0 0 14 8],...
                     'Color','w');
        tlo = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
        title(tlo, sprintf('TeLC%s | %s view — Session Time-Series', anim, upper(vw)),...
              'FontSize',13,'FontWeight','bold');

        for mi = 1:numel(metric_fields)
            nexttile;
            vals = all_rows.(metric_fields{mi});
            valid = ~isnan(vals);
            hold on;
            plot(x_vals(valid), vals(valid), '-','Color',[0.25 0.25 0.25],'LineWidth',1.2);
            scatter(x_vals(valid), vals(valid), 65, pt_col(valid,:),'filled',...
                    'MarkerEdgeColor','k','LineWidth',0.5);
            if any(~valid)
                yl = ylim;
                if yl(1) == yl(2), yl = [yl(1)-1 yl(2)+1]; end
                y_miss = yl(1) + 0.05*(yl(2)-yl(1));
                scatter(x_vals(~valid), y_miss*ones(sum(~valid),1), 55, ...
                        [0.10 0.10 0.10], 'x', 'LineWidth',1.2);
            end
            set(gca,'XTick',x_vals,'XTickLabel',x_labels,...
                    'XTickLabelRotation',40,'FontSize',8,...
                    'Box','off','TickDir','out','Color','w',...
                    'XColor',dark_col,'YColor',dark_col);
            ylabel(metric_labels{mi},'FontSize',9,'Color',dark_col);
            title(metric_labels{mi},'FontSize',9,'FontWeight','bold','Color',dark_col);
            xlim([0.4 n_sess+0.6]);
        end

        fname = fullfile(output_dir, sprintf('01_TeLC%s_%s_timeseries.svg',anim,vw));
        save_svg(fig,fname);
        close(fig);
        fprintf('Saved: %s\n', fname);
    end
end

%% =========================================================================
%  PLOT 2 — SUMMARY BAR (Pre vs Post, mean+SEM across animals)
%% =========================================================================
for vi = 1:numel(views)
    vw   = views{vi};
    vsub = RT(strcmp(RT.view,vw),:);
    legend_handles = gobjects(numel(animal_ids),1);

    pre_mat  = nan(numel(animal_ids), numel(metric_fields));
    post_mat = nan(numel(animal_ids), numel(metric_fields));

    for ai = 1:numel(animal_ids)
        anim = animal_ids{ai};
        asub = vsub(strcmp(vsub.animal,anim),:);
        for mi = 1:numel(metric_fields)
            m = metric_fields{mi};
            pre_v  = asub.(m)(strcmp(asub.condition,'Pre'));
            post_v = asub.(m)(strcmp(asub.condition,'Post'));
            pre_mat(ai,mi)  = mean(pre_v, 'omitnan');
            post_mat(ai,mi) = mean(post_v, 'omitnan');
        end
    end

    fig = figure('Visible','off','Units','inches','Position',[0 0 14 8],...
                 'Color','w');
    tlo = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
    tlo.Position = [0.05 0.14 0.90 0.78];
    title(tlo, sprintf('TeLC Summary — %s view  (mean ± SEM, n=%d)', ...
          upper(vw), numel(animal_ids)),'FontSize',13,'FontWeight','bold');

    for mi = 1:numel(metric_fields)
        nexttile; hold on;
        line_h = gobjects(numel(animal_ids),1);
        pm  = mean(pre_mat(:,mi), 'omitnan');
        pom = mean(post_mat(:,mi), 'omitnan');
        ps  = std(pre_mat(:,mi),  0, 'omitnan') / sqrt(sum(~isnan(pre_mat(:,mi))));
        pos = std(post_mat(:,mi), 0, 'omitnan') / sqrt(sum(~isnan(post_mat(:,mi))));
        pre_vals  = pre_mat(:,mi);
        post_vals = post_mat(:,mi);

        b = bar([1 2],[pm pom],0.50,'FaceColor','flat','EdgeColor',dark_col,'LineWidth',1.2);
        b.CData(1,:) = mix_with_white(colors.Pre, 0.60);
        b.CData(2,:) = mix_with_white(colors.Post, 0.60);
        errorbar([1 2],[pm pom],[ps pos],'k.','LineWidth',1.6,'CapSize',8);
        scatter([1 2],[pm pom],70,'k','filled','d');

        for ai = 1:numel(animal_ids)
            xj = [1+randn*0.04  2+randn*0.04];
            line_h(ai) = plot(xj, [pre_vals(ai) post_vals(ai)],'-o',...
                              'Color',anim_colors(ai,:),...
                              'MarkerFaceColor',anim_colors(ai,:),...
                              'MarkerSize',6,'LineWidth',1.2,...
                              'DisplayName',sprintf('TeLC%s',animal_ids{ai}));
        end

        set(gca,'XTick',[1 2],'XTickLabel',{'Pre','Post'},...
                'FontSize',10,'Box','off','TickDir','out','Color','w',...
                'XColor',dark_col,'YColor',dark_col);
        ylabel(metric_labels{mi},'FontSize',10,'Color',dark_col);
        title(metric_labels{mi},'FontSize',10,'FontWeight','bold','Color',dark_col);
        xlim([0.35 2.65]);
        if mi == 1
            legend_handles = line_h;
        end
    end

    % One figure-level legend (compatible with older MATLAB releases that
    % do not accept TiledChartLayout handles in legend()).
    ax_leg = axes(fig, 'Position',[0.15 0.01 0.70 0.06], 'Visible','off', 'Color','w');
    hleg = legend(ax_leg, legend_handles, strcat('TeLC', animal_ids), ...
                  'Orientation','horizontal', 'NumColumns',numel(animal_ids), ...
                  'FontSize',9, 'Box','off', 'Location','north');
    set(hleg, 'TextColor', dark_col);

    fname = fullfile(output_dir, sprintf('02_TeLC_summary_%s.svg',vw));
    save_svg(fig,fname);
    close(fig);
    fprintf('Saved: %s\n', fname);
end

%% =========================================================================
%  PLOT 3 — POST-INJECTION RECOVERY CURVES (normalised to Pre)
%% =========================================================================
for vi = 1:numel(views)
    vw   = views{vi};
    vsub = RT(strcmp(RT.view,vw),:);

    fig = figure('Visible','off','Units','inches','Position',[0 0 14 9],...
                 'Color','w');
    tlo = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
    title(tlo, sprintf('Post-Injection Recovery — %s view  (normalised to Pre)',...
          upper(vw)),'FontSize',13,'FontWeight','bold');

    for mi = 1:numel(metric_fields)
        m = metric_fields{mi};
        nexttile; hold on;
        ylbl = metric_labels{mi};   % default; overwritten to fold-Pre if normalisation applies

        for ai = 1:numel(animal_ids)
            anim = animal_ids{ai};
            asub = vsub(strcmp(vsub.animal,anim),:);

            pre_val  = mean(asub.(m)(strcmp(asub.condition,'Pre')), 'omitnan');
            post_sub = sortrows(asub(strcmp(asub.condition,'Post'),:),'sess_lbl');
            if isempty(post_sub), continue; end

            post_v = post_sub.(m);
            px     = 1:numel(post_v);

            if ~isnan(pre_val) && pre_val ~= 0
                y_plot = post_v / pre_val;
                yline(1,'--','Color',anim_colors(ai,:),'LineWidth',0.8,'Alpha',0.4);
                ylbl = [metric_labels{mi} '  (fold Pre)'];
            else
                y_plot = post_v;
            end

            plot(px, y_plot,'-o','Color',anim_colors(ai,:),...
                 'MarkerFaceColor',anim_colors(ai,:),'LineWidth',1.8,...
                 'MarkerSize',7,'DisplayName',sprintf('TeLC%s',anim));
        end

        set(gca,'FontSize',9,'Box','off','TickDir','out','Color','w',...
                'XColor',dark_col,'YColor',dark_col);
        xlabel('Post session #','FontSize',9,'Color',dark_col);
        ylabel(ylbl,'FontSize',9,'Color',dark_col);
        title(metric_labels{mi},'FontSize',10,'FontWeight','bold','Color',dark_col);
        if mi == 1
            legend('show','Location','best','FontSize',9,'Box','off');
        end
    end

    fname = fullfile(output_dir, sprintf('03_TeLC_recovery_%s.svg',vw));
    save_svg(fig,fname);
    close(fig);
    fprintf('Saved: %s\n', fname);
end

fprintf('\n=== Analysis complete. All SVGs saved to:\n    %s\n', output_dir);

%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================

function n = get_frame_count(mp4_path)
% Extract total frame count from an MP4 file via VideoReader.
    n = 0;
    try
        v = VideoReader(mp4_path);
        try
            n = v.NumFrames;          % fast path (R2021a+)
        catch
            n = round(v.Duration * v.FrameRate);
        end
    catch ME
        warning('Could not read frame count from %s:\n  %s', mp4_path, ME.message);
    end
end

function save_svg(fig, fpath)
% Save figure as a vector SVG using the painters renderer.
    set(fig,'PaperUnits','inches','PaperPositionMode','auto');
    print(fig, fpath, '-dsvg', '-painters', '-r300');
end

function draw_violin(ax, vals, x0, face_col, half_w)
% Draw a simple violin using kernel density (fallback to jitter if needed).
    vals = vals(~isnan(vals));
    if isempty(vals), return; end
    hold(ax,'on');
    edge_col = max(face_col * 0.55, 0);
    fill_col = mix_with_white(face_col, 0.80);
    if numel(unique(vals)) < 2
        scatter(ax, x0 + randn(size(vals))*0.02, vals, 16, ...
                'MarkerFaceColor',fill_col, 'MarkerEdgeColor',edge_col, ...
                'LineWidth',1.0);
        return;
    end
    try
        [f, yi] = ksdensity(vals);
        if max(f) <= 0, return; end
        f = f ./ max(f) * half_w;
        patch(ax, [x0-f, fliplr(x0+f)], [yi, fliplr(yi)], fill_col, ...
              'EdgeColor',edge_col, 'LineWidth',1.5);
    catch
        scatter(ax, x0 + randn(size(vals))*0.02, vals, 16, ...
                'MarkerFaceColor',fill_col, 'MarkerEdgeColor',edge_col, ...
                'LineWidth',1.0);
    end
end

function c = mix_with_white(c0, keep)
% Blend toward white while keeping strong visibility.
    c = keep .* c0 + (1-keep) .* [1 1 1];
end

function F = build_file_table()
% Returns a table with metadata parsed from the hardcoded path list.

    raw = { ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Pre\IRt_TeLC08_pre_2026_03_31_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Pre\IRt_TeLC08_pre_2026_03_31_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_05\IRt_TeLC08_post_2026_04_05_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_05\IRt_TeLC08_post_2026_04_05_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_06\IRt_TeLC08_post_2026_04_06_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_06\IRt_TeLC08_post_2026_04_06_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_07\IRt_TeLC08_post_2026_04_07_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_07\IRt_TeLC08_post_2026_04_07_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_08\IRt_TeLC08_post_2026_04_08_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_08\IRt_TeLC08_post_2026_04_08_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_09\IRt_TeLC08_post_2026_04_09_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_09\IRt_TeLC08_post_2026_04_09_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_10\IRt_TeLC08_post_2026_04_10_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08_Post\IRt_TeLC08_post_2026_04_10\IRt_TeLC08_post_2026_04_10_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Pre\IRt_TeLC09_pre_2026_04_01_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Pre\IRt_TeLC09_pre_2026_04_01_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_06\IRt_TeLC09_post_2026_04_06_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_06\IRt_TeLC09_post_2026_04_06_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_07\IRt_TeLC09_post_2026_04_07real_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_07\IRt_TeLC09_post_2026_04_07real_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_08\IRt_TeLC09_post_2026_04_08_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_08\IRt_TeLC09_post_2026_04_08_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_09\IRt_TeLC09_post_2026_04_09_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC09_Post\IRt_TeLC09_post_2026_04_09\IRt_TeLC09_post_2026_04_09_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Pre\IRt_TeLC11_pre_2026_03_30_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Pre\IRt_TeLC11_pre_2026_03_30_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_05\IRt_TeLC11_post_2026_04_05_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_05\IRt_TeLC11_post_2026_04_05_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_06\IRt_TeLC11_post_2026_04_06_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_06\IRt_TeLC11_post_2026_04_06_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_07\IRt_TeLC11_post_2026_04_07_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_07\IRt_TeLC11_post_2026_04_07_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_08\IRt_TeLC11_post_2026_04_08_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_08\IRt_TeLC11_post_2026_04_08_side_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_09\IRt_TeLC11_post_2026_04_09_bottom_behavior_100_3.csv'; ...
        'C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC11_Post\IRt_TeLC11_post_2026_04_09\IRt_TeLC11_post_2026_04_09_side_behavior_100_3.csv'; ...
    };

    n         = numel(raw);
    animal    = cell(n,1);
    condition = cell(n,1);
    sess_lbl  = cell(n,1);
    view      = cell(n,1);

    for i = 1:n
        p = raw{i};

        % Animal
        tok = regexp(p,'TeLC(\d+)','tokens','once');
        animal{i} = tok{1};

        % Condition
        if contains(p,'_Pre','IgnoreCase',true)
            condition{i} = 'Pre';
        else
            condition{i} = 'Post';
        end

        % Session label — use last date string in path (YYYY_MM_DD)
        dtok = regexp(p,'(\d{4}_\d{2}_\d{2})','tokens');
        if strcmp(condition{i},'Pre')
            sess_lbl{i} = 'Pre';
        elseif ~isempty(dtok)
            sess_lbl{i} = dtok{end}{1};   % last date match = session date
        else
            sess_lbl{i} = 'Post';
        end

        % View
        if contains(p,'_bottom_')
            view{i} = 'bottom';
        else
            view{i} = 'side';
        end
    end

    F = table(animal, condition, sess_lbl, view, raw, ...
              'VariableNames',{'animal','condition','session_label','view','csv_path'});
end