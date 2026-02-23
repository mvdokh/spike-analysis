#!/usr/bin/env julia
# PSTH + raster aligned to whisker contact interval onsets, per unit.
# Usage: julia psth_raster_whisker_contact.jl [interval_csv] [spikes_csv]

using CSV
using DataFrames
using Plots
using Statistics

const FPS = 350.0
const BIN_S = 0.01   # 10 ms bins
const T_PRE = 0.1    # seconds before interval start
const T_POST = 0.3   # seconds after interval start

function load_intervals(path)
    df = CSV.read(path, DataFrame; types=Dict("Start" => Int, "End" => Int))
    df[!, :start_s] = df.Start ./ FPS
    df[!, :end_s] = df.End ./ FPS
    df
end

function load_spikes(path)
    df = CSV.read(path, DataFrame; header=["time", "unit", "ignore"],
                 types=[Float64, Int, Int], delim=',', ignorerepeated=true)
    df
end

function psth_for_unit(spikes, intervals, unit; bin_s=BIN_S, t_pre=T_PRE, t_post=T_POST)
    t_edges = (-t_pre):bin_s:(t_post + 1e-9)
    n_bins = length(t_edges) - 1
    bin_centers = (t_edges[1:end-1] .+ t_edges[2:end]) ./ 2
    unit_spikes = spikes[spikes.unit .== unit, :].time
    n_trials = nrow(intervals)
    counts = zeros(n_bins)

    for row in eachrow(intervals)
        t0 = row.start_s
        for t in unit_spikes
            rel = t - t0
            if -t_pre <= rel < t_post
                idx = searchsortedlast(t_edges, rel)
                idx = clamp(idx, 1, n_bins)
                counts[idx] += 1
            end
        end
    end

    rate = counts ./ (n_trials * bin_s)
    (; bin_centers, rate, counts, n_trials)
end

"""Return (trial_indices, relative_times) for raster: one spike per entry."""
function raster_for_unit(spikes, intervals, unit; t_pre=T_PRE, t_post=T_POST)
    unit_spikes = spikes[spikes.unit .== unit, :].time
    rel_times = Float64[]
    trial_inds = Int[]

    for (i, row) in enumerate(eachrow(intervals))
        t0 = row.start_s
        for t in unit_spikes
            rel = t - t0
            if -t_pre <= rel < t_post
                push!(rel_times, rel)
                push!(trial_inds, i)
            end
        end
    end

    (; trial_inds, rel_times)
end

function main()
    base = "/Users/martindokholyan/Desktop"
    interval_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(base, "interval_0_whisker_contact.csv")
    spikes_path   = length(ARGS) >= 2 ? ARGS[2] : joinpath(base, "spikes.csv")

    intervals = load_intervals(interval_path)
    spikes = load_spikes(spikes_path)
    units = sort(unique(spikes.unit))

    for unit in units
        p = psth_for_unit(spikes, intervals, unit)
        r = raster_for_unit(spikes, intervals, unit)

        plt = plot(layout=(2, 1), size=(600, 500), link=:x,
                   xlims=(-T_PRE, T_POST))

        # Top: PSTH
        plot!(plt[1], p.bin_centers, p.rate; label="Unit $unit",
              xlabel="", ylabel="Rate (spikes/s)",
              title="PSTH — Unit $unit (n=$(p.n_trials) intervals)")
        vline!(plt[1], [0.0]; label="", color=:black, ls=:dash)

        # Bottom: Raster (vertical tick per spike)
        if !isempty(r.rel_times)
            tick_half = 0.4
            n = length(r.rel_times)
            x_seg = Vector{Float64}(undef, 3n)
            y_seg = Vector{Float64}(undef, 3n)
            for (i, (t, tr)) in enumerate(zip(r.rel_times, r.trial_inds))
                j = 3(i - 1) + 1
                x_seg[j] = x_seg[j+1] = t; x_seg[j+2] = NaN
                y_seg[j] = tr - tick_half; y_seg[j+1] = tr + tick_half; y_seg[j+2] = NaN
            end
            plot!(plt[2], x_seg, y_seg; linecolor=:black, linewidth=1, label="")
        end
        plot!(plt[2]; xlabel="Time from contact onset (s)", ylabel="Trial",
              title="Raster", ylims=(0.5, p.n_trials + 0.5))
        vline!(plt[2], [0.0]; label="", color=:black, ls=:dash)

        out_dir = dirname(interval_path)
        savefig(plt, joinpath(out_dir, "psth_raster_unit_$(unit).png"))
        println("Saved psth_raster_unit_$(unit).png")
    end
end

main()
