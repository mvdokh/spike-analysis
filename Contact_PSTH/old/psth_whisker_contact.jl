#!/usr/bin/env julia
# PSTH aligned to whisker contact interval onsets, per unit.
# Usage: julia psth_whisker_contact.jl [interval_csv] [spikes_csv]
# Default paths: interval_0_whisker_contact.csv, spikes.csv on Desktop.

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
                # bin i = [t_edges[i], t_edges[i+1])
                idx = searchsortedlast(t_edges, rel)
                idx = clamp(idx, 1, n_bins)
                counts[idx] += 1
            end
        end
    end

    # Rate in spikes/s: count per bin / (n_trials * bin_s)
    rate = counts ./ (n_trials * bin_s)
    (; bin_centers, rate, counts, n_trials)
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
        plt = plot(p.bin_centers, p.rate; label="Unit $unit",
                   xlabel="Time from contact onset (s)",
                   ylabel="Firing rate (spikes/s)",
                   title="PSTH — Unit $unit (n=$(p.n_trials) intervals)")
        vline!([0.0]; label="", color=:black, ls=:dash)
        savefig(plt, joinpath(dirname(interval_path), "psth_unit_$(unit).png"))
        println("Saved psth_unit_$(unit).png")
    end
end

main()
