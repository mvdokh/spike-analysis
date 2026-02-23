#!/usr/bin/env Rscript
# PSTH aligned to whisker contact interval onsets, per unit.
# Usage: Rscript psth_whisker_contact.R [interval_csv] [spikes_csv]
# Default paths: interval_0_whisker_contact.csv, spikes.csv on Desktop.

FPS       <- 350
BIN_S     <- 0.01   # 10 ms bins
T_PRE     <- 0.1    # seconds before interval start
T_POST    <- 0.3    # seconds after interval start

load_intervals <- function(path) {
  d <- read.csv(path, colClasses = c("integer", "integer"))
  d$start_s <- d$Start / FPS
  d$end_s   <- d$End / FPS
  d
}

load_spikes <- function(path) {
  d <- read.csv(path, header = FALSE, col.names = c("time", "unit", "ignore"),
                colClasses = c("numeric", "integer", "NULL"), strip.white = TRUE)
  d
}

psth_for_unit <- function(spikes, intervals, unit,
                          bin_s = BIN_S, t_pre = T_PRE, t_post = T_POST) {
  t_edges   <- seq(-t_pre, t_post, by = bin_s)
  n_bins    <- length(t_edges) - 1L
  bin_centers <- (t_edges[-length(t_edges)] + t_edges[-1]) / 2
  unit_spikes <- spikes[spikes$unit == unit, "time"]
  n_trials    <- nrow(intervals)
  counts      <- numeric(n_bins)

  for (i in seq_len(n_trials)) {
    t0 <- intervals$start_s[i]
    for (t in unit_spikes) {
      rel <- t - t0
      if (rel >= -t_pre && rel < t_post) {
        idx <- findInterval(rel, t_edges)
        idx <- max(1L, min(idx, n_bins))
        counts[idx] <- counts[idx] + 1L
      }
    }
  }

  rate <- counts / (n_trials * bin_s)
  list(bin_centers = bin_centers, rate = rate, counts = counts, n_trials = n_trials)
}

main <- function() {
  base <- "/Users/martindokholyan/Desktop"
  args <- commandArgs(trailingOnly = TRUE)
  interval_path <- if (length(args) >= 1) args[1] else file.path(base, "interval_0_whisker_contact.csv")
  spikes_path   <- if (length(args) >= 2) args[2] else file.path(base, "spikes.csv")

  intervals <- load_intervals(interval_path)
  spikes    <- load_spikes(spikes_path)
  units     <- sort(unique(spikes$unit))

  out_dir <- dirname(interval_path)

  for (unit in units) {
    p <- psth_for_unit(spikes, intervals, unit)
    png(file.path(out_dir, paste0("psth_unit_", unit, ".png")),
        width = 640, height = 480, res = 120)
    plot(p$bin_centers, p$rate, type = "l", col = "steelblue", lwd = 2,
         xlab = "Time from contact onset (s)",
         ylab = "Firing rate (spikes/s)",
         main = sprintf("PSTH — Unit %s (n=%d intervals)", unit, p$n_trials))
    abline(v = 0, lty = 2, col = "black")
    dev.off()
    message("Saved psth_unit_", unit, ".png")
  }
}

main()
