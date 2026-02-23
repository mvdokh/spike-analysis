#!/usr/bin/env Rscript
# PSTH + raster aligned to whisker contact interval onsets, per unit.
# Usage: Rscript psth_raster_whisker_contact.R [interval_csv] [spikes_csv]

FPS       <- 350
BIN_S     <- 0.01
T_PRE     <- 0.1
T_POST    <- 0.3

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
  t_edges     <- seq(-t_pre, t_post, by = bin_s)
  n_bins      <- length(t_edges) - 1L
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

raster_for_unit <- function(spikes, intervals, unit, t_pre = T_PRE, t_post = T_POST) {
  unit_spikes <- spikes[spikes$unit == unit, "time"]
  rel_times   <- numeric(0)
  trial_inds  <- integer(0)

  for (i in seq_len(nrow(intervals))) {
    t0 <- intervals$start_s[i]
    for (t in unit_spikes) {
      rel <- t - t0
      if (rel >= -t_pre && rel < t_post) {
        rel_times  <- c(rel_times, rel)
        trial_inds <- c(trial_inds, i)
      }
    }
  }

  list(rel_times = rel_times, trial_inds = trial_inds)
}

main <- function() {
  base <- "/Users/martindokholyan/Desktop"
  args <- commandArgs(trailingOnly = TRUE)
  interval_path <- if (length(args) >= 1) args[1] else file.path(base, "interval_0_whisker_contact.csv")
  spikes_path   <- if (length(args) >= 2) args[2] else file.path(base, "spikes.csv")

  intervals <- load_intervals(interval_path)
  spikes    <- load_spikes(spikes_path)
  units     <- sort(unique(spikes$unit))
  out_dir   <- dirname(interval_path)

  for (unit in units) {
    p <- psth_for_unit(spikes, intervals, unit)
    r <- raster_for_unit(spikes, intervals, unit)

    png(file.path(out_dir, paste0("psth_raster_unit_", unit, ".png")),
        width = 640, height = 560, res = 120)
    par(mfrow = c(2, 1), mar = c(0, 4, 3, 2), oma = c(4, 0, 0, 0))

    # Top: PSTH
    plot(p$bin_centers, p$rate, type = "l", col = "steelblue", lwd = 2,
         xlab = "", xaxt = "n", ylab = "Rate (spikes/s)",
         main = sprintf("PSTH — Unit %s (n=%d intervals)", unit, p$n_trials))
    abline(v = 0, lty = 2, col = "black")

    # Bottom: Raster
    if (length(r$rel_times) > 0) {
      plot(r$rel_times, r$trial_inds, pch = "|", col = "black", cex = 0.6,
           xlab = "Time from contact onset (s)", ylab = "Trial",
           main = "Raster", xlim = c(-T_PRE, T_POST),
           ylim = c(0.5, p$n_trials + 0.5))
    } else {
      plot(NA, xlim = c(-T_PRE, T_POST), ylim = c(0.5, p$n_trials + 0.5),
           xlab = "Time from contact onset (s)", ylab = "Trial", main = "Raster")
    }
    abline(v = 0, lty = 2, col = "black")

    dev.off()
    message("Saved psth_raster_unit_", unit, ".png")
  }
}

main()
