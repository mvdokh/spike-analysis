from whiskertoolbox_python import DigitalEventSeries

# ── Load data ─────────────────────────────────────────────────────────────────
analog       = dm.getData("c2_filt")           # AnalogTimeSeries (clock: time)
pico_pos     = dm.getData("c2_pico_pos")       # DigitalIntervalSeries - 11 windows (clock: time)
pico_cookie  = dm.getData("c2_pico_pos_cookie") # DigitalIntervalSeries - pico intervals (clock: time)

if analog is None:
    print("ERROR: 'c2_filt' not found.")
elif pico_pos is None:
    print("ERROR: 'c2_pico_pos' not found.")
elif pico_cookie is None:
    print("ERROR: 'c2_pico_pos_cookie' not found.")
else:
    # ── Pull analog into plain Python lists ───────────────────────────────────
    analog_values = analog.toList()
    analog_times  = list(range(len(analog_values)))

    # ── Get the 11 c2_pico_pos windows ───────────────────────────────────────
    c2_windows = [(int(w.start), int(w.end)) for w in pico_pos.toList()]
    print(f"c2_pico_pos windows: {len(c2_windows)}")

    # ── Get pico intervals in time clock ──────────────────────────────────────
    all_pico = pico_cookie.toList()
    print(f"c2_pico_pos_cookie intervals: {len(all_pico)}")

    # ── For each pico interval inside a c2_pico_pos window,
    #    find abs max of c2_filt and create an event ───────────────────────────
    peak_events = DigitalEventSeries()
    found = 0

    for pi in all_pico:
        pi_start = int(pi.start)
        pi_end   = int(pi.end)

        # Check if this pico interval falls within any c2_pico_pos window
        inside = any(pi_start >= ws and pi_end <= we for ws, we in c2_windows)
        if not inside:
            continue

        # Collect (time, value) pairs within this pico interval
        window = [(t, v) for t, v in zip(analog_times, analog_values)
                  if pi_start <= t <= pi_end]

        if not window:
            continue

        # Find the point with the largest absolute value
        peak_time, peak_val = max(window, key=lambda tv: abs(tv[1]))

        peak_events.addEvent(peak_time)
        found += 1
        print(f"  pico [{pi_start}, {pi_end}] -> peak at t={peak_time}, value={peak_val:.4f}")

    # ── Register to Data Manager ───────────────────────────────────────────────
    time_key = dm.getTimeKey("c2_filt")
    dm.setData("peak_deflection", peak_events, time_key)

    print(f"\nDone. Created 'peak_deflection' with {found} events.")