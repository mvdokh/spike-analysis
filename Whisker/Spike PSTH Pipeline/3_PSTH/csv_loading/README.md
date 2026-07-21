# PSTH & Raster Plot Analysis

This directory contains tools for creating Peri-Stimulus Time Histogram (PSTH) and raster plots.

## Files

- `psth_raster_analysis.py` - Main script with all analysis functions and print statements
- `psth_interactive_analysis.ipynb` - Jupyter notebook for interactive parameter adjustment

## Data Files Used

- `spikes_time_adjusted.csv` - Contains: time, unit, electrode (electrode column ignored)
- `semi_from_table_designer.csv` - Contains: interval start time, interval end time, interval duration (±4ms tolerance)

## Usage

### Main Script
```python
from psth_raster_analysis import run_psth_analysis

# Run analysis with parameters
fig, trial_data = run_psth_analysis(
    unit=28,           # Which unit to analyze
    duration=750,      # Interval duration (750=20Hz, 300=50Hz, 150=100Hz)
    bin_size_ms=5,     # Bin size in milliseconds
    max_trials=100,    # Maximum trials to plot (optional)
    start_time=None,   # Start time filter (optional)
    end_time=None      # End time filter (optional)
)
```

### Jupyter Notebook
Open `psth_interactive_analysis.ipynb` and modify the parameters in each cell to explore different units and conditions.

## Parameters

1. **unit**: Which unit to plot (available units shown when loading data)
2. **duration**: Pico interval duration to analyze:
   - 750ms → 20 Hz
   - 300ms → 50 Hz  
   - 150ms → 100 Hz
3. **bin_size_ms**: Number of milliseconds per bin for PSTH
4. **start_time/end_time**: Time frame for filtering trials (optional)
5. **max_trials**: Maximum number of trials to display in raster plot

## Output

- **PSTH**: Shows firing rate over time with frequency in title
- **Raster Plot**: Shows individual spike times for each trial
- **Console**: Detailed information about data loading and analysis

Create a PSTH and raster plot. this should have a main script in the 3_PSTH folder and a jupyter notebook where I can run and change parameters. I dont want widgets, I just want to specify in the cell the parameters. make it as short as possible, and put all print statements in the main script

I should be able to select 
    1. Which unit to plot
    2. Number of spikes/bin
    3. Which pico interval duration (i.e. 750 (25 ms), 300 (10 ms), 150 (5 ms)) to plot.
    4. Time frame for the trials (i.e. intervals within these times)

This should show
    1. PSTH with proper interval duration plotted (Also make the header change based on interval duration)
        750 duration = 20 hz, 300 duration = 50 hz, 150 duration = 100 hz
    2. Raster plot with maximum trials plotted

