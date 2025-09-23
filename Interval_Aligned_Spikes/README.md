# Spike Analysis with Custom Time Ranges

This directory contains utilities for analyzing spike data with custom time range specifications.

## Files

- `aligned_utils.py` - Utility functions for spike analysis
- `spike_analysis_custom.ipynb` - Interactive Jupyter notebook for custom analysis

## Key Features

### aligned_utils.py Functions:

1. **`load_data()`** - Loads all data files (spikes, pico intervals, electrode config)

2. **`create_raster_plot_custom(spikes_df, pico_df, time_start, time_end, electrode_subset, save_path)`**
   - Create raster plot for specific time range
   - Optional electrode filtering
   - Optional save to file

3. **`quick_raster(time_start, time_end, electrodes=None)`** - Quick raster plot function

4. **`get_time_range_info(spikes_df=None)`** - Get available time range information

5. **`run_full_analysis(time_start=None, time_end=None, save_plots=True)`** - Complete analysis with custom time window

## Usage Examples

### Python Script Usage:
```python
from aligned_utils import load_data, create_raster_plot_custom

# Load data
spikes_df, pico_df, electrode_df = load_data()

# Create raster plot for first 60 seconds
fig = create_raster_plot_custom(spikes_df, pico_df, 12, 72, save_path='first_60s.png')

# Quick raster plot
from aligned_utils import quick_raster
fig = quick_raster(100, 200)  # 100-200 seconds
```

### Jupyter Notebook Usage:
Open `spike_analysis_custom.ipynb` for interactive analysis with:
- Custom time range specification
- Electrode schematic visualization
- Firing rate heatmaps  
- Activity window analysis
- Interactive plotting

## Time Range Format
- All times are in seconds (timestamps divided by 30,000)
- Available range: ~12s to ~909s (total ~15 minutes)
- Pico intervals are automatically included and shaded in red

## Electrode Information
- 32 total electrodes in config
- 20 electrodes show activity in the data
- Left hemisphere: negative x-coordinates
- Right hemisphere: positive x-coordinates