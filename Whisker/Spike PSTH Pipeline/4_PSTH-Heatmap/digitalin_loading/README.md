# PSTH Heatmap Analysis with digitalin.dat Support

This folder contains the PSTH heatmap analysis tools with support for loading interval data directly from `digitalin.dat` files instead of CSV files.

## Overview

The PSTH heatmap analysis creates 2D visualizations that combine firing rate data from multiple units:
- **Y-axis**: Units (sorted in ascending order)
- **X-axis**: Time relative to interval start (ms)
- **Color**: Firing rate (Hz)

This version extends the original heatmap analysis to support loading interval data from binary digital input files (`digitalin.dat`).

## Files

- `psth_heatmap_analysis.ipynb` - Main Jupyter notebook for interactive analysis
- `heatmap_utils.py` - Core functions for heatmap generation
- `digitalin_loader.py` - Binary digital input file loader (copied from 3_PSTH/digitalin_loading)

## Data Sources

### digitalin.dat Loading (Recommended)
- **File**: `digitalin.dat` - Binary digital input file with TTL events
- **Channel 0**: Pico intervals
- **Channel 1**: Time markers (optional)
- **Advantages**: Direct from acquisition hardware, no preprocessing needed

### CSV Loading (Fallback)
- **File**: `pico_time_adjust.csv` - Preprocessed interval data
- **Advantages**: Human-readable, can be manually edited

## Usage

### 1. Jupyter Notebook (Recommended)

Open `psth_heatmap_analysis.ipynb` and configure the parameters:

```python
# Data source selection
use_digitalin = True              # Set to True for digitalin.dat, False for CSV

# Data files
spikes_file = '../../Data/040425/spikes.csv'
digitalin_file = '../../Data/040425/digitalin.dat'    # When use_digitalin=True
intervals_file = '../../Data/040425/pico_time_adjust.csv'  # When use_digitalin=False
sampling_rate = 30000             # Sampling rate for digitalin.dat

# Analysis parameters
durations_ms = [5, 10, 25]        # Interval durations to analyze (ms)
units = None                      # Units to include (None = all units)
bin_size_ms = 0.1                 # Bin size in milliseconds
pre_interval_ms = 5               # Time before interval start (ms)
post_interval_ms = 10             # Time after interval start (ms)
smooth_window = 5                 # Smoothing window (bins)
```

### 2. Python Script

```python
from heatmap_utils import create_multiple_duration_heatmaps

# Create heatmaps using digitalin.dat
results = create_multiple_duration_heatmaps(
    spikes_file='../../Data/040425/spikes.csv',
    digitalin_file='../../Data/040425/digitalin.dat',
    durations_ms=[5, 10, 25],
    bin_size_ms=0.1,
    pre_interval_ms=5,
    post_interval_ms=10,
    smooth_window=5,
    sampling_rate=30000,
    use_digitalin=True,
    save_dir='../../Output/040425/heatmaps'
)
```

## Function Reference

### `create_psth_heatmap()`

Create a single PSTH heatmap for a specific duration.

**Parameters:**
- `spikes_file` (str): Path to spikes.csv file
- `digitalin_file` (str, optional): Path to digitalin.dat file
- `intervals_file` (str, optional): Path to pico_time_adjust.csv (fallback)
- `duration_ms` (float): Interval duration to analyze in milliseconds
- `units` (List[int], optional): Units to include (None = all units)
- `bin_size_ms` (float): Bin size in milliseconds (default: 0.6)
- `pre_interval_ms` (float): Time before interval in ms (default: 5)
- `post_interval_ms` (float): Time after interval in ms (default: 10)
- `smooth_window` (int, optional): Number of bins for smoothing (default: 5)
- `sampling_rate` (int): Sampling rate for digitalin.dat (default: 30000)
- `use_digitalin` (bool): Whether to use digitalin.dat (default: False)
- `save_path` (str, optional): Path to save the figure

**Returns:**
- Tuple of (figure, heatmap_data, unit_labels, time_bins)

### `create_multiple_duration_heatmaps()`

Create PSTH heatmaps for multiple interval durations.

**Parameters:**
- Same as `create_psth_heatmap()` but with:
- `durations_ms` (List[float]): List of durations to analyze (default: [5, 10, 25])
- `save_dir` (str, optional): Directory to save figures

**Returns:**
- Dictionary mapping duration to (figure, heatmap_data, unit_labels, time_bins)

### `load_data_with_digitalin()`

Load spike and interval data with digitalin.dat support.

**Parameters:**
- `spikes_file` (str, optional): Path to spikes CSV file
- `digitalin_file` (str, optional): Path to digitalin.dat file
- `intervals_file` (str, optional): Path to intervals CSV file (fallback)
- `sampling_rate` (int): Sampling rate for digitalin.dat processing (default: 30000)
- `use_digitalin` (bool): Whether to use digitalin.dat (default: True)

**Returns:**
- Tuple of (spikes_df, intervals_df)

## Dependencies

### Required Files
- `binary_data.py` - Binary data loading utilities (in Full_Pipeline/)
- `ttls.py` - TTL event detection utilities (in Full_Pipeline/)
- `intan.py` - Intan file format support (in Full_Pipeline/)

### Python Packages
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scipy` - Scientific computing

## File Structure

```
4_PSTH-Heatmap/digitalin_loading/
├── README.md
├── psth_heatmap_analysis.ipynb
├── heatmap_utils.py
└── digitalin_loader.py
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure the Full_Pipeline directory is in the Python path
2. **File not found**: Check that digitalin.dat and spikes.csv exist in the specified paths
3. **Empty results**: Verify that the duration values match those in your data
4. **Memory issues**: Try reducing the number of units or time range for large datasets

### Validation

Use the test cell in the notebook to validate digitalin.dat loading:

```python
from digitalin_loader import load_digitalin_intervals, validate_intervals_compatibility

intervals_df = load_digitalin_intervals('path/to/digitalin.dat')
is_valid = validate_intervals_compatibility(intervals_df)
```

## Migration from CSV

To migrate existing analyses from CSV to digitalin.dat:

1. Set `use_digitalin = True` in the configuration
2. Provide path to `digitalin.dat` file
3. Set `sampling_rate` to match your acquisition system
4. Run the test cell to validate data loading
5. Compare results with CSV-based analysis to ensure consistency

## Output

- **Heatmap plots**: PNG files showing firing rate heatmaps
- **Console output**: Data loading statistics and analysis progress
- **Return values**: Raw data arrays for further analysis

The heatmaps are saved with descriptive filenames:
- `psth_heatmap_5ms.png`
- `psth_heatmap_10ms.png`
- `psth_heatmap_25ms.png`p

Create a heatmap (in the 4_PSTH-heatmap folder) that combines the PSTH of each units. Essentially, each unit should be plotted in ascending order on the Y axis, and on the x axis it should be the firing rate at time. 

This means that I will have to specify a pico_time_adjust.csv file and spikes.csv file. 

I want to be able to specify the interval duration from which the PSTH is to be created, and the time before and after, also the bin size

All units should be plotted, different graph for each interval duration
X axis: firing rate at time
Y axis: unit

The only difference for this sub folder is that it should load with digital in data (similar to the PSTH_digitalin notebook under Full_Pipeline/3_PSTH/digitalin_loading)