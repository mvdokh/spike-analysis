# Line Curation - Outlier Removal Analysis

This folder contains tools and results for identifying and removing statistical outliers from line curvature data used in licking and spike analysis.

## Overview

The line curvature data contains time points with corresponding curvature measurements that can have erroneous outlier values. This analysis uses statistical methods to identify and remove these outliers from both the analog output data and corresponding line coordinate data.

## Files Description

### Input Data
- **`analog_output.csv`** - Original time series data with Time and Data (curvature) columns
- **`lines_output.csv`** - Original line coordinate data with Frame, X, and Y columns

### Cleaned Output Data
- **`analog_output_clean.csv`** - Cleaned analog data with outliers removed
- **`lines_output_clean.csv`** - Cleaned line data with corresponding frames removed
- **`outliers_removed.csv`** - Summary of all removed data points with statistical bounds

### Analysis Scripts
- **`remove_outliers.py`** - Main Python script for outlier detection and removal
- **`remove_outliers.R`** - R version of the outlier removal script (requires R installation)

### Visualization
- **`outlier_analysis.png`** - Four-panel visualization showing:
  1. Distribution of original data with outlier boundaries
  2. Distribution of cleaned data
  3. Boxplot comparison (original vs cleaned)
  4. Time series with outliers highlighted

## Statistical Methods Used

The script implements three outlier detection methods:

### 1. Interquartile Range (IQR) Method *(Primary Method)*
- **Formula**: Outliers < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
- **Rationale**: Conservative approach, widely accepted standard
- **Use Case**: General purpose outlier detection for normally distributed data

### 2. Modified Z-Score Method
- **Formula**: |0.6745 × (x - median) / MAD| > 3.5
- **Rationale**: More robust to extreme outliers, uses median instead of mean
- **Use Case**: When data may have extreme outliers that skew the mean

### 3. Standard Z-Score Method
- **Formula**: |(x - mean) / std| > 3
- **Rationale**: Classic statistical approach
- **Use Case**: When data is approximately normally distributed

## Original Data Statistics
```
Min: -32.4471
Max: 25.0455
Mean: -0.0005
Range: 57.4926
```

## Results Summary

The IQR method was used as the primary outlier detection approach. This method:
- Provides a conservative estimate of outliers
- Is robust to non-normal distributions
- Uses quartiles which are less sensitive to extreme values

**Processing Results:**
- Original analog_output.csv: 463,704 rows
- Original lines_output.csv: ~9.3M rows
- Cleaned files created with corresponding outlier rows removed
- Detailed outlier summary saved for review

## Usage Instructions

### Python Script (Recommended)
```bash
cd line_curation
python remove_outliers.py
```

### R Script (Alternative)
```bash
cd line_curation
Rscript remove_outliers.R
```

**Prerequisites:**
- Python: pandas, numpy, matplotlib, seaborn
- R: readr, dplyr, ggplot2 (optional for visualization)

## Understanding the Results

1. **Check the visualization** (`outlier_analysis.png`) to visually inspect the outlier detection
2. **Review outliers_removed.csv** to see exactly which data points were removed
3. **Compare statistics** before and after cleaning in the script output
4. **Verify data integrity** by checking that time points match between cleaned files

## Quality Control

The script performs several quality checks:
- Validates that corresponding time/frame values exist in both datasets
- Provides detailed statistics before and after cleaning  
- Creates visualizations for manual inspection
- Saves a complete record of removed data points

## Customization Options

To adjust outlier detection sensitivity, modify these parameters in the script:

```python
# More conservative (fewer outliers detected)
iqr_multiplier = 2.0  # instead of 1.5

# More aggressive (more outliers detected)  
iqr_multiplier = 1.0  # instead of 1.5

# Use different method
outlier_mask = z_outliers  # instead of iqr_outliers
```

## Data Flow

```
analog_output.csv ──┐
                    ├─► Statistical Analysis ──► Outlier Detection
lines_output.csv  ──┘                            │
                                                 ▼
                    analog_output_clean.csv ──┐
                                              ├─► Final Clean Dataset
                    lines_output_clean.csv ──┘
```

## Notes

- The cleaning process preserves the temporal relationship between analog and line data
- Removed data points are logged for transparency and potential review
- Multiple statistical methods are calculated for comparison, but only one is applied
- The script is designed to handle large datasets efficiently

## Troubleshooting

**Common Issues:**
1. **Memory Error**: Large files may require chunked processing for very large datasets
2. **Missing Dependencies**: Install required packages using pip/conda
3. **File Path Issues**: Ensure working directory is set correctly
4. **R Not Found**: Use Python version if R is not installed

For questions or issues with the outlier removal process, refer to the generated visualizations and outlier summary files for detailed analysis results.