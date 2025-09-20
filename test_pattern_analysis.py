#!/usr/bin/env python3
"""
Test script for pattern analysis utilities
"""

import sys
import os
from utils import SpikePatternAnalyzer, load_spike_data, load_electrode_config, plot_unit_analysis
import matplotlib.pyplot as plt

def test_pattern_analysis():
    """Test the pattern analysis functionality"""
    
    print("Testing spike pattern analysis utilities...")
    
    # Load data
    print("Loading data...")
    spike_data = load_spike_data('spikes.csv')
    electrode_config = load_electrode_config('electrode.cfg')
    
    print(f"Loaded {len(spike_data)} spikes and {len(electrode_config)} electrodes")
    
    # Initialize analyzer
    analyzer = SpikePatternAnalyzer(spike_data, electrode_config)
    
    # Get a sample unit with reasonable spike count
    unit_spike_counts = spike_data['unit'].value_counts()
    print(f"\nUnit spike counts (top 10):")
    print(unit_spike_counts.head(10))
    
    # Select a unit with moderate activity
    test_units = unit_spike_counts[unit_spike_counts >= 50].head(3).index.tolist()
    
    if not test_units:
        print("No units with sufficient spikes found!")
        return
    
    print(f"\nTesting analysis on units: {test_units}")
    
    # Test analysis on first unit
    test_unit = test_units[0]
    print(f"\nRunning comprehensive analysis on Unit {test_unit}...")
    
    # Set test parameters
    params = {
        'burst_max_isi': 0.01,
        'burst_min_spikes': 3,
        'firing_rate_window': 1.0,
        'firing_rate_step': 0.1,
        'rhythm_max_freq': 100.0,
        'correlation_max_lag': 0.1
    }
    
    # Run analysis
    results = analyzer.run_pattern_analysis(test_unit, params)
    
    if 'error' in results:
        print(f"Error in analysis: {results['error']}")
        return
    
    # Print results
    print(f"\nAnalysis Results for Unit {test_unit}:")
    print(f"  Spike count: {results['spike_count']}")
    print(f"  Recording duration: {results['recording_duration']:.2f} seconds")
    print(f"  Mean firing rate: {results['firing_rate_analysis']['mean_rate']:.2f} Hz")
    print(f"  Mean ISI: {results['isi_analysis']['mean']:.4f} seconds")
    print(f"  CV ISI: {results['isi_analysis']['cv']:.3f}")
    print(f"  Burst count: {results['burst_analysis']['burst_count']}")
    print(f"  Pattern type: {results['pattern_classification']['pattern_type']}")
    print(f"  Pattern confidence: {results['pattern_classification']['confidence']:.2f}")
    
    # Test individual functions
    print(f"\nTesting individual analysis functions...")
    
    # Test ISI calculation
    isis = analyzer.calculate_isi(test_unit)
    print(f"  ISI calculation: {len(isis)} intervals computed")
    
    # Test firing rate calculation
    time_bins, firing_rates = analyzer.calculate_firing_rate(test_unit)
    print(f"  Firing rate calculation: {len(firing_rates)} time bins")
    
    # Test burst detection
    bursts = analyzer.detect_bursts(test_unit)
    print(f"  Burst detection: {len(bursts)} bursts found")
    
    # Test rhythmicity analysis
    rhythm = analyzer.analyze_rhythmicity(test_unit)
    print(f"  Rhythmicity analysis: Dominant freq = {rhythm['dominant_frequency']}")
    
    # Test cross-correlation with another unit
    if len(test_units) > 1:
        correlation = analyzer.calculate_spike_train_correlation(test_units[0], test_units[1])
        print(f"  Cross-correlation: Peak lag = {correlation['peak_lag']:.4f} seconds")
    
    # Create and save visualization
    print(f"\nCreating visualization...")
    try:
        fig = plot_unit_analysis(analyzer, test_unit, results, figsize=(15, 10))
        plt.savefig(f'test_unit_{test_unit}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Visualization saved as: test_unit_{test_unit}_analysis.png")
    except Exception as e:
        print(f"  Error creating visualization: {str(e)}")
    
    print(f"\nPattern analysis test completed successfully!")
    
    return results

if __name__ == "__main__":
    test_pattern_analysis()
