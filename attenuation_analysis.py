# %%
# Attenuation Analysis for Different Lenses (3-5cm Range)
# This script analyzes signal attenuation across different lens datasets
# Each dataset directory represents a different lens configuration
# 
# Analysis focuses on the 3-5cm range using median max values for robustness
# Distance calculation: distance = sample_index / 1000 * 1.3 (in cm)

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# %%
# Configuration - Easy to modify
DATA_DIR = "/home/briancottle/Code/SoundPass/new_datasets/2025-09-15"
SAMPLE_LIMIT = 50  # Limit samples per dataset for faster analysis (set to None for all)
PLOT_MAX_VALUES = True  # Whether to plot individual max values
PLOT_AVERAGES = True   # Whether to plot average max values per dataset
PLOT_DISTRIBUTIONS = True  # Whether to plot distribution histograms

# %%
# Load and analyze data from all datasets
def load_dataset_data(dataset_dir, sample_limit=None):
    """Load data from a single dataset directory and return median max value in 3-5cm range"""
    pkl_files = glob.glob(os.path.join(dataset_dir, "*.pkl"))
    
    if sample_limit:
        pkl_files = pkl_files[:sample_limit]
    
    max_values_3_5cm = []
    all_samples_list = []
    
    print(f"Loading {len(pkl_files)} files from {os.path.basename(dataset_dir)}")
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Data structure: [all_samples, distance, orientation_data]
            all_samples = data[0]  # Shape: (num_buffers, num_samples)
            distance = data[1]     # Distance array in cm
            
            # Find indices for 3-5 cm range
            # Distance calculation: distance = sample_index / 1000 * 1.3
            # So sample_index = distance * 1000 / 1.3
            start_idx = int(3.0 * 1000 / 1.3)  # 3 cm
            end_idx = int(5.0 * 1000 / 1.3)    # 5 cm
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, all_samples.shape[1] - 1))
            end_idx = max(start_idx + 1, min(end_idx, all_samples.shape[1]))
            
            # Extract signal in 3-5 cm range
            signal_3_5cm = all_samples[:, start_idx:end_idx]  # Shape: (num_buffers, range_samples)
            
            # Calculate max value for each buffer in the 3-5 cm range
            buffer_maxes_3_5cm = np.max(signal_3_5cm, axis=1)  # Max for each buffer in range
            overall_max_3_5cm = np.max(buffer_maxes_3_5cm)  # Overall max across all buffers in range
            
            max_values_3_5cm.append(overall_max_3_5cm)
            all_samples_list.append(all_samples)
            
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    # Return median max value for this lens configuration (more robust than mean)
    if max_values_3_5cm:
        median_max = np.median(max_values_3_5cm)
        std_max = np.std(max_values_3_5cm)
        return median_max, std_max, len(max_values_3_5cm), all_samples_list
    else:
        return None, None, 0, []

# %%
# Load all datasets
dataset_dirs = sorted([d for d in os.listdir(DATA_DIR) 
                      if os.path.isdir(os.path.join(DATA_DIR, d))])

print(f"Found {len(dataset_dirs)} datasets:")
for i, dataset in enumerate(dataset_dirs):
    print(f"  {i+1}. {dataset}")

# %%
# Analyze each dataset (lens configuration)
dataset_results = {}
lens_median_values = []

for dataset_dir in dataset_dirs:
    full_path = os.path.join(DATA_DIR, dataset_dir)
    median_max, std_max, sample_count, all_samples = load_dataset_data(full_path, SAMPLE_LIMIT)
    
    if median_max is not None:  # Only process if we got data
        dataset_results[dataset_dir] = {
            'median_max': median_max,
            'std_max': std_max,
            'sample_count': sample_count,
            'all_samples': all_samples
        }
        lens_median_values.append(median_max)
        print(f"{dataset_dir}: {sample_count} samples, median max (3-5cm) = {median_max:.2f} ± {std_max:.2f}")

# %%
# Create summary table
print("\n" + "="*80)
print("ATTENUATION ANALYSIS SUMMARY (3-5cm Range)")
print("="*80)
print(f"{'Lens Config':<20} {'Samples':<8} {'Median Max':<12} {'Std Max':<10} {'Lens #':<8}")
print("-"*80)

for i, (dataset, results) in enumerate(dataset_results.items()):
    print(f"{dataset:<20} {results['sample_count']:<8} {results['median_max']:<12.2f} {results['std_max']:<10.2f} {i+1:<8}")

# %%
# Plot 1: Bar chart of median max values for each lens (3-5cm range)
if PLOT_MAX_VALUES:
    plt.figure(figsize=(12, 6))
    
    datasets = list(dataset_results.keys())
    median_values = [dataset_results[d]['median_max'] for d in datasets]
    std_values = [dataset_results[d]['std_max'] for d in datasets]
    
    x_pos = np.arange(len(datasets))
    bars = plt.bar(x_pos, median_values, yerr=std_values, capsize=5, alpha=0.7, 
                   color='skyblue', edgecolor='navy')
    
    plt.xlabel('Lens Configuration')
    plt.ylabel('Median Max Signal Value (3-5cm range)')
    plt.title('Signal Attenuation Comparison Across Different Lenses (3-5cm Range)')
    plt.xticks(x_pos, [f'Lens {i+1}' for i in range(len(datasets))], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (median, std) in enumerate(zip(median_values, std_values)):
        plt.text(i, median + std + max(median_values)*0.01, f'{median:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# %%
# Plot 2: Attenuation comparison with reference line
if PLOT_AVERAGES:
    datasets = list(dataset_results.keys())
    median_values = [dataset_results[d]['median_max'] for d in datasets]
    std_values = [dataset_results[d]['std_max'] for d in datasets]
    
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(datasets))
    
    bars = plt.bar(x_pos, median_values, yerr=std_values, capsize=5, alpha=0.7, 
                   color='skyblue', edgecolor='navy')
    
    # Add reference line (first lens as reference)
    if len(median_values) > 0:
        plt.axhline(y=median_values[0], color='red', linestyle='--', alpha=0.7, 
                   label=f'Reference (Lens 1): {median_values[0]:.1f}')
    
    plt.xlabel('Lens Configuration')
    plt.ylabel('Median Max Signal Value (3-5cm range)')
    plt.title('Signal Attenuation Comparison Across Different Lenses (3-5cm Range)')
    plt.xticks(x_pos, [f'Lens {i+1}' for i in range(len(datasets))], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add value labels on bars
    for i, (median, std) in enumerate(zip(median_values, std_values)):
        plt.text(i, median + std + max(median_values)*0.01, f'{median:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# %%
# Plot 3: Attenuation visualization
if PLOT_DISTRIBUTIONS:
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Attenuation ratios
    plt.subplot(2, 1, 1)
    datasets = list(dataset_results.keys())
    median_values = [dataset_results[d]['median_max'] for d in datasets]
    
    if len(median_values) > 0:
        reference_value = median_values[0]
        attenuation_ratios = [val / reference_value for val in median_values]
        attenuation_db = [20 * np.log10(ratio) for ratio in attenuation_ratios]
        
        x_pos = np.arange(len(datasets))
        bars = plt.bar(x_pos, attenuation_db, alpha=0.7, color='lightcoral', edgecolor='darkred')
        
        plt.xlabel('Lens Configuration')
        plt.ylabel('Attenuation (dB)')
        plt.title('Signal Attenuation Relative to Lens 1 (Reference) - 3-5cm Range')
        plt.xticks(x_pos, [f'Lens {i+1}' for i in range(len(datasets))], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, db in enumerate(attenuation_db):
            plt.text(i, db + max(attenuation_db)*0.01, f'{db:.1f}dB', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Raw values comparison
    plt.subplot(2, 1, 2)
    x_pos = np.arange(len(datasets))
    bars = plt.bar(x_pos, median_values, alpha=0.7, color='lightblue', edgecolor='navy')
    
    plt.xlabel('Lens Configuration')
    plt.ylabel('Median Max Signal Value (3-5cm range)')
    plt.title('Raw Signal Values by Lens Configuration (3-5cm Range)')
    plt.xticks(x_pos, [f'Lens {i+1}' for i in range(len(datasets))], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(median_values):
        plt.text(i, val + max(median_values)*0.01, f'{val:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# %%
# Calculate attenuation ratios (relative to first lens)
if len(dataset_results) > 1:
    print("\n" + "="*60)
    print("ATTENUATION ANALYSIS (Relative to Lens 1) - 3-5cm Range")
    print("="*60)
    
    first_dataset = list(dataset_results.keys())[0]
    first_median = dataset_results[first_dataset]['median_max']
    
    print(f"Reference: {first_dataset} (median max = {first_median:.2f})")
    print(f"{'Lens Config':<20} {'Attenuation Ratio':<18} {'Attenuation (dB)':<15}")
    print("-"*60)
    
    for i, (dataset, results) in enumerate(dataset_results.items()):
        ratio = results['median_max'] / first_median
        attenuation_db = 20 * np.log10(ratio)  # Convert to dB
        print(f"Lens {i+1:<15} {ratio:<18.3f} {attenuation_db:<15.2f}")

# %%
# Statistical analysis
print("\n" + "="*60)
print("STATISTICAL ANALYSIS (3-5cm Range)")
print("="*60)

# Calculate overall statistics
all_median_values = [results['median_max'] for results in dataset_results.values()]
all_std_values = [results['std_max'] for results in dataset_results.values()]

print(f"Overall mean of lens medians: {np.mean(all_median_values):.2f} ± {np.std(all_median_values):.2f}")
print(f"Range of lens medians: {np.min(all_median_values):.2f} to {np.max(all_median_values):.2f}")
print(f"Mean coefficient of variation: {np.mean([std/median for median, std in zip(all_median_values, all_std_values)]):.3f}")

# Find best and worst performing lenses
if len(all_median_values) > 0:
    best_lens_idx = np.argmax(all_median_values)
    worst_lens_idx = np.argmin(all_median_values)
    best_lens = list(dataset_results.keys())[best_lens_idx]
    worst_lens = list(dataset_results.keys())[worst_lens_idx]
    
    print(f"\nBest performing lens: {best_lens} (Lens {best_lens_idx + 1}) - {all_median_values[best_lens_idx]:.2f}")
    print(f"Worst performing lens: {worst_lens} (Lens {worst_lens_idx + 1}) - {all_median_values[worst_lens_idx]:.2f}")
    
    # Calculate total attenuation range
    total_attenuation_db = 20 * np.log10(all_median_values[best_lens_idx] / all_median_values[worst_lens_idx])
    print(f"Total attenuation range: {total_attenuation_db:.2f} dB")

# %%
# Save results to file
results_summary = {
    'lens_configs': list(dataset_results.keys()),
    'median_values': all_median_values,
    'std_values': all_std_values,
    'attenuation_ratios': [median/all_median_values[0] for median in all_median_values],
    'attenuation_db': [20 * np.log10(median/all_median_values[0]) for median in all_median_values],
    'sample_counts': [dataset_results[d]['sample_count'] for d in dataset_results.keys()],
    'analysis_range': '3-5cm',
    'statistic_type': 'median'
}

# Save as pickle for later use
with open('attenuation_analysis_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print(f"\nResults saved to: attenuation_analysis_results.pkl")
print("Analysis complete!")

# %%
# Quick modification section - Easy to change parameters
# Uncomment and modify these lines to quickly test different settings:

# SAMPLE_LIMIT = 10  # Use only 10 samples per dataset for quick testing
# PLOT_MAX_VALUES = False  # Disable individual plots
# PLOT_AVERAGES = True   # Keep only the comparison plot
# PLOT_DISTRIBUTIONS = False  # Disable distribution plots

# %%
# Additional analysis: Signal quality metrics
def analyze_signal_quality(all_samples_list):
    """Analyze signal quality metrics"""
    quality_metrics = []
    
    for samples in all_samples_list:
        # Calculate signal-to-noise ratio approximation
        signal_power = np.mean(np.var(samples, axis=1))
        noise_floor = np.percentile(samples, 10)  # Approximate noise floor
        snr_approx = signal_power / (noise_floor**2 + 1e-10)
        
        # Calculate signal stability (coefficient of variation)
        max_values = np.max(samples, axis=1)
        stability = np.std(max_values) / np.mean(max_values)
        
        quality_metrics.append({
            'snr_approx': snr_approx,
            'stability': stability,
            'signal_power': signal_power
        })
    
    return quality_metrics

# Uncomment to run signal quality analysis:
# print("\n" + "="*60)
# print("SIGNAL QUALITY ANALYSIS")
# print("="*60)
# 
# for i, (dataset, results) in enumerate(dataset_results.items()):
#     quality = analyze_signal_quality(results['all_samples'])
#     avg_snr = np.mean([q['snr_approx'] for q in quality])
#     avg_stability = np.mean([q['stability'] for q in quality])
#     print(f"Lens {i+1}: SNR≈{avg_snr:.2f}, Stability={avg_stability:.3f}")
