#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_length_from_filename(filename):
    """Extract file length from filename like gen.766.115.txt or spam.117.016.txt"""
    # Pattern matches numbers after gen. or spam.
    match = re.search(r'(gen|spam)\.(\d+)\.\d+\.txt', filename)
    if match:
        return int(match.group(2))
    return None

def analyze_performance():
    """Analyze classification performance vs file length"""

    # Read classification results
    with open('lambda_optimal_results.txt', 'r') as f:
        lines = f.readlines()

    # Parse results
    results = []
    for line in lines:
        if '\t' in line:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                predicted_model = parts[0]
                filepath = parts[1]

                # Extract file length and true label
                filename = filepath.split('/')[-1]
                length = extract_length_from_filename(filename)

                if length is not None:
                    true_label = 'gen' if '/gen/' in filepath else 'spam'
                    predicted_label = 'gen' if predicted_model == 'gen-lambda0.005.model' else 'spam'
                    correct = (true_label == predicted_label)

                    results.append({
                        'length': length,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'correct': correct
                    })

    print(f"Analyzed {len(results)} files")

    # Group by length bins for analysis
    length_bins = defaultdict(list)
    for result in results:
        # Bin lengths into groups of 50
        bin_key = (result['length'] // 50) * 50
        length_bins[bin_key].append(result)

    # Calculate accuracy for each bin
    bin_data = []
    for bin_start in sorted(length_bins.keys()):
        bin_results = length_bins[bin_start]
        accuracy = sum(1 for r in bin_results if r['correct']) / len(bin_results)
        bin_data.append({
            'bin_start': bin_start,
            'bin_end': bin_start + 49,
            'count': len(bin_results),
            'accuracy': accuracy
        })

    # Print summary
    print("\nPerformance by file length bins:")
    print("Length Range\tCount\tAccuracy")
    print("-" * 40)
    for data in bin_data:
        if data['count'] >= 5:  # Only show bins with reasonable sample size
            print(f"{data['bin_start']}-{data['bin_end']}\t{data['count']}\t{data['accuracy']:.3f}")

    # Create scatter plot of individual results
    lengths = [r['length'] for r in results]
    correct_vals = [1 if r['correct'] else 0 for r in results]

    plt.figure(figsize=(12, 8))

    # Scatter plot with some jitter for visibility
    jitter = np.random.normal(0, 0.02, len(lengths))
    plt.scatter(lengths, np.array(correct_vals) + jitter, alpha=0.6, s=20)

    # Add trend line using lowess-like smoothing (simple moving average)
    sorted_data = sorted(zip(lengths, correct_vals))
    window_size = max(20, len(sorted_data) // 20)  # Adaptive window size

    smooth_x = []
    smooth_y = []
    for i in range(0, len(sorted_data), window_size // 2):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(sorted_data), i + window_size // 2)
        window_data = sorted_data[window_start:window_end]

        if len(window_data) > 0:
            avg_length = np.mean([d[0] for d in window_data])
            avg_accuracy = np.mean([d[1] for d in window_data])
            smooth_x.append(avg_length)
            smooth_y.append(avg_accuracy)

    if len(smooth_x) > 1:
        plt.plot(smooth_x, smooth_y, 'r-', linewidth=2, label='Trend (smoothed)')

    plt.xlabel('File Length (words)')
    plt.ylabel('Classification Accuracy (1=correct, 0=incorrect)')
    plt.title('Classification Performance vs File Length (λ*=0.005)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig('performance_vs_length.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as performance_vs_length.png")

    # Calculate correlation
    correlation = np.corrcoef(lengths, correct_vals)[0, 1]
    print(f"\nCorrelation between file length and accuracy: {correlation:.3f}")

    return results

if __name__ == "__main__":
    results = analyze_performance()