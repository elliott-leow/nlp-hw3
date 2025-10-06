#!/usr/bin/env python3
"""
Generate graph for Question 3f: Performance vs File Length
Analyzes how add-lambda* (λ=0.005) performance depends on file length
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import scipy.stats as stats

# Read classification results
results = []
with open('classification_results_lambda0.005.txt', 'r') as f:
    for line in f:
        line = line.strip()
        # Skip INFO lines
        if line.startswith('INFO:'):
            continue
        # Look for lines with model predictions
        if '\t' in line and ('gen-lambda' in line or 'spam-lambda' in line or 'gen.model' in line or 'spam.model' in line):
            results.append(line)

# Parse each result: extract file length, actual label, predicted label
data = []
for line in results:
    parts = line.split('\t')
    if len(parts) < 2:
        continue
        
    predicted_model = parts[0]  # gen-lambda0.005.model or spam-lambda0.005.model
    filepath = parts[1]   # ../data/gen_spam/dev/gen/gen.101.156.txt
    
    # Determine predicted category
    if 'gen' in predicted_model.lower():
        predicted = 'gen'
    elif 'spam' in predicted_model.lower():
        predicted = 'spam'
    else:
        continue
    
    # Extract actual category and length from filename
    filename = filepath.split('/')[-1]  # gen.101.156.txt
    if filename.startswith('gen.'):
        actual = 'gen'
        # Extract length: gen.101.156.txt -> 101
        length = int(filename.split('.')[1])
    elif filename.startswith('spam.'):
        actual = 'spam'
        # Extract length: spam.104.052.txt -> 104
        length = int(filename.split('.')[1])
    else:
        continue
    
    correct = (predicted == actual)
    data.append((length, correct, actual, predicted))

print(f"Total files analyzed: {len(data)}")
print(f"Correct: {sum(1 for _, c, _, _ in data if c)}")
print(f"Overall accuracy: {sum(1 for _, c, _, _ in data if c) / len(data) * 100:.2f}%\n")

# Create length bins for analysis
# Use logarithmic-ish bins to handle the wide range
bin_edges = [0, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 5000]
bin_labels = []
for i in range(len(bin_edges) - 1):
    bin_labels.append(f"{bin_edges[i]}-{bin_edges[i+1]}")

# Group by bins
bin_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'lengths': []})

for length, correct, actual, predicted in data:
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= length < bin_edges[i+1]:
            bin_key = bin_labels[i]
            bin_stats[bin_key]['total'] += 1
            if correct:
                bin_stats[bin_key]['correct'] += 1
            bin_stats[bin_key]['lengths'].append(length)
            break

# Calculate error rates for bins with data
bin_centers = []
error_rates = []
bin_counts = []

print("Performance by File Length:")
print("-" * 70)
print(f"{'Length Range':<15} {'Count':<10} {'Correct':<10} {'Error Rate':<15} {'Accuracy':<10}")
print("-" * 70)

for i, label in enumerate(bin_labels):
    bin_stat = bin_stats[label]
    if bin_stat['total'] > 0:
        accuracy = bin_stat['correct'] / bin_stat['total'] * 100
        error_rate = 100 - accuracy
        center = (bin_edges[i] + bin_edges[i+1]) / 2
        
        print(f"{label:<15} {bin_stat['total']:<10} {bin_stat['correct']:<10} {error_rate:<15.1f}% {accuracy:<10.1f}%")
        
        bin_centers.append(center)
        error_rates.append(error_rate)
        bin_counts.append(bin_stat['total'])

print("-" * 70)

# Also create a scatter plot with smoothing
all_lengths = [length for length, _, _, _ in data]
all_correct = [1 if correct else 0 for _, correct, _, _ in data]

# Sort by length for smoothing
sorted_data = sorted(zip(all_lengths, all_correct))
sorted_lengths, sorted_correct = zip(*sorted_data)

# Apply moving average for smoothing
window_size = 20  # Use 20 files for smoothing window
smoothed_lengths = []
smoothed_error_rates = []

for i in range(window_size, len(sorted_lengths) - window_size):
    window_correct = sorted_correct[i-window_size:i+window_size]
    avg_length = sorted_lengths[i]
    avg_error = (1 - sum(window_correct) / len(window_correct)) * 100
    smoothed_lengths.append(avg_length)
    smoothed_error_rates.append(avg_error)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Binned error rates (bar chart)
bars = ax1.bar(range(len(bin_centers)), error_rates, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(bin_centers)))
ax1.set_xticklabels([label for label, bin_stat in zip(bin_labels, [bin_stats[l] for l in bin_labels]) 
                      if bin_stat['total'] > 0], rotation=45, ha='right')
ax1.set_ylabel('Error Rate (%)', fontsize=12)
ax1.set_xlabel('File Length (words)', fontsize=12)
ax1.set_title('3f error rate vs file length\nlambda = 0.005', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, max(error_rates) * 1.2)

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, [bin_stat['total'] for label, bin_stat in zip(bin_labels, [bin_stats[l] for l in bin_labels]) if bin_stat['total'] > 0])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'n={count}', ha='center', va='bottom', fontsize=9)

# Plot 2: Smoothed curve (without individual scatter points)
ax2.plot(smoothed_lengths, smoothed_error_rates, linewidth=3, color='darkred')
ax2.set_xlabel('File Length (words)', fontsize=12)
ax2.set_ylabel('Error Rate (%)', fontsize=12)
ax2.set_title('3f error rate vs file length\nlambda = 0.005', fontsize=13, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(0, max(smoothed_error_rates) * 1.2 if smoothed_error_rates else 50)

plt.tight_layout()
plt.savefig('question_3f_performance_vs_length.png', dpi=150, bbox_inches='tight')
print(f"\nGraph saved as: question_3f_performance_vs_length.png")
plt.show()

# Print key findings
print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)

# Find shortest and longest file ranges
if error_rates:
    max_error_idx = error_rates.index(max(error_rates))
    min_error_idx = error_rates.index(min(error_rates))
    
    print(f"Highest error rate: {error_rates[max_error_idx]:.1f}% for files with {[l for l, s in zip(bin_labels, [bin_stats[l] for l in bin_labels]) if s['total'] > 0][max_error_idx]} words")
    print(f"Lowest error rate: {error_rates[min_error_idx]:.1f}% for files with {[l for l, s in zip(bin_labels, [bin_stats[l] for l in bin_labels]) if s['total'] > 0][min_error_idx]} words")
    
    # Correlation analysis
    if len(bin_centers) > 2:
        correlation, p_value = stats.pearsonr(bin_centers, error_rates)
        print(f"\nCorrelation between file length and error rate: {correlation:.3f}")
        print(f"P-value: {p_value:.4f}")
        
        if abs(correlation) > 0.7:
            direction = "positive" if correlation > 0 else "negative"
            print(f"→ Strong {direction} correlation detected")
            if correlation < 0:
                print("  Interpretation: Longer files tend to have LOWER error rates")
            else:
                print("  Interpretation: Longer files tend to have HIGHER error rates")
                print("  (Note: This may be due to very long files being rare outliers)")
        elif abs(correlation) > 0.3:
            direction = "positive" if correlation > 0 else "negative"
            print(f"→ Moderate {direction} correlation detected")
        else:
            print("→ No strong correlation detected")
        
        # More nuanced analysis focusing on typical files
        typical_bins = [(c, e) for c, e, n in zip(bin_centers, error_rates, bin_counts) if c < 1000 and n >= 10]
        if len(typical_bins) > 3:
            typical_centers, typical_errors = zip(*typical_bins)
            typical_corr, typical_pval = stats.pearsonr(typical_centers, typical_errors)
            print(f"\nFor typical file lengths (< 1000 words, n≥10):")
            print(f"Correlation: {typical_corr:.3f}, p-value: {typical_pval:.4f}")
            if typical_corr < -0.3:
                print("→ Longer files (up to ~1000 words) tend to have lower error rates")

print("="*70)

