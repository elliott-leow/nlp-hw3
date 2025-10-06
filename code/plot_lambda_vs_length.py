#!/usr/bin/env python3
"""
Generate graph for Question 3f: Lambda vs Performance by File Length
Shows how optimal lambda varies (or doesn't) across different file lengths
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import subprocess
import os

# Lambda values to test
lambda_values = [0.0005, 0.005, 0.05, 0.5, 5.0]

# File length bins (words)
length_bins = [
    (0, 50, "Very Short (0-50)"),
    (50, 100, "Short (50-100)"),
    (100, 200, "Medium (100-200)"),
    (200, 500, "Long (200-500)"),
    (500, float('inf'), "Very Long (500+)")
]

print("Generating classifications for all lambda values...")
print("=" * 70)

# Store results for each lambda
all_results = {}

for lam in lambda_values:
    print(f"\nProcessing λ = {lam}...")
    
    # Run classification
    gen_model = f"gen-lambda{lam}.model"
    spam_model = f"spam-lambda{lam}.model"
    output_file = f"classification_results_lambda{lam}.txt"
    
    # Run textcat
    cmd = f"./textcat.py {gen_model} {spam_model} 0.7 ../data/gen_spam/dev/*/*.txt > {output_file} 2>&1"
    result = subprocess.run(cmd, shell=True, cwd="/home/kano/Documents/nlp-hw3/code")
    
    if result.returncode != 0:
        print(f"  WARNING: Classification failed for λ={lam}")
        continue
    
    # Parse results
    results = []
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('INFO:'):
                continue
            if '\t' in line and ('lambda' in line or '.model' in line):
                results.append(line)
    
    # Analyze by file length
    length_stats = {label: {'total': 0, 'correct': 0} 
                    for _, _, label in length_bins}
    
    for line in results:
        parts = line.split('\t')
        if len(parts) < 2:
            continue
            
        predicted_model = parts[0]
        filepath = parts[1]
        
        # Determine predicted category
        if 'gen' in predicted_model.lower():
            predicted = 'gen'
        elif 'spam' in predicted_model.lower():
            predicted = 'spam'
        else:
            continue
        
        # Extract actual category and length
        filename = filepath.split('/')[-1]
        if filename.startswith('gen.'):
            actual = 'gen'
            length = int(filename.split('.')[1])
        elif filename.startswith('spam.'):
            actual = 'spam'
            length = int(filename.split('.')[1])
        else:
            continue
        
        correct = (predicted == actual)
        
        # Assign to length bin
        for min_len, max_len, label in length_bins:
            if min_len <= length < max_len:
                length_stats[label]['total'] += 1
                if correct:
                    length_stats[label]['correct'] += 1
                break
    
    all_results[lam] = length_stats
    
    # Print summary
    print(f"  λ={lam}: {sum(s['correct'] for s in length_stats.values())}/{sum(s['total'] for s in length_stats.values())} correct " +
          f"({sum(s['correct'] for s in length_stats.values())/sum(s['total'] for s in length_stats.values())*100:.1f}%)")

print("\n" + "=" * 70)
print("Results Summary:")
print("=" * 70)

# Calculate error rates for each lambda and length bin
print(f"\n{'Lambda':<10}", end="")
for _, _, label in length_bins:
    print(f"{label:<20}", end="")
print()
print("-" * 110)

for lam in lambda_values:
    print(f"{lam:<10.4f}", end="")
    if lam not in all_results:
        print("  FAILED - Model not available or classification error")
        continue
    for _, _, label in length_bins:
        stats = all_results[lam][label]
        if stats['total'] > 0:
            error_rate = (1 - stats['correct'] / stats['total']) * 100
            print(f"{error_rate:>6.1f}% (n={stats['total']:<3})", end=" ")
        else:
            print(f"{'N/A':<18}", end=" ")
    print()

print("\n" + "=" * 70)

# Create the graph
fig, ax = plt.subplots(figsize=(14, 8))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
markers = ['o', 's', '^', 'D', 'v']

for i, (min_len, max_len, label) in enumerate(length_bins):
    error_rates = []
    lambda_vals = []
    
    for lam in lambda_values:
        if lam not in all_results:
            continue
        stats = all_results[lam][label]
        if stats['total'] >= 5:  # Only plot if we have enough samples
            error_rate = (1 - stats['correct'] / stats['total']) * 100
            error_rates.append(error_rate)
            lambda_vals.append(lam)
    
    if error_rates:
        ax.semilogx(lambda_vals, error_rates, marker=markers[i], 
                   linewidth=2.5, markersize=10, label=label, 
                   color=colors[i], alpha=0.85)

ax.set_xlabel('Lambda (λ) - Log Scale', fontsize=13, fontweight='bold')
ax.set_ylabel('Error Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Question 3f: How Lambda Affects Performance Across File Lengths\n' + 
             'Add-λ Smoothing on Gen/Spam Dev Data (prior = 0.7)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, which='both', linestyle='--')
ax.set_ylim(0, max([max([(1 - all_results[lam][label]['correct'] / all_results[lam][label]['total']) * 100 
                         for lam in lambda_values if lam in all_results and all_results[lam][label]['total'] >= 5], default=0)
                    for _, _, label in length_bins]) * 1.1)

# Add vertical line at optimal lambda
optimal_lambda = 0.005
ax.axvline(x=optimal_lambda, color='red', linestyle=':', linewidth=2, 
           alpha=0.5, label=f'Optimal λ* = {optimal_lambda}')

plt.tight_layout()
plt.savefig('question_3f_lambda_vs_length.png', dpi=150, bbox_inches='tight')
print(f"\nGraph saved as: question_3f_lambda_vs_length.png")
plt.show()

# Analysis
print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)

# Find optimal lambda for each length category
print("\nOptimal λ for each file length range:")
for _, _, label in length_bins:
    best_lam = None
    best_error = float('inf')
    
    for lam in lambda_values:
        if lam not in all_results:
            continue
        stats = all_results[lam][label]
        if stats['total'] >= 5:
            error_rate = (1 - stats['correct'] / stats['total']) * 100
            if error_rate < best_error:
                best_error = error_rate
                best_lam = lam
    
    if best_lam:
        print(f"  {label:<25} λ* = {best_lam:<8.4f} (error rate: {best_error:.1f}%)")

# Check if very short files benefit from different lambda
very_short_optimal = None
medium_optimal = None

for _, _, label in length_bins:
    best_lam = None
    best_error = float('inf')
    
    for lam in lambda_values:
        if lam not in all_results:
            continue
        stats = all_results[lam][label]
        if stats['total'] >= 5:
            error_rate = (1 - stats['correct'] / stats['total']) * 100
            if error_rate < best_error:
                best_error = error_rate
                best_lam = lam
    
    if "Very Short" in label:
        very_short_optimal = best_lam
    elif "Medium" in label:
        medium_optimal = best_lam

print("\n" + "=" * 70)
if very_short_optimal and medium_optimal:
    if very_short_optimal != medium_optimal:
        print(f"→ Short files prefer λ={very_short_optimal} (more smoothing)")
        print(f"→ Medium files prefer λ={medium_optimal} (less smoothing)")
        print("  Shorter files benefit from MORE smoothing due to data sparsity!")
    else:
        print(f"→ Both short and medium files prefer λ={very_short_optimal}")
        print("  The optimal λ is relatively stable across file lengths.")

print("=" * 70)

