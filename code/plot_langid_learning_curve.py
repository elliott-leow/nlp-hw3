#!/usr/bin/env python3
"""
Generate graph for Question 3i: Language ID Learning Curve
Shows how error rate improves with more training data
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re

# Training data sizes to test
sizes = ['1K', '2K', '5K', '10K', '20K', '50K']
size_values = [1, 2, 5, 10, 20, 50]  # For x-axis

print("=" * 70)
print("Question 3i: Language ID Learning Curve")
print("=" * 70)

error_rates = []

for size in sizes:
    print(f"\nProcessing training size: {size}")
    print("-" * 50)
    
    # Model filenames
    en_model = f"en-{size}.model"
    sp_model = f"sp-{size}.model"
    
    # Check if models exist, if not train them
    if not os.path.exists(en_model) or not os.path.exists(sp_model):
        print(f"  Training models for {size}...")
        
        # Train English model
        train_cmd_en = f"./train_lm.py ../data/english_spanish/train/en.{size} --vocab vocab-ensp.txt --smoother add_lambda --lambda 1.0"
        result_en = subprocess.run(train_cmd_en, shell=True, cwd="/home/kano/Documents/nlp-hw3/code", 
                                  capture_output=True, text=True)
        
        # Train Spanish model
        train_cmd_sp = f"./train_lm.py ../data/english_spanish/train/sp.{size} --vocab vocab-ensp.txt --smoother add_lambda --lambda 1.0"
        result_sp = subprocess.run(train_cmd_sp, shell=True, cwd="/home/kano/Documents/nlp-hw3/code",
                                  capture_output=True, text=True)
        
        # Rename models to size-specific names
        subprocess.run(f"mv en.model {en_model}", shell=True, cwd="/home/kano/Documents/nlp-hw3/code")
        subprocess.run(f"mv sp.model {sp_model}", shell=True, cwd="/home/kano/Documents/nlp-hw3/code")
        print(f"  Models trained and saved as {en_model} and {sp_model}")
    else:
        print(f"  Using existing models: {en_model}, {sp_model}")
    
    # Run classification
    output_file = f"classification_results_langid_{size}.txt"
    files_pattern = "$(find ../data/english_spanish/dev -type f -name '*.??')"
    classify_cmd = f"./textcat.py {en_model} {sp_model} 0.7 {files_pattern} > {output_file} 2>&1"
    
    result = subprocess.run(classify_cmd, shell=True, cwd="/home/kano/Documents/nlp-hw3/code",
                          executable='/bin/bash')
    
    # Parse results
    correct = 0
    total = 0
    
    with open(f"/home/kano/Documents/nlp-hw3/code/{output_file}", 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('INFO:'):
                continue
            if '\t' in line and '.model' in line:
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                predicted_model = parts[0]
                filepath = parts[1]
                
                # Determine predicted
                if 'en' in predicted_model:
                    predicted = 'en'
                elif 'sp' in predicted_model:
                    predicted = 'sp'
                else:
                    continue
                
                # Determine actual from path
                if '/english/' in filepath:
                    actual = 'en'
                elif '/spanish/' in filepath:
                    actual = 'sp'
                else:
                    continue
                
                total += 1
                if predicted == actual:
                    correct += 1
    
    if total > 0:
        error_rate = (1 - correct / total) * 100
        error_rates.append(error_rate)
        print(f"  Results: {correct}/{total} correct")
        print(f"  Error rate: {error_rate:.2f}%")
    else:
        print(f"  WARNING: No results found for {size}")
        error_rates.append(None)

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print(f"{'Training Size':<15} {'Error Rate':<15}")
print("-" * 30)
for size, error in zip(sizes, error_rates):
    if error is not None:
        print(f"{size:<15} {error:<15.2f}%")
    else:
        print(f"{size:<15} {'N/A':<15}")

# Filter out None values for plotting
valid_data = [(sv, er) for sv, er in zip(size_values, error_rates) if er is not None]
if valid_data:
    plot_sizes, plot_errors = zip(*valid_data)
else:
    print("No valid data to plot!")
    exit(1)

# Create the learning curve graph
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(plot_sizes, plot_errors, 'o-', linewidth=2.5, markersize=10, color='darkgreen')

# Add labels on points
for size_val, err, size_label in zip(plot_sizes, plot_errors, [s for s, e in zip(sizes, error_rates) if e is not None]):
    ax.annotate(f'{err:.1f}%', xy=(size_val, err), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax.set_xlabel('Training Data Size (thousands of characters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('3i learning curve (language ID)\nEnglish vs Spanish with Add-λ smoothing (λ=1.0)', 
             fontsize=13, fontweight='bold')
ax.set_xticks(plot_sizes)
ax.set_xticklabels([s for s, e in zip(sizes, error_rates) if e is not None])
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(plot_errors) * 1.2)

plt.tight_layout()
plt.savefig('question_3i_langid_learning_curve.png', dpi=150, bbox_inches='tight')
print(f"\nGraph saved as: question_3i_langid_learning_curve.png")
plt.show()

# Analysis
print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)

if len(plot_errors) >= 2:
    improvement_1_to_2 = plot_errors[0] - plot_errors[1] if len(plot_errors) > 1 else 0
    print(f"Improvement from 1K to 2K: {improvement_1_to_2:.2f}% reduction")
    
    if len(plot_errors) >= 3:
        improvement_2_to_5 = plot_errors[1] - plot_errors[2]
        print(f"Improvement from 2K to 5K: {improvement_2_to_5:.2f}% reduction")
    
    final_error = plot_errors[-1]
    print(f"\nFinal error rate at {sizes[len(plot_errors)-1]}: {final_error:.2f}%")
    
    if plot_errors[0] > plot_errors[-1]:
        total_improvement = plot_errors[0] - plot_errors[-1]
        print(f"Total improvement: {total_improvement:.2f}% reduction ({plot_errors[0]:.1f}% → {plot_errors[-1]:.1f}%)")
    
    # Check for diminishing returns
    improvements = [plot_errors[i] - plot_errors[i+1] for i in range(len(plot_errors)-1)]
    if all(improvements[i] >= improvements[i+1] for i in range(len(improvements)-1)):
        print("\n→ Clear diminishing returns: each doubling of data gives less improvement")
    else:
        print("\n→ Improvement pattern varies across training sizes")

print("=" * 70)



