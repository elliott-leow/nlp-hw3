#!/usr/bin/env python3

import subprocess
import re
import matplotlib.pyplot as plt

def run_textcat(gen_model, spam_model, prior=0.7):
    """Run textcat and return classification results"""
    cmd = ['python', 'textcat.py', gen_model, spam_model, str(prior)]

    # Add all dev files
    import glob
    gen_files = glob.glob('../data/gen_spam/dev/gen/*')
    spam_files = glob.glob('../data/gen_spam/dev/spam/*')
    cmd.extend(gen_files + spam_files)

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse results to count correct/incorrect classifications
    lines = result.stdout.strip().split('\n')

    total_files = 0
    correct_gen = 0
    correct_spam = 0
    total_gen = 180  # Known from earlier analysis
    total_spam = 90

    for line in lines:
        if '\t' in line:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                predicted_model = parts[0]
                filepath = parts[1]
                total_files += 1

                true_label = 'gen' if '/gen/' in filepath else 'spam'
                predicted_label = 'gen' if predicted_model.startswith('gen') else 'spam'

                if true_label == predicted_label:
                    if true_label == 'gen':
                        correct_gen += 1
                    else:
                        correct_spam += 1

    # Calculate error rate
    total_correct = correct_gen + correct_spam
    error_rate = (total_files - total_correct) / total_files if total_files > 0 else 0

    return {
        'total_files': total_files,
        'correct': total_correct,
        'error_rate': error_rate,
        'gen_accuracy': correct_gen / total_gen,
        'spam_accuracy': correct_spam / total_spam
    }

def get_training_size(gen_file, spam_file):
    """Get training data size from filenames"""
    sizes = {
        'gen': 1, 'spam': 1,
        'gen-times2': 2, 'spam-times2': 2,
        'gen-times4': 4, 'spam-times4': 4,
        'gen-times8': 8, 'spam-times8': 8
    }

    # Extract base names
    gen_base = gen_file.replace('-lambda0.005.model', '').replace('.model', '')
    spam_base = spam_file.replace('-lambda0.005.model', '').replace('.model', '')

    return sizes.get(gen_base, 1)

def create_learning_curve():
    """Create learning curve for different training data sizes"""

    # Define model pairs and their training sizes
    model_configs = [
        ('gen-lambda0.005.model', 'spam-lambda0.005.model', 1),
        ('gen-times2-lambda0.005.model', 'spam-times2-lambda0.005.model', 2),
        ('gen-times4-lambda0.005.model', 'spam-times4-lambda0.005.model', 4),
        ('gen-times8-lambda0.005.model', 'spam-times8-lambda0.005.model', 8)
    ]

    results = []

    print("Training Size\tError Rate\tGen Accuracy\tSpam Accuracy")
    print("-" * 60)

    for gen_model, spam_model, size in model_configs:
        print(f"Testing training size {size}x...")
        result = run_textcat(gen_model, spam_model)
        result['training_size'] = size

        results.append(result)

        print(f"{size}x\t\t{result['error_rate']:.3f}\t\t{result['gen_accuracy']:.3f}\t\t{result['spam_accuracy']:.3f}")

    # Create learning curve plot
    training_sizes = [r['training_size'] for r in results]
    error_rates = [r['error_rate'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, error_rates, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Training Data Size (multiplier)')
    plt.ylabel('Error Rate')
    plt.title('Learning Curve: Error Rate vs Training Data Size')
    plt.grid(True, alpha=0.3)
    plt.xticks(training_sizes)

    # Add value labels on points
    for i, (size, error) in enumerate(zip(training_sizes, error_rates)):
        plt.annotate(f'{error:.3f}', (size, error), textcoords="offset points",
                    xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
    print(f"\nLearning curve saved as learning_curve.png")

    # Analysis
    print(f"\nLearning Curve Analysis:")
    print(f"Initial error rate (1x): {error_rates[0]:.3f}")
    print(f"Final error rate (8x): {error_rates[-1]:.3f}")
    print(f"Improvement: {error_rates[0] - error_rates[-1]:.3f}")

    if len(error_rates) > 1:
        trend = "decreasing" if error_rates[-1] < error_rates[0] else "increasing"
        print(f"Overall trend: {trend}")

    return results

if __name__ == "__main__":
    results = create_learning_curve()