#!/usr/bin/env python3
"""
Compute error rates for different training sizes
"""
import subprocess
import sys

def compute_error_rate(gen_model, spam_model):
    """Run textcat and compute error rate"""
    # Run textcat and capture output
    result = subprocess.run([
        './textcat.py', gen_model, spam_model, '0.7',
        '../data/gen_spam/dev/gen/*', '../data/gen_spam/dev/spam/*'
    ], shell=True, capture_output=True, text=True)
    
    lines = result.stdout.strip().split('\n')
    
    # Count correct and incorrect classifications
    gen_correct = 0
    gen_incorrect = 0
    spam_correct = 0
    spam_incorrect = 0
    
    for line in lines:
        if line.startswith('gen-') or line.startswith('spam-') or line.startswith('gen.model') or line.startswith('spam.model'):
            parts = line.split('\t')
            if len(parts) == 2:
                predicted = parts[0]
                filepath = parts[1]
                filename = filepath.split('/')[-1]
                
                if filename.startswith('gen.'):
                    # True gen file
                    if 'gen' in predicted:
                        gen_correct += 1
                    else:
                        gen_incorrect += 1
                elif filename.startswith('spam.'):
                    # True spam file  
                    if 'spam' in predicted:
                        spam_correct += 1
                    else:
                        spam_incorrect += 1
    
    total_correct = gen_correct + spam_correct
    total_incorrect = gen_incorrect + spam_incorrect
    total = total_correct + total_incorrect
    
    if total > 0:
        error_rate = total_incorrect / total * 100
    else:
        error_rate = 0
    
    return error_rate, total_correct, total_incorrect, total

# Test each training size
sizes = ['1x', '2x', '4x', '8x']
print("Training Size\tError Rate\tCorrect\tIncorrect\tTotal")

for size in sizes:
    gen_model = f'gen-{size}.model'
    spam_model = f'spam-{size}.model'
    error_rate, correct, incorrect, total = compute_error_rate(gen_model, spam_model)
    print(f"{size}\t\t{error_rate:.1f}%\t\t{correct}\t{incorrect}\t\t{total}")
