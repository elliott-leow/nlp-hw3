#!/usr/bin/env python3

import sys
import subprocess
import re

def get_cross_entropy_data(model_file, test_files):
    """Run fileprob.py and extract total bits and total tokens"""
    cmd = ['python', 'fileprob.py', model_file] + test_files
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse the output to find total bits and total tokens
    lines = result.stdout.strip().split('\n')
    total_bits = None

    for line in lines:
        if 'Total' in line and 'bits' in line:
            # Extract total bits
            match = re.search(r'Total\s+log\s+probability:\s+(-?\d+\.?\d*)\s+bits', line)
            if match:
                total_bits = float(match.group(1))
                break

    if total_bits is None:
        return None, None

    for line in lines:
        if 'Overall cross-entropy' in line:
            # Extract bits per token to calculate total tokens
            match = re.search(r'Overall cross-entropy:\s+(\d+\.?\d*)\s+bits per token', line)
            if match:
                bits_per_token = float(match.group(1))
                total_tokens = abs(total_bits) / bits_per_token
                return abs(total_bits), total_tokens

    return None, None

def calculate_combined_entropy(lambda_val):
    """Calculate combined cross-entropy for a given lambda"""
    # Model names
    gen_model = f"gen-lambda{lambda_val}.model" if lambda_val != 1 else "gen.model"
    spam_model = f"spam-lambda{lambda_val}.model" if lambda_val != 1 else "spam.model"

    # Test files
    import glob
    gen_files = glob.glob('../data/gen_spam/dev/gen/*')
    spam_files = glob.glob('../data/gen_spam/dev/spam/*')

    # Get data for gen model on gen files
    gen_bits, gen_tokens = get_cross_entropy_data(gen_model, gen_files)

    # Get data for spam model on spam files
    spam_bits, spam_tokens = get_cross_entropy_data(spam_model, spam_files)

    if gen_bits is None or spam_bits is None:
        return None

    # Calculate combined cross-entropy
    total_bits = gen_bits + spam_bits
    total_tokens = gen_tokens + spam_tokens
    combined_entropy = total_bits / total_tokens

    return combined_entropy, total_bits, total_tokens

if __name__ == "__main__":
    lambdas = [5, 1, 0.5, 0.05, 0.005, 0.0005]

    results = []
    for lam in lambdas:
        entropy = calculate_combined_entropy(lam)
        if entropy:
            combined_entropy, total_bits, total_tokens = entropy
            results.append((lam, combined_entropy, total_bits, total_tokens))
            print(f"Lambda={lam}: Combined cross-entropy={combined_entropy:.5f} bits per token (Total bits={total_bits:.1f}, Total tokens={total_tokens:.1f})")

    # Find minimum
    if results:
        min_result = min(results, key=lambda x: x[1])
        print(f"\nMinimum combined cross-entropy: Lambda={min_result[0]} with {min_result[1]:.5f} bits per token")