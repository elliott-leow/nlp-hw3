#!/usr/bin/env python3
"""Compute cross-entropy for different lambda values"""
import subprocess
import re

lambdas = [5, 0.5, 0.05, 0.005, 0.0005, 1]

print("Lambda\tGen CE\tSpam CE\tCombined CE")

for lam in lambdas:
    # Get gen cross-entropy
    gen_result = subprocess.run(
        ["python", "fileprob.py", f"gen-lambda-{lam}.model", *[f"../data/gen_spam/dev/gen/{f}" for f in subprocess.run(["ls", "../data/gen_spam/dev/gen/"], capture_output=True, text=True).stdout.strip().split("\n")]],
        capture_output=True,
        text=True
    )
    gen_ce = float(re.search(r'Overall cross-entropy:\s+([\d.]+)', gen_result.stdout).group(1))

    # Get spam cross-entropy
    spam_result = subprocess.run(
        ["python", "fileprob.py", f"spam-lambda-{lam}.model", *[f"../data/gen_spam/dev/spam/{f}" for f in subprocess.run(["ls", "../data/gen_spam/dev/spam/"], capture_output=True, text=True).stdout.strip().split("\n")]],
        capture_output=True,
        text=True
    )
    spam_ce = float(re.search(r'Overall cross-entropy:\s+([\d.]+)', spam_result.stdout).group(1))

    # Compute combined cross-entropy
    # Extract total tokens from each
    gen_lines = gen_result.stdout.strip().split('\n')
    spam_lines = spam_result.stdout.strip().split('\n')

    # Parse the output to get tokens and bits
    from probs import num_tokens
    from pathlib import Path
    import glob

    gen_tokens = sum(num_tokens(Path(f)) for f in glob.glob('../data/gen_spam/dev/gen/*'))
    spam_tokens = sum(num_tokens(Path(f)) for f in glob.glob('../data/gen_spam/dev/spam/*'))

    gen_bits = gen_ce * gen_tokens
    spam_bits = spam_ce * spam_tokens
    combined_ce = (gen_bits + spam_bits) / (gen_tokens + spam_tokens)

    print(f"{lam}\t{gen_ce:.5f}\t{spam_ce:.5f}\t{combined_ce:.5f}")
