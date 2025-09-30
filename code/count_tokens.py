#!/usr/bin/env python3

import glob
import sys

def count_tokens_in_files(file_pattern):
    """Count total tokens in files matching pattern"""
    files = glob.glob(file_pattern)
    total_tokens = 0

    for file_path in files:
        with open(file_path, 'r') as f:
            content = f.read()
            # Split on whitespace and count tokens
            tokens = content.split()
            total_tokens += len(tokens)

    return total_tokens

if __name__ == "__main__":
    gen_tokens = count_tokens_in_files('../data/gen_spam/dev/gen/*')
    spam_tokens = count_tokens_in_files('../data/gen_spam/dev/spam/*')

    print(f"Gen dev tokens: {gen_tokens}")
    print(f"Spam dev tokens: {spam_tokens}")
    print(f"Total dev tokens: {gen_tokens + spam_tokens}")