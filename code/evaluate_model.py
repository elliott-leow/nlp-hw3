#!/usr/bin/env python3
"""
Comprehensive evaluation script for language models
"""
import logging
import math
from pathlib import Path
import sys

from probs import LanguageModel, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def compute_perplexity(model: LanguageModel, file: Path) -> tuple[float, float, int]:
    """
    Compute perplexity and cross-entropy of model on a file.
    Returns: (perplexity, cross_entropy_bits_per_token, token_count)
    """
    log_prob_sum = 0.0
    token_count = 0
    
    for (x, y, z) in read_trigrams(file, model.vocab):
        log_prob = model.log_prob(x, y, z)
        log_prob_sum += log_prob
        token_count += 1
    
    # Average log probability (natural log)
    avg_log_prob = log_prob_sum / token_count
    
    # Convert to log2 for cross-entropy in bits
    avg_log2_prob = avg_log_prob / math.log(2)
    cross_entropy = -avg_log2_prob
    
    # Perplexity = 2^cross_entropy
    perplexity = 2 ** cross_entropy
    
    return perplexity, cross_entropy, token_count

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_file> <test_file1> [test_file2 ...]")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    model_path = Path(sys.argv[1])
    test_files = [Path(f) for f in sys.argv[2:]]
    
    log.info(f"Loading model from {model_path}")
    model = LanguageModel.load(model_path)
    log.info(f"Model loaded: {type(model).__name__}")
    log.info(f"Vocabulary size: {len(model.vocab)}")
    
    log.info(f"\n{'='*70}")
    log.info("EVALUATION RESULTS")
    log.info(f"{'='*70}\n")
    
    total_perplexity = 0.0
    total_cross_entropy = 0.0
    total_tokens = 0
    
    for test_file in test_files:
        log.info(f"Evaluating on: {test_file}")
        
        perplexity, cross_entropy, token_count = compute_perplexity(model, test_file)
        
        log.info(f"  Tokens:        {token_count:,}")
        log.info(f"  Perplexity:    {perplexity:.2f}")
        log.info(f"  Cross-Entropy: {cross_entropy:.4f} bits/token")
        log.info("")
        
        total_tokens += token_count
        # Weight by token count for overall average
        total_perplexity += perplexity * token_count
        total_cross_entropy += cross_entropy * token_count
    
    if len(test_files) > 1:
        avg_perplexity = total_perplexity / total_tokens
        avg_cross_entropy = total_cross_entropy / total_tokens
        
        log.info(f"{'='*70}")
        log.info("OVERALL AVERAGE (weighted by tokens)")
        log.info(f"{'='*70}")
        log.info(f"  Total Tokens:        {total_tokens:,}")
        log.info(f"  Average Perplexity:  {avg_perplexity:.2f}")
        log.info(f"  Average Cross-Entropy: {avg_cross_entropy:.4f} bits/token")

if __name__ == "__main__":
    main()

