#!/usr/bin/env python3
"""
Comprehensive evaluation comparing optimized vs baseline models
"""
import logging
import math
from pathlib import Path
import sys

from probs import LanguageModel, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def compute_perplexity(model: LanguageModel, file: Path) -> tuple[float, float, int]:
    """Compute perplexity and cross-entropy"""
    log_prob_sum = 0.0
    token_count = 0
    
    for (x, y, z) in read_trigrams(file, model.vocab):
        log_prob = model.log_prob(x, y, z)
        log_prob_sum += log_prob
        token_count += 1
    
    avg_log_prob = log_prob_sum / token_count
    avg_log2_prob = avg_log_prob / math.log(2)
    cross_entropy = -avg_log2_prob
    perplexity = 2 ** cross_entropy
    
    return perplexity, cross_entropy, token_count

def evaluate_dataset(model: LanguageModel, data_dir: Path, dataset_name: str):
    """Evaluate model on all files in a directory"""
    files = sorted(data_dir.glob("*.txt"))
    
    if not files:
        log.warning(f"No files found in {data_dir}")
        return None, None, 0
    
    total_log_prob = 0.0
    total_tokens = 0
    
    for file in files:
        log_prob_sum = 0.0
        token_count = 0
        
        for (x, y, z) in read_trigrams(file, model.vocab):
            log_prob = model.log_prob(x, y, z)
            log_prob_sum += log_prob
            token_count += 1
        
        total_log_prob += log_prob_sum
        total_tokens += token_count
    
    avg_log_prob = total_log_prob / total_tokens
    avg_log2_prob = avg_log_prob / math.log(2)
    cross_entropy = -avg_log2_prob
    perplexity = 2 ** cross_entropy
    
    return perplexity, cross_entropy, total_tokens

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_file>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    model_path = Path(sys.argv[1])
    
    log.info(f"{'='*70}")
    log.info(f"COMPREHENSIVE MODEL EVALUATION")
    log.info(f"{'='*70}\n")
    
    log.info(f"Loading model: {model_path}")
    model = LanguageModel.load(model_path)
    log.info(f"Model type: {type(model).__name__}")
    log.info(f"Vocabulary size: {len(model.vocab):,}\n")
    
    # Evaluate on different datasets
    datasets = [
        ("Gen Dev", Path("../data/gen_spam/dev/gen")),
        ("Spam Dev", Path("../data/gen_spam/dev/spam")),
    ]
    
    results = {}
    
    for dataset_name, data_dir in datasets:
        log.info(f"{'='*70}")
        log.info(f"Evaluating on: {dataset_name}")
        log.info(f"{'='*70}")
        
        perplexity, cross_entropy, token_count = evaluate_dataset(model, data_dir, dataset_name)
        
        if perplexity:
            log.info(f"Total tokens:   {token_count:,}")
            log.info(f"Perplexity:     {perplexity:.2f}")
            log.info(f"Cross-Entropy:  {cross_entropy:.4f} bits/token")
            log.info("")
            
            results[dataset_name] = {
                'perplexity': perplexity,
                'cross_entropy': cross_entropy,
                'tokens': token_count
            }
    
    # Summary
    log.info(f"{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")
    for name, res in results.items():
        log.info(f"{name:20s} | Perplexity: {res['perplexity']:8.2f} | "
                f"Cross-Entropy: {res['cross_entropy']:6.4f} bits/token")

if __name__ == "__main__":
    main()

