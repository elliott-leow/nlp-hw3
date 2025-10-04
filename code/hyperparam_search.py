#!/usr/bin/env python3
"""
Hyperparameter search for ImprovedLogLinearLanguageModel
"""
import logging
import sys
from pathlib import Path
from itertools import product
import json

from probs import read_vocab, ImprovedLogLinearLanguageModel, read_trigrams
import math

log = logging.getLogger(Path(__file__).stem)

def compute_perplexity(model, file: Path) -> float:
    """Compute perplexity of the model on a file."""
    log_prob_sum = 0.0
    token_count = 0
    
    for (x, y, z) in read_trigrams(file, model.vocab):
        log_prob = model.log_prob(x, y, z)
        log_prob_sum += log_prob
        token_count += 1
    
    # Perplexity = 2^(-average log2 probability)
    avg_log_prob = log_prob_sum / token_count
    # Convert natural log to log2
    avg_log2_prob = avg_log_prob / math.log(2)
    perplexity = 2 ** (-avg_log2_prob)
    
    return perplexity

def main():
    logging.basicConfig(level=logging.INFO)
    
    # File paths
    vocab_file = Path("vocab-genspam.txt")
    train_file = Path("../data/gen_spam/train/gen")
    dev_file = Path("../data/gen_spam/dev/gen/gen.101.156.txt")
    lexicon_file = Path("../lexicons/words-10.txt")
    
    # Hyperparameter grid (keep small for practical search)
    l2_values = [0.01, 0.1]
    dropout_values = [0.1, 0.2]
    batch_sizes = [32, 64]
    epochs_values = [2]  # Keep small for search
    
    log.info("Starting hyperparameter search...")
    log.info(f"Vocab: {vocab_file}")
    log.info(f"Train: {train_file}")
    log.info(f"Dev: {dev_file}")
    log.info(f"Lexicon: {lexicon_file}")
    
    # Load vocab once
    vocab = read_vocab(vocab_file)
    
    results = []
    best_perplexity = float('inf')
    best_params = None
    
    # Grid search
    total_configs = len(l2_values) * len(dropout_values) * len(batch_sizes) * len(epochs_values)
    config_num = 0
    
    for l2, dropout, batch_size, epochs in product(l2_values, dropout_values, batch_sizes, epochs_values):
        config_num += 1
        log.info(f"\n{'='*60}")
        log.info(f"Configuration {config_num}/{total_configs}")
        log.info(f"l2={l2}, dropout={dropout}, batch_size={batch_size}, epochs={epochs}")
        log.info(f"{'='*60}")
        
        try:
            # Create and train model
            model = ImprovedLogLinearLanguageModel(
                vocab, 
                lexicon_file, 
                l2=l2, 
                epochs=epochs,
                dropout=dropout,
                batch_size=batch_size
            )
            
            model.train(train_file)
            
            # Evaluate on dev set
            dev_perplexity = compute_perplexity(model, dev_file)
            
            log.info(f"Dev Perplexity: {dev_perplexity:.2f}")
            
            result = {
                'l2': l2,
                'dropout': dropout,
                'batch_size': batch_size,
                'epochs': epochs,
                'dev_perplexity': dev_perplexity
            }
            results.append(result)
            
            # Track best
            if dev_perplexity < best_perplexity:
                best_perplexity = dev_perplexity
                best_params = result
                log.info(f"*** NEW BEST MODEL! Perplexity: {best_perplexity:.2f} ***")
            
        except Exception as e:
            log.error(f"Error with config: {e}")
            continue
    
    # Save results
    results_file = Path("hyperparam_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'best_params': best_params,
            'best_perplexity': best_perplexity
        }, f, indent=2)
    
    log.info(f"\n{'='*60}")
    log.info("HYPERPARAMETER SEARCH COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Best parameters: {best_params}")
    log.info(f"Best dev perplexity: {best_perplexity:.2f}")
    log.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()

