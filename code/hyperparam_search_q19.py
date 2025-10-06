#!/usr/bin/env python3
"""
Hyperparameter search for Question 19
Tests C ∈ {0, 0.1, 0.5, 1, 5} and d ∈ {10, 50, 200}
Using only gs-only lexicons
"""
import logging
import sys
from pathlib import Path
from itertools import product
import json
import math

from probs import read_vocab, EmbeddingLogLinearLanguageModel, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def compute_cross_entropy(model, path: Path) -> float:
    """Compute cross-entropy (bits per token) of the model on a file or directory of files."""
    log_prob_sum = 0.0
    token_count = 0
    
    # If path is a directory, process all .txt files in it
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found in directory: {path}")
        for file in files:
            for (x, y, z) in read_trigrams(file, model.vocab):
                log_prob = model.log_prob(x, y, z)
                log_prob_sum += log_prob
                token_count += 1
    else:
        # Process single file
        for (x, y, z) in read_trigrams(path, model.vocab):
            log_prob = model.log_prob(x, y, z)
            log_prob_sum += log_prob
            token_count += 1
    
    # Cross-entropy in bits per token = -average log2 probability
    avg_log_prob = log_prob_sum / token_count
    # Convert natural log to log2
    avg_log2_prob = avg_log_prob / math.log(2)
    cross_entropy = -avg_log2_prob
    
    return cross_entropy

def compute_perplexity(model, file: Path) -> float:
    """Compute perplexity of the model on a file."""
    cross_entropy = compute_cross_entropy(model, file)
    perplexity = 2 ** cross_entropy
    return perplexity

def main():
    logging.basicConfig(level=logging.INFO)
    
    # File paths
    vocab_file = Path("vocab-genspam.txt")
    train_gen = Path("../data/gen_spam/train/gen")
    train_spam = Path("../data/gen_spam/train/spam")
    dev_gen = Path("../data/gen_spam/dev/gen")
    dev_spam = Path("../data/gen_spam/dev/spam")
    
    # Hyperparameter grid as specified in Question 19
    C_values = [0, 0.1, 0.5, 1, 5]
    d_values = [10, 50, 200]
    epochs = 10  # Reasonable number of epochs
    
    # Map d to gs-only lexicon files
    lexicon_map = {
        10: Path("../lexicons/words-gs-only-10.txt"),
        50: Path("../lexicons/words-gs-only-50.txt"),
        200: Path("../lexicons/words-gs-only-200.txt")
    }
    
    log.info("="*70)
    log.info("QUESTION 19 HYPERPARAMETER SEARCH")
    log.info("="*70)
    log.info(f"C values: {C_values}")
    log.info(f"d values: {d_values}")
    log.info(f"Using gs-only lexicons")
    log.info(f"Vocab: {vocab_file}")
    log.info(f"Train: {train_gen} and {train_spam}")
    log.info(f"Dev: {dev_gen} and {dev_spam}")
    log.info(f"Epochs: {epochs}")
    log.info("="*70)
    
    # Load vocab once
    vocab = read_vocab(vocab_file)
    
    results = []
    best_avg_cross_entropy = float('inf')
    best_params = None
    
    # Grid search
    total_configs = len(C_values) * len(d_values)
    config_num = 0
    
    for C, d in product(C_values, d_values):
        config_num += 1
        lexicon_file = lexicon_map[d]
        
        log.info(f"\n{'='*70}")
        log.info(f"Configuration {config_num}/{total_configs}")
        log.info(f"C={C}, d={d}, lexicon={lexicon_file.name}")
        log.info(f"{'='*70}")
        
        # Check if lexicon file exists
        if not lexicon_file.exists():
            log.error(f"Lexicon file not found: {lexicon_file}")
            continue
        
        try:
            # Train model on gen corpus
            log.info(f"Training on gen corpus...")
            model_gen = EmbeddingLogLinearLanguageModel(
                vocab, 
                lexicon_file, 
                l2=C,  # C is the L2 regularization strength
                epochs=epochs
            )
            model_gen.train(train_gen)
            
            # Evaluate gen model on gen dev set
            log.info(f"Evaluating gen model on gen dev set...")
            gen_cross_entropy = compute_cross_entropy(model_gen, dev_gen)
            gen_perplexity = 2 ** gen_cross_entropy
            log.info(f"Gen dev cross-entropy: {gen_cross_entropy:.4f} bits/token")
            log.info(f"Gen dev perplexity: {gen_perplexity:.2f}")
            
            # Train model on spam corpus
            log.info(f"Training on spam corpus...")
            model_spam = EmbeddingLogLinearLanguageModel(
                vocab, 
                lexicon_file, 
                l2=C,
                epochs=epochs
            )
            model_spam.train(train_spam)
            
            # Evaluate spam model on spam dev set
            log.info(f"Evaluating spam model on spam dev set...")
            spam_cross_entropy = compute_cross_entropy(model_spam, dev_spam)
            spam_perplexity = 2 ** spam_cross_entropy
            log.info(f"Spam dev cross-entropy: {spam_cross_entropy:.4f} bits/token")
            log.info(f"Spam dev perplexity: {spam_perplexity:.2f}")
            
            # Compute average cross-entropy (weighted by number of files)
            # Count dev files
            gen_dev_files = list(dev_gen.glob("*.txt"))
            spam_dev_files = list(dev_spam.glob("*.txt"))
            total_dev_files = len(gen_dev_files) + len(spam_dev_files)
            
            # Weighted average
            avg_cross_entropy = (
                (len(gen_dev_files) * gen_cross_entropy + 
                 len(spam_dev_files) * spam_cross_entropy) / 
                total_dev_files
            )
            
            log.info(f"Average cross-entropy: {avg_cross_entropy:.4f} bits/token")
            
            result = {
                'C': C,
                'd': d,
                'lexicon': lexicon_file.name,
                'gen_cross_entropy': gen_cross_entropy,
                'gen_perplexity': gen_perplexity,
                'spam_cross_entropy': spam_cross_entropy,
                'spam_perplexity': spam_perplexity,
                'avg_cross_entropy': avg_cross_entropy,
                'gen_dev_files': len(gen_dev_files),
                'spam_dev_files': len(spam_dev_files)
            }
            results.append(result)
            
            # Track best
            if avg_cross_entropy < best_avg_cross_entropy:
                best_avg_cross_entropy = avg_cross_entropy
                best_params = result
                log.info(f"*** NEW BEST MODEL! Avg Cross-Entropy: {best_avg_cross_entropy:.4f} bits/token ***")
            
        except Exception as e:
            log.error(f"Error with config C={C}, d={d}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    results_file = Path("hyperparam_results_q19.json")
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'best_params': best_params,
            'best_avg_cross_entropy': best_avg_cross_entropy
        }, f, indent=2)
    
    log.info(f"\n{'='*70}")
    log.info("HYPERPARAMETER SEARCH COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Best parameters: C={best_params['C']}, d={best_params['d']}")
    log.info(f"Best average cross-entropy: {best_avg_cross_entropy:.4f} bits/token")
    log.info(f"Results saved to: {results_file}")
    
    # Print summary table
    log.info(f"\n{'='*70}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'C':>5} {'d':>5} {'Gen CE':>10} {'Spam CE':>10} {'Avg CE':>10}")
    log.info(f"{'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        log.info(f"{r['C']:>5.1f} {r['d']:>5} {r['gen_cross_entropy']:>10.4f} {r['spam_cross_entropy']:>10.4f} {r['avg_cross_entropy']:>10.4f}")

if __name__ == "__main__":
    main()

