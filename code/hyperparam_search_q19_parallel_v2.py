#!/usr/bin/env python3
"""
Enhanced Parallelized Hyperparameter search for Question 19
Tests C ∈ {0, 0.1, 0.5, 1, 5} and d ∈ {10, 50, 200}
Using only gs-only lexicons

Features:
- Command-line control of number of processes
- Optional GPU support per process
- Progress tracking
- Configurable epochs
"""
import logging
import sys
import argparse
from pathlib import Path
from itertools import product
import json
import math
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple
import traceback
import os

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

def train_and_evaluate_config(args: Tuple[float, int, Dict]) -> Dict:
    """
    Train and evaluate a single configuration.
    This function runs in a separate process.
    
    Args:
        args: Tuple of (C, d, config_dict) where config_dict contains
              - vocab_file, train_gen, train_spam, dev_gen, dev_spam paths
              - lexicon_file, epochs
    
    Returns:
        Dictionary with results for this configuration
    """
    C, d, config = args
    
    # Set up logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[C={C}, d={d}] %(levelname)s: %(message)s'
    )
    
    try:
        # Extract configuration
        vocab_file = Path(config['vocab_file'])
        train_gen = Path(config['train_gen'])
        train_spam = Path(config['train_spam'])
        dev_gen = Path(config['dev_gen'])
        dev_spam = Path(config['dev_spam'])
        lexicon_file = Path(config['lexicon_file'])
        epochs = config['epochs']
        
        logging.info(f"Starting configuration C={C}, d={d}")
        
        # Load vocab
        vocab = read_vocab(vocab_file)
        
        # Check if lexicon file exists
        if not lexicon_file.exists():
            return {
                'C': C,
                'd': d,
                'error': f"Lexicon file not found: {lexicon_file}",
                'success': False
            }
        
        # Train model on gen corpus
        logging.info(f"Training gen model...")
        model_gen = EmbeddingLogLinearLanguageModel(
            vocab, 
            lexicon_file, 
            l2=C,
            epochs=epochs
        )
        model_gen.train(train_gen)
        
        # Evaluate gen model on gen dev set
        logging.info(f"Evaluating gen model...")
        gen_cross_entropy = compute_cross_entropy(model_gen, dev_gen)
        gen_perplexity = 2 ** gen_cross_entropy
        logging.info(f"Gen dev cross-entropy: {gen_cross_entropy:.4f} bits/token")
        
        # Clean up gen model
        del model_gen
        
        # Train model on spam corpus
        logging.info(f"Training spam model...")
        model_spam = EmbeddingLogLinearLanguageModel(
            vocab, 
            lexicon_file, 
            l2=C,
            epochs=epochs
        )
        model_spam.train(train_spam)
        
        # Evaluate spam model on spam dev set
        logging.info(f"Evaluating spam model...")
        spam_cross_entropy = compute_cross_entropy(model_spam, dev_spam)
        spam_perplexity = 2 ** spam_cross_entropy
        logging.info(f"Spam dev cross-entropy: {spam_cross_entropy:.4f} bits/token")
        
        # Clean up spam model
        del model_spam
        
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
        
        logging.info(f"Average cross-entropy: {avg_cross_entropy:.4f} bits/token")
        logging.info(f"✓ Completed configuration C={C}, d={d}")
        
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
            'spam_dev_files': len(spam_dev_files),
            'success': True
        }
        
        return result
        
    except Exception as e:
        error_msg = f"Error with config C={C}, d={d}: {e}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return {
            'C': C,
            'd': d,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }

def parse_args():
    parser = argparse.ArgumentParser(
        description='Parallelized hyperparameter search for Question 19',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Use all available cores
  %(prog)s -j 4                 # Limit to 4 parallel processes
  %(prog)s -e 15                # Train for 15 epochs
  %(prog)s -j 4 -e 15           # Combine options
  %(prog)s --C 0 0.1 0.5        # Test only C ∈ {0, 0.1, 0.5}
  %(prog)s --d 10 50            # Test only d ∈ {10, 50}
        """
    )
    
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=None,
        metavar='N',
        help='Number of parallel processes (default: all CPUs)'
    )
    
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='Number of training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--C',
        type=float,
        nargs='+',
        default=[0, 0.1, 0.5, 1, 5],
        metavar='VALUE',
        help='C values to test (default: 0 0.1 0.5 1 5)'
    )
    
    parser.add_argument(
        '--d',
        type=int,
        nargs='+',
        default=[10, 50, 200],
        choices=[10, 50, 200],
        metavar='DIM',
        help='d values to test, must be 10, 50, or 200 (default: 10 50 200)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('hyperparam_results_q19_parallel.json'),
        metavar='FILE',
        help='Output JSON file (default: hyperparam_results_q19_parallel.json)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File paths
    vocab_file = Path("vocab-genspam.txt")
    train_gen = Path("../data/gen_spam/train/gen")
    train_spam = Path("../data/gen_spam/train/spam")
    dev_gen = Path("../data/gen_spam/dev/gen")
    dev_spam = Path("../data/gen_spam/dev/spam")
    
    # Validate data files exist
    for path in [vocab_file, train_gen, train_spam, dev_gen, dev_spam]:
        if not path.exists():
            log.error(f"Required file/directory not found: {path}")
            sys.exit(1)
    
    # Hyperparameter grid
    C_values = args.C
    d_values = args.d
    epochs = args.epochs
    
    # Map d to gs-only lexicon files
    lexicon_map = {
        10: Path("../lexicons/words-gs-only-10.txt"),
        50: Path("../lexicons/words-gs-only-50.txt"),
        200: Path("../lexicons/words-gs-only-200.txt")
    }
    
    # Validate lexicons exist
    for d in d_values:
        if not lexicon_map[d].exists():
            log.error(f"Lexicon file not found: {lexicon_map[d]}")
            sys.exit(1)
    
    log.info("="*70)
    log.info("QUESTION 19 HYPERPARAMETER SEARCH (PARALLELIZED v2)")
    log.info("="*70)
    log.info(f"C values: {C_values}")
    log.info(f"d values: {d_values}")
    log.info(f"Using gs-only lexicons")
    log.info(f"Vocab: {vocab_file}")
    log.info(f"Train: {train_gen} and {train_spam}")
    log.info(f"Dev: {dev_gen} and {dev_spam}")
    log.info(f"Epochs: {epochs}")
    
    # Determine number of processes
    n_cpus = cpu_count()
    total_configs = len(C_values) * len(d_values)
    
    if args.jobs is None:
        n_processes = min(n_cpus, total_configs)
    else:
        n_processes = min(args.jobs, total_configs)
    
    log.info(f"Using {n_processes} processes (out of {n_cpus} CPUs)")
    log.info(f"Total configurations to test: {total_configs}")
    log.info(f"Output file: {args.output}")
    log.info("="*70)
    
    # Prepare configurations for parallel processing
    configs = []
    for C, d in product(C_values, d_values):
        config = {
            'vocab_file': str(vocab_file),
            'train_gen': str(train_gen),
            'train_spam': str(train_spam),
            'dev_gen': str(dev_gen),
            'dev_spam': str(dev_spam),
            'lexicon_file': str(lexicon_map[d]),
            'epochs': epochs
        }
        configs.append((C, d, config))
    
    # Run parallel evaluation
    log.info(f"Starting parallel evaluation of {len(configs)} configurations...")
    log.info("This may take 15-60 minutes depending on your hardware...")
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(train_and_evaluate_config, configs)
    
    log.info("All configurations completed!")
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    log.info(f"Successful: {len(successful_results)}/{len(results)}")
    
    if failed_results:
        log.warning(f"{len(failed_results)} configurations failed:")
        for r in failed_results:
            log.warning(f"  C={r['C']}, d={r['d']}: {r.get('error', 'Unknown error')}")
    
    # Find best result
    best_avg_cross_entropy = float('inf')
    best_params = None
    
    for result in successful_results:
        if result['avg_cross_entropy'] < best_avg_cross_entropy:
            best_avg_cross_entropy = result['avg_cross_entropy']
            best_params = result
    
    # Find best C for d=10 (as specified in homework)
    best_c_for_d10 = None
    best_ce_for_d10 = float('inf')
    for result in successful_results:
        if result['d'] == 10 and result['avg_cross_entropy'] < best_ce_for_d10:
            best_ce_for_d10 = result['avg_cross_entropy']
            best_c_for_d10 = result['C']
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'results': successful_results,
            'failed_results': failed_results,
            'best_params': best_params,
            'best_avg_cross_entropy': best_avg_cross_entropy,
            'best_c_for_d10': best_c_for_d10,
            'best_ce_for_d10': best_ce_for_d10,
            'n_processes_used': n_processes,
            'hyperparameters': {
                'C_values': C_values,
                'd_values': d_values,
                'epochs': epochs
            }
        }, f, indent=2)
    
    log.info(f"\n{'='*70}")
    log.info("HYPERPARAMETER SEARCH COMPLETE")
    log.info(f"{'='*70}")
    
    if best_params:
        log.info(f"\nBest overall parameters:")
        log.info(f"  C = {best_params['C']}")
        log.info(f"  d = {best_params['d']}")
        log.info(f"  Average cross-entropy: {best_avg_cross_entropy:.4f} bits/token")
        log.info(f"  Gen cross-entropy: {best_params['gen_cross_entropy']:.4f} bits/token")
        log.info(f"  Spam cross-entropy: {best_params['spam_cross_entropy']:.4f} bits/token")
        
        if best_c_for_d10 is not None:
            log.info(f"\nBest C for d=10 (as specified in Q19):")
            log.info(f"  C* = {best_c_for_d10}")
            log.info(f"  Average cross-entropy: {best_ce_for_d10:.4f} bits/token")
    else:
        log.error("No successful configurations!")
    
    log.info(f"\nResults saved to: {args.output}")
    
    # Print summary table
    if successful_results:
        log.info(f"\n{'='*70}")
        log.info("RESULTS SUMMARY")
        log.info(f"{'='*70}")
        log.info(f"{'C':>6} {'d':>5} {'Gen CE':>10} {'Spam CE':>10} {'Avg CE':>10}")
        log.info(f"{'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
        
        # Sort by d first, then C for better readability
        sorted_results = sorted(successful_results, key=lambda r: (r['d'], r['C']))
        for r in sorted_results:
            marker = " *" if (best_params and r['C'] == best_params['C'] and r['d'] == best_params['d']) else ""
            log.info(f"{r['C']:>6.1f} {r['d']:>5} {r['gen_cross_entropy']:>10.4f} {r['spam_cross_entropy']:>10.4f} {r['avg_cross_entropy']:>10.4f}{marker}")
        
        log.info(f"\n* = Best configuration")

if __name__ == "__main__":
    main()

