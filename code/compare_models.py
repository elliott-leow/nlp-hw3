#!/usr/bin/env python3
"""
Compare baseline vs optimized log-linear models
"""
import logging
import math
from pathlib import Path

from probs import LanguageModel, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def evaluate_model(model: LanguageModel, data_dir: Path, dataset_name: str):
    """Evaluate model on all files in a directory"""
    files = sorted(data_dir.glob("*.txt"))
    
    if not files:
        log.warning(f"No files found in {data_dir}")
        return None
    
    total_log_prob = 0.0
    total_tokens = 0
    
    for file in files:
        for (x, y, z) in read_trigrams(file, model.vocab):
            log_prob = model.log_prob(x, y, z)
            total_log_prob += log_prob
            total_tokens += 1
    
    avg_log_prob = total_log_prob / total_tokens
    avg_log2_prob = avg_log_prob / math.log(2)
    cross_entropy = -avg_log2_prob
    
    # Handle potential overflow in perplexity calculation
    if cross_entropy > 100:
        perplexity_str = f">2^100 (too large)"
    else:
        try:
            perplexity = 2 ** cross_entropy
            perplexity_str = f"{perplexity:.2f}"
        except OverflowError:
            perplexity_str = f"2^{cross_entropy:.2f} (overflow)"
    
    return {
        'tokens': total_tokens,
        'cross_entropy': cross_entropy,
        'perplexity_str': perplexity_str,
        'avg_log_prob': avg_log_prob
    }

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    baseline_path = Path("baseline_loglinear.model")
    optimized_path = Path("optimized_loglinear.model")
    
    log.info(f"\n{'='*80}")
    log.info("BASELINE vs OPTIMIZED LOG-LINEAR MODEL COMPARISON")
    log.info(f"{'='*80}\n")
    
    # Load models
    log.info("Loading models...")
    baseline = LanguageModel.load(baseline_path)
    optimized = LanguageModel.load(optimized_path)
    
    log.info(f"Baseline:  {type(baseline).__name__}")
    log.info(f"Optimized: {type(optimized).__name__}\n")
    
    # Datasets to evaluate
    datasets = [
        ("Gen Dev", Path("../data/gen_spam/dev/gen")),
        ("Spam Dev", Path("../data/gen_spam/dev/spam")),
    ]
    
    results = {}
    
    for dataset_name, data_dir in datasets:
        log.info(f"{'='*80}")
        log.info(f"Evaluating: {dataset_name}")
        log.info(f"{'='*80}")
        
        log.info(f"\nBaseline Model:")
        baseline_res = evaluate_model(baseline, data_dir, dataset_name)
        log.info(f"  Tokens:        {baseline_res['tokens']:,}")
        log.info(f"  Cross-Entropy: {baseline_res['cross_entropy']:.4f} bits/token")
        log.info(f"  Perplexity:    {baseline_res['perplexity_str']}")
        
        log.info(f"\nOptimized Model:")
        optimized_res = evaluate_model(optimized, data_dir, dataset_name)
        log.info(f"  Tokens:        {optimized_res['tokens']:,}")
        log.info(f"  Cross-Entropy: {optimized_res['cross_entropy']:.4f} bits/token")
        log.info(f"  Perplexity:    {optimized_res['perplexity_str']}")
        
        # Calculate improvement
        ce_reduction = baseline_res['cross_entropy'] - optimized_res['cross_entropy']
        ce_improvement = (ce_reduction / baseline_res['cross_entropy']) * 100
        
        log.info(f"\n✨ IMPROVEMENT:")
        log.info(f"  Cross-Entropy Reduction: {ce_reduction:.4f} bits/token ({ce_improvement:.1f}% better)")
        log.info(f"  Average Log-Prob Gain:   {optimized_res['avg_log_prob'] - baseline_res['avg_log_prob']:.4f}\n")
        
        results[dataset_name] = {
            'baseline': baseline_res,
            'optimized': optimized_res,
            'improvement': ce_improvement
        }
    
    # Overall summary
    log.info(f"{'='*80}")
    log.info("SUMMARY")
    log.info(f"{'='*80}\n")
    
    log.info(f"{'Dataset':<20} | {'Baseline CE':>12} | {'Optimized CE':>13} | {'Improvement':>12}")
    log.info(f"{'-'*80}")
    for name, res in results.items():
        log.info(f"{name:<20} | {res['baseline']['cross_entropy']:>11.4f}  | "
                f"{res['optimized']['cross_entropy']:>12.4f}  | {res['improvement']:>11.1f}%")
    
    avg_improvement = sum(r['improvement'] for r in results.values()) / len(results)
    log.info(f"\n{'Average Improvement:':<20} {avg_improvement:>54.1f}%")
    
    log.info(f"\n{'='*80}")
    log.info("KEY OPTIMIZATIONS APPLIED:")
    log.info(f"{'='*80}")
    log.info("✓ Xavier initialization (vs zeros)")
    log.info("✓ Bias terms added")
    log.info("✓ AdamW optimizer (vs SGD)")
    log.info("✓ Learning rate warmup + cosine annealing")
    log.info("✓ Label smoothing (ε=0.1)")
    log.info("✓ Gradient accumulation (effective batch size 128)")
    log.info("✓ Dropout regularization (15%)")
    log.info("✓ Smart L2 regularization (weights only, not biases)")
    log.info("✓ Gradient clipping")
    log.info("✓ Data shuffling per epoch")

if __name__ == "__main__":
    main()

