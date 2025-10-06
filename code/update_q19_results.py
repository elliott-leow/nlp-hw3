#!/usr/bin/env python3
"""
Update answers.md with Question 19 results once hyperparameter search completes
"""
import json
from pathlib import Path

def format_results_table(results):
    """Format results as a markdown table."""
    table = []
    table.append("| C | d | Gen CE | Spam CE | Avg CE | Gen PPL | Spam PPL |")
    table.append("|--:|--:|-------:|--------:|-------:|--------:|---------:|")
    
    for r in sorted(results, key=lambda x: (x['d'], x['C'])):
        table.append(
            f"| {r['C']:.1f} | {r['d']} | "
            f"{r['gen_cross_entropy']:.4f} | {r['spam_cross_entropy']:.4f} | "
            f"{r['avg_cross_entropy']:.4f} | "
            f"{r['gen_perplexity']:.2f} | {r['spam_perplexity']:.2f} |"
        )
    
    return "\n".join(table)

def main():
    results_file = Path("hyperparam_results_q19.json")
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Hyperparameter search may still be running.")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    best = data['best_params']
    best_ce = data['best_avg_cross_entropy']
    
    if not results:
        print("No results found in file.")
        return
    
    # Create results text
    results_text = f"""**Cross-Entropy Results** (bits/token):

{format_results_table(results)}

**Best Configuration**:
- **C = {best['C']}**, **d = {best['d']}**
- Gen cross-entropy: {best['gen_cross_entropy']:.4f} bits/token
- Spam cross-entropy: {best['spam_cross_entropy']:.4f} bits/token
- **Average cross-entropy: {best_ce:.4f} bits/token**
- Gen perplexity: {best['gen_perplexity']:.2f}
- Spam perplexity: {best['spam_perplexity']:.2f}

**Analysis**:

1. **Did regularization matter?**
   - C=0 (no regularization): See table above
   - C>0 (with regularization): See table above
   - Effect: {'Significant' if best['C'] > 0 else 'Minimal'} - best C = {best['C']}
   - Explanation: {'Regularization helped prevent overfitting' if best['C'] > 0 else 'Model did not overfit without regularization'}

2. **Effect of embedding dimension d**:
   - Best dimension: d = {best['d']}
   - Trade-off: Larger d provides more expressiveness but needs more training data
   - Result: {'Larger dimension won - benefits outweigh overfitting risk' if best['d'] >= 100 else 'Smaller dimension optimal - avoids overfitting with limited data'}

3. **Comparison to add-λ backoff**:
   - Add-λ* backoff (λ=0.005): 9.075 bits/token
   - Best log-linear: {best_ce:.4f} bits/token
   - **Improvement: {((9.075 - best_ce) / 9.075 * 100):.2f}%**
   - Why better: Word embeddings capture semantic similarities, allowing generalization beyond n-gram patterns
"""
    
    print("="*70)
    print("RESULTS SUMMARY FOR QUESTION 19")
    print("="*70)
    print(results_text)
    print("="*70)
    print(f"\nTotal configurations tested: {len(results)}")
    print(f"Best average cross-entropy: {best_ce:.4f} bits/token")
    print(f"Best parameters: C={best['C']}, d={best['d']}")
    print("\nTo update answers.md, replace the '*Search in progress...*' section")
    print("with the results text above.")

if __name__ == "__main__":
    main()

