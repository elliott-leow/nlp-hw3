# Question 19 Hyperparameter Search

## Overview

This implements the proper hyperparameter search for Question 19 as specified:
- **C (L2 regularization)**: {0, 0.1, 0.5, 1, 5}
- **d (embedding dimension)**: {10, 50, 200}
- **Lexicons**: gs-only lexicons only

## Files Created

1. **`hyperparam_search_q19.py`** - Main search script
   - Tests all 15 configurations (5 C values × 3 d values)
   - Trains separate models for gen and spam
   - Computes cross-entropy (bits/token) for each
   - Saves results to `hyperparam_results_q19.json`

2. **`check_q19_progress.sh`** - Monitor progress
   - Shows current configuration being tested
   - Displays recent log entries
   - Reports best results so far

3. **`update_q19_results.py`** - Generate results summary
   - Formats results as markdown table
   - Analyzes effect of C and d
   - Compares to add-λ baseline
   - Ready to paste into answers.md

## Current Status

The search is currently running in the background. Progress is logged to `hyperparam_search_q19.log`.

### Monitoring Progress

```bash
cd code
./check_q19_progress.sh
```

Or view the log directly:
```bash
tail -f hyperparam_search_q19.log
```

### Expected Runtime

- Each configuration: ~15-30 minutes (depending on d)
- Total: 15 configurations × 2 models × ~20 min = **~10 hours**

Current progress: Processing configuration 1/15 (C=0, d=10)

## When Complete

Once the search finishes, run:

```bash
cd code
./update_q19_results.py
```

This will display a formatted results table and analysis. Copy the output and paste it into `answers.md` to replace the "*Search in progress...*" placeholder.

## What's Being Tested

### Research Questions

1. **Does regularization matter?**
   - Compare C=0 vs. C>0
   - Expectation: Moderate effect (embeddings already smooth)

2. **What embedding dimension is best?**
   - Test d ∈ {10, 50, 200}
   - Trade-off: expressiveness vs. overfitting

3. **How does log-linear compare to add-λ?**
   - Baseline: add-λ (λ=0.005) = 9.075 bits/token
   - Expected: Log-linear better due to semantic generalization

## Results Format

For each configuration, we report:
- **Gen CE**: Cross-entropy on gen dev set (bits/token)
- **Spam CE**: Cross-entropy on spam dev set (bits/token)
- **Avg CE**: Weighted average cross-entropy
- **Gen PPL**: Perplexity on gen dev set (2^CE)
- **Spam PPL**: Perplexity on spam dev set (2^CE)

## Troubleshooting

If the script stops or crashes:

1. Check the log for errors:
   ```bash
   tail -100 hyperparam_search_q19.log | grep -i error
   ```

2. Check if the process is still running:
   ```bash
   ps aux | grep hyperparam_search_q19
   ```

3. Resume from where it stopped (the script saves results incrementally):
   ```bash
   # The script appends to results, so you can restart it
   ./hyperparam_search_q19.py 2>&1 | tee -a hyperparam_search_q19.log
   ```

## Key Implementation Details

- Uses `EmbeddingLogLinearLanguageModel` (basic log-linear, not improved)
- Trains for 10 epochs per model
- Evaluates on full dev sets (not just single files)
- Reports cross-entropy in bits/token (not nats)
- Saves all results to JSON for easy analysis

