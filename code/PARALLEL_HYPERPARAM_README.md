# Parallelized Hyperparameter Search for Question 19

## Overview

This script parallelizes the hyperparameter search for the gen/spam classification task. It tests different values of regularization parameter C and embedding dimensionality d to find the best configuration.

## Files

- `hyperparam_search_q19_parallel.py`: **Parallel version** - Runs multiple configurations simultaneously using multiprocessing
- `hyperparam_search_q19.py`: Sequential version - Runs configurations one at a time

## Hyperparameters Tested

As specified in Question 19:
- **C (L2 regularization)**: {0, 0.1, 0.5, 1, 5}
- **d (embedding dimensionality)**: {10, 50, 200}
- **Lexicons**: gs-only lexicons (`words-gs-only-{10,50,200}.txt`)
- **Total configurations**: 15 (5 C values × 3 d values)

## Usage

### Running the Parallel Version

```bash
cd code
./hyperparam_search_q19_parallel.py
```

Or:

```bash
cd code
python3 hyperparam_search_q19_parallel.py
```

### Running the Sequential Version

```bash
cd code
./hyperparam_search_q19.py
```

## Performance Comparison

### Parallel Version
- **Speed**: Runs multiple configurations simultaneously
- **CPU Usage**: Uses all available CPU cores (or up to 15 cores for 15 configurations)
- **Memory**: Higher memory usage (multiple models in memory at once)
- **Time**: ~15x faster on a machine with 15+ cores (best case)
- **Output**: Logs from different processes are interleaved

### Sequential Version
- **Speed**: Runs one configuration at a time
- **CPU Usage**: Uses 1 core at a time
- **Memory**: Lower memory usage (one model at a time)
- **Time**: Slower, but more predictable
- **Output**: Clean, ordered logs

## Resource Considerations

### Memory Requirements

Each configuration trains TWO models (one for gen, one for spam):
- d=10: ~10-20 MB per model
- d=50: ~50-100 MB per model
- d=200: ~200-400 MB per model

**Parallel version** may use up to **~6 GB RAM** if all 15 configurations run simultaneously.

### Recommended Usage

- **Many cores (8+)**: Use parallel version for maximum speed
- **Few cores (4 or less)**: Use parallel version, but benefit is smaller
- **Limited memory (<8 GB)**: Use sequential version or reduce parallelism
- **Debugging**: Use sequential version for cleaner logs

## Output

Both versions produce JSON output files:

### Parallel Version
- **File**: `hyperparam_results_q19_parallel.json`

### Sequential Version
- **File**: `hyperparam_results_q19.json`

### JSON Structure

```json
{
  "results": [
    {
      "C": 1.0,
      "d": 10,
      "lexicon": "words-gs-only-10.txt",
      "gen_cross_entropy": 8.5432,
      "gen_perplexity": 370.45,
      "spam_cross_entropy": 7.8901,
      "spam_perplexity": 237.12,
      "avg_cross_entropy": 8.2167,
      "gen_dev_files": 180,
      "spam_dev_files": 90,
      "success": true
    },
    ...
  ],
  "failed_results": [...],
  "best_params": {...},
  "best_avg_cross_entropy": 7.1234,
  "n_processes_used": 8
}
```

## Interpreting Results

The script evaluates each configuration by:

1. Training a model on **gen** corpus
2. Evaluating on **gen** dev set → **gen_cross_entropy**
3. Training a model on **spam** corpus
4. Evaluating on **spam** dev set → **spam_cross_entropy**
5. Computing weighted average → **avg_cross_entropy**

**Lower cross-entropy is better** (indicates better predictions).

The **best configuration** is the one with the **lowest avg_cross_entropy**.

## Strategy for Question 19

According to the homework:
1. Find best C when d=10 → call it C*
2. Hold C* fixed and try different d values
3. Report best results and hyperparameters

### Workflow

1. Run the parallel script:
   ```bash
   ./hyperparam_search_q19_parallel.py
   ```

2. Check results:
   ```bash
   cat hyperparam_results_q19_parallel.json | python3 -m json.tool
   ```

3. Look for best parameters in output:
   - Check terminal output for summary table
   - Or parse JSON for "best_params"

4. Report in answers.md:
   - Best C* for d=10
   - Best overall configuration (C, d)
   - Cross-entropy values for gen and spam

## Troubleshooting

### "Out of memory" errors
- Reduce parallelism by editing the script:
  ```python
  n_processes = min(4, n_cpus)  # Limit to 4 processes
  ```

### Slow progress
- Check if models are training (should see epoch logs)
- Ensure you're in the `code/` directory (paths are relative)
- Check that data files exist in `../data/gen_spam/`

### Import errors
- Ensure `probs.py` is in the same directory
- Check that PyTorch is installed: `python3 -c "import torch; print(torch.__version__)"`

## Implementation Details

### Multiprocessing Strategy

The parallel version uses `multiprocessing.Pool` to:
1. Create worker processes (one per CPU core, up to 15)
2. Distribute configurations to workers
3. Each worker independently trains and evaluates
4. Main process collects results

### Key Differences from Sequential

1. **Process isolation**: Each configuration runs in its own process
2. **Independent logging**: Logs from different processes may interleave
3. **Model cleanup**: Models are explicitly deleted after evaluation
4. **Error handling**: Failed configurations don't stop the entire run

## Tips

- **Monitor progress**: Use `htop` or `top` to watch CPU usage
- **Check logs**: Look for "[C=X, d=Y]" prefixes in output
- **Compare results**: Both versions should give identical cross-entropy values
- **Save output**: Redirect to file for later analysis: `./hyperparam_search_q19_parallel.py 2>&1 | tee search.log`

## Expected Runtime

Rough estimates (on modern CPU):
- **Sequential**: ~2-4 hours (15 configs × 8-15 min each)
- **Parallel (8 cores)**: ~30-60 minutes
- **Parallel (16+ cores)**: ~15-30 minutes

*Actual time depends on CPU, data size, and epochs.*

