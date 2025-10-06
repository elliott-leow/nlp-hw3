# Quick Start Guide: Hyperparameter Search for Question 19

## TL;DR - Run This

```bash
cd code
./hyperparam_search_q19_parallel_v2.py -j 8 -e 10
```

This will test all C and d values using 8 processes and 10 epochs per model.

---

## Available Scripts

### 1. **hyperparam_search_q19_parallel_v2.py** ⭐ RECOMMENDED
**Enhanced parallel version with full control**

```bash
# Basic usage (uses all CPUs)
./hyperparam_search_q19_parallel_v2.py

# Limit to 4 processes (good for 4-core machines)
./hyperparam_search_q19_parallel_v2.py -j 4

# Quick test with fewer epochs
./hyperparam_search_q19_parallel_v2.py -e 5

# Test specific C values only
./hyperparam_search_q19_parallel_v2.py --C 0 0.1 0.5

# Test specific d values only
./hyperparam_search_q19_parallel_v2.py --d 10 50

# Full customization
./hyperparam_search_q19_parallel_v2.py -j 4 -e 15 --C 0.1 1 5 --d 10 50

# Show help
./hyperparam_search_q19_parallel_v2.py --help
```

**Pros:**
- Most flexible
- Command-line control
- Progress indicators
- Best for iterative experimentation

### 2. **hyperparam_search_q19_parallel.py**
**Simple parallel version**

```bash
./hyperparam_search_q19_parallel.py
```

**Pros:**
- Simpler code
- No configuration needed
- Good for one-shot full search

**Cons:**
- No command-line options
- Uses all CPUs (may freeze your machine)

### 3. **hyperparam_search_q19.py**
**Sequential version (original)**

```bash
./hyperparam_search_q19.py
```

**Pros:**
- Easiest to debug
- Cleanest logs
- Lowest memory usage

**Cons:**
- Slowest (15x slower on 15-core machine)

### 4. **run_hyperparam_search.sh**
**Wrapper script**

```bash
# Parallel mode
./run_hyperparam_search.sh

# Sequential mode
./run_hyperparam_search.sh --sequential
```

---

## Which Script Should I Use?

### First Time Running?
→ Use **v2** with limited processes to avoid freezing:
```bash
./hyperparam_search_q19_parallel_v2.py -j 4
```

### Quick Experiment?
→ Use **v2** with fewer epochs and specific values:
```bash
./hyperparam_search_q19_parallel_v2.py -e 5 --C 0.1 1 --d 10
```

### Full Search on Powerful Machine?
→ Use **v2** with all cores:
```bash
./hyperparam_search_q19_parallel_v2.py
```

### Debugging Issues?
→ Use **sequential**:
```bash
./hyperparam_search_q19.py
```

---

## Understanding the Task (Question 19)

From the homework:
> "Report cross-entropy on gen spam with C = 1. Then experiment with other values of C > 0.
> Also experiment with the different lexicons, where d is the embedding dimensionality. Report your
> best results and the hyperparameters that achieved them. For this homework, it's enough to identify
> a pretty good value of C when d = 10—call it C∗—and then hold C∗ fixed while trying different
> lexicons. For this homework, it's also enough to try just C ∈ {0, 0.1, 0.5, 1, 5} and d ∈ {10, 50, 200},
> and to use only the gs-only lexicons."

### Task Breakdown:

1. **First**: Test all C values with d=10 → Find best C* for d=10
2. **Then**: Fix C=C* and test different d values
3. **Report**: Best configuration and cross-entropy values

### Suggested Workflow:

```bash
# Step 1: Test all combinations to find best C for d=10
cd code
./hyperparam_search_q19_parallel_v2.py

# Step 2: Check results
cat hyperparam_results_q19_parallel.json | grep best_c_for_d10

# Step 3: (Optional) Re-run with best C* and all d values
./hyperparam_search_q19_parallel_v2.py --C 0.5 --d 10 50 200

# Step 4: Report in answers.md
# - Best C* for d=10
# - Best overall (C, d)
# - Cross-entropy values for gen and spam
```

---

## Output Files

All scripts produce JSON files with results:

```json
{
  "best_params": {
    "C": 1.0,
    "d": 50,
    "avg_cross_entropy": 7.1234,
    "gen_cross_entropy": 7.5432,
    "spam_cross_entropy": 6.7036
  },
  "best_c_for_d10": 0.5,
  "best_ce_for_d10": 7.8901,
  "results": [...]
}
```

### View Best Results

```bash
# If you have jq installed
cat hyperparam_results_q19_parallel.json | jq '.best_params'

# Without jq
python3 -c "import json; print(json.load(open('hyperparam_results_q19_parallel.json'))['best_params'])"

# Or just look at the terminal output (it shows a summary table)
```

---

## Troubleshooting

### "Out of memory" / System Freezes
→ Reduce parallelism:
```bash
./hyperparam_search_q19_parallel_v2.py -j 2
```

### "ModuleNotFoundError: No module named 'probs'"
→ Make sure you're in the `code/` directory:
```bash
cd /home/kano/Documents/nlp-hw3/code
```

### "File not found: vocab-genspam.txt"
→ Make sure you're in the `code/` directory (paths are relative)

### Takes Too Long
→ Reduce epochs for quick testing:
```bash
./hyperparam_search_q19_parallel_v2.py -e 3
```

### Want to Test Just One Configuration
→ Specify exactly what you want:
```bash
./hyperparam_search_q19_parallel_v2.py --C 1 --d 10 -e 10
```

---

## Performance Estimates

On a modern 8-core CPU:

| Script | Configs | Epochs | Time |
|--------|---------|--------|------|
| Sequential | 15 | 10 | ~2-4 hours |
| Parallel (2 cores) | 15 | 10 | ~1-2 hours |
| Parallel (4 cores) | 15 | 10 | ~30-60 min |
| Parallel (8 cores) | 15 | 10 | ~15-30 min |
| Parallel v2, -e 5 | 15 | 5 | ~8-15 min |
| Parallel v2, --C 1 --d 10 | 1 | 10 | ~5-10 min |

---

## Tips

1. **Monitor progress**: Use `htop` in another terminal to watch CPU usage
2. **Save logs**: Redirect output to file: `./script.py 2>&1 | tee search.log`
3. **Start small**: Test with `-e 3 --C 1 --d 10` first to make sure everything works
4. **Check intermediate results**: The script logs progress, so you can Ctrl+C and still see partial results
5. **Compare versions**: Run sequential and parallel on same config to verify they give identical results

---

## Example Session

```bash
# Navigate to code directory
cd /home/kano/Documents/nlp-hw3/code

# Quick test to make sure everything works
./hyperparam_search_q19_parallel_v2.py -j 2 -e 3 --C 1 --d 10

# If that works, run full search
./hyperparam_search_q19_parallel_v2.py -j 4

# Wait 30-60 minutes...

# Check results
cat hyperparam_results_q19_parallel.json | python3 -m json.tool | grep -A 10 best_params

# Copy best results to answers.md
```

---

## For Your Report

Include in `answers.md`:

```markdown
## Question 19: Neural Language Model Hyperparameters

**Best C for d=10**: C* = 0.5
- Average cross-entropy: 7.8901 bits/token

**Best overall configuration**: C = 1.0, d = 50
- Gen cross-entropy: 7.5432 bits/token
- Spam cross-entropy: 6.7036 bits/token
- Average cross-entropy: 7.1234 bits/token

**Observations**:
- Regularization helps (C=0 performs worse)
- Higher dimensions (d=50) capture more information
- Spam is easier to model than gen (lower cross-entropy)
```

---

## Need More Help?

See `PARALLEL_HYPERPARAM_README.md` for detailed documentation.

