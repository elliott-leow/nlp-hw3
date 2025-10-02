#!/bin/bash
# Test different lambda values and compute cross-entropy

source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate nlp-class

echo "Lambda,Gen_CrossEntropy,Spam_CrossEntropy,Combined_CrossEntropy"

for lambda in 5 0.5 0.05 0.005 0.0005 1; do
    gen_ce=$(python fileprob.py gen-lambda-${lambda}.model ../data/gen_spam/dev/gen/* 2>/dev/null | tail -1 | awk '{print $2}')
    spam_ce=$(python fileprob.py spam-lambda-${lambda}.model ../data/gen_spam/dev/spam/* 2>/dev/null | tail -1 | awk '{print $2}')

    # For combined cross-entropy, we need to compute total bits and total tokens
    gen_output=$(python fileprob.py gen-lambda-${lambda}.model ../data/gen_spam/dev/gen/* 2>/dev/null)
    spam_output=$(python fileprob.py spam-lambda-${lambda}.model ../data/gen_spam/dev/spam/* 2>/dev/null)

    # Extract cross-entropy values
    gen_ce_val=$(echo "$gen_output" | tail -1 | awk '{print $2}')
    spam_ce_val=$(echo "$spam_output" | tail -1 | awk '{print $2}')

    # Count tokens in gen and spam dev sets
    gen_tokens=$(python -c "from probs import num_tokens; from pathlib import Path; import glob; print(sum(num_tokens(Path(f)) for f in glob.glob('../data/gen_spam/dev/gen/*')))")
    spam_tokens=$(python -c "from probs import num_tokens; from pathlib import Path; import glob; print(sum(num_tokens(Path(f)) for f in glob.glob('../data/gen_spam/dev/spam/*')))")

    # Calculate combined cross-entropy
    gen_bits=$(python -c "print($gen_ce_val * $gen_tokens)")
    spam_bits=$(python -c "print($spam_ce_val * $spam_tokens)")
    total_bits=$(python -c "print($gen_bits + $spam_bits)")
    total_tokens=$(python -c "print($gen_tokens + $spam_tokens)")
    combined_ce=$(python -c "print($total_bits / $total_tokens)")

    echo "$lambda,$gen_ce_val,$spam_ce_val,$combined_ce"
done
