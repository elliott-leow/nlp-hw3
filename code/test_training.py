#!/usr/bin/env python3
"""
Test that training is actually working by training a small model
"""

import logging
from pathlib import Path
from probs import ImprovedLogLinearLanguageModel, read_vocab

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

print("="*80)
print("TESTING TRAINING - 3 epochs on gen model")
print("="*80)

# Read vocabulary
vocab = read_vocab(Path("lexicons/chars-10.txt"))

# Create model with 3 epochs for testing
model = ImprovedLogLinearLanguageModel(
    vocab=vocab,
    lexicon_file=Path("lexicons/chars-10.txt"),
    l2=0.0,
    epochs=3,  # Just 3 epochs for testing
    dropout=0.0,
    batch_size=128,
    label_smoothing=0.0,
    warmup_steps=0,
    grad_accum_steps=1
)

# Train on ALL tokens
model.train(Path("data/gen_spam/train/gen"), max_tokens=None)

print("\n" + "="*80)
print("If you see actual loss values (not 0.0000), training is working!")
print("="*80)

