#!/usr/bin/env python3
"""
Train all four models (en, es, gen, spam) with ImprovedLogLinearLanguageModel
Using 1 epoch and limited tokens for faster training
"""

import logging
from pathlib import Path
from probs import ImprovedLogLinearLanguageModel, read_vocab

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

def train_model(model_name, vocab_file, lexicon_file, train_file):
    """Train a single model"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    # Read vocabulary
    vocab = read_vocab(vocab_file)
    
    # Create model with 10 epochs
    model = ImprovedLogLinearLanguageModel(
        vocab=vocab,
        lexicon_file=lexicon_file,
        l2=0.0,  # No L2 regularization
        epochs=10,  # 10 epochs
        dropout=0.0,  # No dropout
        batch_size=128,
        label_smoothing=0.0,
        warmup_steps=0,
        grad_accum_steps=1
    )
    
    # Train on ALL tokens (no max_tokens limit)
    model.train(train_file, max_tokens=None)
    
    # Save model
    output_path = Path("code") / model_name
    model.save(output_path)
    print(f"\n✓ Saved {model_name}")
    
    return model

def main():
    # Paths
    data_root = Path("data")
    lexicon_root = Path("lexicons")
    
    print("="*80)
    print("MODEL TRAINING - 10 epochs, ALL tokens")
    print("="*80)
    
    # Train English model
    train_model(
        model_name="en.model",
        vocab_file=lexicon_root / "words-10.txt",
        lexicon_file=lexicon_root / "words-gs-10.txt",
        train_file=data_root / "english_spanish" / "train" / "en.1K"
    )
    
    # Train Spanish model
    train_model(
        model_name="es.model",
        vocab_file=lexicon_root / "words-10.txt",
        lexicon_file=lexicon_root / "words-gs-10.txt",
        train_file=data_root / "english_spanish" / "train" / "sp.1K"
    )
    
    # Train gen model
    train_model(
        model_name="gen.model",
        vocab_file=lexicon_root / "chars-10.txt",
        lexicon_file=lexicon_root / "chars-10.txt",
        train_file=data_root / "gen_spam" / "train" / "gen"
    )
    
    # Train spam model
    train_model(
        model_name="spam.model",
        vocab_file=lexicon_root / "chars-10.txt",
        lexicon_file=lexicon_root / "chars-10.txt",
        train_file=data_root / "gen_spam" / "train" / "spam"
    )
    
    print("\n" + "="*80)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
