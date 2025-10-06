#!/usr/bin/env python3
"""
Train all four models with improved configuration for better performance
Using ImprovedLogLinearLanguageModel with larger data and better hyperparameters
"""

import logging
from pathlib import Path
from probs import ImprovedLogLinearLanguageModel, read_vocab

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

def train_model(model_name, vocab_file, lexicon_file, train_file, l2=0.1, epochs=10):
    """Train a single model"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    # Read vocabulary
    vocab = read_vocab(vocab_file)
    
    # Create ImprovedLogLinearLanguageModel with proper configuration
    model = ImprovedLogLinearLanguageModel(
        vocab=vocab,
        lexicon_file=lexicon_file,
        l2=l2,  # L2 regularization
        epochs=epochs,
        dropout=0.1,  # Add dropout for regularization
        batch_size=256,  # Larger batch size
        label_smoothing=0.05,  # Small label smoothing
        warmup_steps=100,  # Warmup for better convergence
        grad_accum_steps=1
    )
    
    # Train on ALL tokens
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
    print("IMPROVED MODEL TRAINING")
    print("- 10 epochs")
    print("- Medium training files (20K for langid, times2 for spam)")
    print("- Medium dimension embeddings (20-dim for words and chars)")
    print("- ImprovedLogLinearLanguageModel with dropout & label smoothing")
    print("- L2 regularization")
    print("="*80)
    
    # Train English model - use 20K training file and 20-dim embeddings
    train_model(
        model_name="en.model",
        vocab_file=lexicon_root / "words-20.txt",
        lexicon_file=lexicon_root / "words-20.txt",
        train_file=data_root / "english_spanish" / "train" / "en.20K",
        l2=0.1,  # L2 regularization
        epochs=10
    )
    
    # Train Spanish model - use 20K training file and 20-dim embeddings
    train_model(
        model_name="es.model",
        vocab_file=lexicon_root / "words-20.txt",
        lexicon_file=lexicon_root / "words-20.txt",
        train_file=data_root / "english_spanish" / "train" / "sp.20K",
        l2=0.1,  # L2 regularization
        epochs=10
    )
    
    # Train gen model - use gen-times2 for faster training and 20-dim embeddings
    train_model(
        model_name="gen.model",
        vocab_file=lexicon_root / "chars-20.txt",
        lexicon_file=lexicon_root / "chars-20.txt",
        train_file=data_root / "gen_spam" / "train" / "gen-times2",
        l2=0.05,  # Smaller L2 for character-based model
        epochs=10
    )
    
    # Train spam model - use spam-times2 for faster training and 20-dim embeddings
    train_model(
        model_name="spam.model",
        vocab_file=lexicon_root / "chars-20.txt",
        lexicon_file=lexicon_root / "chars-20.txt",
        train_file=data_root / "gen_spam" / "train" / "spam-times2",
        l2=0.05,  # Smaller L2 for character-based model
        epochs=10
    )
    
    print("\n" + "="*80)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()

