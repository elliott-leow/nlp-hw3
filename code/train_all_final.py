#!/usr/bin/env python3
"""
Final training script for all 4 models with ImprovedLogLinearLanguageModel
Optimized for both language ID and spam detection tasks
"""

import logging
from pathlib import Path
from probs import ImprovedLogLinearLanguageModel, read_vocab

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

def train_model(model_name, vocab_file, lexicon_file, train_file, 
                l2=0.1, epochs=5, dropout=0.1, batch_size=256,
                label_smoothing=0.05, task_type="words"):
    """Train a single model with task-specific hyperparameters"""
    print(f"\n{'='*80}")
    print(f"Training {model_name} ({task_type} task)")
    print(f"{'='*80}")
    
    # Read vocabulary
    vocab = read_vocab(vocab_file)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Adjust hyperparameters based on task type
    if task_type == "chars":
        # Character-level models (spam detection)
        # Need simpler model, more regularization
        dropout = 0.05  # Less dropout for character models
        label_smoothing = 0.01  # Less label smoothing
        warmup_steps = 50  # Less warmup
    else:
        # Word-level models (language ID)
        dropout = 0.1
        label_smoothing = 0.05
        warmup_steps = 100
    
    # Create ImprovedLogLinearLanguageModel
    model = ImprovedLogLinearLanguageModel(
        vocab=vocab,
        lexicon_file=lexicon_file,
        l2=l2,
        epochs=epochs,
        dropout=dropout,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        warmup_steps=warmup_steps,
        grad_accum_steps=1
    )
    
    print(f"Model config: l2={l2}, dropout={dropout}, label_smoothing={label_smoothing}")
    print(f"Training for {epochs} epochs on {train_file}")
    
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
    print("FINAL MODEL TRAINING - ImprovedLogLinearLanguageModel")
    print("- Task-specific hyperparameters")
    print("- 5 epochs for all models")
    print("- 20-dim embeddings")
    print("- Progress logging every 100 tokens")
    print("="*80)
    
    # Train English model (word-level, language ID)
    train_model(
        model_name="en.model",
        vocab_file=lexicon_root / "words-20.txt",
        lexicon_file=lexicon_root / "words-20.txt",
        train_file=data_root / "english_spanish" / "train" / "en.20K",
        l2=0.1,
        epochs=5,
        task_type="words"
    )
    
    # Train Spanish model (word-level, language ID)
    train_model(
        model_name="es.model",
        vocab_file=lexicon_root / "words-20.txt",
        lexicon_file=lexicon_root / "words-20.txt",
        train_file=data_root / "english_spanish" / "train" / "sp.20K",
        l2=0.1,
        epochs=5,
        task_type="words"
    )
    
    # Train gen model (char-level, spam detection)
    train_model(
        model_name="gen.model",
        vocab_file=lexicon_root / "chars-20.txt",
        lexicon_file=lexicon_root / "chars-20.txt",
        train_file=data_root / "gen_spam" / "train" / "gen-times4",
        l2=0.2,  # More regularization for chars
        epochs=5,
        task_type="chars"
    )
    
    # Train spam model (char-level, spam detection)
    train_model(
        model_name="spam.model",
        vocab_file=lexicon_root / "chars-20.txt",
        lexicon_file=lexicon_root / "chars-20.txt",
        train_file=data_root / "gen_spam" / "train" / "spam-times4",
        l2=0.2,  # More regularization for chars
        epochs=5,
        task_type="chars"
    )
    
    print("\n" + "="*80)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()

