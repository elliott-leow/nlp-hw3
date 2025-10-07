#!/usr/bin/env python3
#train all models for submission with d=20, 2x corpus, 5 epochs

import logging
from pathlib import Path
from probs import ImprovedLogLinearLanguageModel, read_vocab

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def train_model(corpus_file: Path, lexicon_file: Path, output_model: Path, l2: float, epochs: int):
    """train a single improved log-linear model"""
    log.info(f"\n{'='*80}")
    log.info(f"training model: {output_model}")
    log.info(f"corpus: {corpus_file}")
    log.info(f"lexicon: {lexicon_file}")
    log.info(f"l2: {l2}, epochs: {epochs}")
    log.info(f"{'='*80}\n")
    
    #read vocabulary from lexicon
    vocab = read_vocab(lexicon_file)
    
    #create model
    model = ImprovedLogLinearLanguageModel(vocab, lexicon_file, l2=l2, epochs=epochs)
    
    #train model
    model.train(corpus_file)
    
    #save model
    model.save(output_model)
    
    log.info(f"model saved to {output_model}\n")

def main():
    #base paths
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data"
    lexicons_path = base_path / "lexicons"
    code_path = base_path / "code"
    
    #hyperparameters
    l2 = 0.0
    epochs = 5
    
    #train english model (en.model)
    log.info("training english model...")
    train_model(
        corpus_file=data_path / "english_spanish" / "train" / "en.2K",
        lexicon_file=lexicons_path / "words-20.txt",
        output_model=code_path / "en.model",
        l2=l2,
        epochs=epochs
    )
    
    #train spanish model (es.model)
    log.info("training spanish model...")
    train_model(
        corpus_file=data_path / "english_spanish" / "train" / "sp.2K",
        lexicon_file=lexicons_path / "words-20.txt",
        output_model=code_path / "es.model",
        l2=l2,
        epochs=epochs
    )
    
    #train gen model (gen.model)
    log.info("training gen model...")
    train_model(
        corpus_file=data_path / "gen_spam" / "train" / "gen-times2",
        lexicon_file=lexicons_path / "words-gs-20.txt",
        output_model=code_path / "gen.model",
        l2=l2,
        epochs=epochs
    )
    
    #train spam model (spam.model)
    log.info("training spam model...")
    train_model(
        corpus_file=data_path / "gen_spam" / "train" / "spam-times2",
        lexicon_file=lexicons_path / "words-gs-20.txt",
        output_model=code_path / "spam.model",
        l2=l2,
        epochs=epochs
    )
    
    log.info("\n" + "="*80)
    log.info("all models trained successfully!")
    log.info("="*80)

if __name__ == "__main__":
    main()

