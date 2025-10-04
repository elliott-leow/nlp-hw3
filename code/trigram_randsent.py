#!/usr/bin/env python3
"""
Samples random sentences from a given smoothed trigram model.
"""
import argparse
import logging
from pathlib import Path
import torch

from probs import LanguageModel

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "num_sentences",
        type=int,
        help="number of sentences to sample",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="maximum length of sampled sentences (default: 20)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    lm = LanguageModel.load(args.model, device=args.device)
    
    log.info(f"Sampling {args.num_sentences} sentences")
    
    for i in range(args.num_sentences):
        sentence = lm.sample_sentence(max_length=args.max_length)
        
        #if sentence was truncated at max_length, add "..." 
        if len(sentence) == args.max_length:
            sentence_str = " ".join(sentence) + " ..."
        else:
            sentence_str = " ".join(sentence)
            
        print(f"{i+1:2d}: {sentence_str}")


if __name__ == "__main__":
    main()
