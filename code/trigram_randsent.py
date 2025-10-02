#!/usr/bin/env python3
"""
Sample random sentences from a trained language model.
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
        help="number of sentences to generate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="maximum length of generated sentences (default: 100)",
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

    # Specify hardware device
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    try:
        torch.set_default_device(args.device)
    except:
        log.warning(f"Could not set default device to {args.device}, using CPU")

    log.info(f"Loading model from {args.model}")
    lm = LanguageModel.load(args.model, device=args.device)

    log.info(f"Generating {args.num_sentences} sentences with max_length={args.max_length}")

    for i in range(args.num_sentences):
        sentence = lm.sample(max_length=args.max_length)

        # Check if we hit max_length (truncate with "...")
        if len(sentence) == args.max_length:
            sentence_str = " ".join(sentence) + " ..."
        else:
            sentence_str = " ".join(sentence)

        print(sentence_str)


if __name__ == "__main__":
    main()
