#!/usr/bin/env python3
"""
Text categorization using Bayes' Theorem with two language models.
Classifies each file according to which model more probably generated it.
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the first trained model",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the second trained model",
    )
    parser.add_argument(
        "prior",
        type=float,
        help="prior probability of model1 (between 0 and 1)",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
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


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)

        if log_prob == -math.inf:
            break

    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Validate prior probability
    if not 0 <= args.prior <= 1:
        log.critical(f"Prior probability must be between 0 and 1, got {args.prior}")
        exit(1)

    # Set up device
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

    log.info("Loading models...")
    lm1 = LanguageModel.load(args.model1, device=args.device)
    lm2 = LanguageModel.load(args.model2, device=args.device)

    # Sanity check: both models must have the same vocabulary
    if lm1.vocab != lm2.vocab:
        log.critical("The two models must have the same vocabulary for fair comparison!")
        exit(1)

    # Get model names from file paths (with .model extension)
    model1_name = args.model1.name
    model2_name = args.model2.name

    # Compute log priors
    log_prior1 = math.log(args.prior)
    log_prior2 = math.log(1 - args.prior)

    # Classification counters
    model1_count = 0
    model2_count = 0

    log.info("Classifying files...")

    for file in args.test_files:
        # Compute log probability under each model
        log_prob1 = file_log_prob(file, lm1)
        log_prob2 = file_log_prob(file, lm2)

        # Compute log posterior probabilities using Bayes' theorem
        # log P(model | doc) = log P(doc | model) + log P(model)
        log_posterior1 = log_prob1 + log_prior1
        log_posterior2 = log_prob2 + log_prior2

        # Classify based on maximum a posteriori (MAP)
        if log_posterior1 > log_posterior2:
            print(f"{model1_name}   {file}")
            model1_count += 1
        else:
            print(f"{model2_name}   {file}")
            model2_count += 1

    # Print summary
    total = model1_count + model2_count
    if total > 0:
        model1_pct = (model1_count / total) * 100
        model2_pct = (model2_count / total) * 100
        print(f"{model1_count} files were more probably from {model1_name} ({model1_pct:.2f}%)")
        print(f"{model2_count} files were more probably from {model2_name} ({model2_pct:.2f}%)")


if __name__ == "__main__":
    main()
