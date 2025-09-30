#!/usr/bin/env python3
"""
Text categorization using Bayes' Theorem with two language models.
Classifies test files into one of two categories based on their likelihood
under each language model and the given prior probabilities.
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
        "prior1",
        type=float,
        help="prior probability of the first category (between 0 and 1)",
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

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file
        # probability to 0 and our cumulative log_prob to -infinity.  In
        # this case we can stop early, since the file probability will stay
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break

    return log_prob


def classify_file(file: Path, lm1: LanguageModel, lm2: LanguageModel,
                  prior1: float, model1_name: str, model2_name: str) -> str:
    """
    Classify a file using Bayes' theorem.
    Returns the name of the more likely model.
    """
    # Calculate log probabilities under each model
    log_prob1 = file_log_prob(file, lm1)
    log_prob2 = file_log_prob(file, lm2)

    # Add log priors: log P(data|model) + log P(model)
    log_posterior1 = log_prob1 + math.log(prior1)
    log_posterior2 = log_prob2 + math.log(1 - prior1)

    # Return the model with higher posterior probability
    if log_posterior1 > log_posterior2:
        return model1_name
    else:
        return model2_name


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Validate prior probability
    if not (0 <= args.prior1 <= 1):
        log.error(f"Prior probability must be between 0 and 1, got {args.prior1}")
        return 1

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

    log.info("Loading models...")
    lm1 = LanguageModel.load(args.model1, device=args.device)
    lm2 = LanguageModel.load(args.model2, device=args.device)

    # Get model names from file paths (without .model extension)
    model1_name = args.model1.stem
    model2_name = args.model2.stem

    log.info("Classifying files...")

    # Track classification results
    model1_count = 0
    model2_count = 0
    total_files = len(args.test_files)

    # Classify each file
    for file in args.test_files:
        predicted_model = classify_file(file, lm1, lm2, args.prior1, model1_name, model2_name)
        print(f"{predicted_model}.model\t{file}")

        if predicted_model == model1_name:
            model1_count += 1
        else:
            model2_count += 1

    # Print summary
    if total_files > 0:
        model1_percent = (model1_count / total_files) * 100
        model2_percent = (model2_count / total_files) * 100

        print(f"{model1_count} files were more probably from {model1_name}.model ({model1_percent:.2f}%)")
        print(f"{model2_count} files were more probably from {model2_name}.model ({model2_percent:.2f}%)")


if __name__ == "__main__":
    main()