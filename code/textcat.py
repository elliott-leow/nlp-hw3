#!/usr/bin/env python3
"""
Classifies text files using two language models and Bayes' theorem.
"""
import argparse
import logging
import math
from pathlib import Path
import sys
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the first language model",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the second language model",
    )
    parser.add_argument(
        "prior1",
        type=float,
        help="prior probability of the first model (0.0 to 1.0)",
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

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    #sanity check prior probability
    if not (0.0 <= args.prior1 <= 1.0):
        log.critical("Prior probability must be between 0.0 and 1.0")
        sys.exit(1)
    
    prior2 = 1.0 - args.prior1

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
    
    #sanity check that both models have same vocabulary
    if lm1.vocab != lm2.vocab:
        log.critical("Both language models must have the same vocabulary")
        sys.exit(1)
    
    #classify each file
    model1_count = 0
    model2_count = 0
    
    for file in args.test_files:
        #compute log probability under each model
        log_prob1 = file_log_prob(file, lm1)
        log_prob2 = file_log_prob(file, lm2)
        
        #apply bayes theorem: p(model|data) = p(data|model) * p(model)
        #in log space: log p(model|data) = log p(data|model) + log p(model)
        posterior1 = log_prob1 + math.log(args.prior1)
        posterior2 = log_prob2 + math.log(prior2)
        
        #classify based on higher posterior probability
        if posterior1 > posterior2:
            print(f"{args.model1}\t{file}")
            model1_count += 1
        else:
            print(f"{args.model2}\t{file}")
            model2_count += 1
    
    #print summary
    total_files = len(args.test_files)
    if total_files > 0:
        model1_percent = (model1_count / total_files) * 100
        model2_percent = (model2_count / total_files) * 100
        print(f"{model1_count} files were more probably from {args.model1} ({model1_percent:.2f}%)")
        print(f"{model2_count} files were more probably from {args.model2} ({model2_percent:.2f}%)")


if __name__ == "__main__":
    main()

