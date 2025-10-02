#!/usr/bin/env python3
"""
Speech recognition using a language model and Bayes' Theorem.
For each utterance file, selects the best candidate transcription by combining
acoustic model scores with language model probabilities.
"""
import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple
import torch

from probs import Wordtype, LanguageModel, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained language model",
    )
    parser.add_argument(
        "utterance_files",
        type=Path,
        nargs="*",
        help="utterance files containing candidate transcriptions"
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


def transcription_log_prob(words: List[Wordtype], lm: LanguageModel) -> float:
    """
    Compute the log probability of a transcription under the language model.
    The transcription should already include BOS and EOS markers.
    """
    log_prob = 0.0

    # Convert words to trigrams and compute probability
    # Note: words should be like ['<s>', '<s>', word1, word2, ..., '</s>']
    for i in range(2, len(words)):
        x, y, z = words[i-2], words[i-1], words[i]
        log_prob += lm.log_prob(x, y, z)

        if log_prob == -math.inf:
            break

    return log_prob


def process_utterance_file(file: Path, lm: LanguageModel) -> Tuple[float, int]:
    """
    Process one utterance file and select the best candidate transcription.

    Returns:
        (word_error_rate, num_words_in_reference): The WER of the selected candidate
                                                     and the length of the reference
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    # First line: reference transcription (just need the length)
    first_line_parts = lines[0].strip().split(maxsplit=1)
    reference_length = int(first_line_parts[0])

    # Process the 9 candidate transcriptions (lines 1-9, 0-indexed)
    best_score = -math.inf
    best_wer = None

    for line in lines[1:10]:  # Lines 2-10 in 1-indexed (lines[1:10] in 0-indexed)
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        wer = float(parts[0])
        acoustic_log_prob = float(parts[1])
        length = int(parts[2])

        # The transcription starts at parts[3] and has 'length + 2' words
        # (length content words plus <s> and </s>)
        transcription = parts[3:3 + length + 2]

        # Compute language model log probability
        lm_log_prob = transcription_log_prob(transcription, lm)

        # Compute combined score: log p(w|u) = log p(u|w) + log p(w)
        # acoustic_log_prob is log p(u|w)
        # lm_log_prob is log p(w)
        combined_score = acoustic_log_prob + lm_log_prob

        log.debug(f"Candidate: WER={wer}, acoustic={acoustic_log_prob:.2f}, "
                  f"lm={lm_log_prob:.2f}, combined={combined_score:.2f}")

        if combined_score > best_score:
            best_score = combined_score
            best_wer = wer

    return best_wer, reference_length


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

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

    log.info("Loading language model...")
    lm = LanguageModel.load(args.model, device=args.device)

    # Process each utterance file
    total_errors = 0.0
    total_words = 0

    for file in args.utterance_files:
        wer, ref_length = process_utterance_file(file, lm)

        # Print individual file result
        print(f"{wer:.3f}\t{file.name}")

        # Accumulate for overall statistics
        # WER is the fraction of errors, so number of errors = WER * ref_length
        num_errors = wer * ref_length
        total_errors += num_errors
        total_words += ref_length

    # Print overall WER
    overall_wer = total_errors / total_words if total_words > 0 else 0.0
    print(f"{overall_wer:.3f}\tOVERALL")


if __name__ == "__main__":
    main()
