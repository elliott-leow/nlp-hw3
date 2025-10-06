#!/usr/bin/env python3
#q9
import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple
import torch

from probs import LanguageModel

log = logging.getLogger(Path(__file__).stem)

# type definitions for clarity
Candidate = Tuple[float, float, int, str]  # (wer, acoustic_score, word_count, transcription)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="speech recognition using language model rescoring")
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained language model",
    )
    parser.add_argument(
        "utterance_files",
        type=Path,
        nargs="+",
        help="utterance files with candidate transcriptions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu', 'cuda', 'mps'],
        help="device to use for pytorch (cpu, cuda, or mps)"
    )
    parser.add_argument(
        "--show-all-candidates",
        action="store_true",
        help="show scores for all candidates, not just the best"
    )

    # verbosity settings
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", 
        dest="logging_level", 
        action="store_const", 
        const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   
        dest="logging_level", 
        action="store_const", 
        const=logging.WARNING
    )

    return parser.parse_args()


def read_utterance_file(file: Path) -> Tuple[str, List[Candidate]]:
    """
    read an utterance file and extract the true transcription and candidates
    
    line 1 - word_count TAB true_transcription
    lines 2-10 - wer TAB acoustic_score TAB word_count TAB candidate_transcription
    
    returns tuple of (true_transcription, list_of_candidates)
    """
    with open(file) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    #parse true transcription
    first_line = lines[0].split('\t')
    true_count = int(first_line[0])
    true_transcription = first_line[1]
    
    #parse candidates
    candidates = []
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) >= 4:
            wer = float(parts[0])
            acoustic_score = float(parts[1])  #already in log2
            word_count = int(parts[2])
            transcription = parts[3]
            candidates.append((wer, acoustic_score, word_count, transcription))
    
    return true_transcription, candidates


def compute_lm_score(transcription: str, lm: LanguageModel) -> float:
    """
    compute language model log-probability for a transcription.
    

        
    returns log2 probability of the transcription
    """
    tokens = transcription.split()
    
    #manually construct trigrams with bos eos boundaries
    #prepend two BOS tokens and append EOS
    from probs import BOS, EOS, OOV
    
    #convert tokens to vocab
    vocab_tokens = []
    for token in tokens:
        if token in lm.vocab:
            vocab_tokens.append(token)
        else:
            vocab_tokens.append(OOV)
    
    # add boundaries
    extended_tokens = [BOS, BOS] + vocab_tokens + [EOS]
    
    #compute log probability as sum of trigram probabilities
    #natural log initially
    log_prob_natural = 0.0
    
    for i in range(2, len(extended_tokens)):
        x = extended_tokens[i-2]
        y = extended_tokens[i-1]
        z = extended_tokens[i]
        log_prob_natural += lm.log_prob(x, y, z)
        
        #stop early if we hit -infinity
        if log_prob_natural == -math.inf:
            break
    
    log_prob_log2 = log_prob_natural / math.log(2)
    
    return log_prob_log2


def compute_posterior_score(candidate: Candidate, lm: LanguageModel) -> float:
    """
    compute the posterior score for a candidate transcription.
    
    according to bayes' theorem:
        p(w|u) ∝ p(u|w) * p(w)
    
    in log space:
        log p(w|u) = log p(u|w) + log p(w) + constant
    
    args:
        candidate: tuple of (wer, acoustic_score, word_count, transcription)
        lm: language model
        
    returns:
        combined log2 posterior score
    """
    wer, acoustic_score, word_count, transcription = candidate
    
    # acoustic score is already log2 p(u|w)
    # compute log2 p(w) from language model
    lm_score = compute_lm_score(transcription, lm)
    
    # combine via addition in log space
    posterior_score = acoustic_score + lm_score
    
    return posterior_score




def process_utterance(file: Path, lm: LanguageModel, show_all: bool = False) -> Tuple[float, str]:
    """
    process a single utterance file and select the best candidate
    
    returns tuple of (selected_wer, best_transcription)
    """
    true_transcription, candidates = read_utterance_file(file)
    
    if not candidates:
        log.warning(f"no candidates found in {file}")
        return 1.0, ""
    
    #compute posterior scores for all candidates
    scored_candidates = []
    for candidate in candidates:
        score = compute_posterior_score(candidate, lm)
        scored_candidates.append((score, candidate))
    
    #sort by score (highest first)
    scored_candidates.sort(reverse=True, key=lambda x: x[0])
    
    if show_all:
        log.info(f"\nall candidates for {file.name}:")
        for score, (wer, acoustic, wc, trans) in scored_candidates:
            lm_score = score - acoustic
            log.info(f"  score={score:8.2f} (acoustic={acoustic:8.2f}, lm={lm_score:8.2f}) "
                    f"wer={wer:.3f} : {trans}")
    
    #select best candidate (highest posterior score)
    best_score, best_candidate = scored_candidates[0]
    best_wer, best_acoustic, best_wc, best_transcription = best_candidate
    
    #log decision
    lm_score = best_score - best_acoustic
    log.debug(f"{file.name}: selected candidate with score={best_score:.2f} "
              f"(acoustic={best_acoustic:.2f}, lm={lm_score:.2f}), wer={best_wer:.3f}")
    
    #use the wer from file
    return best_wer, best_transcription


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    
    # set up pytorch device
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            log.critical("mps not available on this system")
            exit(1)
    torch.set_default_device(args.device)
    
    # load language model
    log.info(f"loading language model from {args.model}")
    lm = LanguageModel.load(args.model, device=args.device)
    log.info(f"model loaded: vocab size = {len(lm.vocab)}")
    
    #process all utterances
    total_wer_sum = 0.0
    total_utterances = 0
    
    log.info(f"\nprocessing {len(args.utterance_files)} utterances...")
    
    for utterance_file in sorted(args.utterance_files):
        wer, selected = process_utterance(
            utterance_file, lm, args.show_all_candidates
        )
        
        total_wer_sum += wer
        total_utterances += 1
        
        #print per-file results
        print(f"{wer:.3f}\t{utterance_file.name}\t{selected}")
    
    #compute average wer
    avg_wer = total_wer_sum / total_utterances if total_utterances > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"summary statistics:")
    print(f"total utterances: {total_utterances}")
    print(f"average word error rate: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"{'='*60}")
    
    log.info(f"speech recognition complete")


if __name__ == "__main__":
    main()

