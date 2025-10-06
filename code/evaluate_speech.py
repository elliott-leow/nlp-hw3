#!/usr/bin/env python3
"""
comprehensive evaluation script for speech recognition.

compares different language models on dev and test sets,
generates detailed statistics and analysis.
"""
import argparse
import logging
from pathlib import Path
import subprocess
import json
from typing import Dict, List, Tuple

log = logging.getLogger(Path(__file__).stem)


def run_speechrec(model: Path, utterance_dir: Path) -> Tuple[float, int]:
    """
    run speechrec on a directory of utterances and extract average wer.
    
    returns:
        tuple of (average_wer, num_utterances)
    """
    # get all utterance files
    utterance_files = sorted(utterance_dir.glob('*'))
    utterance_files = [f for f in utterance_files if f.is_file()]
    
    if not utterance_files:
        return 0.0, 0
    
    # run speechrec.py
    cmd = ['python3', 'speechrec.py', str(model)] + [str(f) for f in utterance_files]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # parse output to extract average wer
    lines = result.stdout.strip().split('\n')
    for line in lines:
        if 'average word error rate:' in line.lower():
            # format: "average word error rate: 0.1624 (16.24%)"
            parts = line.split(':')
            if len(parts) >= 2:
                wer_str = parts[1].strip().split()[0]
                avg_wer = float(wer_str)
                return avg_wer, len(utterance_files)
    
    return 0.0, len(utterance_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--models',
        type=Path,
        nargs='+',
        help='language models to evaluate'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('speech_evaluation_results.json'),
        help='output file for results'
    )
    
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
    
    args = parser.parse_args()
    logging.basicConfig(level=args.logging_level)
    
    # define evaluation sets
    data_root = Path(__file__).parent.parent / 'data' / 'speech'
    eval_sets = {
        'dev_easy': data_root / 'dev' / 'easy',
        'dev_unrestricted': data_root / 'dev' / 'unrestricted',
        'test_easy': data_root / 'test' / 'easy',
        'test_unrestricted': data_root / 'test' / 'unrestricted',
    }
    
    # default models if none specified
    if not args.models:
        code_dir = Path(__file__).parent
        args.models = [
            code_dir / 'switchboard-small.model',
            code_dir / 'switchboard-full.model',
            code_dir / 'switchboard-small-lambda0.012.model',
            code_dir / 'switchboard-full-lambda0.012.model',
        ]
    
    # evaluate all models on all sets
    results = {}
    
    for model in args.models:
        if not model.exists():
            log.warning(f"model not found: {model}, skipping")
            continue
        
        log.info(f"\nevaluating model: {model.name}")
        model_results = {}
        
        for set_name, set_dir in eval_sets.items():
            if not set_dir.exists():
                log.warning(f"evaluation set not found: {set_dir}")
                continue
            
            log.info(f"  evaluating on {set_name}...")
            avg_wer, num_utterances = run_speechrec(model, set_dir)
            
            model_results[set_name] = {
                'wer': avg_wer,
                'wer_percent': avg_wer * 100,
                'num_utterances': num_utterances
            }
            
            log.info(f"    wer: {avg_wer:.4f} ({avg_wer*100:.2f}%) on {num_utterances} utterances")
        
        results[model.name] = model_results
    
    # save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nresults saved to {args.output}")
    
    # print summary table
    print("\n" + "="*80)
    print("speech recognition evaluation summary")
    print("="*80)
    print(f"\n{'model':<50} {'dev easy':<12} {'dev unrest':<12} {'test easy':<12} {'test unrest':<12}")
    print("-"*80)
    
    for model_name, model_results in results.items():
        dev_easy = model_results.get('dev_easy', {}).get('wer_percent', 0)
        dev_unrest = model_results.get('dev_unrestricted', {}).get('wer_percent', 0)
        test_easy = model_results.get('test_easy', {}).get('wer_percent', 0)
        test_unrest = model_results.get('test_unrestricted', {}).get('wer_percent', 0)
        
        print(f"{model_name:<50} {dev_easy:>10.2f}% {dev_unrest:>10.2f}% "
              f"{test_easy:>10.2f}% {test_unrest:>10.2f}%")
    
    print("="*80)


if __name__ == '__main__':
    main()

