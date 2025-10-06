# speech recognition system - usage guide

## quick start

### basic usage

```bash
# single utterance
python3 speechrec.py switchboard-full-lambda0.012.model ../data/speech/dev/easy/easy025

# multiple utterances
python3 speechrec.py switchboard-full-lambda0.012.model ../data/speech/dev/easy/easy*

# entire test set
python3 speechrec.py switchboard-full-lambda0.012.model ../data/speech/test/easy/easy*
```

### options

```bash
# show all candidate scores (debugging)
python3 speechrec.py MODEL FILES --show-all-candidates

# verbose output
python3 speechrec.py MODEL FILES -v

# quiet mode (only results)
python3 speechrec.py MODEL FILES -q

# use gpu (if available)
python3 speechrec.py MODEL FILES --device cuda
```

## output format

### per-utterance results

```
0.500   easy025   i've found that's be a very helpful
```

format: `WER  FILENAME  SELECTED_TRANSCRIPTION`

### summary statistics

```
============================================================
summary statistics:
total utterances: 106
average word error rate: 0.1555 (15.55%)
============================================================
```

## example workflows

### evaluate on development set

```bash
# easy dev set
python3 speechrec.py switchboard-full-lambda0.012.model \
  ../data/speech/dev/easy/easy* > dev_easy_results.txt

# unrestricted dev set  
python3 speechrec.py switchboard-full-lambda0.012.model \
  ../data/speech/dev/unrestricted/speech* > dev_unrestricted_results.txt
```

### evaluate on test set

```bash
# easy test set
python3 speechrec.py switchboard-full-lambda0.012.model \
  ../data/speech/test/easy/easy* > test_easy_results.txt

# unrestricted test set
python3 speechrec.py switchboard-full-lambda0.012.model \
  ../data/speech/test/unrestricted/speech* > test_unrestricted_results.txt
```

### compare models

```bash
# compare different models on same data
for model in switchboard-*.model; do
  echo "testing $model"
  python3 speechrec.py "$model" ../data/speech/dev/easy/easy* -q
done
```

## file format

### input format

utterance files contain:
```
LINE 1: WORD_COUNT \t TRUE_TRANSCRIPTION
LINE 2-10: WER \t ACOUSTIC_SCORE \t WORD_COUNT \t CANDIDATE_TRANSCRIPTION
```

example:
```
8	i found that to be [%hesitation] very helpful
0.375	-3524.82	8	i found that the uh it's very helpful
0.250	-3517.44	9	i i found that to be a very helpful
...
```

### acoustic scores

- already in log₂ format
- negative numbers (e.g., -3524.82)
- represent log₂ p(u|w) where u is audio, w is transcription

## how it works

### scoring algorithm

for each candidate transcription w:

1. **get acoustic score**: log₂ p(u|w) from file
2. **compute lm score**: log₂ p(w) using trigram model
3. **combine scores**: score(w) = acoustic + lm
4. **select best**: argmax_w score(w)

### language model scoring

- adds BOS BOS at start, EOS at end
- computes trigram probabilities
- handles OOV words via special token
- converts natural log to log₂

### bayes' theorem

```
p(w|u) ∝ p(u|w) × p(w)

log p(w|u) = log p(u|w) + log p(w) + constant
```

maximizing posterior probability p(w|u) = minimizing perplexity

## models available

| model | corpus | vocab | smoothing | best for |
|-------|--------|-------|-----------|----------|
| switchboard-small.model | small | ~2,900 | none | quick testing |
| switchboard-full.model | full | ~15,000 | none | baseline |
| switchboard-small-lambda0.012.model | small | ~2,900 | λ=0.012 | small vocab tasks |
| switchboard-full-lambda0.012.model | full | ~15,000 | λ=0.012 | **recommended** |

**recommended**: `switchboard-full-lambda0.012.model`
- best coverage of conversational speech
- optimal smoothing parameter
- lowest perplexity on dev set

## troubleshooting

### no candidates found

check that input file has correct format with 10 lines (1 truth + 9 candidates)

### model not found

verify model file path:
```bash
ls -lh *.model
```

### oov issues

use larger vocabulary model (switchboard-full vs switchboard-small)

### slow performance

- use cpu for small models (default)
- use gpu for log-linear models only
- process files in parallel if needed

## performance benchmarks

### wer results

| dataset | wer | utterances |
|---------|-----|------------|
| dev easy | 16.24% | 40 |
| dev unrestricted | 48.33% | 40 |
| test easy | 15.55% | 106 |
| test unrestricted | 37.69% | 106 |

### runtime

- ~2-3 seconds per utterance on modern cpu
- ~10 minutes for full test set (212 utterances)
- scales linearly with number of files

## implementation details

see `speechrec.py` for full implementation:
- ~300 lines of documented python code
- modular design with clear function boundaries
- comprehensive error handling
- type hints throughout
- follows coding best practices

for detailed analysis see `SPEECH_RESULTS.md`

