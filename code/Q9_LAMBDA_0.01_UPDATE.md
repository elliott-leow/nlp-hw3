# question 9 update - retrained with λ=0.01

## summary of changes

retrained all models with λ=0.01 (instead of λ=0.012) and re-evaluated question 9.

## new models trained

1. **`switchboard-full-lambda0.01.model`**
   - full switchboard corpus (~2.2m tokens)
   - vocabulary: 11,419 word types
   - add-λ smoothing with λ=0.01
   - size: 26 MB

2. **`switchboard-small-lambda0.01.model`**
   - small switchboard corpus (~209k tokens)
   - vocabulary: 2,886 word types
   - add-λ smoothing with λ=0.01
   - size: 3.6 MB

## updated results

### test set performance (λ=0.01)

| dataset | wer | accuracy | utterances |
|---------|-----|----------|------------|
| **test easy** | **15.71%** | 84.29% | 106 |
| **test unrestricted** | **37.58%** | 62.42% | 106 |

### development set performance (λ=0.01)

| dataset | wer | accuracy | utterances |
|---------|-----|----------|------------|
| **dev easy** | **16.74%** | 83.26% | 40 |
| **dev unrestricted** | **48.33%** | 51.67% | 40 |

### model comparison (λ=0.01)

| model | vocab | dev easy wer | dev unrestricted wer |
|-------|-------|--------------|----------------------|
| small | ~2,900 | 20.89% | 49.24% |
| full | ~11,400 | **16.74%** | **48.33%** |

**selected**: `switchboard-full-lambda0.01.model` (better performance on both dev sets)

## comparison with previous λ=0.012 results

| metric | λ=0.012 | λ=0.01 | difference |
|--------|---------|--------|------------|
| test easy wer | 15.55% | 15.71% | +0.16% |
| test unrestricted wer | 37.69% | 37.58% | -0.11% |
| dev easy wer | 16.24% | 16.74% | +0.50% |
| dev unrestricted wer | 48.33% | 48.33% | 0.00% |

**conclusion**: results are very similar. λ=0.01 performs slightly worse on easy set, 
slightly better on unrestricted set. overall performance comparable.

## updated example (easy025)

**with λ=0.01**:
```
truth: "i found that to be %hesitation very helpful"

selected (wer=0.125): "i found that to be a very helpful"
  score = -3573.29 (acoustic=-3517.20, lm=-56.09)

runner-up (wer=0.250): "i i found that to be a very helpful"  
  score = -3576.33 (acoustic=-3517.44, lm=-58.90)
```

**with λ=0.012**:
```
truth: "i found that to be %hesitation very helpful"

selected (wer=0.125): "i found that to be a very helpful"
  score = -3573.76 (acoustic=-3517.20, lm=-56.56)

runner-up (wer=0.250): "i i found that to be a very helpful"  
  score = -3576.81 (acoustic=-3517.44, lm=-59.37)
```

**observation**: both λ values select the same candidate. λ=0.01 gives slightly 
better (less negative) lm scores, indicating less aggressive smoothing.

## files updated

### code files
- ✅ trained new models: `switchboard-{full,small}-lambda0.01.model`
- ✅ generated new result files: `speech_*_lambda0.01_results.txt`

### documentation files
- ✅ `answers.md` - updated question 9 with λ=0.01 results
- ✅ `SPEECH_RESULTS.md` - updated all results tables and examples
- ✅ created `Q9_LAMBDA_0.01_UPDATE.md` (this file)

### key changes in answers.md
- model: λ=0.012 → λ=0.01
- vocab size: ~15,000 → ~11,400
- test easy wer: 15.55% → 15.71%
- test unrestricted wer: 37.69% → 37.58%
- dev easy wer: 16.24% → 16.74%
- updated example to show correct selection
- updated model comparison table

### key changes in SPEECH_RESULTS.md
- updated all wer values
- updated model selection rationale
- updated example from easy025
- updated model comparison table
- updated final performance summary

## validation

all results verified by running:
```bash
# dev sets
python3 speechrec.py switchboard-full-lambda0.01.model ../data/speech/dev/easy/easy* -q
python3 speechrec.py switchboard-full-lambda0.01.model ../data/speech/dev/unrestricted/speech* -q

# test sets
python3 speechrec.py switchboard-full-lambda0.01.model ../data/speech/test/easy/easy* -q
python3 speechrec.py switchboard-full-lambda0.01.model ../data/speech/test/unrestricted/speech* -q
```

## implementation status

✅ models retrained with λ=0.01  
✅ all datasets re-evaluated  
✅ documentation updated  
✅ results verified  
✅ question 9 answers updated  

**status**: complete

---

*updated: october 4, 2025*  
*change: retrained with λ=0.01 instead of λ=0.012*

