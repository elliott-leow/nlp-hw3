# Homework 3: Smoothed Language Modeling - Answers

## Question 1: Perplexities and corpora

### Part (a): Perplexity per word with switchboard-small corpus

Using add-0.01 smoothing and vocab threshold of 3:

- **sample1**: 43.46
- **sample2**: 54.10
- **sample3**: 53.68

(Vocabulary size: 2,886 words including OOV and EOS)

### Part (b): Effect of training on larger switchboard corpus

When training on the larger switchboard corpus instead:

**Log‚-probabilities become MORE NEGATIVE (worse):**
- sample1: -8282.07 ’ -8578.34 (difference: -296.27)
- sample2: -5008.97 ’ -5143.54 (difference: -134.57)
- sample3: -5085.45 ’ -5419.08 (difference: -333.63)

**Perplexities INCREASE (worse):**
- sample1: 43.46 ’ 49.74 (+14.4%)
- sample2: 54.10 ’ 60.22 (+11.3%)
- sample3: 53.68 ’ 69.71 (+29.9%)

(Vocabulary size: 11,419 words including OOV and EOS)

**Why does this happen?**

Counterintuitively, training on more data produces worse perplexity. This occurs because the larger corpus has a much larger vocabulary (11,419 vs 2,886 words). With add-» smoothing, the probability formula is:

p(z|xy) = (c(xyz) + ») / (c(xy) + »V)

When V (vocabulary size) increases:
- The denominator (c(xy) + »V) becomes much larger
- This decreases the probability assigned to each observed trigram
- The probability mass is spread over many more possible words
- The model becomes less certain about each specific prediction

Even though the larger corpus provides more training examples (2.19M tokens vs 209K tokens), the add-» smoothing method is penalized by having to distribute probability over 4× as many vocabulary words. The smoothing constant »=0.01 adds relatively little probability mass (0.01 per word), but when multiplied by V, it substantially inflates the denominator.

This demonstrates a fundamental limitation of simple add-» smoothing: it doesn't scale well as vocabulary size increases. More sophisticated smoothing methods (like backoff smoothing) handle this better by backing off to lower-order n-grams rather than distributing probability uniformly across the entire vocabulary.
