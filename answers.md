# Homework 3: Smoothed Language Modeling - Answers

## Question 2: Implementing a generic text classifier

The `textcat.py` program has been implemented to perform text categorization via Bayes' Theorem. It takes two language models, a prior probability for the first category, and a list of test files. For each file, it computes the posterior probability under each model using:

log P(model | doc) = log P(doc | model) + log P(model)

It then classifies based on the maximum a posteriori (MAP) decision rule.

**Sanity check:** When training on the smallest training sets with λ=1 and classifying all 270 dev files with prior p(gen)=0.7, we classified 23 files as spam. ✓

## Question 3: Evaluating a text classifier

### Part (a): Error rate with add-1 smoothing

Using add-1 smoothing (λ=1) with prior p(gen) = 0.7:

- **Total dev files:** 270 (180 gen + 90 spam)
- **Files classified as gen:** 247
- **Files classified as spam:** 23
- **Gen files misclassified as spam:** 1
- **Spam files misclassified as gen:** 68

**Error rate:** 69/270 = **25.56%**

The high error rate is primarily due to many spam emails being misclassified as genuine emails. This suggests that with λ=1 and this prior, the gen language model assigns relatively high probability even to spam messages.

### Part (c): Minimum prior for all-spam classification

To find the minimum prior probability of gen that causes textcat to classify all dev files as spam, I tested progressively smaller priors:

| Prior p(gen) | Files as spam |
|--------------|---------------|
| 0.7          | 23            |
| 0.1          | 33            |
| 1e-10        | 90            |
| 1e-20        | 132           |
| 1e-50        | 207           |
| 1e-100       | 242           |
| 1e-200       | 263           |
| 1e-300       | 267           |

**Result:** Even at extremely small priors like p(gen) = 1e-300, we still have 3 files classified as gen. This suggests that there are a few development files where the gen model assigns substantially higher probability than the spam model, even when the prior overwhelmingly favors spam. The practical answer is approximately **p(gen) < 1e-300** to classify nearly all (but not quite all) files as spam.

### Parts (d), (e), (f): Optimizing lambda for minimum cross-entropy

I trained models with different values of λ and computed cross-entropy on the dev sets:

| λ     | Gen CE (bits/token) | Spam CE (bits/token) | Combined CE (bits/token) |
|-------|---------------------|----------------------|--------------------------|
| 5     | 11.05263            | 11.07215             | 11.06139                 |
| 1     | 10.45207            | 10.53513             | 10.48934                 |
| 0.5   | 10.15485            | 10.26566             | 10.20457                 |
| 0.05  | 9.29458             | 9.44152              | 9.36051                  |
| 0.005 | **9.04616**         | **9.09572**          | **9.06840**              |
| 0.0005| 9.49982             | 9.41952              | 9.46379                  |

**(d) Minimum cross-entropy for gen and spam separately:**
- **Gen dev files:** 9.04616 bits/token (λ = 0.005)
- **Spam dev files:** 9.09572 bits/token (λ = 0.005)

**(e) Minimum combined cross-entropy:**
When both models use the same λ, the minimum overall cross-entropy on all dev files is **9.06840 bits/token**.

**(f) Optimal lambda (λ\*):**
**λ\* = 0.005**

This value provides the best balance between smoothing (avoiding zero probabilities) and staying faithful to the observed trigram frequencies. Values that are too large (like 5 or 1) over-smooth and wash out important distinctions in the data. Values that are too small (like 0.0005) may not provide enough smoothing for rare trigrams, leading to poor generalization on the dev set.

## Question 1: Perplexities and corpora

### Part (a): Perplexity per word with switchboard-small corpus

Using add-0.01 smoothing and vocab threshold of 3:

- **sample1**: 43.46
- **sample2**: 54.10
- **sample3**: 53.68

(Vocabulary size: 2,886 words including OOV and EOS)

### Part (b): Effect of training on larger switchboard corpus

When training on the larger switchboard corpus instead:

**Log�-probabilities become MORE NEGATIVE (worse):**
- sample1: -8282.07 � -8578.34 (difference: -296.27)
- sample2: -5008.97 � -5143.54 (difference: -134.57)
- sample3: -5085.45 � -5419.08 (difference: -333.63)

**Perplexities INCREASE (worse):**
- sample1: 43.46 � 49.74 (+14.4%)
- sample2: 54.10 � 60.22 (+11.3%)
- sample3: 53.68 � 69.71 (+29.9%)

(Vocabulary size: 11,419 words including OOV and EOS)

**Why does this happen?**

Counterintuitively, training on more data produces worse perplexity. This occurs because the larger corpus has a much larger vocabulary (11,419 vs 2,886 words). With add-� smoothing, the probability formula is:

p(z|xy) = (c(xyz) + �) / (c(xy) + �V)

When V (vocabulary size) increases:
- The denominator (c(xy) + �V) becomes much larger
- This decreases the probability assigned to each observed trigram
- The probability mass is spread over many more possible words
- The model becomes less certain about each specific prediction

Even though the larger corpus provides more training examples (2.19M tokens vs 209K tokens), the add-� smoothing method is penalized by having to distribute probability over 4� as many vocabulary words. The smoothing constant �=0.01 adds relatively little probability mass (0.01 per word), but when multiplied by V, it substantially inflates the denominator.

This demonstrates a fundamental limitation of simple add-� smoothing: it doesn't scale well as vocabulary size increases. More sophisticated smoothing methods (like backoff smoothing) handle this better by backing off to lower-order n-grams rather than distributing probability uniformly across the entire vocabulary.
