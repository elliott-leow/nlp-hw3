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

## Question 4: Analysis

### Part (a): What if V doesn't include OOV?

If we mistakenly set V = 19,999 instead of V = 20,000 (excluding OOV from the count), both the UNIFORM and add-λ estimates would violate a fundamental requirement: **probabilities must sum to 1**.

**For UNIFORM smoothing:**

The formula is: p̂(z|xy) = 1/V for all z

If V = 19,999 but there are actually 20,000 possible outcomes (the 19,999 word types plus OOV), then:
- Each outcome gets probability 1/19,999
- Total probability = 20,000 × (1/19,999) = 20,000/19,999 ≈ 1.00005 > 1

This violates the axiom that probabilities must sum to 1, making the model mathematically invalid.

**For add-λ smoothing:**

The formula is: p̂(z|xy) = (c(xyz) + λ) / (c(xy) + λV)

If V = 19,999 instead of 20,000:
- The denominator c(xy) + λV is too small by exactly λ
- When we sum p̂(z|xy) over all 20,000 actual possible outcomes (including OOV), we get:

  Σ_z p̂(z|xy) = Σ_z (c(xyz) + λ) / (c(xy) + λV)
              = [Σ_z c(xyz) + 20,000λ] / (c(xy) + 19,999λ)
              = [c(xy) + 20,000λ] / (c(xy) + 19,999λ)
              > 1

Again, probabilities don't sum to 1, making the model invalid.

**Why this matters:** Both models need V to match the actual number of possible outcomes in the vocabulary (including special tokens like OOV and EOS) so that the probability distribution is properly normalized.

### Part (b): What goes wrong with λ = 0?

With λ = 0, the add-λ formula becomes:

p̂(z|xy) = (c(xyz) + 0) / (c(xy) + 0) = c(xyz) / c(xy)

This is the **unsmoothed maximum likelihood estimate (MLE)**, which has several serious problems:

1. **Zero probabilities for unseen events:** Any trigram xyz that didn't appear in training gets p̂(z|xy) = 0/c(xy) = 0. This is unrealistic—just because we didn't observe something doesn't mean it's impossible.

2. **Undefined log probabilities:** When evaluating test data, we compute log p̂(z|xy). If p̂(z|xy) = 0, then log(0) = -∞, which breaks our calculations.

3. **Zero probability for entire sentences:** If a test sentence contains even one unseen trigram, the entire sentence gets probability 0 (since probabilities multiply). This makes the model useless for evaluation.

4. **Infinite perplexity:** Perplexity = 2^(cross-entropy). With zero probabilities, cross-entropy = ∞, so perplexity = ∞.

5. **Division by zero:** If c(xy) = 0 (context never seen), we'd have 0/0, which is undefined.

**Testing λ = 0:** If you try training with λ = 0 and test on data with unseen trigrams, fileprob.py would encounter log(0) and likely crash or return -inf for the log probability.

The whole point of smoothing is to avoid these zero probabilities by allocating some probability mass to unseen events. Even λ = 0.001 is better than λ = 0 because it ensures all trigrams have positive probability.

### Part (c): Backoff behavior with novel trigrams

The backoff add-λ formula (from reading section F.3) is:

p̂(z|xy) = (c(xyz) + λV · p̂(z|y)) / (c(xy) + λV)

where p̂(z|y) is the backed-off bigram estimate (computed recursively with the same backoff formula).

**Case 1: c(xyz) = c(xyz') = 0 (both trigrams unseen)**

For both trigrams:
- p̂(z|xy) = (0 + λV · p̂(z|y)) / (c(xy) + λV) = (λV · p̂(z|y)) / (c(xy) + λV)
- p̂(z'|xy) = (0 + λV · p̂(z'|y)) / (c(xy) + λV) = (λV · p̂(z'|y)) / (c(xy) + λV)

**Answer:** No, p̂(z|xy) ≠ p̂(z'|xy) in general! They are only equal if p̂(z|y) = p̂(z'|y).

The probability is: **p̂(z|xy) = [λV/(c(xy) + λV)] · p̂(z|y)**

This is proportional to the backed-off bigram probability. The backoff model uses lower-order information to distinguish between unseen trigrams.

**Contrast with regular add-λ (no backoff):** Without backoff, both would equal λ/(c(xy) + λV), treating all unseen trigrams identically regardless of their bigram contexts.

**Case 2: c(xyz) = c(xyz') = 1 (both seen once)**

- p̂(z|xy) = (1 + λV · p̂(z|y)) / (c(xy) + λV)
- p̂(z'|xy) = (1 + λV · p̂(z'|y)) / (c(xy) + λV)

**Answer:** Again, p̂(z|xy) ≠ p̂(z'|xy) in general unless p̂(z|y) = p̂(z'|y).

Now both trigrams have the same observed count (1), but their probabilities still differ based on their bigram contexts. The backed-off bigram probabilities still influence the final estimates, though less strongly than in Case 1.

### Part (d): Effect of increasing λ in backoff smoothing

In the backoff formula:

p̂(z|xy) = (c(xyz) + λV · p̂(z|y)) / (c(xy) + λV)

**As λ increases:**

1. **Numerator:** c(xyz) + λV · p̂(z|y) becomes dominated by the term λV · p̂(z|y)
2. **Denominator:** c(xy) + λV becomes dominated by λV
3. **Ratio approaches:** (λV · p̂(z|y)) / λV = p̂(z|y)

Therefore, larger λ causes the trigram estimate to **back off more strongly** to the bigram estimate p̂(z|y). The observed trigram count c(xyz) matters less and less.

**Effect on probability estimates:**
- Trigrams with the same bigram context y become more similar to each other (since they all converge toward p̂(z|y))
- Rare trigrams benefit more from the backed-off information
- The model relies more on lower-order n-grams and less on specific trigram observations
- The distinction from part (c) becomes more pronounced: unseen trigrams are differentiated by their bigram contexts rather than being treated uniformly

**As λ → 0:**
- The observed counts c(xyz) dominate
- Less backing off occurs
- The model approaches the unsmoothed MLE (with all its problems from part b)
- The trigram estimates become less reliable for rare events

**In summary:** λ controls the trade-off between trusting the observed trigram counts versus backing off to lower-order models. Larger λ means more smoothing and stronger reliance on backed-off estimates.
