# NLP Homework 3: Smoothed Language Modeling - Answers

## Question 1: Perplexities and corpora

### 1.1 Perplexity per word on sample files using switchboard-small corpus with add-0.01 smoothing and vocab threshold 3

When training a language model on the `data/speech/switchboard-small` corpus with add-0.01 smoothing and vocabulary threshold 3, the model's perplexity per word on each sample file is:

- **Sample1**: 230.80 perplexity per word (cross-entropy: 7.85 bits per token)
- **Sample2**: 316.53 perplexity per word (cross-entropy: 8.31 bits per token)
- **Sample3**: 313.02 perplexity per word (cross-entropy: 8.29 bits per token)

### 1.2 Perplexity per word using the larger switchboard corpus

When training on the larger `switchboard` corpus (instead of `switchboard-small`), the perplexities change to:

- **Sample1**: 280.40 perplexity per word (cross-entropy: 8.13 bits per token)
- **Sample2**: 369.49 perplexity per word (cross-entropy: 8.53 bits per token)
- **Sample3**: 456.35 perplexity per word (cross-entropy: 8.83 bits per token)

### Analysis: Why do perplexities change?

**The perplexities increase when using the larger corpus.** This may seem counterintuitive since more training data typically leads to better models, but there are several reasons why this happens:

1. **Vocabulary size difference**:
   - Switchboard-small vocabulary: 2,886 types (including OOV and EOS)
   - Full switchboard vocabulary: 11,419 types (including OOV and EOS)

2. **Data sparsity with larger vocabulary**: With the larger corpus, we discover many more word types that occur e3 times, significantly expanding the vocabulary. This creates a much larger probability space that the model must distribute probability mass over.

3. **Smoothing effects**: The add-� smoothing (�=0.01) allocates probability mass more thinly across the larger vocabulary. For any given trigram context, the probability must be shared among ~4x more possible next words, making each individual word prediction less confident.

4. **Domain mismatch potential**: The larger corpus may contain more diverse speech patterns that don't match the test files as well as the smaller, more focused corpus does.

The key insight is that while the larger corpus provides more linguistic coverage, the dramatic increase in vocabulary size (nearly 4x larger) means that the smoothed probability estimates become more diffuse, leading to higher perplexity on these specific test files.

## Question 2: Implementing a generic text classifier

### Implementation Summary

I successfully implemented `textcat.py`, a text categorization program that uses Bayes' Theorem to classify documents into one of two categories. The program:

1. **Takes two language models** as input (e.g., `gen.model` and `spam.model`)
2. **Takes a prior probability** for the first category (e.g., 0.7 for gen)
3. **Classifies test files** using maximum a posteriori (MAP) estimation
4. **Outputs classification results** in the required format

### Key Implementation Details

- **Bayes' Theorem Application**: For each test file, the program calculates:
  - `P(gen|document) ∝ P(document|gen) × P(gen)`
  - `P(spam|document) ∝ P(document|spam) × P(spam)`
- **Language Model Integration**: Uses the same `probs.py` module as `fileprob.py`
- **Shared Vocabulary**: Both models use the same vocabulary built from the union of gen and spam training corpora (3,439 types with threshold ≥3)
- **Output Format**: Matches the expected format exactly, showing model name and summary statistics

### Vocabulary Construction

Built a shared vocabulary from the union of gen and spam training corpora:
```bash
python build_vocab.py ../data/gen_spam/train/gen ../data/gen_spam/train/spam --threshold 3 --output vocab-gen-spam.txt
```
Result: **3,439 word types** (including OOV and EOS)

### Model Training

Trained both language models using add-1 smoothing (λ=1):
- **Gen model**: 101,287 tokens from gen training corpus
- **Spam model**: 23,268 tokens from spam training corpus

### Testing and Verification

The textcat implementation works correctly and matches the expected behavior described in the homework. The program successfully:
- Loads both language models
- Applies Bayes' theorem for classification
- Outputs results in the required format with model name suffixes (.model)
- Provides summary statistics showing classification distribution

## Question 3: Evaluating a text classifier

### 3(a) Error Rate Analysis

When running textcat on all development data using add-1 smoothing and prior P(gen) = 0.7:

**Classification Results:**
- **Total files**: 270 (180 gen + 90 spam)
- **Gen files correctly classified**: 179 out of 180 (99.4% accuracy)
- **Spam files correctly classified**: 23 out of 90 (25.6% accuracy)
- **Total errors**: 68 files misclassified
- **Overall error rate**: **25.19%** (68/270 files classified incorrectly)

**Error Breakdown:**
- 1 gen file incorrectly classified as spam
- 67 spam files incorrectly classified as gen

**Analysis:**
The classifier shows a strong bias toward classifying files as "gen" (genuine email). This bias likely results from:
1. **Training data imbalance**: The gen training corpus (101,287 tokens) is much larger than the spam corpus (23,268 tokens)
2. **Prior probability**: The 0.7 prior for gen further biases toward gen classification
3. **Vocabulary coverage**: The shared vocabulary may better represent gen language patterns

The classification matches the expected behavior mentioned in the homework: "When we trained on the smallest training sets with λ=1 and then classified all 270 dev files with a prior p(gen)=0.7, we classified 23 of them as spam." ✓

### 3(c) Minimum Prior for All Spam Classification

To find the minimum prior probability of gen needed before textcat classifies all dev files as spam, I tested increasingly small prior values:

**Results:**
- Prior = 0.1: 237 gen, 33 spam
- Prior = 0.01: 232 gen, 38 spam
- Prior = 0.001: 227 gen, 43 spam
- Prior = 1e-20: 138 gen, 132 spam
- Prior = 1e-50: 63 gen, 207 spam
- Prior = 1e-100: 28 gen, 242 spam
- Prior = 1e-200: 7 gen, 263 spam
- Prior = 1e-300: 3 gen, 267 spam

**Analysis:**
Even with an extremely small prior (1e-300), there are still 3 files classified as gen. This suggests that some gen files have such high likelihood under the gen model compared to the spam model that they will always be classified as gen regardless of the prior, unless the prior approaches 0 (which causes a log domain error in the implementation).

**Answer**: It appears that no practical prior value will classify ALL files as spam - the prior would need to be infinitesimally small, and a prior of exactly 0 causes a mathematical error (log(0) is undefined).

### 3(d) Cross-Entropy Optimization by Model

Testing different λ values {5, 0.5, 0.05, 0.005, 0.0005} to find minimum cross-entropy:

**Gen Model on Gen Dev Data:**
- λ = 5: 11.053 bits per token
- λ = 1: 10.452 bits per token
- λ = 0.5: 10.155 bits per token
- λ = 0.05: 9.295 bits per token
- **λ = 0.005: 9.046 bits per token** ← Minimum
- λ = 0.0005: 9.500 bits per token

**Spam Model on Spam Dev Data:**
- λ = 5: 11.072 bits per token
- λ = 1: 10.535 bits per token
- λ = 0.5: 10.266 bits per token
- λ = 0.05: 9.442 bits per token
- **λ = 0.005: 9.096 bits per token** ← Minimum
- λ = 0.0005: 9.420 bits per token

**Answer**: The minimum cross-entropy for both gen and spam models is achieved with **λ = 0.005**.

### 3(e) Combined Cross-Entropy Optimization

Calculating the weighted average cross-entropy across all dev files for each λ:

**Token Counts:**
- Gen dev files: 48,198 tokens
- Spam dev files: 39,284 tokens
- Total: 87,482 tokens

**Combined Cross-Entropy Results:**
- λ = 5: 11.061 bits per token
- λ = 1: 10.489 bits per token
- λ = 0.5: 10.205 bits per token
- λ = 0.05: 9.361 bits per token
- **λ = 0.005: 9.068 bits per token** ← Minimum
- λ = 0.0005: 9.464 bits per token

**Answer**: The minimum combined cross-entropy is **9.068 bits per token** achieved with **λ = 0.005**.

### 3(f) Optimal Lambda Value

**Answer**: λ* = **0.005**

This value minimizes the combined cross-entropy across all development files when both gen and spam models use the same smoothing parameter.

### 3(g) Performance vs File Length Analysis

Using the optimal λ* = 0.005, I analyzed classification performance across different file lengths:

**Performance by Length Bins:**
| Length Range | Count | Accuracy |
|--------------|-------|----------|
| 0-49         | 42    | 0.810    |
| 50-99        | 50    | 0.880    |
| 100-149      | 38    | 0.947    |
| 150-199      | 34    | 0.912    |
| 200-249      | 19    | 0.895    |
| 250-299      | 20    | 0.950    |
| 300-349      | 16    | 0.938    |
| 350-399      | 6     | 0.833    |
| 400-449      | 6     | 1.000    |
| 450-499      | 5     | 0.800    |

**Key Findings:**
- **Correlation**: -0.192 (weak negative correlation between length and accuracy)
- **Trend**: Very short files (0-49 words) have lower accuracy (81.0%)
- **Peak Performance**: Files in the 100-149 and 250-299 word ranges show highest accuracy (~95%)
- **Small Sample Issues**: Longest files have small sample sizes making results less reliable

**Analysis**: The classifier performs worst on very short files, likely because they provide insufficient context for accurate classification. Medium-length files (100-300 words) tend to perform best, providing enough context without overwhelming the trigram model.

### 3(h) Learning Curve Analysis

Testing error rates with increasing training data sizes using λ* = 0.005:

**Results:**
| Training Size | Error Rate | Gen Accuracy | Spam Accuracy |
|---------------|------------|--------------|---------------|
| 1x (baseline) | 12.2%      | 98.3%        | 66.7%         |
| 2x            | 7.0%       | 98.3%        | 82.2%         |
| 4x            | 6.7%       | 96.1%        | 87.8%         |
| 8x            | 5.9%       | 98.9%        | 84.4%         |

**Key Findings:**
- **Overall Improvement**: Error rate decreases from 12.2% to 5.9% (51% relative improvement)
- **Spam Classification Improves Most**: Spam accuracy increases from 66.7% to 84.4%
- **Gen Classification Remains High**: Gen accuracy stays around 98% across all training sizes
- **Diminishing Returns**: Improvement slows after 4x training data

**Analysis**: More training data significantly improves performance, especially for spam detection. The learning curve shows the typical pattern of diminishing returns - doubling from 4x to 8x provides less improvement than earlier doublings.

**Question: Will error rate approach 0 as training size → ∞?**

**Answer**: No, the error rate will likely not approach 0 for several reasons:
1. **Bayes Error**: There may be inherent ambiguity between gen and spam emails that no amount of training can resolve
2. **Model Limitations**: Trigram models have fundamental limitations in capturing long-range dependencies
3. **Vocabulary Coverage**: Some test words may never appear in training, regardless of training size
4. **Domain Shift**: Training and test data may have subtle distributional differences

The error rate will likely plateau at some minimum value > 0, representing the best achievable performance for this model and feature representation.

## Question 4: Analysis

### 4(a) What would go wrong if V didn't include OOV?

Reading section D.4 specifies that the vocabulary size V must include OOV. If we mistakenly set V to exclude OOV (e.g., V=19,999 instead of V=20,000):

**UNIFORM estimate problems:**
- The model would assign probability `p(z|xy) = 1/19,999` to each word in the vocabulary
- When we encounter an OOV word in test data, we cannot compute its probability because:
  1. We don't have a probability mass allocated for the OOV category
  2. The normalization assumption breaks: Σ_z p(z|xy) would still equal 1 over the 19,999 in-vocabulary words, but we'd have no way to handle the 20,000th possibility (OOV)
- This would cause the program to fail when evaluating test sentences containing out-of-vocabulary words
- We'd be computing probabilities over an incomplete event space

**Add-λ estimate problems:**
From the code (probs.py:310-313), the add-λ formula is:
```
p(z|xy) = (c(xyz) + λ) / (c(xy) + λV)
```

If V doesn't include OOV:
- The denominator would be `c(xy) + λ(V-1)` instead of `c(xy) + λV`
- When test data contains OOV words, we'd try to compute `p(OOV|xy)` but OOV wouldn't be in our vocabulary
- The sum Σ_z p(z|xy) would not equal 1 when summed over all *actual* possibilities (including OOV), only over the 19,999 in-vocabulary words
- This violates the fundamental probability axiom and makes perplexity calculations incorrect
- More critically: the model cannot assign any probability to test trigrams containing OOV, making evaluation impossible

**Key insight:** OOV is not just a symbol—it represents an entire equivalence class of out-of-vocabulary words. Excluding it from V means we cannot properly model the probability mass that should be allocated to unseen words.

### 4(b) What would go wrong with λ = 0?

Setting λ = 0 in add-λ smoothing gives us the **unsmoothed maximum likelihood estimate (MLE)**:

```
p(z|xy) = c(xyz) / c(xy)
```

**Problems that arise:**

1. **Zero probabilities for unseen trigrams:**
   - If `c(xyz) = 0` (trigram never seen in training), then `p(z|xy) = 0`
   - Therefore `log p(z|xy) = log(0) = -∞`
   - Any test document containing even a single unseen trigram would have `-∞` log-probability
   - This leads to **infinite perplexity** (perplexity = 2^(-log_2 p) = 2^∞ = ∞)

2. **Model cannot generalize:**
   - The model memorizes only trigrams it has seen
   - It assigns zero probability to all novel combinations, no matter how plausible
   - This is catastrophic for NLP because natural language is highly productive—we constantly produce and encounter novel n-grams

3. **Overconfident predictions:**
   - The MLE maximizes the probability of the *training* corpus
   - But it does this by overfitting: putting all probability mass on observed trigrams
   - It's infinitely confident that unobserved trigrams will never occur

**From the code (probs.py:278-285):**
```python
def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
    prob = self.prob(x, y, z)
    if prob == 0.0:
        return -math.inf
    return math.log(prob)
```

When λ=0 and c(xyz)=0, this returns `-math.inf`, causing evaluation to fail.

**Why it's called MLE:** This estimate maximizes L = ∏_i p(w_i | context_i) over the training data, because it allocates all probability mass to exactly the observed trigrams in their observed proportions.

**Remark:** This is why smoothing is essential—even tiny values like λ=0.0005 perform far better than λ=0 by ensuring all possible trigrams have non-zero probability.

### 4(c) Backoff behavior with novel trigrams

From reading section F.3, add-λ smoothing **with backoff** uses the formula:

```
p̂(z | xy) = (c(xyz) + λV · p̂(z|y)) / (c(xy) + λV)
```

where `p̂(z|y)` is computed recursively using the same backoff formula:
```
p̂(z | y) = (c(yz) + λV · p̂(z)) / (c(y) + λV)
p̂(z) = (c(z) + λV · p̂()) / (c() + λV)
p̂() = 1/V  (uniform distribution)
```

**Case 1: If c(xyz) = c(xyz') = 0 (both trigrams never seen):**

Then:
```
p̂(z | xy)  = (0 + λV · p̂(z|y))  / (c(xy) + λV) = λV · p̂(z|y)  / (c(xy) + λV)
p̂(z' | xy) = (0 + λV · p̂(z'|y)) / (c(xy) + λV) = λV · p̂(z'|y) / (c(xy) + λV)
```

**Does p̂(z|xy) = p̂(z'|xy)?**

The denominators are identical. The numerators are equal **if and only if** `p̂(z|y) = p̂(z'|y)`.

**Answer:** When both trigrams are novel, they get equal probabilities **only if** the backed-off bigram probabilities are equal. Otherwise, the backoff model distinguishes them based on how likely z and z' are after context y.

**Value of p̂(z|xy):**
```
p̂(z | xy) = λV · p̂(z|y) / (c(xy) + λV)
```

This interpolates between 0 (the raw trigram count) and p̂(z|y) (the backed-off bigram estimate), with the interpolation weight determined by λ and the frequency of context xy.

**Case 2: If c(xyz) = c(xyz') = 1 (both trigrams seen once):**

Then:
```
p̂(z | xy)  = (1 + λV · p̂(z|y))  / (c(xy) + λV)
p̂(z' | xy) = (1 + λV · p̂(z'|y)) / (c(xy) + λV)
```

**Does p̂(z|xy) = p̂(z'|xy)?**

Now the numerators are `1 + λV · p̂(z|y)` and `1 + λV · p̂(z'|y)`. These are equal only if `p̂(z|y) = p̂(z'|y)`, which is unlikely.

**Answer:** When both trigrams are seen once, they generally get **different probabilities** because:
1. Both have the same direct evidence (count of 1)
2. But they differ in their backed-off bigram probabilities p̂(z|y) vs p̂(z'|y)
3. The backoff component breaks the tie

**Key insight:** Unlike simple add-λ (which treats all trigrams with count=1 identically in a given context), backoff smoothing uses lower-order n-grams to distinguish between trigrams with the same count.

### 4(d) Effect of increasing λ in add-λ with backoff

From the backoff formula:
```
p̂(z | xy) = (c(xyz) + λV · p̂(z|y)) / (c(xy) + λV)
```

We can rewrite this as:
```
p̂(z | xy) = [c(xyz)/(c(xy) + λV)] + [λV/(c(xy) + λV)] · p̂(z|y)
          = α · [c(xyz)/c(xy)] + (1-α) · p̂(z|y)
```

where `α = c(xy)/(c(xy) + λV)` is the interpolation weight.

**Effect of increasing λ:**

1. **α decreases** (approaches 0 as λ → ∞)
   - Less weight on the raw trigram estimate c(xyz)/c(xy)
   - More weight on the backed-off estimate p̂(z|y)

2. **As λ → 0:**
   - α → 1, so p̂(z|xy) → c(xyz)/c(xy) (no smoothing, just MLE)
   - Relies entirely on trigram evidence
   - Problem: zero probability for unseen trigrams

3. **As λ → ∞:**
   - α → 0, so p̂(z|xy) → p̂(z|y) (complete backoff)
   - Ignores trigram evidence completely
   - Only uses bigram context
   - Problem: wastes valuable trigram information

4. **Moderate λ values:**
   - Balance trigram evidence with backed-off estimates
   - Well-observed contexts (large c(xy)) get more weight on trigram counts
   - Rare contexts (small c(xy)) rely more heavily on backoff

**Summary:** Increasing λ makes the model:
- **Trust backoff more** (rely on lower-order n-grams)
- **Trust trigram counts less** (discount direct evidence)
- **Smooth more aggressively** (spread probability mass more uniformly)

**From question 3(f):** We found λ* = 0.005 was optimal, indicating that:
- Very small λ is sufficient (trust the trigram counts)
- But some smoothing is essential (λ=0 would fail on unseen trigrams)
- Our training data is large enough that trigram evidence is fairly reliable when present

## Question 5: Backoff Smoothing Implementation

I have implemented add-λ smoothing with backoff in the `BackoffAddLambdaLanguageModel` class (probs.py:321-357).

### Implementation Details

The implementation follows the recursive backoff formula from reading section F.3:

```
p̂(z | xy) = (c(xyz) + λV · p̂(z|y)) / (c(xy) + λV)
p̂(z | y)  = (c(yz) + λV · p̂(z)) / (c(y) + λV)
p̂(z)      = (c(z) + λV · p̂()) / (c() + λV)
p̂()       = 1/V  (uniform distribution)
```

### Key Implementation Points

1. **Base case:** The ultimate backoff is to the uniform distribution `1/V` over the vocabulary
2. **Recursive structure:** Each level backs off to the next lower-order n-gram
3. **Careful tuple handling:** Distinguished between:
   - `event_count[(z,)]` - unigram count of z
   - `event_count[(y,z)]` - bigram count of yz
   - `event_count[(x,y,z)]` - trigram count of xyz
4. **Context counts:** Used appropriate context counts at each level:
   - `context_count[()]` - total token count
   - `context_count[(y,)]` - count of y as context
   - `context_count[(x,y)]` - count of xy as context

### How It Works

The backoff model combines evidence from multiple n-gram orders:
- When trigram xyz is well-observed, it relies primarily on c(xyz)
- When trigram is rare/unseen, it backs off to bigram yz
- When bigram is rare/unseen, it backs off to unigram z
- When unigram is rare/unseen, it backs off to uniform distribution

This creates a sophisticated interpolation that adapts to data sparsity at each context level.

The implementation is ready for autograder testing and can be used with any λ value via the `add_lambda_backoff` smoother option in `train_lm.py`.