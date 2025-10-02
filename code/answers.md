
## Question 2: Implementing a generic text classifier

I implemented `textcat.py` based on `fileprob.py` to perform text classification using Bayes' Theorem. The program:

1. Takes two language models (one for each category) as input
2. Takes a prior probability for the first category
3. Classifies each test file using the Maximum A Posteriori (MAP) decision rule

**Implementation approach:**

For each document, I compute the posterior probability of each class using Bayes' Theorem:
- log P(class₁ | doc) = log P(doc | class₁) + log P(class₁)
- log P(class₂ | doc) = log P(doc | class₂) + log P(class₂)

The document is classified according to whichever posterior probability is higher. Working in log space avoids numerical underflow issues when dealing with very small probabilities.

**Verification:**

When training on the smallest training sets with λ=1 and classifying all 270 dev files with prior p(gen)=0.7, the classifier correctly identifies **23 files as spam** (matching the expected result from the homework instructions).

The implementation includes:
- Vocabulary consistency check to ensure both models use the same vocabulary (required for fair comparison)
- Proper handling of prior probabilities via log transformation
- Output format matching the homework specification exactly
