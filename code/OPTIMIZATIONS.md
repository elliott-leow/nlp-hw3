# Optimized Log-Linear Language Model

## Architecture
The model maintains the pure log-linear form:
```
f(x,y,z) = x^T X z + y^T Y z + bias_x(z) + bias_y(z)
```

where:
- `x`, `y` are context word embeddings
- `z` is target word embedding
- `X`, `Y` are learned transformation matrices (dim × dim)
- `bias_x`, `bias_y` are per-word bias terms

This is still a **log-linear model** (linear in log space), not a deep neural network.

## Key Optimizations

### 1. **Better Parameter Initialization**
- **Xavier/Glorot Uniform** initialization for `X` and `Y` matrices
- Helps avoid vanishing/exploding gradients
- Scales initialization based on layer sizes
- Small random initialization for bias terms (-0.01 to 0.01)

### 2. **Bias Terms**
- Added learnable bias vectors for each vocabulary word
- Allows model to capture word frequency priors
- Still maintains log-linear property: `log p(z|x,y) ∝ f(x,y,z)`

### 3. **AdamW Optimizer**
- **Adam** with **decoupled weight decay** instead of vanilla SGD
- Adaptive learning rates per parameter
- Better momentum handling with β1=0.9, β2=0.999
- More stable convergence, especially with sparse gradients

### 4. **Learning Rate Schedule**
- **Warmup phase**: Linear increase for first 500 steps
  - Prevents large updates early in training
  - Helps with stability
- **Cosine annealing**: Gradual decay after warmup
  - `lr = base_lr * 0.5 * (1 + cos(π * progress))`
  - Smooth decay leads to better convergence

### 5. **Label Smoothing**
- Regularization technique that softens one-hot targets
- Target gets confidence (1 - ε), other words get ε/(V-1)
- Default ε = 0.1
- Prevents overconfidence, improves generalization
- Particularly helpful for language modeling

### 6. **Gradient Accumulation**
- Accumulate gradients over 2 mini-batches before updating
- Effective batch size = batch_size × grad_accum_steps = 128
- More stable gradients without memory overhead
- Better gradient estimates

### 7. **Gradient Clipping**
- Clip gradient norm to max=1.0
- Prevents exploding gradients
- Stabilizes training

### 8. **Smart Regularization**
- L2 regularization applied **only to X and Y matrices**
- **Not** applied to bias terms
- Biases need flexibility to capture frequency priors
- Standard practice in modern deep learning

### 9. **Dropout Regularization**
- Applied to word embeddings during training (rate=0.15)
- Forces model to not rely on specific embedding dimensions
- Improves generalization

### 10. **Embedding Caching**
- Cache all vocabulary embeddings in memory
- Avoid repeated dictionary lookups
- Significant speedup during training

### 11. **Data Shuffling**
- Shuffle training data each epoch
- Prevents order-dependent learning
- Better exploration of parameter space

## Training Configuration

Default hyperparameters (optimized for gen/spam task):
```python
dropout = 0.15              # Embedding dropout rate
batch_size = 64             # Mini-batch size  
label_smoothing = 0.1       # Label smoothing factor
warmup_steps = 500          # LR warmup steps
grad_accum_steps = 2        # Gradient accumulation
base_lr = 0.001             # Base learning rate
l2 = 0.01                   # L2 regularization strength
```

## Theoretical Justification

### Why These Techniques Work for Log-Linear Models:

1. **Xavier Initialization**: Based on maintaining variance of activations across layers. Even though we only have one "layer", it ensures X and Y have appropriate scale relative to embedding magnitudes.

2. **Bias Terms**: In log-linear models, biases act as unigram priors. The model learns: `log p(z|x,y) = x^T X z + y^T Y z + b_x(z) + b_y(z) - log Z(x,y)`. The biases capture word frequency information independent of context.

3. **AdamW**: Particularly effective for NLP because:
   - Word embeddings are high-dimensional and sparse
   - Adaptive learning rates handle varying gradient magnitudes
   - Decoupled weight decay is more principled than L2 in Adam

4. **Label Smoothing**: Prevents the model from becoming overconfident. In language modeling, there's inherent uncertainty (multiple plausible next words), so softening targets is theoretically justified.

5. **Learning Rate Scheduling**: 
   - Warmup prevents instability from large random initialization
   - Cosine annealing allows fine-tuning in later epochs
   - Well-established in transformer training (BERT, GPT, etc.)

## Performance Expectations

Compared to baseline log-linear model, expect:
- **10-30% lower perplexity** on held-out data
- **Faster convergence** (fewer epochs needed)
- **Better generalization** (smaller train-test gap)
- **More stable training** (less variance across runs)

## References

- Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization" (AdamW)
- Szegedy et al. (2016): "Rethinking the Inception Architecture for Computer Vision" (Label Smoothing)
- Vaswani et al. (2017): "Attention is All You Need" (LR warmup + cosine annealing)

