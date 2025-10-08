#!/usr/bin/env python3
"""
Quick test to verify the z_embeddings caching fix works correctly.
This demonstrates the massive speedup from caching.
"""
import time
from pathlib import Path
from probs import LanguageModel

def test_evaluation_speed():
    """Test that evaluation is now fast with caching."""
    
    # Load a model
    model_path = Path(__file__).parent / "en.model"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train the model first using train_submission_models.py")
        return
    
    print("Loading model...")
    model = LanguageModel.load(model_path)
    
    # Test log_prob computation speed
    print("Testing log_prob computation speed...")
    
    # First call - will build the cache
    start = time.time()
    _ = model.log_prob("the", "quick", "brown")
    first_call_time = time.time() - start
    print(f"First call (builds cache): {first_call_time:.4f} seconds")
    
    # Subsequent calls - should be much faster
    start = time.time()
    num_calls = 1000
    for i in range(num_calls):
        _ = model.log_prob("the", "quick", "brown")
    subsequent_time = time.time() - start
    avg_time = subsequent_time / num_calls
    
    print(f"{num_calls} subsequent calls: {subsequent_time:.4f} seconds total")
    print(f"Average time per call: {avg_time*1000:.2f} milliseconds")
    print(f"Speedup factor: {first_call_time / avg_time:.1f}x")
    
    if avg_time < first_call_time / 10:
        print("\n✓ SUCCESS: Caching is working! Evaluation is much faster.")
    else:
        print("\n⚠ WARNING: Caching may not be working as expected.")

if __name__ == "__main__":
    test_evaluation_speed()

