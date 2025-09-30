#!/usr/bin/env python3

# Token counts from dev data
gen_tokens = 48198
spam_tokens = 39284
total_tokens = 87482

# Cross-entropy results for each lambda
results = {
    5: {"gen": 11.05263, "spam": 11.07215},
    1: {"gen": 10.45207, "spam": 10.53513},
    0.5: {"gen": 10.15485, "spam": 10.26566},
    0.05: {"gen": 9.29458, "spam": 9.44152},
    0.005: {"gen": 9.04616, "spam": 9.09572},
    0.0005: {"gen": 9.49982, "spam": 9.41952}
}

print("Combined cross-entropy for each lambda:")
print("="*50)

combined_results = []
for lam in sorted(results.keys(), reverse=True):
    gen_entropy = results[lam]["gen"]
    spam_entropy = results[lam]["spam"]

    # Calculate weighted average
    combined_entropy = (gen_entropy * gen_tokens + spam_entropy * spam_tokens) / total_tokens

    combined_results.append((lam, combined_entropy))
    print(f"Lambda={lam}: {combined_entropy:.5f} bits per token")

# Find minimum
min_result = min(combined_results, key=lambda x: x[1])
print(f"\nMinimum combined cross-entropy:")
print(f"Lambda={min_result[0]}: {min_result[1]:.5f} bits per token")