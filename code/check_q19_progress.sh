#!/bin/bash
# Monitor progress of Question 19 hyperparameter search

echo "==================================================================="
echo "Question 19 Hyperparameter Search Progress"
echo "==================================================================="
echo ""

if [ ! -f hyperparam_search_q19.log ]; then
    echo "Log file not found. Search may not have started."
    exit 1
fi

# Count completed configurations
completed=$(grep -c "NEW BEST MODEL\|Average cross-entropy:" hyperparam_search_q19.log)
echo "Configurations evaluated: $completed / 15"
echo ""

# Show current configuration
echo "Current status:"
grep "Configuration.*15" hyperparam_search_q19.log | tail -1
echo ""

# Show last few lines
echo "Recent log entries:"
tail -20 hyperparam_search_q19.log
echo ""

# If results file exists, show best so far
if [ -f hyperparam_results_q19.json ]; then
    echo "==================================================================="
    echo "Best results so far:"
    python3 -c "
import json
with open('hyperparam_results_q19.json', 'r') as f:
    data = json.load(f)
    if data['best_params']:
        print(f\"C={data['best_params']['C']}, d={data['best_params']['d']}\")
        print(f\"Average cross-entropy: {data['best_avg_cross_entropy']:.4f} bits/token\")
        print(f\"Gen CE: {data['best_params']['gen_cross_entropy']:.4f} bits/token\")
        print(f\"Spam CE: {data['best_params']['spam_cross_entropy']:.4f} bits/token\")
"
fi

