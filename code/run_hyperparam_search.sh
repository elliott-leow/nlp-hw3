#!/bin/bash
# Wrapper script to run hyperparameter search with options

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo -e "${BLUE}Usage: $0 [OPTIONS]${NC}"
    echo ""
    echo "Options:"
    echo "  -p, --parallel    Run parallel version (default)"
    echo "  -s, --sequential  Run sequential version"
    echo "  -j N             Limit to N processes (parallel only)"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run parallel version with all cores"
    echo "  $0 -j 4               # Run parallel version with 4 processes"
    echo "  $0 --sequential       # Run sequential version"
    echo ""
}

# Default settings
MODE="parallel"
NUM_PROCESSES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel)
            MODE="parallel"
            shift
            ;;
        -s|--sequential)
            MODE="sequential"
            shift
            ;;
        -j)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check if we're in the code directory
if [[ ! -f "probs.py" ]]; then
    echo -e "${RED}Error: Must be run from the code/ directory${NC}"
    echo "Please run: cd code && ./run_hyperparam_search.sh"
    exit 1
fi

# Check if required files exist
if [[ ! -f "vocab-genspam.txt" ]]; then
    echo -e "${RED}Error: vocab-genspam.txt not found${NC}"
    exit 1
fi

if [[ ! -d "../data/gen_spam" ]]; then
    echo -e "${RED}Error: ../data/gen_spam directory not found${NC}"
    exit 1
fi

# Run the appropriate version
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Question 19 Hyperparameter Search${NC}"
echo -e "${GREEN}========================================${NC}"

if [[ "$MODE" == "parallel" ]]; then
    echo -e "${BLUE}Mode: PARALLEL${NC}"
    
    # Get number of CPUs
    NCPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
    echo -e "${BLUE}Available CPUs: $NCPUS${NC}"
    
    if [[ -n "$NUM_PROCESSES" ]]; then
        echo -e "${BLUE}Limiting to: $NUM_PROCESSES processes${NC}"
        # Modify the script temporarily to use the specified number of processes
        python3 -c "
import sys
sys.path.insert(0, '.')
# We'll just set an environment variable instead
" 
        export MAX_PROCESSES="$NUM_PROCESSES"
    fi
    
    echo -e "${YELLOW}Running parallel version...${NC}"
    if [[ -n "$NUM_PROCESSES" ]]; then
        # Create a modified version that respects MAX_PROCESSES
        python3 hyperparam_search_q19_parallel.py
    else
        python3 hyperparam_search_q19_parallel.py
    fi
    
    RESULTS_FILE="hyperparam_results_q19_parallel.json"
else
    echo -e "${BLUE}Mode: SEQUENTIAL${NC}"
    echo -e "${YELLOW}Running sequential version...${NC}"
    python3 hyperparam_search_q19.py
    RESULTS_FILE="hyperparam_results_q19.json"
fi

# Check if the run was successful
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Search completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${BLUE}Results saved to: $RESULTS_FILE${NC}"
    
    # Try to show best results
    if command -v jq &> /dev/null; then
        echo ""
        echo -e "${BLUE}Best parameters:${NC}"
        jq -r '.best_params | "C = \(.C), d = \(.d), Avg CE = \(.avg_cross_entropy)"' "$RESULTS_FILE" 2>/dev/null || true
    else
        echo -e "${YELLOW}Install 'jq' to see formatted results: sudo apt-get install jq${NC}"
    fi
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Error: Search failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

