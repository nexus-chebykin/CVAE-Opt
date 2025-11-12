#!/bin/bash
# ------------------------------------------------------------------------------+
# Master Bash Script for Hyperparameter Tuning
#
# This script runs hyperparameter tuning for all 9 optimizers in sequence.
# This is a simpler alternative to run_all_tuning.py for users who prefer bash.
#
# Usage:
#   ./run_all_tuning.sh <model_path> <instances_path> [n_trials] [tune_n_instances]
#
# Example:
#   ./run_all_tuning.sh \
#     models/tsp_100_model.pt \
#     instances/tsp/test/tsp100_10inst_w_optimal.pkl \
#     50 \
#     3
#
# For more options, use the Python version: run_all_tuning.py
# ------------------------------------------------------------------------------+

set -e  # Exit on error

# Default values
MODEL_PATH="${1:-models/tsp_100_model.pt}"
INSTANCES_PATH="${2:-instances/tsp/test/tsp100_10inst_w_optimal.pkl}"
N_TRIALS="${3:-50}"
TUNE_N_INSTANCES="${4:-3}"
BATCH_SIZE="${5:-600}"
TIME_LIMIT="${6:-75}"
SEED="${7:-1234}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo "================================================================================"
echo "MASTER HYPERPARAMETER TUNING - BASH VERSION"
echo "================================================================================"
echo "Model:            $MODEL_PATH"
echo "Instances:        $INSTANCES_PATH"
echo "Trials:           $N_TRIALS"
echo "Tune instances:   $TUNE_N_INSTANCES"
echo "Batch size:       $BATCH_SIZE"
echo "Time limit:       ${TIME_LIMIT}s"
echo "Seed:             $SEED"
echo "================================================================================"

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$INSTANCES_PATH" ]; then
    echo -e "${RED}Error: Instances file not found: $INSTANCES_PATH${NC}"
    exit 1
fi

# List of optimizers
OPTIMIZERS=(
    "de:Standard DE:optuna_de_tune.py"
    "ode:EvoX ODE:optuna_ode_tune.py"
    "code:EvoX CoDE:optuna_code_tune.py"
    "shade:EvoX SHADE:optuna_shade_tune.py"
    "sade:EvoX SaDE:optuna_sade_tune.py"
    "cmaes:Standard CMA-ES:optuna_cmaes_tune.py"
    "ipop:IPOP-CMA-ES:optuna_ipop_tune.py"
    "bipop:BIPOP-CMA-ES:optuna_bipop_tune.py"
    "scipy_de:SciPy DE:optuna_scipy_de_tune.py"
)

TOTAL=${#OPTIMIZERS[@]}
SUCCESS=0
FAILED=0
START_TIME=$(date +%s)

echo ""
echo "Total optimizers to tune: $TOTAL"
echo "Estimated total time: ~$(echo "$TOTAL * 3.1" | bc) hours (with 50 trials each)"
echo "================================================================================"
echo ""

# Function to run tuning for one optimizer
run_tuning() {
    local key=$1
    local name=$2
    local script=$3
    local idx=$4

    echo "================================================================================"
    echo "OPTIMIZER $idx/$TOTAL: $name ($key)"
    echo "================================================================================"
    echo "Script: $script"
    echo ""

    local opt_start=$(date +%s)

    if uv run python "$script" \
        --model_path "$MODEL_PATH" \
        --instances_path "$INSTANCES_PATH" \
        --n_trials "$N_TRIALS" \
        --tune_n_instances "$TUNE_N_INSTANCES" \
        --batch_size "$BATCH_SIZE" \
        --time_limit "$TIME_LIMIT" \
        --seed "$SEED"; then

        local opt_end=$(date +%s)
        local opt_duration=$((opt_end - opt_start))
        local opt_minutes=$((opt_duration / 60))

        echo ""
        echo -e "${GREEN}✓ COMPLETED: $name${NC}"
        echo "  Duration: ${opt_minutes} minutes"
        ((SUCCESS++))
    else
        local opt_end=$(date +%s)
        local opt_duration=$((opt_end - opt_start))
        local opt_minutes=$((opt_duration / 60))

        echo ""
        echo -e "${RED}✗ FAILED: $name${NC}"
        echo "  Duration before failure: ${opt_minutes} minutes"
        ((FAILED++))
    fi

    echo "================================================================================"

    # Calculate progress
    local completed=$idx
    local remaining=$((TOTAL - idx))
    local elapsed=$((opt_end - START_TIME))
    local elapsed_hours=$(echo "scale=2; $elapsed / 3600" | bc)

    if [ $completed -gt 0 ]; then
        local avg_time=$((elapsed / completed))
        local est_remaining=$((avg_time * remaining))
        local est_remaining_hours=$(echo "scale=2; $est_remaining / 3600" | bc)
        local est_total_hours=$(echo "scale=2; ($elapsed + $est_remaining) / 3600" | bc)

        echo ""
        echo "Progress: $completed/$TOTAL optimizers completed"
        echo "Elapsed time: ${elapsed_hours} hours"
        echo "Estimated remaining: ${est_remaining_hours} hours"
        echo "Estimated total: ${est_total_hours} hours"
        echo ""
    fi
}

# Run tuning for each optimizer
idx=1
for optimizer in "${OPTIMIZERS[@]}"; do
    IFS=':' read -r key name script <<< "$optimizer"
    run_tuning "$key" "$name" "$script" "$idx"
    ((idx++))
done

# Print summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$(echo "scale=2; $TOTAL_DURATION / 3600" | bc)
TOTAL_MINUTES=$(echo "scale=1; $TOTAL_DURATION / 60" | bc)

echo ""
echo "================================================================================"
echo "MASTER TUNING SUMMARY"
echo "================================================================================"
echo "Total optimizers: $TOTAL"
echo -e "${GREEN}Successful: $SUCCESS${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo "Failed: $FAILED"
fi
echo "Total duration: ${TOTAL_HOURS} hours (${TOTAL_MINUTES} minutes)"
echo "================================================================================"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✓ All optimizers tuned successfully!${NC}\n"
    exit 0
else
    echo -e "\n${YELLOW}⚠ $FAILED optimizer(s) failed. Check logs for details.${NC}\n"
    exit 1
fi
