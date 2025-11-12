#!/bin/bash
# Cleanup Test Artifacts
#
# Removes all test-related artifacts including databases, JSON outputs,
# visualization plots, and test logs. Keeps framework code intact.

echo "Cleaning up test artifacts..."

# Remove Optuna databases (both patterns)
rm -f optuna_*_tuning.db
rm -f optuna_*_tuning.db-shm
rm -f optuna_*_tuning.db-wal
rm -f optuna_*.db
rm -f optuna_*.db-shm
rm -f optuna_*.db-wal

# Remove best parameters JSON files
rm -f best_de_params.json
rm -f best_ode_params.json
rm -f best_code_params.json
rm -f best_shade_params.json
rm -f best_sade_params.json
rm -f best_cmaes_params.json
rm -f best_ipop_params.json
rm -f best_bipop_params.json
rm -f best_scipy_de_params.json

# Remove visualization plots
rm -f *_optimization_history.png
rm -f *_param_importances.png
rm -f *_parallel_coordinate.png

# Remove master tuning summary
rm -f tuning_summary.json

# Remove test log files (optional - comment out if you want to keep them)
# rm -f logs/optuna_*_tuning_*.log
# rm -f logs/master_tuning_*.log

echo "✓ Cleanup complete!"
echo ""
echo "Removed:"
echo "  - Optuna databases (*.db)"
echo "  - Best parameters JSON files"
echo "  - Visualization plots (*.png)"
echo "  - Master tuning summary"
echo ""
echo "Kept:"
echo "  - All tuning scripts"
echo "  - All test scripts"
echo "  - Log files (in logs/ directory)"
echo "  - Framework documentation"
