#!/usr/bin/env python
"""
Comprehensive test for remaining untested optimizers.
Tests: CoDE, SHADE, SaDE, IPOP-CMA-ES, BIPOP-CMA-ES, SciPy DE
"""
import subprocess
import sys
import json
import os
import sqlite3
from pathlib import Path
import time

# Optimizers to test with their respective tuning scripts
OPTIMIZERS = [
    ("CoDE", "optuna_code_tune.py", "optuna_code.db"),
    ("SHADE", "optuna_shade_tune.py", "optuna_shade.db"),
    ("SaDE", "optuna_sade_tune.py", "optuna_sade.db"),
    ("IPOP-CMA-ES", "optuna_ipop_tune.py", "optuna_ipop.db"),
    ("BIPOP-CMA-ES", "optuna_bipop_tune.py", "optuna_bipop.db"),
    ("SciPy DE", "optuna_scipy_de_tune.py", "optuna_scipy_de.db"),
]

# Test configuration
MODEL_PATH = "models/tsp_100_model_63108.pt"
INSTANCES_PATH = "instances/tsp/test/tsp100_1inst_w_optimal.pkl"
N_TRIALS = 2  # Minimal for smoke test
N_INSTANCES = 1
BATCH_SIZE = 100
SEARCH_TIMELIMIT = 10  # 10 seconds per trial

def cleanup_artifacts():
    """Clean up any existing test artifacts."""
    print("\n🧹 Cleaning up previous test artifacts...")
    subprocess.run(["bash", "cleanup_test_artifacts.sh"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_optimizer_test(name, script, db_name):
    """Run a single optimizer tuning test."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    start_time = time.time()

    # Run the tuning script
    cmd = [
        "uv", "run", "python", script,
        "--model_path", MODEL_PATH,
        "--instances_path", INSTANCES_PATH,
        "--n_trials", str(N_TRIALS),
        "--tune_n_instances", str(N_INSTANCES),
        "--batch_size", str(BATCH_SIZE),
        "--search_timelimit", str(SEARCH_TIMELIMIT),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ {name} FAILED - Script error")
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return False

    print(f"✓ Script completed in {elapsed:.1f}s")

    # Check 1: Database exists
    db_exists = os.path.exists(db_name)
    print(f"{'✓' if db_exists else '✗'} Database created: {db_name}")

    if not db_exists:
        return False

    # Check 2: Database has trials
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trials")
        trial_count = cursor.fetchone()[0]
        conn.close()
        print(f"{'✓' if trial_count >= N_TRIALS else '✗'} Database has {trial_count} trials (expected {N_TRIALS})")
        if trial_count < N_TRIALS:
            return False
    except Exception as e:
        print(f"✗ Database check failed: {e}")
        return False

    # Check 3: JSON output exists and is valid
    runs_dir = Path("runs")
    # Try multiple patterns: best_params_X.json, best_X_params.json
    json_files = list(runs_dir.glob(f"optuna_tune_*/best_*params*.json"))

    if json_files:
        # Filter to files that match the optimizer name
        # Use a more flexible matching strategy
        name_lower = name.lower()
        if "ipop" in name_lower:
            json_files = [f for f in json_files if "ipop" in f.name.lower()]
        elif "bipop" in name_lower:
            json_files = [f for f in json_files if "bipop" in f.name.lower()]
        elif "scipy" in name_lower:
            json_files = [f for f in json_files if "scipy" in f.name.lower()]
        else:
            # For other optimizers, use the first word
            first_word = name_lower.split()[0].split('-')[0]
            json_files = [f for f in json_files if first_word in f.name.lower()]

    if not json_files:
        print(f"✗ JSON output not found in runs/optuna_tune_*/ directories")
        return False

    json_path = sorted(json_files)[-1]  # Get most recent
    print(f"✓ JSON output: {json_path}")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        has_best_params = "best_params" in data
        has_best_value = "best_value" in data
        has_trials = "n_trials" in data

        print(f"{'✓' if has_best_params else '✗'} JSON has best_params")
        print(f"{'✓' if has_best_value else '✗'} JSON has best_value (gap: {data.get('best_value', 'N/A')})")
        print(f"{'✓' if has_trials else '✗'} JSON has n_trials: {data.get('n_trials', 'N/A')}")

        if not (has_best_params and has_best_value and has_trials):
            return False

    except Exception as e:
        print(f"✗ JSON validation failed: {e}")
        return False

    # Check 4: Log file exists
    log_files = list(runs_dir.glob(f"optuna_tune_*/*.log"))
    log_exists = len(log_files) > 0
    print(f"{'✓' if log_exists else '✗'} Log file created")

    if not log_exists:
        return False

    print(f"\n✅ {name} PASSED all checks!")
    return True

def main():
    """Run tests for all remaining optimizers."""
    print("="*70)
    print("REMAINING OPTIMIZERS COMPREHENSIVE TEST")
    print("="*70)
    print(f"Testing {len(OPTIMIZERS)} optimizers:")
    for name, _, _ in OPTIMIZERS:
        print(f"  - {name}")
    print(f"\nConfiguration:")
    print(f"  Trials per optimizer: {N_TRIALS}")
    print(f"  Instances per trial: {N_INSTANCES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Search time limit: {SEARCH_TIMELIMIT}s")

    # Check if model and instances exist
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found: {MODEL_PATH}")
        return 1
    if not os.path.exists(INSTANCES_PATH):
        print(f"\n❌ Instances not found: {INSTANCES_PATH}")
        return 1

    results = {}

    for name, script, db_name in OPTIMIZERS:
        cleanup_artifacts()
        time.sleep(1)  # Brief pause between tests

        try:
            success = run_optimizer_test(name, script, db_name)
            results[name] = success
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nOverall: {passed}/{total} optimizers passed")

    if passed == total:
        print("\n🎉 All optimizers working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} optimizer(s) need fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())
