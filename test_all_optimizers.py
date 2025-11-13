#!/usr/bin/env python
"""
Final comprehensive test for ALL working optimizers.
Tests all 8 validated optimizers to ensure framework is production-ready.
"""
import subprocess
import sys
import json
import os
import sqlite3
from pathlib import Path
import time

# All working optimizers (all 9 optimizers now working!)
OPTIMIZERS = [
    ("DE", "optuna_de_tune.py", "optuna_de.db"),
    ("CMA-ES", "optuna_cmaes_tune.py", "optuna_cmaes.db"),
    ("ODE", "optuna_ode_tune.py", "optuna_ode.db"),
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
    subprocess.run(["bash", "cleanup_test_artifacts.sh"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_optimizer_test(name, script, db_name):
    """Run a single optimizer tuning test."""
    print(f"\nTesting {name}... ", end='', flush=True)

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

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ FAILED ({elapsed:.1f}s)")
        return False

    # Quick validation checks
    db_exists = os.path.exists(db_name)
    if not db_exists:
        print(f"❌ FAILED - no database ({elapsed:.1f}s)")
        return False

    # Check database has trials
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trials")
        trial_count = cursor.fetchone()[0]
        conn.close()
        if trial_count < N_TRIALS:
            print(f"❌ FAILED - {trial_count}/{N_TRIALS} trials ({elapsed:.1f}s)")
            return False
    except Exception:
        print(f"❌ FAILED - DB check error ({elapsed:.1f}s)")
        return False

    # Check JSON exists
    runs_dir = Path("runs")
    json_files = list(runs_dir.glob(f"optuna_tune_*/best_*params*.json"))

    if json_files:
        name_lower = name.lower()
        if "ipop" in name_lower:
            json_files = [f for f in json_files if "ipop" in f.name.lower()]
        elif "bipop" in name_lower:
            json_files = [f for f in json_files if "bipop" in f.name.lower()]
        elif "scipy" in name_lower:
            json_files = [f for f in json_files if "scipy" in f.name.lower()]
        else:
            first_word = name_lower.split()[0].split('-')[0]
            json_files = [f for f in json_files if first_word in f.name.lower()]

    if not json_files:
        print(f"❌ FAILED - no JSON ({elapsed:.1f}s)")
        return False

    # Validate JSON content
    try:
        json_path = sorted(json_files)[-1]
        with open(json_path, 'r') as f:
            data = json.load(f)

        if not all(k in data for k in ["best_params", "best_value", "n_trials"]):
            print(f"❌ FAILED - invalid JSON ({elapsed:.1f}s)")
            return False

        gap = data.get('best_value', 999)
        print(f"✅ PASSED ({elapsed:.1f}s, gap: {gap:.2f}%)")
        return True

    except Exception:
        print(f"❌ FAILED - JSON read error ({elapsed:.1f}s)")
        return False

def main():
    """Run tests for all working optimizers."""
    print("="*70)
    print("COMPREHENSIVE FRAMEWORK VALIDATION TEST")
    print("="*70)
    print(f"Testing {len(OPTIMIZERS)} production-ready optimizers")
    print(f"Configuration: {N_TRIALS} trials, {N_INSTANCES} instance(s), {BATCH_SIZE} batch size, {SEARCH_TIMELIMIT}s time limit")
    print("="*70)

    # Check if model and instances exist
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found: {MODEL_PATH}")
        return 1
    if not os.path.exists(INSTANCES_PATH):
        print(f"\n❌ Instances not found: {INSTANCES_PATH}")
        return 1

    results = {}
    total_start = time.time()

    for name, script, db_name in OPTIMIZERS:
        cleanup_artifacts()
        time.sleep(0.5)  # Brief pause between tests

        try:
            success = run_optimizer_test(name, script, db_name)
            results[name] = success
        except Exception as e:
            print(f"❌ FAILED - Exception: {e}")
            results[name] = False

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    passed = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    print(f"\n✅ PASSED ({len(passed)}/{len(results)}):")
    for name in passed:
        print(f"   • {name}")

    if failed:
        print(f"\n❌ FAILED ({len(failed)}/{len(results)}):")
        for name in failed:
            print(f"   • {name}")

    print(f"\nTotal test time: {total_elapsed:.1f}s")

    if len(passed) == len(results):
        print("\n" + "="*70)
        print("🎉 SUCCESS! ALL OPTIMIZERS VALIDATED!")
        print("The hyperparameter tuning framework is production-ready.")
        print("="*70)
        return 0
    else:
        print(f"\n⚠️  {len(failed)} optimizer(s) need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
