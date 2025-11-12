#!/usr/bin/env python3
"""
Individual Optimizer Tests

Tests each optimizer separately with minimal configuration to validate
that modified optimizer files accept new tunable parameters correctly.

Usage:
    uv run python test_optimizers.py [optimizer_key]

    If optimizer_key is not provided, tests all optimizers.
"""

import sys
import subprocess
import time
import argparse
from pathlib import Path

# Test configuration (minimal)
TEST_CONFIG = {
    'model_path': 'models/tsp_100_model_63108.pt',
    'instances_path': 'instances/tsp/test/tsp100_1inst_w_optimal.pkl',
    'n_trials': 3,
    'tune_n_instances': 1,
    'batch_size': 200,
    'search_timelimit': 20,
    'seed': 1234
}

# All optimizers
OPTIMIZERS = {
    'de': {'script': 'optuna_de_tune.py', 'name': 'Standard DE'},
    'ode': {'script': 'optuna_ode_tune.py', 'name': 'EvoX ODE'},
    'code': {'script': 'optuna_code_tune.py', 'name': 'EvoX CoDE'},
    'shade': {'script': 'optuna_shade_tune.py', 'name': 'EvoX SHADE'},
    'sade': {'script': 'optuna_sade_tune.py', 'name': 'EvoX SaDE'},
    'cmaes': {'script': 'optuna_cmaes_tune.py', 'name': 'Standard CMA-ES'},
    'ipop': {'script': 'optuna_ipop_tune.py', 'name': 'IPOP-CMA-ES'},
    'bipop': {'script': 'optuna_bipop_tune.py', 'name': 'BIPOP-CMA-ES'},
    'scipy_de': {'script': 'optuna_scipy_de_tune.py', 'name': 'SciPy DE'}
}


def test_optimizer(optimizer_key):
    """Test a single optimizer."""
    if optimizer_key not in OPTIMIZERS:
        print(f"Error: Unknown optimizer '{optimizer_key}'")
        print(f"Available: {', '.join(OPTIMIZERS.keys())}")
        return False

    opt = OPTIMIZERS[optimizer_key]
    print(f"\n{'='*80}")
    print(f"Testing: {opt['name']} ({optimizer_key})")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        'uv', 'run', 'python', opt['script'],
        '--model_path', TEST_CONFIG['model_path'],
        '--instances_path', TEST_CONFIG['instances_path'],
        '--n_trials', str(TEST_CONFIG['n_trials']),
        '--tune_n_instances', str(TEST_CONFIG['tune_n_instances']),
        '--batch_size', str(TEST_CONFIG['batch_size']),
        '--search_timelimit', str(TEST_CONFIG['search_timelimit']),
        '--seed', str(TEST_CONFIG['seed'])
    ]

    print(f"Command: {' '.join(cmd)}\n")

    # Run test
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output directly
            timeout=600  # 10 minute timeout
        )

        duration = time.time() - start_time
        success = (result.returncode == 0)

        print(f"\n{'-'*80}")
        if success:
            print(f"✓ Test PASSED in {duration:.1f}s")
        else:
            print(f"✗ Test FAILED with exit code {result.returncode} after {duration:.1f}s")
        print(f"{'-'*80}\n")

        return success

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"\n{'-'*80}")
        print(f"✗ Test TIMEOUT after {duration:.1f}s")
        print(f"{'-'*80}\n")
        return False

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n{'-'*80}")
        print(f"✗ Test EXCEPTION: {e}")
        print(f"{'-'*80}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test individual optimizers')
    parser.add_argument('optimizer', nargs='?', default=None,
                       help='Optimizer to test (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available optimizers')
    args = parser.parse_args()

    if args.list:
        print("Available optimizers:")
        for key, opt in OPTIMIZERS.items():
            print(f"  {key:12} - {opt['name']}")
        return 0

    # Determine which optimizers to test
    if args.optimizer:
        optimizers_to_test = [args.optimizer]
    else:
        optimizers_to_test = list(OPTIMIZERS.keys())

    print("="*80)
    print("INDIVIDUAL OPTIMIZER TESTS")
    print("="*80)
    print(f"Testing {len(optimizers_to_test)} optimizer(s): {', '.join(optimizers_to_test)}")
    print(f"Configuration: {TEST_CONFIG['n_trials']} trials, {TEST_CONFIG['batch_size']} batch size")

    # Run tests
    results = {}
    start_time = time.time()

    for opt_key in optimizers_to_test:
        results[opt_key] = test_optimizer(opt_key)

    total_time = time.time() - start_time

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for success in results.values() if success)
    failed = len(results) - passed

    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if passed > 0:
        print("\n✓ Passed:")
        for key, success in results.items():
            if success:
                print(f"  {key}: {OPTIMIZERS[key]['name']}")

    if failed > 0:
        print("\n✗ Failed:")
        for key, success in results.items():
            if not success:
                print(f"  {key}: {OPTIMIZERS[key]['name']}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
