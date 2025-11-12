#!/usr/bin/env python3
"""
Smoke Test for Hyperparameter Tuning Framework

This script runs ultra-fast smoke tests to validate that the tuning framework
is working correctly. It tests a representative sample of optimizers with
minimal configuration.

Usage:
    uv run python test_tuning_smoke.py
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Test configuration (ultra-fast)
TEST_CONFIG = {
    'model_path': 'models/tsp_20_model_74446.pt',
    'instances_path': 'instances/tsp/test/tsp100_1inst_w_optimal.pkl',
    'n_trials': 2,
    'tune_n_instances': 1,
    'batch_size': 100,
    'search_timelimit': 10,
    'seed': 1234
}

# Optimizers to test (representative sample)
TEST_OPTIMIZERS = [
    {'key': 'de', 'script': 'optuna_de_tune.py', 'name': 'Standard DE'},
    {'key': 'cmaes', 'script': 'optuna_cmaes_tune.py', 'name': 'CMA-ES'},
    {'key': 'ode', 'script': 'optuna_ode_tune.py', 'name': 'EvoX ODE'}
]


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"{text}")
    print(f"{'='*80}\n")


def print_success(text):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text):
    """Print error message in red."""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_info(text):
    """Print info message in blue."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.NC}")


def check_prerequisites():
    """Check that required files exist."""
    print_header("CHECKING PREREQUISITES")

    all_good = True

    # Check model exists
    if not os.path.exists(TEST_CONFIG['model_path']):
        print_error(f"Model not found: {TEST_CONFIG['model_path']}")
        all_good = False
    else:
        print_success(f"Model found: {TEST_CONFIG['model_path']}")

    # Check instances exist
    if not os.path.exists(TEST_CONFIG['instances_path']):
        print_error(f"Instances not found: {TEST_CONFIG['instances_path']}")
        all_good = False
    else:
        print_success(f"Instances found: {TEST_CONFIG['instances_path']}")

    # Check tuning scripts exist
    for opt in TEST_OPTIMIZERS:
        if not os.path.exists(opt['script']):
            print_error(f"Script not found: {opt['script']}")
            all_good = False
        else:
            print_success(f"Script found: {opt['script']}")

    return all_good


def test_imports():
    """Test that all modules can be imported."""
    print_header("TESTING IMPORTS")

    modules_to_test = [
        'optuna_de_tune',
        'optuna_cmaes_tune',
        'optuna_ode_tune',
        'run_all_tuning'
    ]

    all_good = True
    for module in modules_to_test:
        try:
            result = subprocess.run(
                ['python', '-c', f'import {module}'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print_success(f"Import {module}")
            else:
                print_error(f"Import {module} failed")
                print(f"  Error: {result.stderr}")
                all_good = False
        except Exception as e:
            print_error(f"Import {module} exception: {e}")
            all_good = False

    return all_good


def cleanup_test_artifacts(optimizer_key):
    """Remove test artifacts for an optimizer."""
    artifacts = [
        f'best_{optimizer_key}_params.json',
        f'optuna_{optimizer_key}_tuning.db',
        f'optuna_{optimizer_key}_tuning.db-shm',
        f'optuna_{optimizer_key}_tuning.db-wal',
        f'{optimizer_key}_optimization_history.png',
        f'{optimizer_key}_param_importances.png',
        f'{optimizer_key}_parallel_coordinate.png'
    ]

    for artifact in artifacts:
        if os.path.exists(artifact):
            try:
                os.remove(artifact)
            except:
                pass


def run_optimizer_test(optimizer):
    """
    Run a single optimizer tuning test.

    Returns:
        dict with test results
    """
    print_header(f"TESTING: {optimizer['name']} ({optimizer['key']})")

    # Cleanup previous artifacts
    cleanup_test_artifacts(optimizer['key'])

    # Build command
    cmd = [
        'uv', 'run', 'python', optimizer['script'],
        '--model_path', TEST_CONFIG['model_path'],
        '--instances_path', TEST_CONFIG['instances_path'],
        '--n_trials', str(TEST_CONFIG['n_trials']),
        '--tune_n_instances', str(TEST_CONFIG['tune_n_instances']),
        '--batch_size', str(TEST_CONFIG['batch_size']),
        '--search_timelimit', str(TEST_CONFIG['search_timelimit']),
        '--seed', str(TEST_CONFIG['seed'])
    ]

    print_info(f"Command: {' '.join(cmd)}")
    print()

    result = {
        'optimizer': optimizer['name'],
        'key': optimizer['key'],
        'success': False,
        'error': None,
        'duration': 0,
        'checks': {}
    }

    # Run test
    start_time = time.time()
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        result['duration'] = time.time() - start_time
        result['success'] = (process.returncode == 0)

        if not result['success']:
            result['error'] = f"Process returned exit code {process.returncode}"
            print_error(f"Tuning failed with exit code {process.returncode}")
            print("STDERR:")
            print(process.stderr[-1000:])  # Last 1000 chars
        else:
            print_success(f"Tuning completed in {result['duration']:.1f}s")

    except subprocess.TimeoutExpired:
        result['duration'] = time.time() - start_time
        result['error'] = "Test timed out"
        print_error("Test timed out")
    except Exception as e:
        result['duration'] = time.time() - start_time
        result['error'] = str(e)
        print_error(f"Exception: {e}")

    # Validate outputs (only if test succeeded)
    if result['success']:
        print()
        print_info("Validating outputs...")

        # Check database (corrected naming pattern)
        db_file = f"optuna_{optimizer['key']}.db"
        result['checks']['database'] = os.path.exists(db_file)
        if result['checks']['database']:
            print_success(f"Database created: {db_file}")
        else:
            print_error(f"Database missing: {db_file}")

        # Check JSON output in runs directory (most recent run)
        runs_dir = Path('runs')
        json_found = False
        if runs_dir.exists():
            run_dirs = sorted([d for d in runs_dir.glob('optuna_tune_*') if d.is_dir()],
                            key=lambda x: x.stat().st_mtime, reverse=True)
            for run_dir in run_dirs[:3]:  # Check last 3 runs
                json_file = run_dir / f"best_{optimizer['key']}_params.json"
                if json_file.exists():
                    json_found = True
                    result['checks']['json'] = True
                    print_success(f"JSON created: {json_file}")

                    # Validate JSON content
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)

                        required_keys = ['best_params', 'best_value', 'n_trials']
                        has_required = all(k in data for k in required_keys)
                        result['checks']['json_valid'] = has_required

                        if has_required:
                            print_success(f"  JSON valid: {data['n_trials']} trials, best gap: {data['best_value']:.2f}%")
                        else:
                            print_error(f"  JSON missing required keys")
                    except Exception as e:
                        result['checks']['json_valid'] = False
                        print_error(f"  JSON parsing failed: {e}")
                    break

        if not json_found:
            result['checks']['json'] = False
            print_error(f"JSON missing in runs directory")

        # Check log file in runs directory
        log_found = False
        if runs_dir.exists():
            for run_dir in run_dirs[:3]:  # Check last 3 runs
                log_files = list(run_dir.glob('optuna_tuning_*.log'))
                if log_files:
                    log_found = True
                    result['checks']['log'] = True
                    print_success(f"Log file created: {run_dir.name}/{log_files[0].name}")
                    break

        if not log_found:
            result['checks']['log'] = False
            print_error("Log file missing in runs directory")

    return result


def run_all_tests():
    """Run all smoke tests."""
    print_header("HYPERPARAMETER TUNING FRAMEWORK - SMOKE TEST")
    print_info(f"Testing {len(TEST_OPTIMIZERS)} optimizers with minimal configuration")
    print_info(f"Configuration: {TEST_CONFIG['n_trials']} trials, {TEST_CONFIG['tune_n_instances']} instance(s), {TEST_CONFIG['batch_size']} batch size")

    # Check prerequisites
    if not check_prerequisites():
        print_error("\nPrerequisite checks failed. Aborting tests.")
        return False

    # Test imports
    if not test_imports():
        print_error("\nImport tests failed. Aborting tests.")
        return False

    # Run optimizer tests
    results = []
    start_time = time.time()

    for optimizer in TEST_OPTIMIZERS:
        result = run_optimizer_test(optimizer)
        results.append(result)

    total_time = time.time() - start_time

    # Print summary
    print_header("TEST SUMMARY")

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f}s")
    print()

    # Detailed results
    if successful > 0:
        print("✓ Successful tests:")
        for r in results:
            if r['success']:
                checks_passed = sum(1 for v in r['checks'].values() if v)
                checks_total = len(r['checks'])
                print(f"  {Colors.GREEN}✓{Colors.NC} {r['optimizer']} ({r['key']}) - {r['duration']:.1f}s - {checks_passed}/{checks_total} checks passed")

    if failed > 0:
        print()
        print("✗ Failed tests:")
        for r in results:
            if not r['success']:
                print(f"  {Colors.RED}✗{Colors.NC} {r['optimizer']} ({r['key']}) - {r.get('error', 'Unknown error')}")

    # Overall result
    print()
    if failed == 0:
        print_success("ALL SMOKE TESTS PASSED!")
        print_info("Framework is working correctly. Ready for full tuning runs.")
        return True
    else:
        print_error(f"{failed} TEST(S) FAILED")
        print_info("Check errors above and fix issues before proceeding.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
