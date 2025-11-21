#!/usr/bin/env python3
# ------------------------------------------------------------------------------+
# Master Orchestration Script for Hyperparameter Tuning
#
# This script runs hyperparameter tuning for all 10 optimizers in sequence.
#
# Usage:
#   uv run python run_all_tuning.py \
#     --model_path models/tsp_100_model.pt \
#     --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
#     --n_trials 50 \
#     --tune_n_instances 3 \
#     --batch_size 600 \
#     --time_limit 75 \
#     --seed 1234
#
# Options:
#   --optimizers: Comma-separated list of optimizers to tune (default: all)
#                 Example: --optimizers de,ode,cmaes
#   --skip: Comma-separated list of optimizers to skip
#           Example: --skip jade,shade
#   --resume: Resume existing studies if they exist
#
# Optimizers:
#   1. de         - Standard DE
#   2. ode        - EvoX ODE (Oppositional DE)
#   3. code       - EvoX CoDE (Composite DE)
#   4. shade      - EvoX SHADE
#   5. sade       - EvoX SaDE
#   6. cmaes      - Standard CMA-ES
#   7. ipop       - IPOP-CMA-ES
#   8. bipop      - BIPOP-CMA-ES
#   9. scipy_de   - SciPy DE
#   10. jade      - EvoX JADE
#
# Outputs:
#   - Individual tuning results for each optimizer
#   - Master summary: tuning_summary.json
#   - Master log: logs/master_tuning_TIMESTAMP.log
# ------------------------------------------------------------------------------+

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Define all available optimizers with their tuning scripts
OPTIMIZERS = {
    'de': {
        'script': 'optuna_de_tune.py',
        'name': 'Standard DE',
        'params': ['mutate', 'recombination']
    },
    'ode': {
        'script': 'optuna_ode_tune.py',
        'name': 'EvoX ODE',
        'params': ['base_vector', 'num_difference_vectors', 'differential_weight', 'cross_probability']
    },
    'code': {
        'script': 'optuna_code_tune.py',
        'name': 'EvoX CoDE',
        'params': ['diff_padding_num', 'replace']
    },
    'shade': {
        'script': 'optuna_shade_tune.py',
        'name': 'EvoX SHADE',
        'params': ['diff_padding_num']
    },
    'sade': {
        'script': 'optuna_sade_tune.py',
        'name': 'EvoX SaDE',
        'params': ['diff_padding_num', 'LP']
    },
    'cmaes': {
        'script': 'optuna_cmaes_tune.py',
        'name': 'Standard CMA-ES',
        'params': ['sigma0', 'CMA_rankmu', 'CMA_rankone']
    },
    'ipop': {
        'script': 'optuna_ipop_tune.py',
        'name': 'IPOP-CMA-ES',
        'params': ['sigma0', 'CMA_rankmu', 'CMA_rankone']
    },
    'bipop': {
        'script': 'optuna_bipop_tune.py',
        'name': 'BIPOP-CMA-ES',
        'params': ['sigma0', 'CMA_rankmu', 'CMA_rankone']
    },
    'scipy_de': {
        'script': 'optuna_scipy_de_tune.py',
        'name': 'SciPy DE',
        'params': ['strategy', 'mutation', 'recombination_scipy', 'updating']
    },
    'jade': {
        'script': 'optuna_jade_tune.py',
        'name': 'EvoX JADE',
        'params': ['jade_c', 'jade_num_diff_vectors']
    }
}


def setup_logging(log_dir="logs"):
    """Setup logging configuration for master script."""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"master_tuning_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def run_optimizer_tuning(optimizer_key, args):
    """
    Run hyperparameter tuning for a single optimizer.

    Args:
        optimizer_key: Key identifying the optimizer (e.g., 'de', 'cmaes')
        args: Namespace with configuration

    Returns:
        dict with results: {
            'optimizer': str,
            'success': bool,
            'duration': float,
            'error': str (if failed)
        }
    """
    optimizer_info = OPTIMIZERS[optimizer_key]
    script_path = optimizer_info['script']
    optimizer_name = optimizer_info['name']

    logging.info("=" * 80)
    logging.info(f"STARTING TUNING: {optimizer_name} ({optimizer_key})")
    logging.info("=" * 80)
    logging.info(f"Script: {script_path}")
    logging.info(f"Tunable parameters: {', '.join(optimizer_info['params'])}")

    # Build command
    cmd = [
        'uv', 'run', 'python', script_path,
        '--model_path', args.model_path,
        '--instances_path', args.instances_path,
        '--n_trials', str(args.n_trials),
        '--tune_n_instances', str(args.tune_n_instances),
        '--batch_size', str(args.batch_size),
        '--search_timelimit', str(int(args.time_limit)),
        '--seed', str(args.seed),
        '--load_if_exists'  # Always allow resuming/loading existing studies
    ]

    logging.info(f"Command: {' '.join(cmd)}")
    logging.info("-" * 80)

    # Run tuning
    start_time = time.time()
    result = {
        'optimizer': optimizer_name,
        'key': optimizer_key,
        'script': script_path,
        'start_time': datetime.now().isoformat()
    }

    try:
        # Run subprocess and capture output
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output go to console
            text=True
        )

        duration = time.time() - start_time
        result['success'] = True
        result['duration'] = duration
        result['end_time'] = datetime.now().isoformat()

        logging.info("-" * 80)
        logging.info(f"✓ COMPLETED: {optimizer_name}")
        logging.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logging.info("=" * 80)

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        result['success'] = False
        result['duration'] = duration
        result['error'] = f"Process returned non-zero exit code: {e.returncode}"
        result['end_time'] = datetime.now().isoformat()

        logging.error("-" * 80)
        logging.error(f"✗ FAILED: {optimizer_name}")
        logging.error(f"  Error: {result['error']}")
        logging.error(f"  Duration before failure: {duration:.1f} seconds")
        logging.error("=" * 80)

    except Exception as e:
        duration = time.time() - start_time
        result['success'] = False
        result['duration'] = duration
        result['error'] = str(e)
        result['end_time'] = datetime.now().isoformat()

        logging.error("-" * 80)
        logging.error(f"✗ FAILED: {optimizer_name}")
        logging.error(f"  Exception: {result['error']}")
        logging.error(f"  Duration before failure: {duration:.1f} seconds")
        logging.error("=" * 80)

    return result


def save_summary(results, args, log_file, start_time):
    """Save master tuning summary to JSON."""
    total_duration = time.time() - start_time
    total_optimizers = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_optimizers - successful

    summary = {
        "total_optimizers": total_optimizers,
        "successful": successful,
        "failed": failed,
        "total_duration_seconds": total_duration,
        "total_duration_minutes": total_duration / 60,
        "total_duration_hours": total_duration / 3600,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "configuration": {
            "model_path": args.model_path,
            "instances_path": args.instances_path,
            "n_trials": args.n_trials,
            "tune_n_instances": args.tune_n_instances,
            "batch_size": args.batch_size,
            "time_limit": args.time_limit,
            "seed": args.seed,
            "resume": args.resume
        },
        "log_file": str(log_file),
        "results": results
    }

    output_file = "tuning_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info("=" * 80)
    logging.info("MASTER TUNING SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total optimizers: {total_optimizers}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Total duration: {total_duration/3600:.2f} hours ({total_duration/60:.1f} minutes)")
    logging.info(f"Summary saved to: {output_file}")
    logging.info("=" * 80)

    if successful > 0:
        logging.info("\nSuccessful tunings:")
        for r in results:
            if r['success']:
                logging.info(f"  ✓ {r['optimizer']} ({r['key']}) - {r['duration']/60:.1f} min")

    if failed > 0:
        logging.info("\nFailed tunings:")
        for r in results:
            if not r['success']:
                logging.info(f"  ✗ {r['optimizer']} ({r['key']}) - {r.get('error', 'Unknown error')}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Master script for running hyperparameter tuning on all optimizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune all optimizers with default settings
  uv run python run_all_tuning.py \\
    --model_path models/tsp_100_model.pt \\
    --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl

  # Tune only specific optimizers
  uv run python run_all_tuning.py \\
    --model_path models/tsp_100_model.pt \\
    --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \\
    --optimizers de,cmaes,ipop

  # Skip certain optimizers
  uv run python run_all_tuning.py \\
    --model_path models/tsp_100_model.pt \\
    --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \\
    --skip shade,sade

  # Resume existing studies with more trials
  uv run python run_all_tuning.py \\
    --model_path models/tsp_100_model.pt \\
    --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \\
    --n_trials 100 \\
    --resume

Available optimizers: de, ode, code, shade, sade, jade, cmaes, ipop, bipop, scipy_de
        """
    )

    # Configuration parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--instances_path', type=str, required=True,
                       help='Path to problem instances (.pkl file)')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials per optimizer (default: 50)')
    parser.add_argument('--tune_n_instances', type=int, default=3,
                       help='Number of instances for tuning (default: 3)')
    parser.add_argument('--batch_size', type=int, default=600,
                       help='Population size (default: 600)')
    parser.add_argument('--time_limit', type=float, default=75.0,
                       help='Time limit per instance in seconds (default: 75)')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed (default: 1234)')

    # Optimizer selection
    parser.add_argument('--optimizers', type=str, default=None,
                       help='Comma-separated list of optimizers to tune (default: all)')
    parser.add_argument('--skip', type=str, default=None,
                       help='Comma-separated list of optimizers to skip')

    # Execution options
    parser.add_argument('--resume', action='store_true',
                       help='Resume existing studies if they exist')

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging()

    # Determine which optimizers to run
    if args.optimizers:
        selected_optimizers = [o.strip() for o in args.optimizers.split(',')]
        # Validate optimizer names
        invalid = [o for o in selected_optimizers if o not in OPTIMIZERS]
        if invalid:
            logging.error(f"Invalid optimizer names: {', '.join(invalid)}")
            logging.error(f"Available optimizers: {', '.join(OPTIMIZERS.keys())}")
            sys.exit(1)
    else:
        selected_optimizers = list(OPTIMIZERS.keys())

    # Remove skipped optimizers
    if args.skip:
        skip_list = [o.strip() for o in args.skip.split(',')]
        selected_optimizers = [o for o in selected_optimizers if o not in skip_list]
        logging.info(f"Skipping optimizers: {', '.join(skip_list)}")

    # Print header
    logging.info("=" * 80)
    logging.info("MASTER HYPERPARAMETER TUNING ORCHESTRATION")
    logging.info("=" * 80)
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Instances: {args.instances_path}")
    logging.info(f"Trials per optimizer: {args.n_trials}")
    logging.info(f"Tuning instances: {args.tune_n_instances}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Time limit: {args.time_limit}s per instance")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Resume existing studies: {args.resume}")
    logging.info(f"Log file: {log_file}")
    logging.info("-" * 80)
    logging.info(f"Total optimizers to tune: {len(selected_optimizers)}")
    logging.info(f"Optimizers: {', '.join(selected_optimizers)}")
    logging.info(f"Estimated total time: {len(selected_optimizers) * 3.1:.1f} hours")
    logging.info("  (Assuming ~3.1 hours per optimizer with 50 trials)")
    logging.info("=" * 80)

    # Run tuning for each optimizer
    start_time = time.time()
    results = []

    for idx, optimizer_key in enumerate(selected_optimizers, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"OPTIMIZER {idx}/{len(selected_optimizers)}")
        logging.info(f"{'='*80}\n")

        result = run_optimizer_tuning(optimizer_key, args)
        results.append(result)

        # Print progress
        completed = idx
        remaining = len(selected_optimizers) - idx
        elapsed = time.time() - start_time
        avg_time_per_optimizer = elapsed / completed
        estimated_remaining = avg_time_per_optimizer * remaining

        logging.info(f"\nProgress: {completed}/{len(selected_optimizers)} optimizers completed")
        logging.info(f"Elapsed time: {elapsed/3600:.2f} hours")
        logging.info(f"Estimated remaining time: {estimated_remaining/3600:.2f} hours")
        logging.info(f"Estimated total time: {(elapsed + estimated_remaining)/3600:.2f} hours\n")

    # Save summary
    save_summary(results, args, log_file, start_time)

    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r['success'])
    if failed_count > 0:
        logging.warning(f"\n{failed_count} optimizer(s) failed. Check logs for details.")
        sys.exit(1)
    else:
        logging.info("\n✓ All optimizers tuned successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
