# ------------------------------------------------------------------------------+
# Optuna-based Hyperparameter Tuning for EvoX JADE Optimizer
#
# This script performs automated hyperparameter search for the JADE algorithm
# using the Optuna optimization framework.
#
# Usage:
#   uv run python optuna_jade_tune.py \
#     --model_path models/tsp_100_model.pt \
#     --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
#     --n_trials 100 \
#     --tune_n_instances 5
# ------------------------------------------------------------------------------+

import argparse
import datetime
import json
import logging
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import optuna
import torch
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

import train
import tsp
import cvrp
from search_control import solve_instance, decode
from utils import read_instance_pkl
from VAE_8 import VAE_8


def setup_logging(output_dir: str) -> Tuple[str, logging.Logger]:
    """
    Setup logging for the tuning process.

    Args:
        output_dir: Directory to save log files

    Returns:
        Tuple of (log_filename, logger)
    """
    now = datetime.datetime.now()
    timestamp = f"{now.year:04d}{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}{now.second:02d}"
    log_filename = f"optuna_tuning_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)

    # Create logger
    logger = logging.getLogger('optuna_jade_tuning')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler (for log file)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (for terminal output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return log_filename, logger


def create_config_for_trial(base_config, jade_c: float, jade_num_diff_vectors: int):
    """
    Create a configuration object for a trial with specific JADE parameters.

    Args:
        base_config: Base configuration with model path, device, etc.
        jade_c: JADE learning rate parameter
        jade_num_diff_vectors: Number of difference vectors

    Returns:
        Config object with JADE parameters set
    """
    # Create a simple config object (namespace)
    class Config:
        pass

    config = Config()

    # Copy base parameters
    config.device = base_config.device
    config.search_space_bound = base_config.search_space_bound
    config.search_space_size = base_config.search_space_size
    config.search_iterations = base_config.search_iterations
    config.search_timelimit = base_config.search_timelimit
    config.search_evaluations = base_config.search_evaluations
    config.optimizer = 'evox_jade'
    config.problem = base_config.problem
    config.problem_size = base_config.problem_size

    # Set JADE-specific parameters (these are what we're tuning)
    config.jade_c = jade_c
    config.jade_num_diff_vectors = jade_num_diff_vectors
    config.jade_mean = None  # Keep uniform initialization
    config.jade_stdev = None  # Keep uniform initialization

    # Other optimizer parameters (not used by JADE but required by interface)
    config.de_mutate = 0.3
    config.de_recombine = 0.95

    return config


def objective(trial: optuna.Trial, args) -> float:
    """
    Optuna objective function to minimize.

    Args:
        trial: Optuna trial object
        args: Tuple containing (model, base_config, instances, solutions, cost_fn,
                                batch_size, tune_n_instances, logger)

    Returns:
        Mean optimality gap (%) across tuning instances
    """
    model, base_config, instances, solutions, cost_fn, batch_size, tune_n_instances, logger = args

    # Suggest hyperparameters
    jade_c = trial.suggest_float('jade_c', 0.01, 0.5, log=True)
    jade_num_diff_vectors = trial.suggest_categorical('jade_num_diff_vectors', [1, 2])

    # Log trial start
    logger.info(f"=" * 80)
    logger.info(f"Trial {trial.number + 1} started")
    logger.info(f"  Parameters: jade_c={jade_c:.6f}, jade_num_diff_vectors={jade_num_diff_vectors}")

    # Create config for this trial
    config = create_config_for_trial(base_config, jade_c, jade_num_diff_vectors)

    # Determine which instances to use for tuning
    if tune_n_instances is not None and tune_n_instances < len(instances):
        tuning_instances = instances[:tune_n_instances]
        tuning_solutions = solutions[:tune_n_instances] if solutions else None
        logger.info(f"  Using first {tune_n_instances} instances for tuning")
    else:
        tuning_instances = instances
        tuning_solutions = solutions
        logger.info(f"  Using all {len(instances)} instances for tuning")

    # Evaluate on tuning instances
    gaps = []
    trial_start_time = time.time()

    for i, instance in enumerate(tuning_instances):
        instance_start = time.time()

        # Solve instance with current hyperparameters
        objective_value, solution, convergence_history, time_history, timing_breakdown = solve_instance(
            model, instance, config, cost_fn, batch_size
        )

        # Calculate gap if optimal solution is available
        if tuning_solutions:
            optimal_value = cost_fn(
                torch.Tensor(instance).unsqueeze(0),
                torch.Tensor(tuning_solutions[i]).long().unsqueeze(0)
            ).item()
            gap = (objective_value / optimal_value - 1) * 100
            gaps.append(gap)

            instance_time = time.time() - instance_start
            logger.info(f"  Instance {i}: gap={gap:.2f}%, cost={objective_value:.4f}, "
                       f"optimal={optimal_value:.4f}, time={instance_time:.1f}s")
        else:
            # If no optimal solution, use raw cost (to be minimized)
            gaps.append(objective_value)
            instance_time = time.time() - instance_start
            logger.info(f"  Instance {i}: cost={objective_value:.4f}, time={instance_time:.1f}s")

    # Calculate mean gap
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    trial_time = time.time() - trial_start_time

    logger.info(f"  Trial {trial.number + 1} completed:")
    logger.info(f"    Mean gap: {mean_gap:.4f}%")
    logger.info(f"    Std gap: {std_gap:.4f}%")
    logger.info(f"    Total time: {trial_time:.2f}s")
    logger.info(f"=" * 80)

    return mean_gap


def main():
    parser = argparse.ArgumentParser(
        description="Optuna-based hyperparameter tuning for EvoX JADE optimizer"
    )

    # Required parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--instances_path', type=str, required=True,
                       help='Path to instances pickle file (with optimal solutions if available)')

    # Tuning parameters
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of Optuna trials to run (default: 100)')
    parser.add_argument('--tune_n_instances', type=int, default=None,
                       help='Number of instances to use for tuning (default: None, uses all)')
    parser.add_argument('--batch_size', type=int, default=600,
                       help='Population size for JADE (default: 600)')

    # Optuna study parameters
    parser.add_argument('--study_name', type=str, default='jade_tuning',
                       help='Name for Optuna study (default: jade_tuning)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_jade.db',
                       help='Optuna storage backend (default: sqlite:///optuna_jade.db)')
    parser.add_argument('--load_if_exists', action='store_true',
                       help='Load existing study if it exists (default: False, creates new study)')

    # Search parameters
    parser.add_argument('--search_iterations', type=int, default=300,
                       help='Maximum iterations per instance (default: 300)')
    parser.add_argument('--search_timelimit', type=int, default=None,
                       help='Time limit in seconds per instance (default: None)')
    parser.add_argument('--search_evaluations', type=int, default=None,
                       help='Maximum evaluations per instance (default: None)')
    parser.add_argument('--search_space_size', type=int, default=100,
                       help='Dimensionality of search space (default: 100)')

    # Other parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu, default: cuda)')
    parser.add_argument('--problem', type=str, default=None,
                       help='Problem type (TSP or CVRP, default: auto-detect from model)')
    parser.add_argument('--problem_size', type=int, default=None,
                       help='Problem size (default: auto-detect from model)')
    parser.add_argument('--output_path', type=str, default='',
                       help='Output directory (default: current directory)')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility (default: 1234)')

    args = parser.parse_args()

    # Setup output directory
    if args.output_path == '':
        args.output_path = os.getcwd()

    now = datetime.datetime.now()
    run_id = f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    output_dir = os.path.join(
        args.output_path, 'runs',
        f"optuna_tune_{now.day}.{now.month}.{now.year}_{run_id}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_filename, logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("OPTUNA HYPERPARAMETER TUNING FOR EVOX JADE")
    logger.info("=" * 80)
    logger.info(f"Log file: {os.path.join(output_dir, log_filename)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    device = torch.device(args.device)
    model_data = torch.load(args.model_path, device, weights_only=False)

    # Create base config
    class BaseConfig:
        pass

    base_config = BaseConfig()
    base_config.device = device
    base_config.search_space_bound = model_data['Z_bound']
    base_config.search_space_size = args.search_space_size
    base_config.search_iterations = args.search_iterations
    base_config.search_timelimit = args.search_timelimit
    base_config.search_evaluations = args.search_evaluations
    base_config.problem = args.problem if args.problem else model_data['problem']
    base_config.problem_size = args.problem_size if args.problem_size else model_data['problem_size']
    base_config.instances_path = args.instances_path  # Add instances_path for read_instance_pkl

    logger.info(f"Problem: {base_config.problem}{base_config.problem_size}")
    logger.info(f"Search space bound: {base_config.search_space_bound}")
    logger.info(f"Search space size: {base_config.search_space_size}")
    logger.info(f"Search iterations: {base_config.search_iterations}")

    # Load model
    model = VAE_8(base_config).to(device)
    model.load_state_dict(model_data['parameters'])
    model.eval()
    logger.info("Model loaded successfully")

    # Load instances
    logger.info(f"Loading instances from: {args.instances_path}")
    instances, solutions = read_instance_pkl(base_config)

    if instances is None or len(instances) == 0:
        # Try reading with custom config
        class InstanceConfig:
            instances_path = args.instances_path
            problem = base_config.problem

        instances, solutions = read_instance_pkl(InstanceConfig())

    logger.info(f"Loaded {len(instances)} instances")
    if solutions:
        logger.info(f"Optimal solutions available: Yes")
    else:
        logger.info(f"Optimal solutions available: No (will minimize raw cost)")

    # Setup cost function
    if base_config.problem == "TSP":
        cost_fn = tsp.tours_length
    elif base_config.problem == "CVRP":
        cost_fn = cvrp.tours_length
        if solutions:
            solutions = [cvrp.solution_to_single_tour(solution) for solution in solutions]
    else:
        raise ValueError(f"Unknown problem type: {base_config.problem}")

    # Log tuning configuration
    logger.info("")
    logger.info("TUNING CONFIGURATION:")
    logger.info(f"  Number of trials: {args.n_trials}")
    logger.info(f"  Instances for tuning: {args.tune_n_instances if args.tune_n_instances else 'all (' + str(len(instances)) + ')'}")
    logger.info(f"  Population size: {args.batch_size}")
    logger.info(f"  Study name: {args.study_name}")
    logger.info(f"  Storage: {args.storage}")
    logger.info("")
    logger.info("HYPERPARAMETER SEARCH SPACE:")
    logger.info(f"  jade_c: LogUniform[0.01, 0.5]")
    logger.info(f"  jade_num_diff_vectors: Categorical[1, 2]")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',
        load_if_exists=args.load_if_exists,
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )

    # Prepare arguments for objective function
    objective_args = (
        model, base_config, instances, solutions, cost_fn,
        args.batch_size, args.tune_n_instances, logger
    )

    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    logger.info("")

    study_start_time = time.time()

    try:
        study.optimize(
            lambda trial: objective(trial, objective_args),
            n_trials=args.n_trials,
            show_progress_bar=False  # We have custom logging
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    study_time = time.time() - study_start_time

    # Log results
    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total time: {study_time:.2f}s ({study_time/60:.1f} minutes)")
    logger.info(f"Trials completed: {len(study.trials)}")
    logger.info("")
    logger.info("BEST PARAMETERS:")
    logger.info(f"  jade_c: {study.best_params['jade_c']:.6f}")
    logger.info(f"  jade_num_diff_vectors: {study.best_params['jade_num_diff_vectors']}")
    logger.info(f"  Best mean gap: {study.best_value:.4f}%")
    logger.info("")

    # Save best parameters to JSON
    best_params_file = os.path.join(output_dir, 'best_jade_params.json')
    best_params_data = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'study_name': args.study_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'instances_path': args.instances_path,
        'model_path': args.model_path,
        'tune_n_instances': args.tune_n_instances if args.tune_n_instances else len(instances),
        'batch_size': args.batch_size
    }

    with open(best_params_file, 'w') as f:
        json.dump(best_params_data, f, indent=2)

    logger.info(f"Best parameters saved to: {best_params_file}")
    logger.info("")

    # Generate Optuna visualizations
    logger.info("Generating optimization visualizations...")

    try:
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(output_dir, 'optimization_history.png'))
        logger.info("  - Optimization history plot saved")

        # Parameter importances (only if enough trials)
        if len(study.trials) >= 10:
            fig2 = plot_param_importances(study)
            fig2.write_image(os.path.join(output_dir, 'param_importances.png'))
            logger.info("  - Parameter importances plot saved")

        # Parallel coordinate plot
        fig3 = plot_parallel_coordinate(study)
        fig3.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
        logger.info("  - Parallel coordinate plot saved")

    except Exception as e:
        logger.warning(f"Could not generate some visualizations: {e}")
        logger.warning("Install kaleido for static image export: uv add kaleido")

    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("")
    logger.info("To use the optimized parameters, run:")
    logger.info(f"  uv run python search.py \\")
    logger.info(f"    --optimizer evox_jade \\")
    logger.info(f"    --jade_c {study.best_params['jade_c']:.6f} \\")
    logger.info(f"    --jade_num_diff_vectors {study.best_params['jade_num_diff_vectors']} \\")
    logger.info(f"    --model_path {args.model_path} \\")
    logger.info(f"    --instances_path {args.instances_path} \\")
    logger.info(f"    --batch_sizes {args.batch_size} \\")
    logger.info(f"    --save_plots")
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
