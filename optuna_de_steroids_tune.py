# ------------------------------------------------------------------------------+
# Optuna-based Hyperparameter Tuning for DE on Steroids Optimizer
#
# This script performs automated hyperparameter search for the DE on Steroids
# algorithm using the Optuna optimization framework.
#
# DE on Steroids supports 6 mutation strategies:
#   - rand1bin, rand2bin, best1bin, best2bin, currenttobest1bin, randtobest1bin
#
# Usage:
#   uv run python optuna_de_steroids_tune.py \
#     --model_path models/tsp_100_model.pt \
#     --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
#     --n_trials 50 \
#     --tune_n_instances 3
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
    logger = logging.getLogger('optuna_de_steroids_tuning')
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


def create_config_for_trial(base_config, strategy: str, mutate: float, recombination: float):
    """
    Create a configuration object for a trial with specific DE on Steroids parameters.

    Args:
        base_config: Base configuration with model path, device, etc.
        strategy: Mutation strategy (e.g., 'rand1bin', 'best1bin')
        mutate: Mutation factor (F)
        recombination: Crossover probability (CR)

    Returns:
        Config object with DE on Steroids parameters set
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
    config.seed = base_config.seed
    config.optimizer = 'de_on_steroids'
    config.problem = base_config.problem
    config.problem_size = base_config.problem_size

    # Set DE on Steroids-specific parameters (these are what we're tuning)
    config.steroids_strategy = strategy
    config.steroids_mutate = mutate
    config.steroids_recombine = recombination

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
    strategy = trial.suggest_categorical('strategy',
        ['rand1bin', 'rand2bin', 'best1bin', 'best2bin', 'currenttobest1bin', 'randtobest1bin'])
    mutate = trial.suggest_float('mutate', 0.1, 1.0, log=True)
    recombination = trial.suggest_float('recombination', 0.5, 0.95)

    # Log trial start
    logger.info(f"=" * 80)
    logger.info(f"Trial {trial.number + 1} started")
    logger.info(f"  Parameters: strategy={strategy}, mutate={mutate:.6f}, recombination={recombination:.6f}")

    # Create config for this trial
    config = create_config_for_trial(base_config, strategy, mutate, recombination)

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
        description="Optuna-based hyperparameter tuning for DE on Steroids optimizer"
    )

    # Required parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--instances_path', type=str, required=True,
                       help='Path to instances pickle file (with optimal solutions if available)')

    # Tuning parameters
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials to run (default: 50)')
    parser.add_argument('--tune_n_instances', type=int, default=3,
                       help='Number of instances to use for tuning (default: 3)')
    parser.add_argument('--batch_size', type=int, default=600,
                       help='Population size for DE (default: 600)')

    # Optuna study parameters
    parser.add_argument('--study_name', type=str, default='de_steroids_tuning',
                       help='Name for Optuna study (default: de_steroids_tuning)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_de_steroids.db',
                       help='Optuna storage backend (default: sqlite:///optuna_de_steroids.db)')
    parser.add_argument('--load_if_exists', action='store_true',
                       help='Load existing study if it exists (default: False, creates new study)')

    # Search parameters
    parser.add_argument('--search_iterations', type=int, default=300,
                       help='Maximum iterations per instance (default: 300)')
    parser.add_argument('--search_timelimit', type=int, default=75,
                       help='Time limit in seconds per instance (default: 75)')
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
    logger.info("OPTUNA HYPERPARAMETER TUNING FOR DE ON STEROIDS")
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
    base_config.seed = args.seed

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
    logger.info(f"  Instances for tuning: {args.tune_n_instances}")
    logger.info(f"  Population size: {args.batch_size}")
    logger.info(f"  Study name: {args.study_name}")
    logger.info(f"  Storage: {args.storage}")
    logger.info("")
    logger.info("HYPERPARAMETER SEARCH SPACE:")
    logger.info(f"  strategy: Categorical['rand1bin', 'rand2bin', 'best1bin', 'best2bin', 'currenttobest1bin', 'randtobest1bin']")
    logger.info(f"  mutate (F): LogUniform[0.1, 1.0]")
    logger.info(f"  recombination (CR): Uniform[0.5, 0.95]")
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

    # Enqueue default parameters as first trial
    study.enqueue_trial({'strategy': 'rand1bin', 'mutate': 0.232165, 'recombination': 0.875693})
    logger.info("Default parameters enqueued as first trial (strategy='rand1bin', mutate=0.232165, recombination=0.875693)")
    logger.info("")

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
    logger.info(f"  strategy: {study.best_params['strategy']}")
    logger.info(f"  mutate (F): {study.best_params['mutate']:.6f}")
    logger.info(f"  recombination (CR): {study.best_params['recombination']:.6f}")
    logger.info(f"  Best mean gap: {study.best_value:.4f}%")
    logger.info("")

    # Save best parameters to JSON
    best_params_file = os.path.join(output_dir, 'best_de_steroids_params.json')
    best_params_data = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'study_name': args.study_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'instances_path': args.instances_path,
        'model_path': args.model_path,
        'tune_n_instances': args.tune_n_instances,
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
        logger.info("Visualizations skipped (requires Chrome/Chromium for image export)")

    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
