#!/usr/bin/env python3
# ------------------------------------------------------------------------------+
# Optuna-based Hyperparameter Tuning for BIPOP-CMA-ES Optimizer
#
# This script performs automated hyperparameter search for the BIPOP-CMA-ES
# algorithm using the Optuna optimization framework.
#
# Usage:
#   uv run python optuna_bipop_tune.py \
#     --model_path models/tsp_100_model.pt \
#     --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
#     --n_trials 50 \
#     --tune_n_instances 3 \
#     --batch_size 600 \
#     --time_limit 75 \
#     --seed 1234
#
# Tunable Parameters:
#   - sigma0: Initial step size (range: [0.15, 1.0])
#   - CMA_rankmu: Rank-mu update learning rate multiplier (range: [0.5, 2.0])
#   - CMA_rankone: Rank-one update learning rate multiplier (range: [0.5, 2.0])
#
# Fixed Parameters:
#   - restarts: 5
#   - incpopsize: 2.0
#   - use_lhs: True
#
# Outputs:
#   - best_bipop_params.json: Best hyperparameters found
#   - optuna_bipop_tuning.db: SQLite database with study results
#   - Visualization plots (if kaleido is installed)
# ------------------------------------------------------------------------------+

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import optuna

# Import optimizer
import bipop_cmaes

# Import cost function and search control
from tsp_solution_cost import tsp_solution_cost
from cvrp_solution_cost import cvrp_solution_cost


def setup_logging(log_dir="logs"):
    """Setup logging configuration."""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"optuna_bipop_tuning_{timestamp}.log"

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


def load_model(model_path, device):
    """Load the trained CVAE model."""
    logging.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def load_instances(instances_path):
    """Load problem instances from pickle file."""
    logging.info(f"Loading instances from {instances_path}")
    with open(instances_path, 'rb') as f:
        instances = pickle.load(f)
    return instances


def determine_problem_type(instances_path):
    """Determine if problem is TSP or CVRP based on path."""
    path_lower = str(instances_path).lower()
    if 'tsp' in path_lower:
        return 'tsp'
    elif 'cvrp' in path_lower:
        return 'cvrp'
    else:
        raise ValueError(f"Cannot determine problem type from path: {instances_path}")


def get_cost_function(problem_type):
    """Get the appropriate cost function for the problem type."""
    if problem_type == 'tsp':
        return tsp_solution_cost
    elif problem_type == 'cvrp':
        return cvrp_solution_cost
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def calculate_gap(objective_value, optimal_value):
    """Calculate optimality gap percentage."""
    if optimal_value == 0:
        return 0.0
    return ((objective_value - optimal_value) / optimal_value) * 100.0


def solve_instance(model, instance, sigma0, CMA_rankmu, CMA_rankone, cost_fn,
                   batch_size, time_limit, device):
    """
    Solve a single instance with given hyperparameters.

    Returns:
        objective_value: Best cost found
        solution: Best solution found
        elapsed_time: Time taken
    """
    # Extract instance data
    coordinates = instance['coordinates']

    # Handle both dict and direct array formats
    if isinstance(coordinates, dict):
        coordinates = coordinates['coordinates']

    # Move to device
    if isinstance(coordinates, np.ndarray):
        coordinates = torch.from_numpy(coordinates).float()
    coordinates = coordinates.to(device)

    # Get search space configuration from model
    search_space_bound = model.Z_bound
    search_space_size = model.latent_size

    # Define cost function for optimizer
    def cost_func(Z_batch):
        """Vectorized cost function for BIPOP-CMA-ES optimizer."""
        with torch.no_grad():
            solutions = model.decode(Z_batch)
            costs = cost_fn(solutions, coordinates)
        return costs.cpu().numpy()

    # Run BIPOP-CMA-ES optimizer
    best_cost, best_solution, _, elapsed_time = bipop_cmaes.minimize(
        cost_func=cost_func,
        args=(),
        search_space_bound=search_space_bound,
        search_space_size=search_space_size,
        popsize=batch_size,
        sigma0=sigma0,
        maxiter=None,
        maxtime=time_limit,
        maxevaluations=None,
        restarts=5,  # Fixed
        incpopsize=2.0,  # Fixed
        use_lhs=True,  # Fixed
        CMA_rankmu=CMA_rankmu,
        CMA_rankone=CMA_rankone,
        seed=1234
    )

    return best_cost, best_solution, elapsed_time


def objective(trial: optuna.Trial, args) -> float:
    """
    Optuna objective function for BIPOP-CMA-ES hyperparameter tuning.

    Args:
        trial: Optuna trial object
        args: Namespace with configuration

    Returns:
        Mean optimality gap (%) across tuning instances
    """
    # Suggest hyperparameters
    sigma0 = trial.suggest_float('sigma0', 0.15, 1.0)
    CMA_rankmu = trial.suggest_float('CMA_rankmu', 0.5, 2.0)
    CMA_rankone = trial.suggest_float('CMA_rankone', 0.5, 2.0)

    logging.info(f"Trial {trial.number} started")
    logging.info(f"  Parameters: sigma0={sigma0:.3f}, CMA_rankmu={CMA_rankmu:.3f}, "
                f"CMA_rankone={CMA_rankone:.3f}")
    logging.info(f"  Using first {args.tune_n_instances} instances for tuning")

    # Evaluate on tuning instances
    gaps = []
    for idx, instance in enumerate(args.tuning_instances):
        optimal_value = instance.get('optimal_tour_length', instance.get('optimal_value'))

        # Solve instance
        objective_value, solution, elapsed_time = solve_instance(
            model=args.model,
            instance=instance,
            sigma0=sigma0,
            CMA_rankmu=CMA_rankmu,
            CMA_rankone=CMA_rankone,
            cost_fn=args.cost_fn,
            batch_size=args.batch_size,
            time_limit=args.time_limit,
            device=args.device
        )

        # Calculate gap
        gap = calculate_gap(objective_value, optimal_value)
        gaps.append(gap)

        logging.info(f"  Instance {idx}: gap={gap:.2f}%, cost={objective_value:.1f}, "
                    f"optimal={optimal_value:.1f}, time={elapsed_time:.1f}s")

    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    logging.info(f"  Trial {trial.number} completed:")
    logging.info(f"    Mean gap: {mean_gap:.2f}%")
    logging.info(f"    Std gap: {std_gap:.2f}%")
    logging.info("=" * 80)

    return mean_gap


def save_results(study, args, log_file):
    """Save tuning results to JSON file."""
    best_params = study.best_params
    best_value = study.best_value

    results = {
        "best_params": best_params,
        "best_value": float(best_value),
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "timestamp": datetime.now().isoformat(),
        "instances_path": str(args.instances_path),
        "model_path": str(args.model_path),
        "tune_n_instances": args.tune_n_instances,
        "batch_size": args.batch_size,
        "time_limit": args.time_limit,
        "seed": args.seed,
        "log_file": str(log_file),
        "fixed_params": {
            "restarts": 5,
            "incpopsize": 2.0,
            "use_lhs": True
        }
    }

    output_file = "best_bipop_params.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {output_file}")
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best mean gap: {best_value:.2f}%")

    return output_file


def generate_visualizations(study):
    """Generate Optuna visualization plots."""
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate
        )

        logging.info("Generating visualizations...")

        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image("bipop_optimization_history.png")
        logging.info("  Saved: bipop_optimization_history.png")

        # Parameter importances (requires ≥10 trials)
        if len(study.trials) >= 10:
            fig = plot_param_importances(study)
            fig.write_image("bipop_param_importances.png")
            logging.info("  Saved: bipop_param_importances.png")

        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_image("bipop_parallel_coordinate.png")
        logging.info("  Saved: bipop_parallel_coordinate.png")

    except ImportError:
        logging.warning("Visualization skipped: Install kaleido for plot export")
        logging.warning("  Run: uv add kaleido")
    except Exception as e:
        logging.warning(f"Visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for BIPOP-CMA-ES optimizer using Optuna"
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--instances_path', type=str, required=True,
                       help='Path to problem instances (.pkl file)')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials (default: 50)')
    parser.add_argument('--tune_n_instances', type=int, default=3,
                       help='Number of instances for tuning (default: 3)')
    parser.add_argument('--batch_size', type=int, default=600,
                       help='Population size (default: 600)')
    parser.add_argument('--time_limit', type=float, default=75.0,
                       help='Time limit per instance in seconds (default: 75)')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed (default: 1234)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_bipop_tuning.db',
                       help='Optuna storage database (default: sqlite:///optuna_bipop_tuning.db)')
    parser.add_argument('--load_if_exists', action='store_true',
                       help='Resume existing study if it exists')

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging()
    logging.info("=" * 80)
    logging.info("OPTUNA HYPERPARAMETER TUNING FOR BIPOP-CMA-ES OPTIMIZER")
    logging.info("=" * 80)
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Instances: {args.instances_path}")
    logging.info(f"Trials: {args.n_trials}")
    logging.info(f"Tuning instances: {args.tune_n_instances}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Time limit: {args.time_limit}s per instance")
    logging.info(f"Seed: {args.seed}")
    logging.info("Fixed parameters: restarts=5, incpopsize=2.0, use_lhs=True")
    logging.info("=" * 80)

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load model and instances
    model = load_model(args.model_path, device)
    instances = load_instances(args.instances_path)

    # Determine problem type and cost function
    problem_type = determine_problem_type(args.instances_path)
    cost_fn = get_cost_function(problem_type)
    logging.info(f"Problem type: {problem_type.upper()}")

    # Select tuning instances (first N instances)
    tuning_instances = instances[:args.tune_n_instances]
    logging.info(f"Selected {len(tuning_instances)} instances for tuning")

    # Store in args for objective function
    args.model = model
    args.tuning_instances = tuning_instances
    args.cost_fn = cost_fn
    args.device = device

    # Create Optuna study
    logging.info("Creating Optuna study...")
    study = optuna.create_study(
        study_name='bipop_tuning',
        storage=args.storage,
        direction='minimize',
        load_if_exists=args.load_if_exists,
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )

    # Run optimization
    logging.info(f"Starting optimization with {args.n_trials} trials...")
    logging.info("=" * 80)

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=False  # We use custom logging
    )

    # Save results
    logging.info("=" * 80)
    logging.info("OPTIMIZATION COMPLETED")
    logging.info("=" * 80)
    save_results(study, args, log_file)

    # Generate visualizations
    generate_visualizations(study)

    logging.info("=" * 80)
    logging.info("TUNING FINISHED SUCCESSFULLY")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()
