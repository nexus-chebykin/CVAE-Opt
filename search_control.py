import torch
import tsp, cvrp
import numpy as np
import time
from de import minimize
import logging
import os
import matplotlib.pyplot as plt


def decode(Z, model, config, instance, cost_fn):
    Z = torch.Tensor(Z).to(config.device)
    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance, Z, config)
    costs = cost_fn(instance, tour_idx)
    return tour_idx, costs.tolist()


def evaluate(Z, model, config, instance, cost_fn):
    Z = torch.Tensor(Z).to(config.device)
    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance, Z, config)
    costs = cost_fn(instance, tour_idx)
    return costs.tolist()


def plot_convergence_comparison_iterations(instance_idx, convergence_data, output_path):
    """
    Plot comparison of convergence histories for different batch sizes vs iterations.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different batch sizes
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653']

    # Plot each batch size
    for idx, (batch_size, (convergence_history, time_history)) in enumerate(sorted(convergence_data.items())):
        iterations = list(range(1, len(convergence_history) + 1))
        color = colors[idx % len(colors)]
        plt.plot(iterations, convergence_history, linewidth=2.5, color=color,
                 label=f'Batch Size: {batch_size}', marker='o', markevery=max(1, len(iterations)//10), markersize=6)

    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Best Objective Value', fontsize=13)
    plt.title(f'Convergence Comparison - Instance {instance_idx}', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_iterations.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved iterations-based comparison plot for instance {instance_idx}")


def plot_convergence_comparison(instance_idx, convergence_data, output_path):
    """
    Plot comparison of convergence histories for different batch sizes on the same graph.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different batch sizes
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653']

    # Plot each batch size
    for idx, (batch_size, (convergence_history, time_history)) in enumerate(sorted(convergence_data.items())):
        iterations = list(range(1, len(convergence_history) + 1))
        evaluations = [iter_num * batch_size for iter_num in iterations]
        color = colors[idx % len(colors)]
        plt.plot(evaluations, convergence_history, linewidth=2.5, color=color,
                 label=f'Batch Size: {batch_size}', marker='o', markevery=max(1, len(evaluations)//10), markersize=6)

    plt.xlabel('Objective Function Evaluations', fontsize=13)
    plt.ylabel('Best Objective Value', fontsize=13)
    plt.title(f'Convergence Comparison - Instance {instance_idx}', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved comparison plot for instance {instance_idx}")


def plot_convergence_comparison_time(instance_idx, convergence_data, output_path):
    """
    Plot comparison of convergence histories for different batch sizes vs wall-clock time.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different batch sizes
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653']

    # Plot each batch size
    for idx, (batch_size, (convergence_history, time_history)) in enumerate(sorted(convergence_data.items())):
        color = colors[idx % len(colors)]
        plt.plot(time_history, convergence_history, linewidth=2.5, color=color,
                 label=f'Batch Size: {batch_size}', marker='o', markevery=max(1, len(time_history)//10), markersize=6)

    plt.xlabel('Wall-Clock Time (seconds)', fontsize=13)
    plt.ylabel('Best Objective Value', fontsize=13)
    plt.title(f'Convergence Comparison - Instance {instance_idx}', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_time.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved time-based comparison plot for instance {instance_idx}")


def solve_instance_de(model, instance, config, cost_fn, batch_size):
    instance = torch.Tensor(instance)
    instance = instance.unsqueeze(0).expand(batch_size, -1, -1)
    instance = instance.to(config.device)
    model.reset_decoder(batch_size, config)

    result_cost, result_tour, convergence_history, time_history = minimize(decode, (model, config, instance, cost_fn), config.search_space_bound,
                                        config.search_space_size, popsize=batch_size,
                                        mutate=config.de_mutate, recombination=config.de_recombine,
                                        maxiter=config.search_iterations, maxtime=config.search_timelimit)
    solution = decode(np.array([result_tour] * batch_size), model, config, instance, cost_fn)[0][0].tolist()
    return result_cost, solution, convergence_history, time_history


def solve_instance_set(model, config, instances, solutions=None, verbose=True):
    model.eval()

    if config.problem == "TSP":
        cost_fn = tsp.tours_length
    elif config.problem == "CVRP":
        cost_fn = cvrp.tours_length
        if solutions:
            solutions = [cvrp.solution_to_single_tour(solution) for solution in solutions]

    # Create search output directory if saving plots
    if config.save_plots:
        search_output_dir = os.path.join(config.output_path, "search")
        os.makedirs(search_output_dir, exist_ok=True)

    # Store results for each batch size
    all_results = {bs: {'gap_values': [], 'cost_values': [], 'runtime_values': []}
                   for bs in config.batch_sizes}

    for i, instance in enumerate(instances):
        logging.info(f"Solving instance {i + 1}/{len(instances)}")
        convergence_data = {}

        # Run search for each batch size
        for batch_size in config.batch_sizes:
            logging.info(f"  Batch size: {batch_size}")
            start_time = time.time()
            objective_value, solution, convergence_history, time_history = solve_instance_de(model, instance, config, cost_fn, batch_size)
            runtime = time.time() - start_time

            # Store convergence history and time history for comparison plots
            convergence_data[batch_size] = (convergence_history, time_history)

            # Calculate gap if solutions provided
            if solutions:
                optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                        torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
                gap = (objective_value / optimal_value - 1) * 100
                all_results[batch_size]['gap_values'].append(gap)
                logging.info(f"    Objective: {objective_value:.4f}, Optimal: {optimal_value:.4f}, Gap: {gap:.2f}%")
            else:
                all_results[batch_size]['gap_values'].append(0)
                logging.info(f"    Objective: {objective_value:.4f}")

            all_results[batch_size]['cost_values'].append(objective_value)
            all_results[batch_size]['runtime_values'].append(runtime)
            logging.info(f"    Runtime: {runtime:.2f}s")

        # Save convergence comparison plots if enabled
        if config.save_plots:
            # Only create comparison plots (all batch sizes on same graph)
            if len(config.batch_sizes) > 1:
                plot_convergence_comparison_iterations(i, convergence_data, search_output_dir)
                plot_convergence_comparison(i, convergence_data, search_output_dir)
                plot_convergence_comparison_time(i, convergence_data, search_output_dir)
            else:
                # If single batch size, still create plots but they'll only have one curve
                plot_convergence_comparison_iterations(i, convergence_data, search_output_dir)
                plot_convergence_comparison(i, convergence_data, search_output_dir)
                plot_convergence_comparison_time(i, convergence_data, search_output_dir)

    # Log final results for each batch size
    logging.info("=" * 60)
    logging.info("Final search results:")
    for batch_size in config.batch_sizes:
        results = all_results[batch_size]
        logging.info(f"\nBatch size {batch_size}:")
        logging.info(f"  Mean cost: {np.mean(results['cost_values']):.4f}")
        logging.info(f"  Mean runtime: {np.mean(results['runtime_values']):.2f}s")
        if solutions:
            logging.info(f"  Mean gap: {np.mean(results['gap_values']):.2f}%")
            logging.info(f"  Std gap: {np.std(results['gap_values']):.2f}%")

        # Save results to file
        if verbose and not solutions:
            output_file = os.path.join(config.output_path, "search", f'results_bs{batch_size}.txt')
            results_array = np.array(list(zip(results['cost_values'], results['runtime_values'])))
            np.savetxt(output_file, results_array, delimiter=',', fmt=['%s', '%s'],
                       header="cost, runtime")

    # Return results for the first batch size (for backward compatibility)
    first_batch_size = config.batch_sizes[0]
    return (np.mean(all_results[first_batch_size]['gap_values']),
            np.mean(all_results[first_batch_size]['runtime_values']),
            all_results[first_batch_size]['cost_values'])
