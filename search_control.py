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


def plot_convergence_comparison_iterations(instance_idx, convergence_data, output_path, max_iterations):
    """
    Plot comparison of convergence histories for different batch sizes vs iterations.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
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
    plt.title(f'Convergence Comparison - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_iterations.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved iterations-based comparison plot for instance {instance_idx}")


def plot_convergence_comparison(instance_idx, convergence_data, output_path, max_iterations):
    """
    Plot comparison of convergence histories for different batch sizes on the same graph.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
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
    plt.title(f'Convergence Comparison - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved comparison plot for instance {instance_idx}")


def plot_convergence_comparison_time(instance_idx, convergence_data, output_path, max_iterations):
    """
    Plot comparison of convergence histories for different batch sizes vs wall-clock time.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
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
    plt.title(f'Convergence Comparison - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_time.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved time-based comparison plot for instance {instance_idx}")


def plot_convergence_comparison_iterations_pct(instance_idx, convergence_data, output_path, max_iterations):
    """
    Plot comparison of convergence histories as percentage of initial value vs iterations.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different batch sizes
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653']

    # Find the maximum initial value across all batch sizes for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in convergence_data.values())

    # Plot each batch size
    for idx, (batch_size, (convergence_history, time_history)) in enumerate(sorted(convergence_data.items())):
        iterations = list(range(1, len(convergence_history) + 1))
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors[idx % len(colors)]
        plt.plot(iterations, convergence_pct, linewidth=2.5, color=color,
                 label=f'Batch Size: {batch_size}', marker='o', markevery=max(1, len(iterations)//10), markersize=6)

    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Convergence Comparison - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_iterations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved iterations-based percentage comparison plot for instance {instance_idx}")


def plot_convergence_comparison_pct(instance_idx, convergence_data, output_path, max_iterations):
    """
    Plot comparison of convergence histories as percentage of initial value vs evaluations.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different batch sizes
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653']

    # Find the maximum initial value across all batch sizes for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in convergence_data.values())

    # Plot each batch size
    for idx, (batch_size, (convergence_history, time_history)) in enumerate(sorted(convergence_data.items())):
        iterations = list(range(1, len(convergence_history) + 1))
        evaluations = [iter_num * batch_size for iter_num in iterations]
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors[idx % len(colors)]
        plt.plot(evaluations, convergence_pct, linewidth=2.5, color=color,
                 label=f'Batch Size: {batch_size}', marker='o', markevery=max(1, len(evaluations)//10), markersize=6)

    plt.xlabel('Objective Function Evaluations', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Convergence Comparison - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved evaluations-based percentage comparison plot for instance {instance_idx}")


def plot_convergence_comparison_time_pct(instance_idx, convergence_data, output_path, max_iterations):
    """
    Plot comparison of convergence histories as percentage of initial value vs wall-clock time.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different batch sizes
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653']

    # Find the maximum initial value across all batch sizes for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in convergence_data.values())

    # Plot each batch size
    for idx, (batch_size, (convergence_history, time_history)) in enumerate(sorted(convergence_data.items())):
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors[idx % len(colors)]
        plt.plot(time_history, convergence_pct, linewidth=2.5, color=color,
                 label=f'Batch Size: {batch_size}', marker='o', markevery=max(1, len(time_history)//10), markersize=6)

    plt.xlabel('Wall-Clock Time (seconds)', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Convergence Comparison - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_time_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved time-based percentage comparison plot for instance {instance_idx}")


def plot_average_convergence_iterations_pct(averaged_data, output_path, max_iterations, num_instances):
    """
    Plot averaged convergence histories as percentage of initial value vs iterations.

    Args:
        averaged_data: Dict mapping batch_size -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different batch sizes
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653']

    # Find the maximum initial value across all batch sizes for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in averaged_data.values())

    # Plot each batch size
    for idx, (batch_size, (convergence_history, time_history)) in enumerate(sorted(averaged_data.items())):
        iterations = list(range(1, len(convergence_history) + 1))
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors[idx % len(colors)]
        plt.plot(iterations, convergence_pct, linewidth=2.5, color=color,
                 label=f'Batch Size: {batch_size}', marker='o', markevery=max(1, len(iterations)//10), markersize=6)

    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Average Convergence Comparison (Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'average_convergence_iterations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved averaged iterations-based percentage comparison plot")


def plot_average_convergence_evaluations_pct(averaged_data, output_path, max_iterations, num_instances):
    """
    Plot averaged convergence histories as percentage of initial value vs evaluations.

    Args:
        averaged_data: Dict mapping batch_size -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different batch sizes
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653']

    # Find the maximum initial value across all batch sizes for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in averaged_data.values())

    # Plot each batch size
    for idx, (batch_size, (convergence_history, time_history)) in enumerate(sorted(averaged_data.items())):
        iterations = list(range(1, len(convergence_history) + 1))
        evaluations = [iter_num * batch_size for iter_num in iterations]
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors[idx % len(colors)]
        plt.plot(evaluations, convergence_pct, linewidth=2.5, color=color,
                 label=f'Batch Size: {batch_size}', marker='o', markevery=max(1, len(evaluations)//10), markersize=6)

    plt.xlabel('Objective Function Evaluations', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Average Convergence Comparison (Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'average_convergence_evaluations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved averaged evaluations-based percentage comparison plot")


def plot_average_convergence_time_pct(averaged_data, output_path, max_iterations, num_instances):
    """
    Plot averaged convergence histories as percentage of initial value vs wall-clock time.

    Args:
        averaged_data: Dict mapping batch_size -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different batch sizes
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653']

    # Find the maximum initial value across all batch sizes for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in averaged_data.values())

    # Plot each batch size
    for idx, (batch_size, (convergence_history, time_history)) in enumerate(sorted(averaged_data.items())):
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors[idx % len(colors)]
        plt.plot(time_history, convergence_pct, linewidth=2.5, color=color,
                 label=f'Batch Size: {batch_size}', marker='o', markevery=max(1, len(time_history)//10), markersize=6)

    plt.xlabel('Wall-Clock Time (seconds)', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Average Convergence Comparison (Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'average_convergence_time_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved averaged time-based percentage comparison plot")


def compute_averaged_convergence(all_instances_data, batch_sizes):
    """
    Compute averaged convergence histories across all instances.

    Args:
        all_instances_data: Dict {batch_size: {'convergence': [...], 'time': [...]}}
                          where each list contains convergence/time histories from all instances
        batch_sizes: List of batch sizes

    Returns:
        averaged_data: Dict {batch_size: (avg_convergence_history, avg_time_history)}
    """
    averaged_data = {}

    for batch_size in batch_sizes:
        convergence_histories = all_instances_data[batch_size]['convergence']
        time_histories = all_instances_data[batch_size]['time']

        # Find maximum length across all instances for this batch size
        max_conv_len = max(len(h) for h in convergence_histories)
        max_time_len = max(len(h) for h in time_histories)

        # Pad convergence histories with last value to make them same length
        padded_convergence = []
        for history in convergence_histories:
            padded = history + [history[-1]] * (max_conv_len - len(history))
            padded_convergence.append(padded)

        # Pad time histories with last value to make them same length
        padded_time = []
        for history in time_histories:
            padded = history + [history[-1]] * (max_time_len - len(history))
            padded_time.append(padded)

        # Compute element-wise average
        avg_convergence = np.mean(padded_convergence, axis=0).tolist()
        avg_time = np.mean(padded_time, axis=0).tolist()

        averaged_data[batch_size] = (avg_convergence, avg_time)

    return averaged_data


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

    # Accumulator for averaging convergence data across instances
    all_instances_data = {bs: {'convergence': [], 'time': []}
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

            # Accumulate data for averaging across instances
            all_instances_data[batch_size]['convergence'].append(convergence_history)
            all_instances_data[batch_size]['time'].append(time_history)

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

        # Save convergence comparison plots if enabled (only for per-instance mode)
        if config.save_plots and config.plot_mode == 'per_instance':
            # Create absolute value comparison plots (all batch sizes on same graph)
            if len(config.batch_sizes) > 1:
                plot_convergence_comparison_iterations(i, convergence_data, search_output_dir, config.search_iterations)
                plot_convergence_comparison(i, convergence_data, search_output_dir, config.search_iterations)
                plot_convergence_comparison_time(i, convergence_data, search_output_dir, config.search_iterations)
                # Create percentage-based comparison plots
                plot_convergence_comparison_iterations_pct(i, convergence_data, search_output_dir, config.search_iterations)
                plot_convergence_comparison_pct(i, convergence_data, search_output_dir, config.search_iterations)
                plot_convergence_comparison_time_pct(i, convergence_data, search_output_dir, config.search_iterations)
            else:
                # If single batch size, still create plots but they'll only have one curve
                plot_convergence_comparison_iterations(i, convergence_data, search_output_dir, config.search_iterations)
                plot_convergence_comparison(i, convergence_data, search_output_dir, config.search_iterations)
                plot_convergence_comparison_time(i, convergence_data, search_output_dir, config.search_iterations)
                # Create percentage-based plots
                plot_convergence_comparison_iterations_pct(i, convergence_data, search_output_dir, config.search_iterations)
                plot_convergence_comparison_pct(i, convergence_data, search_output_dir, config.search_iterations)
                plot_convergence_comparison_time_pct(i, convergence_data, search_output_dir, config.search_iterations)

    # Generate averaged plots if in average mode
    if config.save_plots and config.plot_mode == 'average' and len(config.batch_sizes) > 1:
        logging.info("Computing averaged convergence data across all instances...")
        averaged_data = compute_averaged_convergence(all_instances_data, config.batch_sizes)

        # Generate averaged percentage plots
        plot_average_convergence_iterations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances))
        plot_average_convergence_evaluations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances))
        plot_average_convergence_time_pct(averaged_data, search_output_dir, config.search_iterations, len(instances))

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
