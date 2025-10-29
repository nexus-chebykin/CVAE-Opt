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


def plot_convergence_comparison_iterations(instance_idx, convergence_data, output_path, max_iterations, optimizer='DE'):
    """
    Plot comparison of convergence histories for different batch sizes vs iterations.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        optimizer: Optimizer name for plot title (default: 'DE')
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
    plt.title(f'Convergence Comparison [{optimizer}] - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_iterations.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved iterations-based comparison plot for instance {instance_idx}")


def plot_convergence_comparison(instance_idx, convergence_data, output_path, max_iterations, optimizer='DE'):
    """
    Plot comparison of convergence histories for different batch sizes on the same graph.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        optimizer: Optimizer name for plot title (default: 'DE')
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
    plt.title(f'Convergence Comparison [{optimizer}] - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved comparison plot for instance {instance_idx}")


def plot_convergence_comparison_time(instance_idx, convergence_data, output_path, max_iterations, optimizer='DE'):
    """
    Plot comparison of convergence histories for different batch sizes vs wall-clock time.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        optimizer: Optimizer name for plot title (default: 'DE')
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
    plt.title(f'Convergence Comparison [{optimizer}] - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_time.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved time-based comparison plot for instance {instance_idx}")


def plot_convergence_comparison_iterations_pct(instance_idx, convergence_data, output_path, max_iterations, optimizer='DE'):
    """
    Plot comparison of convergence histories as percentage of initial value vs iterations.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        optimizer: Optimizer name for plot title (default: 'DE')
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
    plt.title(f'Convergence Comparison [{optimizer}] - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_iterations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved iterations-based percentage comparison plot for instance {instance_idx}")


def plot_convergence_comparison_pct(instance_idx, convergence_data, output_path, max_iterations, optimizer='DE'):
    """
    Plot comparison of convergence histories as percentage of initial value vs evaluations.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        optimizer: Optimizer name for plot title (default: 'DE')
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
    plt.title(f'Convergence Comparison [{optimizer}] - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved evaluations-based percentage comparison plot for instance {instance_idx}")


def plot_convergence_comparison_time_pct(instance_idx, convergence_data, output_path, max_iterations, optimizer='DE'):
    """
    Plot comparison of convergence histories as percentage of initial value vs wall-clock time.

    Args:
        instance_idx: Index of the instance
        convergence_data: Dict mapping batch_size -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        optimizer: Optimizer name for plot title (default: 'DE')
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
    plt.title(f'Convergence Comparison [{optimizer}] - Instance {instance_idx} (Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_comparison_time_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved time-based percentage comparison plot for instance {instance_idx}")


def plot_average_convergence_iterations_pct(averaged_data, output_path, max_iterations, num_instances, optimizer='DE'):
    """
    Plot averaged convergence histories as percentage of initial value vs iterations.

    Args:
        averaged_data: Dict mapping batch_size -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        optimizer: Optimizer name for plot title (default: 'DE')
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
    plt.title(f'Average Convergence Comparison [{optimizer}] (Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'average_convergence_iterations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved averaged iterations-based percentage comparison plot")


def plot_average_convergence_evaluations_pct(averaged_data, output_path, max_iterations, num_instances, optimizer='DE'):
    """
    Plot averaged convergence histories as percentage of initial value vs evaluations.

    Args:
        averaged_data: Dict mapping batch_size -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        optimizer: Optimizer name for plot title (default: 'DE')
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
    plt.title(f'Average Convergence Comparison [{optimizer}] (Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'average_convergence_evaluations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved averaged evaluations-based percentage comparison plot")


def plot_average_convergence_time_pct(averaged_data, output_path, max_iterations, num_instances, optimizer='DE'):
    """
    Plot averaged convergence histories as percentage of initial value vs wall-clock time.

    Args:
        averaged_data: Dict mapping batch_size -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        optimizer: Optimizer name for plot title (default: 'DE')
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
    plt.title(f'Average Convergence Comparison [{optimizer}] (Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'average_convergence_time_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved averaged time-based percentage comparison plot")


def plot_sigma_comparison_iterations_pct(averaged_data, output_path, max_iterations, num_instances, batch_size):
    """
    Plot CMA-ES sigma comparison as percentage of initial value vs iterations.

    Args:
        averaged_data: Dict mapping sigma_value -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        batch_size: Fixed batch size used for all sigma values
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different sigma values
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653', '#9B59B6']

    # Find the maximum initial value across all sigma values for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in averaged_data.values())

    # Plot each sigma value
    for idx, (sigma_value, (convergence_history, time_history)) in enumerate(sorted(averaged_data.items())):
        iterations = list(range(1, len(convergence_history) + 1))
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors[idx % len(colors)]
        plt.plot(iterations, convergence_pct, linewidth=2.5, color=color,
                 label=f'Sigma: {sigma_value}', marker='o', markevery=max(1, len(iterations)//10), markersize=6)

    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'CMA-ES Sigma Comparison (Batch Size: {batch_size}, Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'sigma_comparison_iterations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved sigma comparison iterations-based percentage plot")


def plot_sigma_comparison_evaluations_pct(averaged_data, output_path, max_iterations, num_instances, batch_size):
    """
    Plot CMA-ES sigma comparison as percentage of initial value vs evaluations.

    Args:
        averaged_data: Dict mapping sigma_value -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        batch_size: Fixed batch size used for all sigma values
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different sigma values
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653', '#9B59B6']

    # Find the maximum initial value across all sigma values for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in averaged_data.values())

    # Plot each sigma value
    for idx, (sigma_value, (convergence_history, time_history)) in enumerate(sorted(averaged_data.items())):
        iterations = list(range(1, len(convergence_history) + 1))
        evaluations = [iter_num * batch_size for iter_num in iterations]
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors[idx % len(colors)]
        plt.plot(evaluations, convergence_pct, linewidth=2.5, color=color,
                 label=f'Sigma: {sigma_value}', marker='o', markevery=max(1, len(evaluations)//10), markersize=6)

    plt.xlabel('Objective Function Evaluations', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'CMA-ES Sigma Comparison (Batch Size: {batch_size}, Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'sigma_comparison_evaluations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved sigma comparison evaluations-based percentage plot")


def plot_sigma_comparison_time_pct(averaged_data, output_path, max_iterations, num_instances, batch_size):
    """
    Plot CMA-ES sigma comparison as percentage of initial value vs wall-clock time.

    Args:
        averaged_data: Dict mapping sigma_value -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        batch_size: Fixed batch size used for all sigma values
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different sigma values
    colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653', '#9B59B6']

    # Find the maximum initial value across all sigma values for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in averaged_data.values())

    # Plot each sigma value
    for idx, (sigma_value, (convergence_history, time_history)) in enumerate(sorted(averaged_data.items())):
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors[idx % len(colors)]
        plt.plot(time_history, convergence_pct, linewidth=2.5, color=color,
                 label=f'Sigma: {sigma_value}', marker='o', markevery=max(1, len(time_history)//10), markersize=6)

    plt.xlabel('Wall-Clock Time (seconds)', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'CMA-ES Sigma Comparison (Batch Size: {batch_size}, Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'sigma_comparison_time_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved sigma comparison time-based percentage plot")


def plot_optimizer_comparison_iterations_pct(optimizer_data, output_path, max_iterations, num_instances, batch_size):
    """
    Plot DE vs CMA-ES comparison as percentage of initial value vs iterations.

    Args:
        optimizer_data: Dict mapping optimizer_name -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        batch_size: Fixed batch size used for both optimizers
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F'}

    # Find the maximum initial value across both optimizers for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in optimizer_data.values())

    # Plot each optimizer
    for optimizer_name, (convergence_history, time_history) in sorted(optimizer_data.items()):
        iterations = list(range(1, len(convergence_history) + 1))
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors.get(optimizer_name, '#264653')
        plt.plot(iterations, convergence_pct, linewidth=2.5, color=color,
                 label=optimizer_name, marker='o', markevery=max(1, len(iterations)//10), markersize=6)

    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Optimizer Comparison (Batch Size: {batch_size}, Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'optimizer_comparison_iterations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved optimizer comparison iterations-based percentage plot")


def plot_optimizer_comparison_evaluations_pct(optimizer_data, output_path, max_iterations, num_instances, batch_size):
    """
    Plot DE vs CMA-ES comparison as percentage of initial value vs evaluations.

    Args:
        optimizer_data: Dict mapping optimizer_name -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        batch_size: Fixed batch size used for both optimizers
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F'}

    # Find the maximum initial value across both optimizers for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in optimizer_data.values())

    # Plot each optimizer
    for optimizer_name, (convergence_history, time_history) in sorted(optimizer_data.items()):
        iterations = list(range(1, len(convergence_history) + 1))
        evaluations = [iter_num * batch_size for iter_num in iterations]
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors.get(optimizer_name, '#264653')
        plt.plot(evaluations, convergence_pct, linewidth=2.5, color=color,
                 label=optimizer_name, marker='o', markevery=max(1, len(evaluations)//10), markersize=6)

    plt.xlabel('Objective Function Evaluations', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Optimizer Comparison (Batch Size: {batch_size}, Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'optimizer_comparison_evaluations_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved optimizer comparison evaluations-based percentage plot")


def plot_optimizer_comparison_time_pct(optimizer_data, output_path, max_iterations, num_instances, batch_size):
    """
    Plot DE vs CMA-ES comparison as percentage of initial value vs wall-clock time.

    Args:
        optimizer_data: Dict mapping optimizer_name -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        batch_size: Fixed batch size used for both optimizers
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F'}

    # Find the maximum initial value across both optimizers for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in optimizer_data.values())

    # Plot each optimizer
    for optimizer_name, (convergence_history, time_history) in sorted(optimizer_data.items()):
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors.get(optimizer_name, '#264653')
        plt.plot(time_history, convergence_pct, linewidth=2.5, color=color,
                 label=optimizer_name, marker='o', markevery=max(1, len(time_history)//10), markersize=6)

    plt.xlabel('Wall-Clock Time (seconds)', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Optimizer Comparison (Batch Size: {batch_size}, Across {num_instances} Instances, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, 'optimizer_comparison_time_pct.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved optimizer comparison time-based percentage plot")


def plot_optimizer_comparison_iterations_pct_per_instance(optimizer_results, output_path, max_iterations, batch_size, instance_idx, problem, problem_size):
    """
    Plot DE vs CMA-ES comparison for a single instance as percentage of initial value vs iterations.

    Args:
        optimizer_results: Dict mapping optimizer_name -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        batch_size: Fixed batch size used for both optimizers
        instance_idx: Index of the instance
        problem: Problem name
        problem_size: Problem size
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F'}

    # Find the maximum initial value across both optimizers for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in optimizer_results.values())

    # Plot each optimizer
    for optimizer_name, (convergence_history, time_history) in sorted(optimizer_results.items()):
        iterations = list(range(1, len(convergence_history) + 1))
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors.get(optimizer_name, '#264653')
        plt.plot(iterations, convergence_pct, linewidth=2.5, color=color,
                 label=optimizer_name, marker='o', markevery=max(1, len(iterations)//10), markersize=6)

    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Optimizer Comparison - Instance {instance_idx} ({problem}{problem_size}, Batch Size: {batch_size}, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'optimizer_comparison_iterations_pct_inst{instance_idx}.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_optimizer_comparison_evaluations_pct_per_instance(optimizer_results, output_path, max_iterations, batch_size, instance_idx, problem, problem_size):
    """
    Plot DE vs CMA-ES comparison for a single instance as percentage of initial value vs evaluations.

    Args:
        optimizer_results: Dict mapping optimizer_name -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        batch_size: Fixed batch size used for both optimizers
        instance_idx: Index of the instance
        problem: Problem name
        problem_size: Problem size
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F'}

    # Find the maximum initial value across both optimizers for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in optimizer_results.values())

    # Plot each optimizer
    for optimizer_name, (convergence_history, time_history) in sorted(optimizer_results.items()):
        # Calculate number of evaluations per iteration
        evaluations = [i * batch_size for i in range(1, len(convergence_history) + 1)]
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors.get(optimizer_name, '#264653')
        plt.plot(evaluations, convergence_pct, linewidth=2.5, color=color,
                 label=optimizer_name, marker='o', markevery=max(1, len(evaluations)//10), markersize=6)

    plt.xlabel('Number of Evaluations', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Optimizer Comparison - Instance {instance_idx} ({problem}{problem_size}, Batch Size: {batch_size}, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'optimizer_comparison_evaluations_pct_inst{instance_idx}.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_optimizer_comparison_time_pct_per_instance(optimizer_results, output_path, max_iterations, batch_size, instance_idx, problem, problem_size):
    """
    Plot DE vs CMA-ES comparison for a single instance as percentage of initial value vs wall-clock time.

    Args:
        optimizer_results: Dict mapping optimizer_name -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        batch_size: Fixed batch size used for both optimizers
        instance_idx: Index of the instance
        problem: Problem name
        problem_size: Problem size
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F'}

    # Find the maximum initial value across both optimizers for normalization
    max_initial_value = max(convergence_history[0] for convergence_history, _ in optimizer_results.values())

    # Plot each optimizer
    for optimizer_name, (convergence_history, time_history) in sorted(optimizer_results.items()):
        # Convert to percentage of maximum initial value
        convergence_pct = [(value / max_initial_value) * 100 for value in convergence_history]
        color = colors.get(optimizer_name, '#264653')
        plt.plot(time_history, convergence_pct, linewidth=2.5, color=color,
                 label=optimizer_name, marker='o', markevery=max(1, len(time_history)//10), markersize=6)

    plt.xlabel('Wall-Clock Time (seconds)', fontsize=13)
    plt.ylabel('Objective Value (% of Max Initial)', fontsize=13)
    plt.title(f'Optimizer Comparison - Instance {instance_idx} ({problem}{problem_size}, Batch Size: {batch_size}, Max Iterations: {max_iterations})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f'optimizer_comparison_time_pct_inst{instance_idx}.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()


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


def solve_instance(model, instance, config, cost_fn, batch_size, sigma0=None):
    """
    Solve a single instance using the configured optimizer (DE or CMA-ES).

    Args:
        model: Neural network model
        instance: Problem instance to solve
        config: Configuration object with optimizer settings
        cost_fn: Cost function for evaluating tours
        batch_size: Population size for the optimizer
        sigma0: Optional sigma0 for CMA-ES (overrides config.cmaes_sigma0 if provided)

    Returns:
        result_cost: Best objective value found
        solution: Best solution (tour)
        convergence_history: History of best fitness per iteration
        time_history: History of elapsed time per iteration
    """
    instance = torch.Tensor(instance)
    instance = instance.unsqueeze(0).expand(batch_size, -1, -1)
    instance = instance.to(config.device)
    model.reset_decoder(batch_size, config)

    # Select optimizer based on config
    if config.optimizer == 'de':
        from de import minimize
        result_cost, result_tour, convergence_history, time_history = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=config.search_iterations,
            maxtime=config.search_timelimit
        )
    elif config.optimizer == 'cmaes':
        from cmaes import minimize
        # Use provided sigma0 or fall back to config value
        cmaes_sigma = sigma0 if sigma0 is not None else config.cmaes_sigma0
        result_cost, result_tour, convergence_history, time_history = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            sigma0=cmaes_sigma,
            maxiter=config.search_iterations,
            maxtime=config.search_timelimit
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

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

    # Detect special modes
    optimizer_comparison_mode = config.compare_optimizers
    sigma_sweep_mode = config.cmaes_sigma_sweep is not None

    if optimizer_comparison_mode:
        # Optimizer comparison mode: run both DE and CMA-ES with fixed batch size
        fixed_batch_size = config.batch_sizes[0]
        logging.info(f"Running optimizer comparison mode with batch size {fixed_batch_size}")
        logging.info(f"Comparing DE vs CMA-ES (sigma={config.cmaes_sigma0})")

        # Store results for each optimizer
        all_results = {'DE': {'gap_values': [], 'cost_values': [], 'runtime_values': []},
                       'CMA-ES': {'gap_values': [], 'cost_values': [], 'runtime_values': []}}

        # Accumulator for averaging convergence data across instances
        all_instances_data = {'DE': {'convergence': [], 'time': []},
                              'CMA-ES': {'convergence': [], 'time': []}}
    elif sigma_sweep_mode:
        # Sigma sweep mode: loop over sigma values with fixed batch size
        sweep_values = config.cmaes_sigma_sweep
        fixed_batch_size = config.batch_sizes[0]
        logging.info(f"Running CMA-ES sigma sweep mode with batch size {fixed_batch_size}")
        logging.info(f"Sigma values: {sweep_values}")

        # Store results for each sigma value
        all_results = {sigma: {'gap_values': [], 'cost_values': [], 'runtime_values': []}
                       for sigma in sweep_values}

        # Accumulator for averaging convergence data across instances
        all_instances_data = {sigma: {'convergence': [], 'time': []}
                              for sigma in sweep_values}
    else:
        # Normal mode: loop over batch sizes
        # Store results for each batch size
        all_results = {bs: {'gap_values': [], 'cost_values': [], 'runtime_values': []}
                       for bs in config.batch_sizes}

        # Accumulator for averaging convergence data across instances
        all_instances_data = {bs: {'convergence': [], 'time': []}
                              for bs in config.batch_sizes}

    for i, instance in enumerate(instances):
        logging.info(f"Solving instance {i + 1}/{len(instances)}")
        convergence_data = {}

        if optimizer_comparison_mode:
            # Run both DE and CMA-ES
            for optimizer_name in ['DE', 'CMA-ES']:
                logging.info(f"  Optimizer: {optimizer_name}")
                start_time = time.time()

                # Temporarily change config.optimizer for solve_instance
                original_optimizer = config.optimizer
                config.optimizer = 'de' if optimizer_name == 'DE' else 'cmaes'

                objective_value, solution, convergence_history, time_history = solve_instance(
                    model, instance, config, cost_fn, fixed_batch_size)

                # Restore original optimizer
                config.optimizer = original_optimizer
                runtime = time.time() - start_time

                # Store convergence history and time history for comparison plots
                convergence_data[optimizer_name] = (convergence_history, time_history)

                # Accumulate data for averaging across instances
                all_instances_data[optimizer_name]['convergence'].append(convergence_history)
                all_instances_data[optimizer_name]['time'].append(time_history)

                # Calculate gap if solutions provided
                if solutions:
                    optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                            torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
                    gap = (objective_value / optimal_value - 1) * 100
                    all_results[optimizer_name]['gap_values'].append(gap)
                    logging.info(f"    Objective: {objective_value:.4f}, Optimal: {optimal_value:.4f}, Gap: {gap:.2f}%")
                else:
                    all_results[optimizer_name]['gap_values'].append(0)
                    logging.info(f"    Objective: {objective_value:.4f}")

                all_results[optimizer_name]['cost_values'].append(objective_value)
                all_results[optimizer_name]['runtime_values'].append(runtime)
                logging.info(f"    Runtime: {runtime:.2f}s")

            # Generate per-instance optimizer comparison plots if in per_instance mode
            if config.save_plots and config.plot_mode == 'per_instance':
                plot_optimizer_comparison_iterations_pct_per_instance(
                    convergence_data, search_output_dir, config.search_iterations,
                    fixed_batch_size, i, config.problem, config.problem_size)
                plot_optimizer_comparison_evaluations_pct_per_instance(
                    convergence_data, search_output_dir, config.search_iterations,
                    fixed_batch_size, i, config.problem, config.problem_size)
                plot_optimizer_comparison_time_pct_per_instance(
                    convergence_data, search_output_dir, config.search_iterations,
                    fixed_batch_size, i, config.problem, config.problem_size)
        elif sigma_sweep_mode:
            # Run search for each sigma value with fixed batch size
            for sigma_value in sweep_values:
                logging.info(f"  Sigma: {sigma_value}")
                start_time = time.time()
                objective_value, solution, convergence_history, time_history = solve_instance(
                    model, instance, config, cost_fn, fixed_batch_size, sigma0=sigma_value)
                runtime = time.time() - start_time

                # Store convergence history and time history for comparison plots
                convergence_data[sigma_value] = (convergence_history, time_history)

                # Accumulate data for averaging across instances
                all_instances_data[sigma_value]['convergence'].append(convergence_history)
                all_instances_data[sigma_value]['time'].append(time_history)

                # Calculate gap if solutions provided
                if solutions:
                    optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                            torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
                    gap = (objective_value / optimal_value - 1) * 100
                    all_results[sigma_value]['gap_values'].append(gap)
                    logging.info(f"    Objective: {objective_value:.4f}, Optimal: {optimal_value:.4f}, Gap: {gap:.2f}%")
                else:
                    all_results[sigma_value]['gap_values'].append(0)
                    logging.info(f"    Objective: {objective_value:.4f}")

                all_results[sigma_value]['cost_values'].append(objective_value)
                all_results[sigma_value]['runtime_values'].append(runtime)
                logging.info(f"    Runtime: {runtime:.2f}s")
        else:
            # Run search for each batch size
            for batch_size in config.batch_sizes:
                logging.info(f"  Batch size: {batch_size}")
                start_time = time.time()
                objective_value, solution, convergence_history, time_history = solve_instance(model, instance, config, cost_fn, batch_size)
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
            # Format optimizer name for display
            optimizer_name = 'CMA-ES' if config.optimizer == 'cmaes' else 'DE'

            # Create absolute value comparison plots (all batch sizes on same graph)
            if len(config.batch_sizes) > 1:
                plot_convergence_comparison_iterations(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_time(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                # Create percentage-based comparison plots
                plot_convergence_comparison_iterations_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_time_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
            else:
                # If single batch size, still create plots but they'll only have one curve
                plot_convergence_comparison_iterations(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_time(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                # Create percentage-based plots
                plot_convergence_comparison_iterations_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_time_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)

    # Generate averaged plots
    # For optimizer_comparison_mode and sigma_sweep_mode, always generate averaged plots regardless of plot_mode
    # For normal mode, only generate averaged plots if plot_mode == 'average'
    if config.save_plots:
        if optimizer_comparison_mode:
            # Optimizer comparison mode: generate optimizer comparison plots (always, regardless of plot_mode)
            logging.info("Computing averaged convergence data across all instances for optimizer comparison...")
            averaged_data = compute_averaged_convergence(all_instances_data, ['CMA-ES', 'DE'])

            # Generate optimizer comparison plots
            plot_optimizer_comparison_iterations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
            plot_optimizer_comparison_evaluations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
            plot_optimizer_comparison_time_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
        elif sigma_sweep_mode:
            # Sigma sweep mode: generate sigma comparison plots (always, regardless of plot_mode)
            logging.info("Computing averaged convergence data across all instances for sigma sweep...")
            averaged_data = compute_averaged_convergence(all_instances_data, sweep_values)

            # Generate sigma comparison plots
            plot_sigma_comparison_iterations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
            plot_sigma_comparison_evaluations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
            plot_sigma_comparison_time_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
        elif config.plot_mode == 'average' and len(config.batch_sizes) > 1:
            # Normal mode with average plot_mode: generate batch size comparison plots
            logging.info("Computing averaged convergence data across all instances...")
            averaged_data = compute_averaged_convergence(all_instances_data, config.batch_sizes)

            # Format optimizer name for display
            optimizer_name = 'CMA-ES' if config.optimizer == 'cmaes' else 'DE'

            # Generate averaged percentage plots
            plot_average_convergence_iterations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), optimizer_name)
            plot_average_convergence_evaluations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), optimizer_name)
            plot_average_convergence_time_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), optimizer_name)

    # Log final results
    logging.info("=" * 60)
    logging.info("Final search results:")

    if optimizer_comparison_mode:
        # Log results for each optimizer
        for optimizer_name in ['DE', 'CMA-ES']:
            results = all_results[optimizer_name]
            logging.info(f"\n{optimizer_name}:")
            logging.info(f"  Mean cost: {np.mean(results['cost_values']):.4f}")
            logging.info(f"  Mean runtime: {np.mean(results['runtime_values']):.2f}s")
            if solutions:
                logging.info(f"  Mean gap: {np.mean(results['gap_values']):.2f}%")
                logging.info(f"  Std gap: {np.std(results['gap_values']):.2f}%")
    elif sigma_sweep_mode:
        # Log results for each sigma value
        for sigma_value in sweep_values:
            results = all_results[sigma_value]
            logging.info(f"\nSigma: {sigma_value}:")
            logging.info(f"  Mean cost: {np.mean(results['cost_values']):.4f}")
            logging.info(f"  Mean runtime: {np.mean(results['runtime_values']):.2f}s")
            if solutions:
                logging.info(f"  Mean gap: {np.mean(results['gap_values']):.2f}%")
                logging.info(f"  Std gap: {np.std(results['gap_values']):.2f}%")
    else:
        # Log results for each batch size
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

    # Return results (for backward compatibility)
    if optimizer_comparison_mode:
        # Return results for DE (first optimizer)
        return (np.mean(all_results['DE']['gap_values']),
                np.mean(all_results['DE']['runtime_values']),
                all_results['DE']['cost_values'])
    elif sigma_sweep_mode:
        # Return results for the first sigma value
        first_sigma = sweep_values[0]
        return (np.mean(all_results[first_sigma]['gap_values']),
                np.mean(all_results[first_sigma]['runtime_values']),
                all_results[first_sigma]['cost_values'])
    else:
        # Return results for the first batch size
        first_batch_size = config.batch_sizes[0]
        return (np.mean(all_results[first_batch_size]['gap_values']),
                np.mean(all_results[first_batch_size]['runtime_values']),
                all_results[first_batch_size]['cost_values'])
