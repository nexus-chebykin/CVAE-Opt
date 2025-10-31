# ------------------------------------------------------------------------------+
# Plotting functions for optimizer convergence analysis
#
# Extracted from search_control.py for better organization
# Supports comparison across batch sizes, optimizers (DE, CMA-ES, Portfolio),
# and CMA-ES sigma values
# ------------------------------------------------------------------------------+

import matplotlib.pyplot as plt
import numpy as np
import logging
import os


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
    Plot optimizer comparison as percentage of initial value vs iterations.
    Supports DE, CMA-ES, and Portfolio optimizers.

    Args:
        optimizer_data: Dict mapping optimizer_name -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        batch_size: Fixed batch size used for all optimizers
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers (supports 3 optimizers now)
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F', 'Portfolio': '#F1A208'}

    # Find the maximum initial value across all optimizers for normalization
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
    Plot optimizer comparison as percentage of initial value vs evaluations.
    Supports DE, CMA-ES, and Portfolio optimizers.

    Args:
        optimizer_data: Dict mapping optimizer_name -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        batch_size: Fixed batch size used for all optimizers
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers (supports 3 optimizers now)
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F', 'Portfolio': '#F1A208'}

    # Find the maximum initial value across all optimizers for normalization
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
    Plot optimizer comparison as percentage of initial value vs wall-clock time.
    Supports DE, CMA-ES, and Portfolio optimizers.

    Args:
        optimizer_data: Dict mapping optimizer_name -> (avg_convergence_history, avg_time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        num_instances: Number of instances that were averaged
        batch_size: Fixed batch size used for all optimizers
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers (supports 3 optimizers now)
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F', 'Portfolio': '#F1A208'}

    # Find the maximum initial value across all optimizers for normalization
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
    Plot optimizer comparison for a single instance as percentage of initial value vs iterations.
    Supports DE, CMA-ES, and Portfolio optimizers.

    Args:
        optimizer_results: Dict mapping optimizer_name -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        batch_size: Fixed batch size used for all optimizers
        instance_idx: Index of the instance
        problem: Problem name
        problem_size: Problem size
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers (supports 3 optimizers now)
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F', 'Portfolio': '#F1A208'}

    # Find the maximum initial value across all optimizers for normalization
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
    Plot optimizer comparison for a single instance as percentage of initial value vs evaluations.
    Supports DE, CMA-ES, and Portfolio optimizers.

    Args:
        optimizer_results: Dict mapping optimizer_name -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        batch_size: Fixed batch size used for all optimizers
        instance_idx: Index of the instance
        problem: Problem name
        problem_size: Problem size
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers (supports 3 optimizers now)
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F', 'Portfolio': '#F1A208'}

    # Find the maximum initial value across all optimizers for normalization
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
    Plot optimizer comparison for a single instance as percentage of initial value vs wall-clock time.
    Supports DE, CMA-ES, and Portfolio optimizers.

    Args:
        optimizer_results: Dict mapping optimizer_name -> (convergence_history, time_history)
        output_path: Directory to save plots
        max_iterations: Maximum number of iterations
        batch_size: Fixed batch size used for all optimizers
        instance_idx: Index of the instance
        problem: Problem name
        problem_size: Problem size
    """
    plt.figure(figsize=(12, 7))

    # Define colors for optimizers (supports 3 optimizers now)
    colors = {'DE': '#E63946', 'CMA-ES': '#2A9D8F', 'Portfolio': '#F1A208'}

    # Find the maximum initial value across all optimizers for normalization
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
