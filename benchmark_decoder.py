#!/usr/bin/env python3
"""
Benchmark script to measure decoder evaluation time across different batch sizes.

This script measures the time it takes to evaluate the decoder (model.decode + cost computation)
for batch sizes ranging from 1 to 1000, in intervals of 10.
"""

import torch
import numpy as np
import time
import argparse
import os
import csv
import matplotlib.pyplot as plt
from VAE_8 import VAE_8
from utils import read_instance_pkl
import tsp
import cvrp


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark decoder evaluation time for different batch sizes"
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--instances_path', type=str, required=True,
                        help='Path to test instances (will use first instance)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_path', type=str, default='.',
                        help='Directory to save results')
    parser.add_argument('--max_batch_size', type=int, default=1000,
                        help='Maximum batch size to test (default: 1000)')
    parser.add_argument('--interval', type=int, default=10,
                        help='Batch size interval (default: 10)')

    return parser.parse_args()


def load_model_and_config(model_path, device):
    """Load the trained model and extract configuration."""
    print(f"Loading model from: {model_path}")
    model_data = torch.load(model_path, device, weights_only=False)

    # Extract model configuration
    config = argparse.Namespace()
    config.device = torch.device(device)
    config.search_space_bound = model_data['Z_bound']
    config.search_space_size = 100  # Standard latent space size
    config.problem = model_data['problem']
    config.problem_size = model_data['problem_size']

    # Additional config needed for decode
    config.decode_len = config.problem_size
    config.decode_type = 'sampling'

    print(f"  Problem: {config.problem}{config.problem_size}")
    print(f"  Search space bound (Z_bound): {config.search_space_bound:.4f}")
    print(f"  Search space size: {config.search_space_size}")

    # Load model
    model = VAE_8(config).to(config.device)
    model.load_state_dict(model_data['parameters'])
    model.eval()
    print("  Model loaded and set to eval mode")

    return model, config


def load_instance(instances_path, config):
    """Load test instance (use first instance)."""
    print(f"\nLoading instances from: {instances_path}")

    # Create a temporary config with instances_path
    temp_config = argparse.Namespace()
    temp_config.instances_path = instances_path
    temp_config.problem = config.problem

    instances = read_instance_pkl(temp_config)
    instance = instances[0]  # Use first instance

    print(f"  Loaded {len(instances)} instances, using first one")
    print(f"  Instance shape: {instance.shape if hasattr(instance, 'shape') else len(instance)}")

    # Convert to torch tensor
    instance_tensor = torch.Tensor(instance).unsqueeze(0).to(config.device)

    return instance_tensor


def get_cost_function(config):
    """Get the cost function based on problem type."""
    if config.problem == 'TSP':
        return tsp.tours_length
    elif config.problem == 'CVRP':
        return cvrp.tours_length
    else:
        raise ValueError(f"Unknown problem type: {config.problem}")


def decode_batch(model, config, instance, batch_size):
    """
    Decode a batch of random latent vectors and compute costs.
    This mimics what happens during optimization.

    Args:
        model: The VAE model
        config: Configuration
        instance: Problem instance
        batch_size: Number of solutions to evaluate

    Returns:
        Time taken in seconds
    """
    # Generate random latent vectors in the search space
    Z = np.random.uniform(
        -config.search_space_bound,
        config.search_space_bound,
        (batch_size, config.search_space_size)
    )
    Z = torch.Tensor(Z).to(config.device)

    # Replicate instance for batch
    instance_batch = instance.repeat(batch_size, 1, 1)

    # Get cost function
    cost_fn = get_cost_function(config)

    # Reset decoder before decoding (initializes dummy_solution and instance_hidden)
    model.reset_decoder(batch_size, config)

    # Time the decode operation
    torch.cuda.synchronize() if config.device.type == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance_batch, Z, config)
        costs = cost_fn(instance_batch, tour_idx)

    torch.cuda.synchronize() if config.device.type == 'cuda' else None
    elapsed_time = time.time() - start_time

    return elapsed_time


def warmup(model, config, instance, num_warmup=3):
    """Warmup GPU before benchmarking."""
    print("\nWarming up GPU...")
    for i in range(num_warmup):
        _ = decode_batch(model, config, instance, batch_size=100)
    print("  Warmup complete")


def benchmark_decoder(model, config, instance, max_batch_size, interval):
    """
    Benchmark decoder across different batch sizes.

    Args:
        model: The VAE model
        config: Configuration
        instance: Problem instance
        max_batch_size: Maximum batch size to test
        interval: Interval between batch sizes

    Returns:
        List of (batch_size, time) tuples
    """
    results = []

    # Generate batch sizes: 1, then 10, 20, 30, ..., max_batch_size
    batch_sizes = [1] + list(range(interval, max_batch_size + 1, interval))

    print(f"\nBenchmarking decoder for {len(batch_sizes)} batch sizes...")
    print(f"Batch sizes: 1, {interval}, {2*interval}, ..., {max_batch_size}")
    print("-" * 60)

    for i, batch_size in enumerate(batch_sizes):
        elapsed_time = decode_batch(model, config, instance, batch_size)
        results.append((batch_size, elapsed_time))

        # Print progress
        progress = (i + 1) / len(batch_sizes) * 100
        print(f"[{progress:5.1f}%] Batch size {batch_size:4d}: {elapsed_time:.6f} seconds")

    print("-" * 60)
    print(f"Benchmark complete! Tested {len(batch_sizes)} batch sizes.")

    return results


def save_results_csv(results, output_path):
    """Save benchmark results to CSV file."""
    csv_file = os.path.join(output_path, 'decoder_benchmark_results.csv')

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch_size', 'time_seconds'])
        writer.writerows(results)

    print(f"\nResults saved to: {csv_file}")


def plot_results(results, output_path):
    """Create and save plot of benchmark results."""
    batch_sizes = [r[0] for r in results]
    times = [r[1] for r in results]

    plt.figure(figsize=(12, 7))
    plt.plot(batch_sizes, times, linewidth=2.5, color='#2A9D8F',
             marker='o', markersize=4, markerfacecolor='#E63946')

    plt.xlabel('Batch Size', fontsize=13)
    plt.ylabel('Evaluation Time (seconds)', fontsize=13)
    plt.title('Decoder Evaluation Time vs Batch Size', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(output_path, 'decoder_benchmark.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {plot_file}")

    # Print some statistics
    print("\nStatistics:")
    print(f"  Minimum time: {min(times):.6f} seconds (batch size {batch_sizes[times.index(min(times))]})")
    print(f"  Maximum time: {max(times):.6f} seconds (batch size {batch_sizes[times.index(max(times))]})")
    print(f"  Average time: {np.mean(times):.6f} seconds")
    print(f"  Time per evaluation (avg): {np.mean([t/bs for bs, t in results]):.6f} seconds")


def main():
    """Main benchmark function."""
    args = get_args()

    print("=" * 60)
    print("Decoder Benchmark Script")
    print("=" * 60)

    # Create output directory if needed
    os.makedirs(args.output_path, exist_ok=True)

    # Load model and configuration
    model, config = load_model_and_config(args.model_path, args.device)

    # Load instance
    instance = load_instance(args.instances_path, config)

    # Warmup GPU
    if config.device.type == 'cuda':
        warmup(model, config, instance)

    # Run benchmark
    results = benchmark_decoder(model, config, instance, args.max_batch_size, args.interval)

    # Save results
    save_results_csv(results, args.output_path)
    plot_results(results, args.output_path)

    print("\n" + "=" * 60)
    print("Benchmark completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
