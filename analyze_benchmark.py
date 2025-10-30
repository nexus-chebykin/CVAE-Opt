#!/usr/bin/env python3
"""
Analyze decoder benchmark results using linear regression.
Shows the trade-off between batch size and evaluation time.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


def analyze_benchmark(csv_path):
    """Perform linear regression on benchmark data."""

    # Load data
    print("=" * 60)
    print("Decoder Benchmark Analysis - Linear Regression")
    print("=" * 60)
    print(f"\nLoading data from: {csv_path}")

    # Read CSV manually
    batch_sizes = []
    times = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_sizes.append(float(row['batch_size']))
            times.append(float(row['time_seconds']))

    X = np.array(batch_sizes).reshape(-1, 1)
    y = np.array(times)
    print(f"Loaded {len(X)} data points")

    # Fit linear regression using numpy (y = mx + b)
    # Calculate slope (m) and intercept (b) using least squares
    X_flat = X.flatten()
    n = len(X_flat)

    # Calculate means
    x_mean = np.mean(X_flat)
    y_mean = np.mean(y)

    # Calculate slope and intercept
    numerator = np.sum((X_flat - x_mean) * (y - y_mean))
    denominator = np.sum((X_flat - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Make predictions
    y_pred = slope * X_flat + intercept

    # Calculate metrics
    ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - y_mean) ** 2)  # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    mse = ss_res / n
    rmse = np.sqrt(mse)

    # Print results
    print("\n" + "-" * 60)
    print("LINEAR REGRESSION RESULTS")
    print("-" * 60)
    print(f"Equation: time = {intercept:.6f} + {slope:.6f} × batch_size")
    print(f"\nIntercept (base time): {intercept:.6f} seconds")
    print(f"Slope (time per unit): {slope:.6f} seconds/batch")
    print(f"\nR² Score: {r2:.6f}")
    print(f"RMSE: {rmse:.6f} seconds")

    # Calculate time per evaluation
    time_per_eval = slope
    print(f"\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    print(f"Base overhead (setup): {intercept:.6f} seconds")
    print(f"Additional time per item in batch: {time_per_eval*1000:.3f} milliseconds")
    print(f"Time per evaluation (marginal): {time_per_eval*1000:.3f} ms")

    # Efficiency analysis
    print(f"\n" + "-" * 60)
    print("EFFICIENCY ANALYSIS")
    print("-" * 60)

    # Calculate actual time per evaluation for different batch sizes
    batch_sizes_to_analyze = [1, 10, 50, 100, 300, 600, 1000]

    print(f"{'Batch Size':<12} {'Total Time (s)':<16} {'Time/Eval (ms)':<16} {'Throughput (eval/s)':<20}")
    print("-" * 64)

    for bs in batch_sizes_to_analyze:
        if bs <= X_flat.max():
            predicted_time = slope * bs + intercept
            time_per_eval_actual = predicted_time / bs
            throughput = bs / predicted_time
            print(f"{bs:<12} {predicted_time:<16.6f} {time_per_eval_actual*1000:<16.3f} {throughput:<20.2f}")

    # Speedup analysis
    print(f"\n" + "-" * 60)
    print("SPEEDUP ANALYSIS (vs batch_size=1)")
    print("-" * 60)

    time_bs1 = slope * 1 + intercept
    print(f"Time for batch_size=1: {time_bs1:.6f} seconds")
    print(f"\n{'Batch Size':<12} {'Speedup Factor':<20} {'Efficiency':<15}")
    print("-" * 47)

    for bs in batch_sizes_to_analyze[1:]:  # Skip batch_size=1
        if bs <= X_flat.max():
            time_bs = slope * bs + intercept
            # Speedup = (time to do bs evaluations individually) / (time to do bs evaluations in batch)
            time_sequential = time_bs1 * bs
            speedup = time_sequential / time_bs
            efficiency = speedup / bs * 100  # Percentage of ideal speedup
            print(f"{bs:<12} {speedup:<20.2f}x {efficiency:<15.1f}%")

    # Create visualization
    plt.figure(figsize=(14, 10))

    # Plot 1: Actual data and regression line
    plt.subplot(2, 2, 1)
    plt.scatter(X_flat, y, alpha=0.5, s=10, color='#2A9D8F', label='Actual data')
    plt.plot(X_flat, y_pred, color='#E63946', linewidth=2, label='Linear regression')
    plt.xlabel('Batch Size', fontsize=11)
    plt.ylabel('Time (seconds)', fontsize=11)
    plt.title(f'Linear Regression Fit (R² = {r2:.4f})', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Residuals
    plt.subplot(2, 2, 2)
    residuals = y - y_pred
    plt.scatter(X_flat, residuals, alpha=0.5, s=10, color='#F1A208')
    plt.axhline(y=0, color='#E63946', linestyle='--', linewidth=2)
    plt.xlabel('Batch Size', fontsize=11)
    plt.ylabel('Residuals (seconds)', fontsize=11)
    plt.title('Residual Plot', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 3: Time per evaluation
    plt.subplot(2, 2, 3)
    time_per_eval_array = y / X_flat
    plt.scatter(X_flat, time_per_eval_array * 1000, alpha=0.5, s=10, color='#264653')
    predicted_time_per_eval = slope * 1000
    plt.axhline(y=predicted_time_per_eval, color='#E63946', linestyle='--', linewidth=2,
                label=f'Average: {predicted_time_per_eval:.3f} ms')
    plt.xlabel('Batch Size', fontsize=11)
    plt.ylabel('Time per Evaluation (milliseconds)', fontsize=11)
    plt.title('Time per Evaluation vs Batch Size', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Throughput
    plt.subplot(2, 2, 4)
    throughput = X_flat / y
    plt.scatter(X_flat, throughput, alpha=0.5, s=10, color='#9B59B6')
    plt.xlabel('Batch Size', fontsize=11)
    plt.ylabel('Throughput (evaluations/second)', fontsize=11)
    plt.title('Throughput vs Batch Size', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = csv_path.replace('.csv', '_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n" + "=" * 60)
    print(f"Analysis plot saved to: {output_path}")
    print("=" * 60)

    return slope, intercept, r2, rmse


if __name__ == "__main__":
    csv_path = "runs/decoder_benchmark_results_1000.csv"
    slope, intercept, r2, rmse = analyze_benchmark(csv_path)
