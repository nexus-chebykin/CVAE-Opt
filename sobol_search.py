# ------------------------------------------------------------------------------+
# Sobol Quasi-Random Search optimizer for CVAE-Opt
#
# Uses low-discrepancy Sobol sequences for better space coverage compared to
# uniform random sampling. Provides strong baseline for comparing adaptive
# optimization methods.
#
# Key features:
# - Deterministic quasi-random sampling (reproducible)
# - Better space-filling properties: O(n^-1) vs O(n^-0.5) for random
# - Scrambling for improved high-dimensional performance
# - Ideal for exploring 100D latent spaces
# ------------------------------------------------------------------------------+

import numpy as np
import time
from scipy.stats import qmc


def minimize(cost_func, args, search_space_bound, search_space_size, popsize,
             mutate, recombination, maxiter, maxtime, maxevaluations=None,
             scramble=True, seed=1234):
    """
    Sobol quasi-random search optimizer.

    Generates candidate solutions using Sobol sequences (low-discrepancy)
    and tracks the best solution found. Provides superior space coverage
    compared to pure random search, especially in high dimensions.

    Args:
        cost_func: Objective function that takes (Z_batch, *args) and returns (tours, costs)
        args: Additional arguments for cost_func (model, config, instance, cost_fn)
        search_space_bound: Symmetric bounds for search space [-bound, +bound]
        search_space_size: Dimensionality of search space
        popsize: Population size (batch size for evaluation)
        mutate: Mutation factor (dummy parameter for interface compatibility)
        recombination: Crossover rate (dummy parameter for interface compatibility)
        maxiter: Maximum number of iterations (None = no limit)
        maxtime: Maximum wall-clock time in seconds (None = no limit)
        maxevaluations: Maximum number of function evaluations (None = no limit)
        scramble: Use Owen scrambling for better high-dimensional performance (default: True)
        seed: Random seed for reproducibility (used for scrambling)

    Returns:
        best_fitness: Best objective value found
        best_solution: Best solution vector (1D array)
        convergence_history: List of best fitness per iteration
        time_history: List of elapsed time per iteration
        timing_breakdown: Dict with 'ask_time', 'eval_time', 'tell_time', 'iterations'
    """

    # --- INITIALIZE ----------------+
    start_time = time.time()
    best_fitness = float('inf')
    best_solution = None
    convergence_history = []
    time_history = []
    evaluations_done = 0

    # Track timing breakdown
    ask_time_total = 0.0
    eval_time_total = 0.0
    tell_time_total = 0.0

    # Create Sobol sampler with scrambling
    # Scrambling improves performance in high dimensions and enables different seeds
    sampler = qmc.Sobol(d=search_space_size, scramble=scramble, seed=seed)

    # --- INITIAL EVALUATION (iteration 0) ----------------+
    # Generate initial batch using Sobol sequence
    ask_start = time.time()
    # For best Sobol properties, use power of 2 batch sizes when possible
    # But we support arbitrary batch sizes by generating and slicing
    n_power = int(np.ceil(np.log2(popsize)))
    initial_samples = sampler.random_base2(m=n_power)[:popsize]

    # Scale from [0,1]^d to [-search_space_bound, +search_space_bound]^d
    lower_bounds = np.full(search_space_size, -search_space_bound)
    upper_bounds = np.full(search_space_size, search_space_bound)
    initial_population = qmc.scale(initial_samples, lower_bounds, upper_bounds)
    ask_time_total += time.time() - ask_start

    # Evaluate initial population
    eval_start = time.time()
    _, initial_fitness = cost_func(initial_population, *args)
    initial_fitness = np.array(initial_fitness)
    evaluations_done += popsize
    eval_time_total += time.time() - eval_start

    # Track best from initial population
    tell_start = time.time()
    min_idx = np.argmin(initial_fitness)
    best_fitness = initial_fitness[min_idx]
    best_solution = initial_population[min_idx].copy()
    tell_time_total += time.time() - tell_start

    # Record iteration 0
    convergence_history.append(best_fitness)
    time_history.append(time.time() - start_time)

    # --- MAIN SEARCH LOOP ----------------+
    iteration = 0
    while True:
        iteration += 1

        # Check stopping criteria
        if maxtime is not None and time.time() - start_time > maxtime:
            break
        if maxiter is not None and iteration > maxiter:
            break
        if maxevaluations is not None and evaluations_done >= maxevaluations:
            break

        # --- ASK: Generate Sobol samples ----------------+
        ask_start = time.time()
        # Continue Sobol sequence from where we left off
        # sampler remembers its state and generates next points in sequence
        sobol_samples = sampler.random(n=popsize)

        # Scale to search space bounds
        population = qmc.scale(sobol_samples, lower_bounds, upper_bounds)
        ask_time_total += time.time() - ask_start

        # --- EVALUATE: Batch evaluation ----------------+
        eval_start = time.time()
        _, fitness = cost_func(population, *args)
        fitness = np.array(fitness)
        evaluations_done += popsize
        eval_time_total += time.time() - eval_start

        # --- TELL: Update best solution ----------------+
        tell_start = time.time()
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            best_solution = population[min_idx].copy()
        tell_time_total += time.time() - tell_start

        # --- TRACK CONVERGENCE ----------------+
        convergence_history.append(best_fitness)
        time_history.append(time.time() - start_time)

    # --- RETURN RESULTS ----------------+
    timing_breakdown = {
        'ask_time': ask_time_total,
        'eval_time': eval_time_total,
        'tell_time': tell_time_total,
        'iterations': iteration
    }

    return best_fitness, best_solution, convergence_history, time_history, timing_breakdown
