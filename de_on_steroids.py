# ------------------------------------------------------------------------------+
# DE on Steroids - Vectorized Differential Evolution with Multiple Strategies
#
# Based on de_vectorized.py with support for all scipy DE mutation strategies.
# Implements 6 binomial crossover strategies for hyperparameter optimization.
#
# MIT License - See de.py for full license text
# ------------------------------------------------------------------------------+

import numpy as np
import time


# Available strategies
STRATEGIES = ['rand1bin', 'rand2bin', 'best1bin', 'best2bin', 'currenttobest1bin', 'randtobest1bin']


def _generate_random_indices(rng, popsize, num_indices, all_indices):
    """
    Generate random indices for mutation, ensuring no duplicates and no self-selection.

    Args:
        rng: numpy random generator
        popsize: population size
        num_indices: number of random indices needed per individual
        all_indices: array of indices [0, 1, ..., popsize-1]

    Returns:
        List of index arrays [r0, r1, r2, ...] where each is shape (popsize,)
    """
    indices = []

    for i in range(num_indices):
        # Generate random indices avoiding already selected ones
        r = rng.integers(0, popsize - 1 - i, size=popsize)

        # Sort excluded indices for efficient adjustment
        if i == 0:
            # Only avoid self (j)
            r = np.where(r >= all_indices, r + 1, r)
        else:
            # Avoid self and previously selected indices
            excluded = np.column_stack([all_indices] + indices)
            excluded_sorted = np.sort(excluded, axis=1)
            for col in range(excluded_sorted.shape[1]):
                r = np.where(r >= excluded_sorted[:, col], r + 1, r)

        indices.append(r)

    return indices


def minimize(cost_func, args, search_space_bound, search_space_size, popsize,
             mutate=0.232165, recombination=0.875693, maxiter=None, maxtime=None,
             maxevaluations=None, seed=1234, strategy='rand1bin'):
    """
    DE on Steroids - Vectorized Differential Evolution with multiple mutation strategies.

    Args:
        cost_func: Objective function to minimize
        args: Additional arguments for cost_func
        search_space_bound: Bounds for the search space [-bound, bound]
        search_space_size: Dimensionality of the search space
        popsize: Population size
        mutate: Mutation factor F (default: 0.232165)
        recombination: Crossover probability CR (default: 0.875693)
        maxiter: Maximum iterations (default: None)
        maxtime: Maximum time in seconds (default: None)
        maxevaluations: Maximum function evaluations (default: None)
        seed: Random seed for reproducibility (default: 1234)
        strategy: Mutation strategy (default: 'rand1bin')
                  Options: 'rand1bin', 'rand2bin', 'best1bin', 'best2bin',
                           'currenttobest1bin', 'randtobest1bin'

    Returns:
        best_cost: Best objective value found
        best_solution: Best solution vector
        convergence_history: List of best fitness per iteration
        time_history: List of elapsed time per iteration
        timing_breakdown: Dict with timing statistics
    """

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {STRATEGIES}")

    # --- INITIALIZE A POPULATION (step #1) ----------------+
    start_time = time.time()
    children = np.zeros((popsize, search_space_size))
    iterations_without_improvement = 0
    gen_best = np.inf
    convergence_history = []
    time_history = []
    evaluations_done = 0

    # Track timing breakdown
    ask_time_total = 0.0
    eval_time_total = 0.0
    tell_time_total = 0.0

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    population = rng.uniform(-search_space_bound, search_space_bound,
                             (popsize, search_space_size))

    # Evaluate initial population
    eval_start = time.time()
    _, population_cost_initial = cost_func(population, *args)
    population_cost = np.array(population_cost_initial)
    evaluations_done += popsize
    eval_time_total += time.time() - eval_start

    # Record initial best (iteration 0)
    gen_best = np.min(population_cost)
    best_idx = np.argmin(population_cost)
    convergence_history.append(gen_best)
    time_history.append(time.time() - start_time)

    # Pre-allocate index array for random selection
    all_indices = np.arange(popsize)

    # Determine number of random indices needed based on strategy
    if strategy in ['rand1bin', 'best1bin', 'currenttobest1bin']:
        num_random_indices = 3  # r0, r1, r2 (or just r0, r1 for best1bin)
    elif strategy in ['randtobest1bin']:
        num_random_indices = 3
    elif strategy in ['rand2bin', 'best2bin']:
        num_random_indices = 5  # r0, r1, r2, r3, r4

    # --- SOLVE --------------------------------------------+

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

        # --- ASK: Generate candidate solutions (VECTORIZED) ----------------+
        ask_start = time.time()

        # Get current best individual
        best_idx = np.argmin(population_cost)
        x_best = population[best_idx]

        # Generate random indices based on strategy requirements
        if strategy == 'rand1bin':
            # b' = x_r0 + F * (x_r1 - x_r2)
            r0, r1, r2 = _generate_random_indices(rng, popsize, 3, all_indices)
            children = population[r0] + mutate * (population[r1] - population[r2])

        elif strategy == 'rand2bin':
            # b' = x_r0 + F * (x_r1 + x_r2 - x_r3 - x_r4)
            r0, r1, r2, r3, r4 = _generate_random_indices(rng, popsize, 5, all_indices)
            children = population[r0] + mutate * (population[r1] + population[r2] - population[r3] - population[r4])

        elif strategy == 'best1bin':
            # b' = x_best + F * (x_r0 - x_r1)
            r0, r1 = _generate_random_indices(rng, popsize, 2, all_indices)
            children = x_best + mutate * (population[r0] - population[r1])

        elif strategy == 'best2bin':
            # b' = x_best + F * (x_r0 + x_r1 - x_r2 - x_r3)
            r0, r1, r2, r3 = _generate_random_indices(rng, popsize, 4, all_indices)
            children = x_best + mutate * (population[r0] + population[r1] - population[r2] - population[r3])

        elif strategy == 'currenttobest1bin':
            # b' = x_i + F * (x_best - x_i) + F * (x_r0 - x_r1)
            r0, r1 = _generate_random_indices(rng, popsize, 2, all_indices)
            children = population + mutate * (x_best - population) + mutate * (population[r0] - population[r1])

        elif strategy == 'randtobest1bin':
            # b' = x_r0 + F * (x_best - x_r0) + F * (x_r1 - x_r2)
            r0, r1, r2 = _generate_random_indices(rng, popsize, 3, all_indices)
            children = population[r0] + mutate * (x_best - population[r0]) + mutate * (population[r1] - population[r2])

        # --- RECOMBINATION (step #3.B) - Binomial Crossover ----------------+
        crossover_mask = rng.uniform(0, 1, (popsize, search_space_size)) > recombination
        # Where mask is True, keep parent gene; where False, keep mutant gene
        children = np.where(crossover_mask, population, children)

        # Ensure bounds
        children = np.clip(children, -search_space_bound, search_space_bound)

        ask_time_total += time.time() - ask_start

        # --- EVALUATE: Objective function evaluation ------+
        eval_start = time.time()
        _, scores_trial = cost_func(children, *args)
        scores_trial = np.array(scores_trial)
        evaluations_done += popsize
        eval_time_total += time.time() - eval_start

        # --- TELL: Update population with results ---------+
        tell_start = time.time()

        iterations_without_improvement += 1
        if np.min(population_cost) > np.min(scores_trial):
            iterations_without_improvement = 0

        improvement = population_cost > scores_trial
        population[improvement] = children[improvement]
        population_cost[improvement] = scores_trial[improvement]

        tell_time_total += time.time() - tell_start

        # --- SCORE KEEPING --------------------------------+
        gen_best = np.min(population_cost)
        convergence_history.append(gen_best)
        time_history.append(time.time() - start_time)

    # Return timing breakdown
    timing_breakdown = {
        'ask_time': ask_time_total,
        'eval_time': eval_time_total,
        'tell_time': tell_time_total,
        'iterations': iteration
    }

    return gen_best, population[np.argmin(population_cost)], convergence_history, time_history, timing_breakdown
