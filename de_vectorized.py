# ------------------------------------------------------------------------------+
# Vectorized Differential Evolution Implementation
# Based on the original de.py implementation by Nathan A. Rooy
#
# This version replaces the per-individual loop with fully vectorized numpy
# operations for improved performance.
#
# MIT License - See de.py for full license text
# ------------------------------------------------------------------------------+

import numpy as np
import time


def minimize(cost_func, args, search_space_bound, search_space_size, popsize, mutate, recombination, maxiter, maxtime, maxevaluations=None, seed=1234):
    """
    Vectorized Differential Evolution optimizer.

    Same interface as de.py but with fully vectorized mutation and crossover
    operations for improved performance.
    """

    # --- INITIALIZE A POPULATION (step #1) ----------------+
    start_time = time.time()
    children = np.zeros((popsize, search_space_size))
    iterations_without_improvement = 0
    gen_best = np.inf
    convergence_history = []
    time_history = []
    evaluations_done = 0  # Track total number of evaluations

    # Track timing breakdown
    ask_time_total = 0.0
    eval_time_total = 0.0
    tell_time_total = 0.0

    # Set random seed for reproducibility (using legacy MT19937 to match de.py)
    np.random.seed(seed)

    population = np.random.uniform(-search_space_bound, search_space_bound,
                                   (popsize, search_space_size))

    # Evaluate initial population
    eval_start = time.time()
    _, population_cost_initial = cost_func(population, *args)
    population_cost = np.array(population_cost_initial)
    evaluations_done += popsize
    eval_time_total += time.time() - eval_start

    # Record initial best (iteration 0)
    gen_best = np.min(population_cost)
    convergence_history.append(gen_best)
    time_history.append(time.time() - start_time)

    # Pre-allocate index array for random selection
    all_indices = np.arange(popsize)

    # --- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    iteration = 0
    while True:
        iteration += 1

        # Check stopping criteria
        # Check time limit if specified
        if maxtime is not None and time.time() - start_time > maxtime:
            break
        # Check iteration limit if specified
        if maxiter is not None and iteration > maxiter:
            break
        # Check evaluation limit if specified
        if maxevaluations is not None and evaluations_done >= maxevaluations:
            break

        # --- ASK: Generate candidate solutions (VECTORIZED) ----------------+
        ask_start = time.time()

        # Generate random indices for all individuals at once
        # For each individual j, we need 3 random indices != j
        # Strategy: generate random indices and shift to avoid self-selection
        r0 = np.random.randint(0, popsize - 1, size=popsize)
        r1 = np.random.randint(0, popsize - 2, size=popsize)
        r2 = np.random.randint(0, popsize - 3, size=popsize)

        # Adjust indices to avoid selecting self (j) and previously selected indices
        # For r0: if r0 >= j, increment by 1
        r0 = np.where(r0 >= all_indices, r0 + 1, r0)

        # For r1: avoid j and r0
        r1 = np.where(r1 >= np.minimum(all_indices, r0), r1 + 1, r1)
        r1 = np.where(r1 >= np.maximum(all_indices, r0), r1 + 1, r1)

        # For r2: avoid j, r0, and r1
        sorted_exclude = np.sort(np.stack([all_indices, r0, r1], axis=1), axis=1)
        r2 = np.where(r2 >= sorted_exclude[:, 0], r2 + 1, r2)
        r2 = np.where(r2 >= sorted_exclude[:, 1], r2 + 1, r2)
        r2 = np.where(r2 >= sorted_exclude[:, 2], r2 + 1, r2)

        # --- MUTATION (step #3.A) - Vectorized ---------------------+
        # child = x_r0 + F * (x_r1 - x_r2)
        x_diff = population[r1] - population[r2]
        children = population[r0] + mutate * x_diff

        # --- RECOMBINATION (step #3.B) - Vectorized ----------------+
        # Generate crossover mask for all individuals at once
        crossover_mask = np.random.uniform(0, 1, (popsize, search_space_size)) > recombination
        # Where mask is True, keep parent gene; where False, keep mutant gene
        children = np.where(crossover_mask, population, children)

        # Ensure bounds
        children = np.clip(children, -search_space_bound, search_space_bound)

        ask_time_total += time.time() - ask_start

        # --- EVALUATE: Objective function evaluation ------+
        eval_start = time.time()
        _, scores_trial = cost_func(children, *args)
        scores_trial = np.array(scores_trial)
        evaluations_done += popsize  # Increment evaluation counter
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
        gen_best = np.min(population_cost)  # fitness of best individual
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
