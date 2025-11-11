# ------------------------------------------------------------------------------+
# Nevergrad NGOpt (Neural Gradient Optimization)
# Implementation using the Nevergrad optimization library
#
# Reference: https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimization.optimizerlib.NGOpt
# NGOpt uses a competence map approach to automatically select optimization
# strategies based on problem characteristics, requiring no manual parameter tuning.
# ------------------------------------------------------------------------------+

import nevergrad as ng
import numpy as np
import time


def minimize(cost_func, args, search_space_bound, search_space_size, popsize,
             mutate, recombination, maxiter, maxtime, maxevaluations=None):
    """
    Minimize objective function using Nevergrad NGOpt algorithm.

    NGOpt (Neural Gradient Optimization) is a meta-optimizer that uses competence
    maps to automatically select appropriate optimization strategies based on
    problem characteristics. It requires no manual parameter tuning.

    Args:
        cost_func: Objective function that takes (X_batch, *args) and returns (tours, costs)
        args: Tuple of arguments to pass to cost_func (model, config, instance, cost_fn)
        search_space_bound: Symmetric search space bounds [-bound, +bound]
        search_space_size: Dimensionality of the search space
        popsize: Population size / batch size (num_workers in NGOpt)
        mutate: Mutation factor (not used by NGOpt, kept for signature consistency)
        recombination: Crossover rate (not used by NGOpt, kept for signature consistency)
        maxiter: Maximum number of iterations (None = no limit)
        maxtime: Maximum wall-clock time in seconds (None = no limit)
        maxevaluations: Maximum number of function evaluations (None = no limit)

    Returns:
        best_fitness: Best objective value found
        best_solution: Best solution vector (numpy array)
        convergence_history: List of best fitness values per iteration
        time_history: List of elapsed times per iteration
        timing_breakdown: Dict with 'ask_time', 'eval_time', 'tell_time', 'iterations'
    """
    start_time = time.time()

    # Initialize tracking variables
    convergence_history = []
    time_history = []
    evaluations_done = 0
    best_fitness = float('inf')
    best_solution = None

    # Timing breakdown
    ask_time_total = 0.0
    eval_time_total = 0.0
    tell_time_total = 0.0

    # Create bounded parametrization for the search space
    # NGOpt will respect these bounds during optimization
    parametrization = ng.p.Array(shape=(search_space_size,)).set_bounds(
        lower=-search_space_bound,
        upper=search_space_bound
    )

    # Initialize NGOpt optimizer
    # Set budget to large value and use external stopping criteria
    # num_workers enables parallel/batch evaluation
    optimizer = ng.optimizers.NGOpt(
        parametrization=parametrization,
        budget=999999,  # Large value, use external stopping criteria instead
        num_workers=popsize  # Enable batch evaluation with popsize workers
    )

    # Main optimization loop
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

        # --- ASK PHASE: Get batch of candidate solutions ----------------+
        ask_start = time.time()
        candidates = [optimizer.ask() for _ in range(popsize)]
        # Extract parameter values from candidate objects
        X_batch = np.array([cand.value for cand in candidates])
        ask_time_total += time.time() - ask_start

        # --- EVALUATE PHASE: Batch evaluation of candidates -------------+
        eval_start = time.time()
        _, costs = cost_func(X_batch, *args)
        costs_array = np.array(costs)
        evaluations_done += popsize
        eval_time_total += time.time() - eval_start

        # --- TELL PHASE: Report fitness values back to optimizer --------+
        tell_start = time.time()
        for candidate, cost in zip(candidates, costs_array):
            optimizer.tell(candidate, cost)
        tell_time_total += time.time() - tell_start

        # --- SCORE KEEPING -------------------------------------------+
        # Track best solution found so far
        min_idx = np.argmin(costs_array)
        min_cost = costs_array[min_idx]
        if min_cost < best_fitness:
            best_fitness = min_cost
            best_solution = X_batch[min_idx].copy()

        # Record convergence history
        convergence_history.append(best_fitness)
        time_history.append(time.time() - start_time)

    # Verify with optimizer's recommendation (should match or be close to our tracked best)
    recommendation = optimizer.provide_recommendation()
    recommended_fitness = recommendation.loss if hasattr(recommendation, 'loss') else best_fitness

    # Use manually tracked best if it's better
    if best_solution is None:
        best_solution = recommendation.value

    # Prepare timing breakdown
    timing_breakdown = {
        'ask_time': ask_time_total,
        'eval_time': eval_time_total,
        'tell_time': tell_time_total,
        'iterations': iteration
    }

    return best_fitness, best_solution, convergence_history, time_history, timing_breakdown
