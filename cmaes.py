# ------------------------------------------------------------------------------+
# CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer
# Wrapper for the pycma library with ask-tell interface
#
# Uses batch evaluation pattern compatible with the existing DE implementation
# ------------------------------------------------------------------------------+

import numpy as np
import time
import cma


def minimize(cost_func, args, search_space_bound, search_space_size, popsize, sigma0, maxiter, maxtime, maxevaluations=None):
    """
    CMA-ES optimizer matching DE interface.

    Args:
        cost_func: Objective function that accepts batch of solutions
        args: Additional arguments passed to cost_func
        search_space_bound: Box constraints [-bound, +bound] for all dimensions
        search_space_size: Dimensionality of search space
        popsize: Population size (lambda in CMA-ES terminology)
        sigma0: Initial step size (typically 0.2-0.5 of search range)
        maxiter: Maximum number of iterations
        maxtime: Maximum wall-clock time in seconds

    Returns:
        gen_best: Best fitness value found
        best_solution: Best solution vector
        convergence_history: List of best fitness at each iteration
        time_history: List of elapsed time at each iteration
    """

    # --- INITIALIZE CMA-ES ----------------+
    start_time = time.time()
    convergence_history = []
    time_history = []
    evaluations_done = 0  # Track total number of evaluations

    # Initial mean: start at origin (center of search space)
    x0 = np.zeros(search_space_size)

    # CMA-ES options
    # If maxiter is None, use a very large number so time limit is the constraint
    cmaes_maxiter = maxiter if maxiter is not None else 1000000

    opts = {
        'popsize': popsize,
        'bounds': [-search_space_bound, search_space_bound],
        'maxiter': cmaes_maxiter,
        'verbose': -9,  # Suppress output
        'verb_disp': 0,  # No display
        'verb_log': 0    # No logging
    }

    # Initialize CMA-ES evolution strategy
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # --- OPTIMIZE WITH ASK-TELL PATTERN ----------------+

    while not es.stop():
        # Check stopping criteria
        # Always check time limit
        if time.time() - start_time > maxtime:
            break
        # Only check evaluation limit if specified (overrides time)
        if maxevaluations is not None and evaluations_done >= maxevaluations:
            break

        # ASK: Generate new population of candidate solutions
        solutions = es.ask()

        # EVALUATE: Batch evaluation of all candidates
        solutions_array = np.array(solutions)
        _, fitness_values = cost_func(solutions_array, *args)
        fitness_values = np.array(fitness_values)
        evaluations_done += popsize  # Increment evaluation counter

        # TELL: Update CMA-ES distribution based on fitness values
        es.tell(solutions, fitness_values.tolist())

        # --- TRACK CONVERGENCE --------------------------------+
        gen_best = es.result.fbest  # Best fitness so far
        convergence_history.append(gen_best)
        time_history.append(time.time() - start_time)

    # --- RETURN RESULTS ----------------+
    best_solution = es.result.xbest
    best_fitness = es.result.fbest

    return best_fitness, best_solution, convergence_history, time_history
