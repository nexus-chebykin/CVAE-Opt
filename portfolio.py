# ------------------------------------------------------------------------------+
# Portfolio optimizer from nevergrad library
# Uses a portfolio of multiple optimization algorithms
#
# Nevergrad's Portfolio automatically manages a collection of optimizers,
# distributing the evaluation budget among them based on their performance
# ------------------------------------------------------------------------------+

import numpy as np
import time
import nevergrad as ng


def minimize(cost_func, args, search_space_bound, search_space_size, popsize, maxiter, maxtime, maxevaluations=None):
    """
    Portfolio optimizer matching CMA-ES and DE interface.

    Args:
        cost_func: Objective function that accepts batch of solutions
        args: Additional arguments passed to cost_func
        search_space_bound: Box constraints [-bound, +bound] for all dimensions
        search_space_size: Dimensionality of search space
        popsize: Population size (number of workers/candidates per iteration)
        maxiter: Maximum number of iterations
        maxtime: Maximum wall-clock time in seconds
        maxevaluations: Maximum number of function evaluations (optional)

    Returns:
        best_fitness: Best fitness value found
        best_solution: Best solution vector
        convergence_history: List of best fitness at each iteration
        time_history: List of elapsed time at each iteration
    """

    # --- INITIALIZE PORTFOLIO OPTIMIZER ----------------+
    start_time = time.time()
    convergence_history = []
    time_history = []
    evaluations_done = 0  # Track total number of evaluations

    # Define the parametrization: continuous bounded variables
    param = ng.p.Array(shape=(search_space_size,))
    param.set_bounds(-search_space_bound, search_space_bound)

    # Initialize Portfolio optimizer with popsize workers
    # If maxiter is None, use a very large number so time limit is the constraint
    budget = maxiter if maxiter is not None else 1000000
    if maxevaluations is not None:
        budget = min(budget, maxevaluations // popsize)

    optimizer = ng.optimizers.Portfolio(parametrization=param, budget=budget, num_workers=popsize)

    # Track best solution found so far
    best_fitness = np.inf
    best_solution = None

    # --- OPTIMIZE WITH ASK-TELL PATTERN ----------------+
    iteration = 0
    while True:
        iteration += 1

        # Check stopping criteria
        # Always check time limit
        if time.time() - start_time > maxtime:
            break
        # Only check iteration limit if specified
        if maxiter is not None and iteration > maxiter:
            break
        # Only check evaluation limit if specified
        if maxevaluations is not None and evaluations_done >= maxevaluations:
            break

        # ASK: Generate batch of candidate solutions
        candidates = []
        for _ in range(popsize):
            try:
                candidate = optimizer.ask()
                candidates.append(candidate)
            except Exception:
                # Budget exhausted or optimizer stopped
                break

        if len(candidates) == 0:
            break

        # EVALUATE: Batch evaluation of all candidates
        solutions_array = np.array([c.value for c in candidates])
        _, fitness_values = cost_func(solutions_array, *args)
        fitness_values = np.array(fitness_values)
        evaluations_done += len(candidates)

        # TELL: Update optimizer with fitness values
        for candidate, fitness in zip(candidates, fitness_values):
            optimizer.tell(candidate, fitness)

            # Track global best
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = candidate.value.copy()

        # --- TRACK CONVERGENCE --------------------------------+
        convergence_history.append(best_fitness)
        time_history.append(time.time() - start_time)

    # --- RETURN RESULTS ----------------+
    # If no solution was found, get recommendation from optimizer
    if best_solution is None:
        recommendation = optimizer.provide_recommendation()
        best_solution = recommendation.value
        # Evaluate the recommendation to get its fitness
        _, fitness_values = cost_func(best_solution.reshape(1, -1), *args)
        best_fitness = fitness_values[0]

    return best_fitness, best_solution, convergence_history, time_history
