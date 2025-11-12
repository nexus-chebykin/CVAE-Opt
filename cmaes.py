# ------------------------------------------------------------------------------+
# CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer
# Wrapper for the pycma library with ask-tell interface
#
# Uses batch evaluation pattern compatible with the existing DE implementation
# ------------------------------------------------------------------------------+

import numpy as np
import time
import cma
from lhs_init import generate_lhs_population


def minimize(
    cost_func,
    args,
    search_space_bound,
    search_space_size,
    popsize,
    sigma0,
    maxiter,
    maxtime,
    maxevaluations=None,
    use_lhs=True,
    CMA_rankmu=1.0,
    CMA_rankone=1.0,
    seed=None,
):
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
        maxevaluations: Maximum number of function evaluations (None = no limit)
        use_lhs: If True, initialize with Latin Hypercube Sampling using "prime the pump" approach
        CMA_rankmu: Rank-mu update learning rate multiplier (default: 1.0)
        CMA_rankone: Rank-one update learning rate multiplier (default: 1.0)
        seed: Random seed for reproducibility (used for LHS initialization)

    Returns:
        gen_best: Best fitness value found
        best_solution: Best solution vector
        convergence_history: List of best fitness at each iteration
        time_history: List of elapsed time at each iteration
        timing_breakdown: Dictionary with timing statistics
    """

    # --- INITIALIZE CMA-ES ----------------+
    start_time = time.time()
    convergence_history = []
    time_history = []
    evaluations_done = 0  # Track total number of evaluations

    # Track timing breakdown
    ask_time_total = 0.0
    eval_time_total = 0.0
    tell_time_total = 0.0

    # Initial mean: random point across search space
    x0 = np.random.uniform(-search_space_bound, search_space_bound, search_space_size)

    # CMA-ES options
    # If maxiter is None, use a very large number so time limit is the constraint
    cmaes_maxiter = maxiter if maxiter is not None else 1000000

    opts = {
        "popsize": popsize,
        "bounds": [-search_space_bound, search_space_bound],
        "maxiter": cmaes_maxiter,
        "verbose": -9,  # Suppress output
        "verb_disp": 0,  # No display
        "verb_log": 0,  # No logging
        "CMA_rankmu": CMA_rankmu,  # Rank-mu update learning rate multiplier
        "CMA_rankone": CMA_rankone  # Rank-one update learning rate multiplier
        # Disable internal stopping criteria to respect only time limit
        #'tolx': 1e100,  # Disable stopping based on small x-changes
        #'tolfun': 1e100,  # Disable stopping based on small function value changes
        #'tolstagnation': 1e100,  # Disable stopping based on stagnation
        #'tolfacupx': 1e100  # Disable stopping based on large step-size
    }

    # Initialize CMA-ES evolution strategy
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # --- LHS INITIALIZATION ("Prime the Pump") ----------------+
    # If use_lhs is True, we "prime" the CMA-ES by generating an LHS population,
    # evaluating it, and feeding it to CMA-ES via tell() before the main loop.
    # This gives CMA-ES a better starting distribution than a single point at origin.
    if use_lhs:
        # CMA-ES requires ask() before tell() to initialize internal state
        # We call ask() to generate initial population, but then replace it with LHS
        ask_start = time.time()
        _ = es.ask()  # Discard the uniform random population
        ask_time_total += time.time() - ask_start

        # Generate LHS population to replace the ask() result
        lhs_population = generate_lhs_population(
            num_samples=popsize,
            dimension=search_space_size,
            lower_bound=-search_space_bound,
            upper_bound=search_space_bound,
            seed=seed
        )

        # Evaluate LHS population
        eval_start = time.time()
        _, lhs_fitness = cost_func(lhs_population, *args)
        lhs_fitness = np.array(lhs_fitness)
        eval_time_total += time.time() - eval_start
        evaluations_done += popsize

        # Prime the pump: tell CMA-ES about the LHS population
        # This becomes "Generation 0" for CMA-ES
        tell_start = time.time()
        es.tell(lhs_population.tolist(), lhs_fitness.tolist())
        tell_time_total += time.time() - tell_start
    else:
        # No LHS: Evaluate initial random population from CMA-ES
        ask_start = time.time()
        initial_population = es.ask()
        ask_time_total += time.time() - ask_start

        # Evaluate initial population
        eval_start = time.time()
        _, initial_fitness = cost_func(np.array(initial_population), *args)
        initial_fitness = np.array(initial_fitness)
        eval_time_total += time.time() - eval_start
        evaluations_done += popsize

        # Tell CMA-ES about initial population
        tell_start = time.time()
        es.tell(initial_population, initial_fitness.tolist())
        tell_time_total += time.time() - tell_start

    # Record initial convergence point (iteration 0)
    gen_best = es.result.fbest
    convergence_history.append(gen_best)
    time_history.append(time.time() - start_time)

    # --- OPTIMIZE WITH ASK-TELL PATTERN ----------------+

    iteration = 0
    while True:
        iteration += 1

        # Check stopping criteria (before ask/tell to avoid extra evaluations)
        # Check time limit if specified
        if maxtime is not None and time.time() - start_time > maxtime:
            break
        # Check iteration limit if specified
        if maxiter is not None and iteration > maxiter:
            break
        # Check evaluation limit if specified
        if maxevaluations is not None and evaluations_done >= maxevaluations:
            break

        # ASK: Generate new population of candidate solutions
        ask_start = time.time()
        solutions = es.ask()
        ask_time_total += time.time() - ask_start

        # EVALUATE: Batch evaluation of all candidates
        eval_start = time.time()
        solutions_array = np.array(solutions)
        _, fitness_values = cost_func(solutions_array, *args)
        fitness_values = np.array(fitness_values)
        evaluations_done += popsize  # Increment evaluation counter
        eval_time_total += time.time() - eval_start

        # TELL: Update CMA-ES distribution based on fitness values
        tell_start = time.time()
        es.tell(solutions, fitness_values.tolist())
        tell_time_total += time.time() - tell_start

        # --- TRACK CONVERGENCE --------------------------------+
        gen_best = es.result.fbest  # Best fitness so far
        convergence_history.append(gen_best)
        time_history.append(time.time() - start_time)

    # --- RETURN RESULTS ----------------+
    best_solution = es.result.xbest
    best_fitness = es.result.fbest

    # Return timing breakdown
    timing_breakdown = {
        "ask_time": ask_time_total,
        "eval_time": eval_time_total,
        "tell_time": tell_time_total,
        "iterations": iteration,
    }

    return (
        best_fitness,
        best_solution,
        convergence_history,
        time_history,
        timing_breakdown,
    )
