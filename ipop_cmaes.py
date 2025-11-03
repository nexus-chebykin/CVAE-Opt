# ------------------------------------------------------------------------------+
# IPOP-CMA-ES (Increasing Population CMA-ES) optimizer
# CMA-ES with restart strategy using increasing population sizes
#
# Uses batch evaluation pattern compatible with the existing DE implementation
# ------------------------------------------------------------------------------+

import numpy as np
import time
import cma


def minimize(cost_func, args, search_space_bound, search_space_size, popsize=None, sigma0=0.5, maxiter=None, maxtime=None, maxevaluations=None, restarts=5, incpopsize=2.0):
    """
    IPOP-CMA-ES optimizer: CMA-ES with increasing population restarts.

    Args:
        cost_func: Objective function that accepts batch of solutions
        args: Additional arguments passed to cost_func
        search_space_bound: Box constraints [-bound, +bound] for all dimensions
        search_space_size: Dimensionality of search space
        popsize: Initial population size (default: None, uses CMA-ES library default)
        sigma0: Initial step size (typically 0.2-0.5 of search range)
        maxiter: Maximum number of iterations
        maxtime: Maximum wall-clock time in seconds
        maxevaluations: Maximum number of function evaluations
        restarts: Number of restarts with increasing population (default: 5)
        incpopsize: Population size multiplier for each restart (default: 2.0)

    Returns:
        gen_best: Best fitness value found
        best_solution: Best solution vector
        convergence_history: List of best fitness at each iteration
        time_history: List of elapsed time at each iteration
        timing_breakdown: Dict with ask_time, eval_time, tell_time
    """

    # --- INITIALIZE ----------------+
    start_time = time.time()
    convergence_history = []
    time_history = []
    evaluations_done = 0

    # Track timing breakdown
    ask_time_total = 0.0
    eval_time_total = 0.0
    tell_time_total = 0.0

    # Track best across all restarts
    global_best_fitness = np.inf
    global_best_solution = None

    # Initial population size (use library default if not specified)
    current_popsize = popsize

    # --- RESTART LOOP ----------------+
    for restart_idx in range(restarts + 1):  # +1 for initial run

        # Initialize CMA-ES for this restart
        x0 = np.zeros(search_space_size)  # Start at origin (center of search space)

        # If maxiter is None, use a very large number so time limit is the constraint
        cmaes_maxiter = maxiter if maxiter is not None else 1000000

        opts = {
            'bounds': [-search_space_bound, search_space_bound],
            'maxiter': cmaes_maxiter,
            'verbose': -9,  # Suppress output
            'verb_disp': 0,
            'verb_log': 0,
            # Disable some internal stopping criteria to respect only our limits
            'tolx': 1e-11,  # Allow convergence based on small x-changes
            'tolfun': 1e-11,  # Allow convergence based on small function value changes
        }

        # Only set popsize if specified by user (otherwise let CMA-ES decide)
        if current_popsize is not None:
            opts['popsize'] = current_popsize

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        # Get actual population size from CMA-ES (needed if popsize was None)
        actual_popsize = es.popsize

        # --- OPTIMIZE WITH ASK-TELL PATTERN ----------------+
        restart_iteration = 0
        while True:
            restart_iteration += 1

            # Check stopping criteria before ask/tell to avoid extra evaluations
            if maxtime is not None and time.time() - start_time > maxtime:
                break
            if maxiter is not None and len(convergence_history) >= maxiter:
                break
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
            evaluations_done += actual_popsize
            eval_time_total += time.time() - eval_start

            # TELL: Update CMA-ES distribution based on fitness values
            tell_start = time.time()
            es.tell(solutions, fitness_values.tolist())
            tell_time_total += time.time() - tell_start

            # --- TRACK CONVERGENCE --------------------------------+
            gen_best = es.result.fbest  # Best fitness so far in this restart
            convergence_history.append(gen_best)
            time_history.append(time.time() - start_time)

            # Update global best across all restarts
            if gen_best < global_best_fitness:
                global_best_fitness = gen_best
                global_best_solution = es.result.xbest

        # --- CHECK IF WE SHOULD DO ANOTHER RESTART ----------------+
        if restart_idx < restarts:
            # Check stopping criteria before restarting
            if maxtime is not None and time.time() - start_time > maxtime:
                break
            if maxiter is not None and len(convergence_history) >= maxiter:
                break
            if maxevaluations is not None and evaluations_done >= maxevaluations:
                break

            # Increase population size for next restart
            # Use actual_popsize (from CMA-ES) in case popsize was None initially
            current_popsize = int(actual_popsize * incpopsize)

    # --- RETURN RESULTS ----------------+
    # Use global best from all restarts
    best_solution = global_best_solution if global_best_solution is not None else es.result.xbest
    best_fitness = global_best_fitness if global_best_fitness != np.inf else es.result.fbest

    # Return timing breakdown
    timing_breakdown = {
        'ask_time': ask_time_total,
        'eval_time': eval_time_total,
        'tell_time': tell_time_total
    }

    return best_fitness, best_solution, convergence_history, time_history, timing_breakdown
