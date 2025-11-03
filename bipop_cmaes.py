# ------------------------------------------------------------------------------+
# BIPOP-CMA-ES (Bi-Population CMA-ES) optimizer
# CMA-ES with restart strategy alternating between small and large populations
#
# Uses batch evaluation pattern compatible with the existing DE implementation
# ------------------------------------------------------------------------------+

import numpy as np
import time
import cma


def minimize(cost_func, args, search_space_bound, search_space_size, popsize=None, sigma0=0.5, maxiter=None, maxtime=None, maxevaluations=None, restarts=5, incpopsize=2.0, maxpopsize=None):
    """
    BIPOP-CMA-ES optimizer: CMA-ES with bi-population restart strategy.

    Alternates between small (exploration) and large (exploitation) populations
    with budget management to balance exploration and exploitation.

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
        restarts: Number of restarts (default: 5)
        incpopsize: Population size multiplier for large population restarts (default: 2.0)
        maxpopsize: Maximum population size (default: None, no limit)

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

    # --- BIPOP PARAMETERS ----------------+
    budget_small = 0  # Total evaluations used by small population runs
    budget_large = 0  # Total evaluations used by large population runs

    # If popsize not specified, CMA-ES will determine it on first run
    # We'll initialize popsize_large and popsize_small after first CMA-ES creation
    popsize_large = popsize  # Current large population size (can be None initially)
    popsize_small = int(popsize / 2) if popsize is not None else None  # Initial small population size

    restart_idx = 0
    use_large_population = True  # Start with large population
    first_run = True  # Track first run to initialize population sizes

    # --- RESTART LOOP ----------------+
    while restart_idx <= restarts:

        # --- DETERMINE POPULATION SIZE FOR THIS RESTART ----------------+
        if use_large_population:
            current_popsize = popsize_large
            current_sigma = sigma0  # Use standard sigma for large populations
            x0 = np.zeros(search_space_size)  # Start at origin for large populations
        else:
            # Small population with randomization for exploration
            current_popsize = popsize_small
            # Vary sigma for small populations (exploration)
            current_sigma = sigma0 * (0.5 + np.random.rand())
            # Random initial point for small populations
            x0 = np.random.uniform(-search_space_bound, search_space_bound, search_space_size)

        # --- INITIALIZE CMA-ES FOR THIS RESTART ----------------+
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

        # Only set popsize if specified (otherwise let CMA-ES decide)
        if current_popsize is not None:
            opts['popsize'] = current_popsize

        # Set maximum population size if specified
        if maxpopsize is not None:
            opts['maxpopsize'] = maxpopsize

        es = cma.CMAEvolutionStrategy(x0, current_sigma, opts)

        # Get actual population size from CMA-ES (needed if popsize was None)
        actual_popsize = es.popsize

        # Initialize popsize_large and popsize_small on first run if they were None
        if first_run and popsize is None:
            popsize_large = actual_popsize
            popsize_small = int(actual_popsize / 2)
            first_run = False

        # Track evaluations used in this restart
        restart_evals = 0

        # --- OPTIMIZE WITH ASK-TELL PATTERN ----------------+
        while True:

            # Check stopping criteria before ask/tell to avoid extra evaluations
            if maxtime is not None and time.time() - start_time > maxtime:
                break
            if maxiter is not None and len(convergence_history) >= maxiter:
                break
            if maxevaluations is not None and evaluations_done >= maxevaluations:
                break

            # Check CMA-ES internal stopping criteria (convergence)
            if es.stop():
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
            restart_evals += actual_popsize
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

        # --- UPDATE BUDGETS ----------------+
        if use_large_population:
            budget_large += restart_evals
        else:
            budget_small += restart_evals

        # --- CHECK IF WE SHOULD DO ANOTHER RESTART ----------------+
        if restart_idx < restarts:
            # Check stopping criteria before restarting
            if maxtime is not None and time.time() - start_time > maxtime:
                break
            if maxiter is not None and len(convergence_history) >= maxiter:
                break
            if maxevaluations is not None and evaluations_done >= maxevaluations:
                break

            # --- DECIDE NEXT RESTART STRATEGY (BIPOP LOGIC) ----------------+
            # Run small populations until budget_small >= budget_large
            # Then switch to large population and increase its size
            if budget_small < budget_large:
                # Use small population
                use_large_population = False
                # Randomly vary small population size (exploration)
                # Use the initial popsize_large (determined from first run) as base
                popsize_small = int(popsize_large * (0.5 + 0.5 * np.random.rand()))
                # Ensure minimum population size
                if popsize_small < 4:
                    popsize_small = 4
            else:
                # Use large population and increase its size (IPOP-style)
                use_large_population = True
                popsize_large = int(popsize_large * incpopsize)

        restart_idx += 1

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
