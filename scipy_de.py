# ------------------------------------------------------------------------------+
# SciPy Differential Evolution wrapper for CVAE-Opt
#
# Implements Differential Evolution using scipy.optimize.differential_evolution
# with adaptive mutation dithering for automatic F tuning
#
# Key features:
# - Adaptive mutation via dithering (F varies between 0.5-1.0 automatically)
# - Fixed high crossover rate (CR = 0.9, recommended value)
# - Vectorized batch evaluation for GPU efficiency
# - Strategy: 'best1bin' (most common and robust)
# - Compatible with CVAE-Opt's ask-evaluate-tell timing structure
# ------------------------------------------------------------------------------+

from scipy.optimize import differential_evolution
import numpy as np
import time


def minimize(cost_func, args, search_space_bound, search_space_size, popsize,
             mutate, recombination, maxiter, maxtime, maxevaluations=None,
             strategy='best1bin', mutation=(0.5, 1.0), recombination_scipy=0.9,
             updating='deferred', seed=None):
    """
    Minimize using SciPy's Differential Evolution with tunable parameters.

    Args:
        cost_func: Objective function that takes (Z_batch, *args) and returns (tours, costs)
        args: Additional arguments for cost_func (model, config, instance, cost_fn)
        search_space_bound: Symmetric bounds for search space [-bound, +bound]
        search_space_size: Dimensionality of search space
        popsize: Absolute population size (will be converted to SciPy multiplier)
        mutate: Mutation factor F (IGNORED - use mutation parameter instead)
        recombination: Crossover rate CR (IGNORED - use recombination_scipy parameter instead)
        maxiter: Maximum number of iterations (None = no limit)
        maxtime: Maximum wall-clock time in seconds (None = no limit)
        maxevaluations: Maximum number of function evaluations (None = no limit)
        strategy: DE mutation strategy (e.g., 'best1bin', 'rand1bin', default: 'best1bin')
        mutation: Mutation factor (float) or dithering range (tuple), default: (0.5, 1.0)
        recombination_scipy: Crossover probability (default: 0.9)
        updating: Population update strategy ('deferred' or 'immediate', default: 'deferred')

    Returns:
        best_cost: Best objective value found
        best_solution: Best solution vector (1D array)
        convergence_history: List of best fitness per iteration
        time_history: List of elapsed time per iteration
        timing_breakdown: Dict with 'ask_time', 'eval_time', 'tell_time', 'iterations'
    """

    # Convert absolute batch_size to SciPy's popsize multiplier
    # SciPy uses: total_population = popsize * search_space_size
    scipy_popsize = max(1, popsize // search_space_size)

    # Tracking variables
    start_time = time.time()
    convergence_history = []
    time_history = []
    iteration_count = [0]  # Use list for mutability in callback
    evaluations_done = [0]  # Track total evaluations
    stop_flag = {'stop': False, 'reason': None}  # Flag to stop optimization

    # Callback function to capture convergence history per iteration
    def callback(xk, convergence=None):
        """
        Callback function called after each iteration by SciPy's differential_evolution.

        Args:
            xk: Current best solution vector
            convergence: Convergence metric (not used here)

        Returns:
            True to stop optimization, False to continue
        """
        iteration_count[0] += 1
        elapsed_time = time.time() - start_time

        # Evaluate current best to get its cost
        _, costs = cost_func(xk.reshape(1, -1), *args)
        best_cost = costs[0]

        convergence_history.append(best_cost)
        time_history.append(elapsed_time)

        # Increment evaluation counter (callback evaluates best solution)
        evaluations_done[0] += 1

        # Check stopping criteria
        # Time limit check
        if maxtime is not None and elapsed_time > maxtime:
            stop_flag['stop'] = True
            stop_flag['reason'] = 'maxtime'
            return True

        # Iteration limit check
        if maxiter is not None and iteration_count[0] >= maxiter:
            stop_flag['stop'] = True
            stop_flag['reason'] = 'maxiter'
            return True

        # Evaluation limit check (approximate, as SciPy doesn't expose exact count)
        # Each iteration evaluates approximately scipy_popsize * search_space_size individuals
        if maxevaluations is not None and evaluations_done[0] >= maxevaluations:
            stop_flag['stop'] = True
            stop_flag['reason'] = 'maxevaluations'
            return True

        return False  # Continue optimization

    # Wrapper for vectorized cost function
    # SciPy's vectorized mode may pass arrays in different orientations depending on version
    def vectorized_objective(Z_batch):
        """
        Vectorized objective function for SciPy's differential_evolution.

        Args:
            Z_batch: 2D array, either (N_population, N_dimensions) or (N_dimensions, N_population)

        Returns:
            costs: 1D array of shape (N_population,)
        """
        # SciPy's vectorized mode may pass Z_batch as (N_dims, N_pop) in some cases
        # Ensure it's (N_pop, N_dims) by checking which dimension matches search_space_size
        if Z_batch.ndim == 2:
            if Z_batch.shape[1] == search_space_size:
                # Already correct shape: (N_pop, N_dims)
                pass
            elif Z_batch.shape[0] == search_space_size:
                # Transposed: (N_dims, N_pop) -> transpose to (N_pop, N_dims)
                Z_batch = Z_batch.T
            else:
                # Neither dimension matches - this shouldn't happen but use as-is
                pass

        # Update evaluation counter
        evaluations_done[0] += Z_batch.shape[0]

        # Call the cost function (returns tours, costs)
        _, costs = cost_func(Z_batch, *args)
        return np.array(costs)

    # Set bounds for all dimensions
    bounds = [(-search_space_bound, search_space_bound)] * search_space_size

    # Set random seed for reproducibility
    np.random.seed(seed if seed is not None else 1234)

    # Evaluate initial population (full batch) for iteration 0 baseline
    # This ensures scipy_de starts from the same quality as other optimizers
    initial_population = np.random.uniform(-search_space_bound, search_space_bound, (popsize, search_space_size))
    _, initial_costs = cost_func(initial_population, *args)
    initial_costs_array = np.array(initial_costs)
    initial_cost = np.min(initial_costs_array)  # Best of initial population
    convergence_history.append(initial_cost)
    time_history.append(time.time() - start_time)
    evaluations_done[0] += popsize  # Track all evaluations

    # Run SciPy's differential evolution
    # Key parameters:
    # - strategy='best1bin': Most common DE strategy (DE/best/1/bin)
    # - mutation=(0.5, 1.0): Adaptive dithering - F varies randomly between 0.5 and 1.0 each generation
    # - recombination=0.9: Fixed high crossover rate (tuple form not supported by SciPy)
    # - vectorized=True: Batch evaluation for GPU efficiency
    # - workers=1: Sequential evaluation (parallelization handled by GPU in cost_func)
    optimization_start = time.time()

    try:
        result = differential_evolution(
            func=vectorized_objective,
            bounds=bounds,
            strategy=strategy,  # Tunable mutation strategy
            mutation=mutation,  # Tunable mutation factor (float or tuple for dithering)
            recombination=recombination_scipy,  # Tunable crossover rate
            vectorized=True,
            popsize=scipy_popsize,
            maxiter=maxiter if maxiter is not None else 1000,  # SciPy requires a value
            callback=callback,
            workers=1,  # No parallelization (GPU handles batching)
            polish=False,  # Disable final polish step (not useful for discrete problems)
            atol=1000.0,  # Set very high to prevent convergence-based stopping
            tol=1000.0,   # Convergence check: std(pop) <= atol + tol*|mean(pop)|. High values prevent early stop.
            updating=updating,  # Tunable update strategy ('deferred' or 'immediate')
            init='latinhypercube',  # FIXED: Use Latin Hypercube Sampling for initialization
            seed=seed  # Use provided seed (passed from config.seed)
        )
    except StopIteration:
        # SciPy uses StopIteration internally, catch it if raised
        pass

    optimization_time = time.time() - optimization_start

    # Extract results
    best_solution = result.x
    best_cost = result.fun

    # If convergence_history is empty (optimization stopped before first callback),
    # add at least the final result
    if len(convergence_history) == 0:
        convergence_history.append(best_cost)
        time_history.append(time.time() - start_time)
        iteration_count[0] = 1

    # Timing breakdown: Follow pygmo_de pattern (all time in ask_time, eval/tell = 0)
    # SciPy's DE combines ask-evaluate-tell internally, so we can't separate phases
    timing_breakdown = {
        'ask_time': optimization_time,
        'eval_time': 0.0,
        'tell_time': 0.0,
        'iterations': iteration_count[0]
    }

    return best_cost, best_solution, convergence_history, time_history, timing_breakdown
