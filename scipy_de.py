# ------------------------------------------------------------------------------+
# SciPy Differential Evolution wrapper for CVAE-Opt
#
# Implements adaptive Differential Evolution using scipy.optimize.differential_evolution
# with adaptive=True (JADE-like self-adaptation of F and CR parameters)
#
# Key features:
# - Adaptive mutation and crossover rates (ignores manual F/CR parameters)
# - Vectorized batch evaluation for GPU efficiency
# - Strategy: 'best1bin' (most common and robust)
# - Compatible with CVAE-Opt's ask-evaluate-tell timing structure
# ------------------------------------------------------------------------------+

from scipy.optimize import differential_evolution
import numpy as np
import time


def minimize(cost_func, args, search_space_bound, search_space_size, popsize,
             mutate, recombination, maxiter, maxtime, maxevaluations=None):
    """
    Minimize using SciPy's adaptive Differential Evolution.

    Args:
        cost_func: Objective function that takes (Z_batch, *args) and returns (tours, costs)
        args: Additional arguments for cost_func (model, config, instance, cost_fn)
        search_space_bound: Symmetric bounds for search space [-bound, +bound]
        search_space_size: Dimensionality of search space
        popsize: Absolute population size (will be converted to SciPy multiplier)
        mutate: Mutation factor F (IGNORED - adaptive=True auto-tunes this)
        recombination: Crossover rate CR (IGNORED - adaptive=True auto-tunes this)
        maxiter: Maximum number of iterations (None = no limit)
        maxtime: Maximum wall-clock time in seconds (None = no limit)
        maxevaluations: Maximum number of function evaluations (None = no limit)

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
    # SciPy's vectorized mode expects a function that takes a 2D array (N_pop, N_dims)
    # and returns a 1D array of costs (N_pop,)
    def vectorized_objective(Z_batch):
        """
        Vectorized objective function for SciPy's differential_evolution.

        Args:
            Z_batch: 2D array of shape (N_population, N_dimensions)

        Returns:
            costs: 1D array of shape (N_population,)
        """
        # Update evaluation counter
        evaluations_done[0] += Z_batch.shape[0]

        # Call the cost function (returns tours, costs)
        _, costs = cost_func(Z_batch, *args)
        return np.array(costs)

    # Set bounds for all dimensions
    bounds = [(-search_space_bound, search_space_bound)] * search_space_size

    # Run SciPy's differential evolution
    # Key parameters:
    # - strategy='best1bin': Most common DE strategy (DE/best/1/bin)
    # - adaptive=True: JADE-like self-adaptation of F and CR (ignores mutation/recombination args)
    # - vectorized=True: Batch evaluation for GPU efficiency
    # - workers=1: Sequential evaluation (parallelization handled by GPU in cost_func)
    optimization_start = time.time()

    try:
        result = differential_evolution(
            func=vectorized_objective,
            bounds=bounds,
            strategy='best1bin',
            adaptive=True,
            vectorized=True,
            popsize=scipy_popsize,
            maxiter=maxiter if maxiter is not None else 1000,  # SciPy requires a value
            callback=callback,
            workers=1,  # No parallelization (GPU handles batching)
            polish=False,  # Disable final polish step (not useful for discrete problems)
            atol=0,  # Disable absolute tolerance stopping
            tol=0.0,  # Disable relative tolerance stopping
            updating='deferred',  # Classic DE (evaluate all candidates before updating)
            seed=None  # Use global random state (controlled by np.random.seed in search.py)
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
