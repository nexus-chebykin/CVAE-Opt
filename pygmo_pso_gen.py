# ------------------------------------------------------------------------------+
# Pygmo PSO Generational optimizer
# Wrapper for the pygmo library's PSO_gen algorithm with batch evaluation
#
# Uses pygmo's Particle Swarm Optimization (generational variant) with batch
# fitness evaluation via set_bfe() for efficient GPU-accelerated neural network
# inference.
# ------------------------------------------------------------------------------+

import numpy as np
import time
import pygmo as pg


class PygmoProblem:
    """
    Wrapper class that adapts the cost function to pygmo's problem interface.
    Enables batch fitness evaluation for efficient neural network inference.
    """

    def __init__(self, cost_func, args, search_space_bound, search_space_size):
        """
        Args:
            cost_func: Objective function that accepts batch of solutions
            args: Additional arguments passed to cost_func (tuple)
            search_space_bound: Box constraints [-bound, +bound] for all dimensions
            search_space_size: Dimensionality of search space
        """
        self.cost_func = cost_func
        self.args = args
        self.search_space_bound = search_space_bound
        self.search_space_size = search_space_size

    def get_bounds(self):
        """
        Define the search space bounds for each dimension.

        Returns:
            tuple: (lower_bounds, upper_bounds) as lists
        """
        lower_bounds = [-self.search_space_bound] * self.search_space_size
        upper_bounds = [self.search_space_bound] * self.search_space_size
        return (lower_bounds, upper_bounds)

    def has_batch_fitness(self):
        """
        Tells pygmo that our fitness function can handle batch evaluation.

        Returns:
            bool: True, indicating batch fitness evaluation is supported
        """
        return True

    def fitness(self, x):
        """
        Evaluates a single solution (mandatory for pygmo).
        Wraps the single solution in a batch and calls batch_fitness.

        Args:
            x: Single solution vector

        Returns:
            list: Single-element list containing the fitness value
        """
        # Call batch_fitness with a single solution
        # batch_fitness expects a flattened 1D array, so we pass x directly
        batch_result = self.batch_fitness(x)
        return [batch_result[0]]

    def batch_fitness(self, solutions_batch):
        """
        Evaluates a batch of solutions.

        Args:
            solutions_batch: Flattened array from pygmo (popsize * dimensions elements)

        Returns:
            list: Fitness values for each solution in the batch
        """
        # CRITICAL FIX: Pygmo passes batch_fitness as a flattened 1D array
        # We need to reshape it to (popsize, search_space_size)
        solutions_array = np.asarray(solutions_batch, dtype=float)

        # Reshape from 1D to 2D: (popsize * dimensions,) -> (popsize, dimensions)
        num_solutions = len(solutions_array) // self.search_space_size
        solutions_2d = solutions_array.reshape(num_solutions, self.search_space_size)

        # FIX: Flatten GRU parameters before every evaluation
        # Pygmo's internal serialization can cause GRU weights to lose contiguous memory layout.
        # We restore it here before each batch evaluation to prevent PyTorch warnings.
        # The model is passed as args[0] in the cost_func arguments tuple.
        if len(self.args) > 0:
            model = self.args[0]
            # Check if model has modules() method (it's a PyTorch model)
            if hasattr(model, 'modules'):
                import torch.nn as nn
                for module in model.modules():
                    if isinstance(module, nn.GRU):
                        module.flatten_parameters()

        # cost_func returns (tours, costs) - we only need costs
        _, costs = self.cost_func(solutions_2d, *self.args)
        return costs


def minimize(
    cost_func,
    args,
    search_space_bound,
    search_space_size,
    popsize,
    mutate,
    recombination,
    maxiter,
    maxtime,
    maxevaluations=None,
):
    """
    PSO Generational optimizer using pygmo library.

    Uses gen=1 in a manual loop to enable iteration-by-iteration tracking
    of convergence history and time history, matching the interface of
    the existing DE and CMA-ES optimizers.

    Args:
        cost_func: Objective function that accepts batch of solutions
        args: Additional arguments passed to cost_func (tuple)
        search_space_bound: Box constraints [-bound, +bound] for all dimensions
        search_space_size: Dimensionality of search space
        popsize: Population size (swarm size)
        mutate: Not used by PSO (kept for interface compatibility)
        recombination: Not used by PSO (kept for interface compatibility)
        maxiter: Maximum number of iterations
        maxtime: Maximum wall-clock time in seconds
        maxevaluations: Maximum number of function evaluations (optional)

    Returns:
        best_fitness: Best fitness value found
        best_solution: Best solution vector
        convergence_history: List of best fitness at each iteration
        time_history: List of elapsed time at each iteration
        timing_breakdown: Dict with ask_time, eval_time, tell_time, iterations
    """

    # --- INITIALIZE -------------------------------------+
    # Start timing from the beginning (including initial population evaluation)
    # This ensures maxtime limits total wall-clock runtime, matching user expectations
    start_time = time.time()
    convergence_history = []
    time_history = []
    evaluations_done = 0

    # Track timing breakdown
    # Note: pygmo's evolve() combines ask-evaluate-tell into a single call,
    # so we track the total evolution time as "ask_time" for consistency
    ask_time_total = 0.0
    eval_time_total = 0.0  # Included in ask_time for pygmo
    tell_time_total = 0.0  # Included in ask_time for pygmo

    # 1. Create the pygmo problem wrapper
    problem = PygmoProblem(cost_func, args, search_space_bound, search_space_size)
    prob = pg.problem(problem)

    # 2. Create the batch fitness evaluator
    # pg.bfe() uses single-process evaluation, perfect for GPU batch inference
    # This MUST be created before the population to enable batch evaluation during initialization
    bfe = pg.bfe()

    # 3. Create the initial population with batch evaluation enabled
    # CRITICAL: Pass the bfe to population constructor to enable batch evaluation during initialization
    # Without this, the initial population will be evaluated one-by-one (600 individual calls)
    pop = pg.population(prob=prob, size=popsize, b=bfe, seed=1234)

    # 4. Create the pygmo algorithm (PSO_gen)
    # We set gen=1 to manually control generations for iteration-by-iteration tracking
    # PSO_gen default parameters:
    # - omega (inertia weight): 0.7298
    # - eta1 (social component): 2.05
    # - eta2 (cognitive component): 2.05
    # - max_vel: 0.5 (maximum allowed particle velocity)
    # - variant: 6 (PSO variant - 6 is canonical/global best, more standard than variant 5)
    pso_algo = pg.pso_gen(gen=1, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5, variant=6, seed=1234)

    # CRITICAL: Attach batch fitness evaluator to the algorithm for evolution
    # This enables true batch processing during the evolve() call
    pso_algo.set_bfe(bfe)

    algo = pg.algorithm(pso_algo)

    # Track the initial best fitness (from initial population evaluation)
    gen_best = pop.champion_f[0]
    convergence_history.append(gen_best)
    time_history.append(time.time() - start_time)
    evaluations_done += popsize

    # --- EVOLUTION LOOP (replaces ask-tell) -------------+
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

        # Single evolve() call performs one complete generation:
        # - UPDATE VELOCITIES: PSO updates particle velocities based on personal/global best
        # - UPDATE POSITIONS: Move particles according to velocities
        # - EVALUATE: Batch evaluation via our batch_fitness method (enabled by set_bfe)
        # - UPDATE BEST: Update personal and global best positions
        ask_start = time.time()
        pop = algo.evolve(pop)
        ask_time_total += time.time() - ask_start

        evaluations_done += popsize

        # --- SCORE KEEPING ----------------------------------+
        gen_best = pop.champion_f[0]  # Best fitness in current population
        convergence_history.append(gen_best)
        time_history.append(time.time() - start_time)

        # Check if we've exceeded maxtime after completing this iteration
        # This prevents starting another expensive iteration when already over budget
        if maxtime is not None and time.time() - start_time > maxtime:
            break

    # --- RETURN RESULTS ---------------------------------+
    best_fitness = pop.champion_f[0]
    best_solution = pop.champion_x

    # Create timing breakdown for compatibility with other optimizers
    # Note: In pygmo, the evolve() call combines all three phases
    timing_breakdown = {
        "ask_time": ask_time_total,
        "eval_time": eval_time_total,  # Not separately tracked in pygmo
        "tell_time": tell_time_total,  # Not separately tracked in pygmo
        "iterations": iteration,
    }

    return (
        best_fitness,
        best_solution,
        convergence_history,
        time_history,
        timing_breakdown,
    )
