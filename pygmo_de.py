# ------------------------------------------------------------------------------+
# Pygmo Differential Evolution optimizer
# Wrapper for the pygmo library's DE algorithm with batch evaluation
#
# Uses ask-tell pattern compatible with the existing DE implementation
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
        batch_result = self.batch_fitness(np.array([x]))
        return [batch_result[0]]

    def batch_fitness(self, solutions_batch):
        """
        Evaluates a batch of solutions.

        Args:
            solutions_batch: NumPy array of shape (popsize, search_space_size)

        Returns:
            list: Fitness values for each solution in the batch
        """
        # cost_func returns (tours, costs) - we only need costs
        _, costs = self.cost_func(solutions_batch, *self.args)
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
    Differential Evolution optimizer using pygmo library.

    Uses gen=1 in a manual loop to enable iteration-by-iteration tracking
    of convergence history and time history, matching the interface of
    the existing DE and CMA-ES optimizers.

    Args:
        cost_func: Objective function that accepts batch of solutions
        args: Additional arguments passed to cost_func (tuple)
        search_space_bound: Box constraints [-bound, +bound] for all dimensions
        search_space_size: Dimensionality of search space
        popsize: Population size
        mutate: Mutation factor F (typical range: 0.1-1.0)
        recombination: Crossover rate CR (typical range: 0.0-1.0)
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

    # 2. Create the pygmo algorithm (DE)
    # We set gen=1 to manually control generations for iteration-by-iteration tracking
    # variant=6 corresponds to "rand/1/bin" strategy, matching de.py
    # mutate -> F (mutation factor)
    # recombination -> CR (crossover rate)
    de_algo = pg.de(gen=1, F=mutate, CR=recombination, variant=6, ftol=1e-6, xtol=1e-6)
    algo = pg.algorithm(de_algo)

    # 3. Create the initial population
    # pygmo creates a random initial population and evaluates it
    pop = pg.population(prob=prob, size=popsize)

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
        # - ASK: Generate candidate solutions (mutation + crossover)
        # - EVALUATE: Batch evaluation via our batch_fitness method
        # - TELL: Update population based on fitness values
        ask_start = time.time()
        pop = algo.evolve(pop)
        ask_time_total += time.time() - ask_start

        evaluations_done += popsize

        # --- SCORE KEEPING ----------------------------------+
        gen_best = pop.champion_f[0]  # Best fitness in current population
        convergence_history.append(gen_best)
        time_history.append(time.time() - start_time)

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
