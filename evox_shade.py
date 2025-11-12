# ------------------------------------------------------------------------------+
# EvoX SHADE (Success-History based Adaptive Differential Evolution)
# Implementation using the EvoX optimization library
#
# Reference: https://evox.readthedocs.io/en/latest/apidocs/evox/evox.algorithms.so.de_variants.shade.html
# SHADE: Tanabe, R., & Fukunaga, A. (2013). Success-history based parameter adaptation
# for differential evolution. In 2013 IEEE congress on evolutionary computation (pp. 71-78). IEEE.
# ------------------------------------------------------------------------------+

import numpy as np
import time
import torch
from evox.algorithms.so.de_variants import SHADE
from evox.workflows import StdWorkflow


class CVAEOptProblem(torch.nn.Module):
    """Wrapper to adapt CVAE-Opt cost function to EvoX problem interface."""

    def __init__(self, cost_func, args):
        super().__init__()
        self.cost_func = cost_func
        self.args = args
        # Track best solution across all evaluations
        self.best_fitness = float('inf')
        self.best_solution = None

    def evaluate(self, X):
        """
        Evaluate population X.

        Args:
            X: PyTorch tensor of shape (popsize, search_space_size)

        Returns:
            PyTorch tensor of fitness values, shape (popsize,)
        """
        # Convert PyTorch tensor to numpy for compatibility with PyTorch model
        X_np = X.cpu().numpy()

        # Evaluate using the cost function
        _, costs = self.cost_func(X_np, *self.args)
        costs_array = np.array(costs)

        # Track best solution
        min_idx = np.argmin(costs_array)
        min_cost = costs_array[min_idx]
        if min_cost < self.best_fitness:
            self.best_fitness = min_cost
            self.best_solution = X_np[min_idx]

        # Convert back to PyTorch tensor on the same device as input
        return torch.tensor(costs_array, device=X.device, dtype=torch.float32)


def minimize(cost_func, args, search_space_bound, search_space_size, popsize,
             mutate, recombination, maxiter, maxtime, maxevaluations=None, seed=1234):
    """
    Minimize objective function using EvoX SHADE algorithm.

    SHADE improves upon JADE by maintaining a historical memory of successful
    parameter values (F and CR), leading to better parameter adaptation.

    Args:
        cost_func: Objective function that takes (X_batch, *args) and returns (tours, costs)
        args: Tuple of arguments to pass to cost_func (model, config, instance, cost_fn)
        search_space_bound: Symmetric search space bounds [-bound, +bound]
        search_space_size: Dimensionality of the search space
        popsize: Population size
        mutate: Mutation factor (not used by SHADE, which adapts F internally via history)
        recombination: Crossover rate (not used by SHADE, which adapts CR internally via history)
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

    # Get the device from args
    model, config, instance, cost_fn = args
    device = config.device if hasattr(config, 'device') else torch.device('cpu')

    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set PyTorch default device for EvoX compatibility
    # Save original default device to restore later
    original_device = torch.get_default_device() if hasattr(torch, 'get_default_device') else None
    if device.type == 'cuda':
        torch.set_default_device(device)

    # Create problem wrapper
    problem = CVAEOptProblem(cost_func, args)

    # Initialize SHADE algorithm
    # SHADE uses internal historical memory to store successful F and CR values
    # The EvoX implementation manages historical memory internally
    # Reference: https://evox.readthedocs.io/en/latest/apidocs/evox/evox.algorithms.so.de_variants.shade.html
    lb = torch.full((search_space_size,), -search_space_bound, device=device)
    ub = torch.full((search_space_size,), search_space_bound, device=device)

    algorithm = SHADE(
        pop_size=popsize,
        lb=lb,
        ub=ub,
        device=device
    )

    # Create workflow
    workflow = StdWorkflow(algorithm, problem)

    # Initialize workflow with first evaluation
    workflow.init_step()

    # Main optimization loop
    iteration = 0

    # Track initial evaluation
    evaluations_done += popsize

    # Record initial best (iteration 0)
    convergence_history.append(problem.best_fitness)
    time_history.append(time.time() - start_time)

    while True:
        iteration += 1

        # Check stopping criteria
        if maxtime is not None and time.time() - start_time > maxtime:
            break
        if maxiter is not None and iteration > maxiter:
            break
        if maxevaluations is not None and evaluations_done >= maxevaluations:
            break

        # --- STEP: One iteration of optimization ----------------+
        # EvoX workflow combines ask, evaluate, and tell into step()
        workflow.step()
        evaluations_done += popsize

        # --- SCORE KEEPING ------------------------------------+
        # Record convergence history
        convergence_history.append(problem.best_fitness)
        time_history.append(time.time() - start_time)

    # Restore original default device if changed
    if device.type == 'cuda' and original_device is not None:
        torch.set_default_device(original_device)

    # Get final best solution from problem
    best_fitness = problem.best_fitness
    best_solution = problem.best_solution

    # Calculate total algorithm time
    total_time = time.time() - start_time

    # Prepare timing breakdown
    timing_breakdown = {
        'ask_time': 0.0,  # Not tracked separately
        'eval_time': total_time,  # Total time reported as eval_time for consistency
        'tell_time': 0.0,  # Not tracked separately
        'iterations': iteration
    }

    return best_fitness, best_solution, convergence_history, time_history, timing_breakdown
