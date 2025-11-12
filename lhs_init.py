"""
Latin Hypercube Sampling (LHS) initialization utility for optimizers.

This module provides a shared function to generate initial populations using
Latin Hypercube Sampling, which provides better space coverage than uniform
random sampling.
"""

import numpy as np
from scipy.stats import qmc


def generate_lhs_population(num_samples, dimension, lower_bound, upper_bound, seed=None):
    """
    Generate a population using Latin Hypercube Sampling.

    Latin Hypercube Sampling (LHS) is a stratified sampling method that ensures
    better coverage of the search space compared to uniform random sampling by
    dividing each dimension into equal intervals and sampling once from each interval.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate (population size)
    dimension : int
        Dimensionality of the search space
    lower_bound : float or array-like
        Lower bound(s) of the search space. Can be:
        - A scalar (same bound for all dimensions)
        - An array of shape (dimension,) with per-dimension bounds
    upper_bound : float or array-like
        Upper bound(s) of the search space. Can be:
        - A scalar (same bound for all dimensions)
        - An array of shape (dimension,) with per-dimension bounds
    seed : int, optional
        Random seed for reproducibility. If None, uses random state.

    Returns
    -------
    population : np.ndarray
        Array of shape (num_samples, dimension) containing LHS samples
        scaled to [lower_bound, upper_bound]

    Examples
    --------
    >>> # Generate 100 samples in 10D space with symmetric bounds [-5, 5]
    >>> pop = generate_lhs_population(100, 10, -5.0, 5.0, seed=1234)
    >>> pop.shape
    (100, 10)

    >>> # Generate 50 samples with asymmetric bounds
    >>> lb = np.array([-5, -10, -3])
    >>> ub = np.array([5, 10, 7])
    >>> pop = generate_lhs_population(50, 3, lb, ub, seed=42)
    >>> pop.shape
    (50, 3)
    """
    # Convert bounds to numpy arrays for consistency
    if np.isscalar(lower_bound):
        lower_bound = np.full(dimension, lower_bound, dtype=np.float64)
    else:
        lower_bound = np.asarray(lower_bound, dtype=np.float64)

    if np.isscalar(upper_bound):
        upper_bound = np.full(dimension, upper_bound, dtype=np.float64)
    else:
        upper_bound = np.asarray(upper_bound, dtype=np.float64)

    # Validate bounds
    if lower_bound.shape != (dimension,):
        raise ValueError(f"lower_bound must be scalar or have shape ({dimension},), got {lower_bound.shape}")
    if upper_bound.shape != (dimension,):
        raise ValueError(f"upper_bound must be scalar or have shape ({dimension},), got {upper_bound.shape}")
    if np.any(lower_bound >= upper_bound):
        raise ValueError("lower_bound must be strictly less than upper_bound for all dimensions")

    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)

    # Generate samples in [0, 1]^d
    lhs_samples = sampler.random(n=num_samples)

    # Scale samples to [lower_bound, upper_bound]
    # Formula: x_scaled = lb + (ub - lb) * x_unit
    population = lower_bound + (upper_bound - lower_bound) * lhs_samples

    return population
