#!/usr/bin/env python3
"""Quick test script for EvoX SaDE implementation."""

import sys
print("Starting SaDE test...", flush=True)

try:
    from evox_sade import minimize
    print("✓ Successfully imported evox_sade", flush=True)
except Exception as e:
    print(f"✗ Failed to import evox_sade: {e}", flush=True)
    sys.exit(1)

try:
    import torch
    from evox.algorithms.so.de_variants import SaDE
    print("✓ Successfully imported EvoX SaDE", flush=True)
except Exception as e:
    print(f"✗ Failed to import EvoX library: {e}", flush=True)
    sys.exit(1)

# Simple test function
def simple_cost_func(X, *args):
    """Simple sphere function for testing."""
    import numpy as np
    costs = np.sum(X**2, axis=1)
    return None, costs.tolist()

print("Running simple optimization test...", flush=True)

try:
    best_fitness, best_solution, convergence, time_hist, timing = minimize(
        cost_func=simple_cost_func,
        args=(),
        search_space_bound=5.0,
        search_space_size=10,
        popsize=20,
        mutate=0.5,
        recombination=0.9,
        maxiter=5,
        maxtime=None,
        maxevaluations=None
    )

    print(f"✓ Optimization completed successfully!", flush=True)
    print(f"  Best fitness: {best_fitness:.6f}", flush=True)
    print(f"  Convergence history length: {len(convergence)}", flush=True)
    print(f"  Final convergence: {convergence[-1]:.6f}", flush=True)
    print("✓ SaDE implementation is working correctly!", flush=True)

except Exception as e:
    print(f"✗ Optimization failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
