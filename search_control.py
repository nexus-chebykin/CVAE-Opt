import torch
import tsp, cvrp
import numpy as np
import time
import logging
import os
from plotting import *

# Track last batch size to avoid unnecessary decoder resets
_last_batch_size = None


def decode(Z, model, config, instance, cost_fn):
    global _last_batch_size
    Z = torch.Tensor(Z).to(config.device)
    batch_size = Z.shape[0]

    # Check if population size exceeds pre-expanded instance size
    if batch_size > instance.shape[0]:
        logging.warning(f"Population size ({batch_size}) exceeds pre-allocated instance size ({instance.shape[0]}). Re-expanding instance tensor.")
        # Get original instance (first element of the batch)
        original_instance = instance[0]
        # Re-expand to new batch size
        instance = original_instance.unsqueeze(0).expand(batch_size, -1, -1)
        instance_batch = instance
    else:
        # Slice instance to match Z's batch size (for variable population sizes)
        instance_batch = instance[:batch_size]

    # Reset decoder only when batch size changes (typically once per restart, not per iteration)
    if batch_size != _last_batch_size:
        model.reset_decoder(batch_size, config)
        _last_batch_size = batch_size

    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance_batch, Z, config)
    costs = cost_fn(instance_batch, tour_idx)
    return tour_idx, costs.tolist()


def evaluate(Z, model, config, instance, cost_fn):
    global _last_batch_size
    Z = torch.Tensor(Z).to(config.device)
    batch_size = Z.shape[0]

    # Check if population size exceeds pre-expanded instance size
    if batch_size > instance.shape[0]:
        logging.warning(f"Population size ({batch_size}) exceeds pre-allocated instance size ({instance.shape[0]}). Re-expanding instance tensor.")
        # Get original instance (first element of the batch)
        original_instance = instance[0]
        # Re-expand to new batch size
        instance = original_instance.unsqueeze(0).expand(batch_size, -1, -1)
        instance_batch = instance
    else:
        # Slice instance to match Z's batch size (for variable population sizes)
        instance_batch = instance[:batch_size]

    # Reset decoder only when batch size changes (typically once per restart, not per iteration)
    if batch_size != _last_batch_size:
        model.reset_decoder(batch_size, config)
        _last_batch_size = batch_size

    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance_batch, Z, config)
    costs = cost_fn(instance_batch, tour_idx)
    return costs.tolist()


def solve_instance(model, instance, config, cost_fn, batch_size, sigma0=None,
                   override_maxiter=None, override_maxtime=None, override_maxevaluations=None):
    """
    Solve a single instance using the configured optimizer (DE, CMA-ES, IPOP-CMA-ES, or BIPOP-CMA-ES).

    Args:
        model: Neural network model
        instance: Problem instance to solve
        config: Configuration object with optimizer settings
        cost_fn: Cost function for evaluating tours
        batch_size: Population size for the optimizer
        sigma0: Optional sigma0 for CMA-ES (overrides config.cmaes_sigma0 if provided)
        override_maxiter: Optional override for maximum iterations (if None, uses config.search_iterations)
        override_maxtime: Optional override for maximum time (if None, uses config.search_timelimit)
        override_maxevaluations: Optional override for maximum evaluations (if None, uses config.search_evaluations)

    Returns:
        result_cost: Best objective value found
        solution: Best solution (tour)
        convergence_history: History of best fitness per iteration
        time_history: History of elapsed time per iteration
    """
    global _last_batch_size

    # Reset batch size tracking for this instance
    _last_batch_size = None

    # Pre-expand instance to 3x batch_size to accommodate CoDE (3 strategies × popsize)
    # Other optimizers will slice to their actual needs (safe due to expand() creating view)
    instance = torch.Tensor(instance)
    instance = instance.unsqueeze(0).expand(batch_size * 3, -1, -1)
    instance = instance.to(config.device)
    model.reset_decoder(batch_size * 3, config)

    # Update tracking (decoder was just reset to batch_size * 3)
    _last_batch_size = batch_size * 3

    # Determine stopping criteria (use overrides if provided, otherwise use config values)
    maxiter = override_maxiter if override_maxiter is not None else config.search_iterations
    maxtime = override_maxtime if override_maxtime is not None else config.search_timelimit
    maxevaluations = override_maxevaluations if override_maxevaluations is not None else config.search_evaluations

    # Select optimizer based on config
    if config.optimizer == 'de':
        from de import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            seed=config.seed
        )
    elif config.optimizer == 'de_vectorized':
        from de_vectorized import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            seed=config.seed
        )
    elif config.optimizer == 'de_on_steroids':
        from de_on_steroids import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.steroids_mutate,
            recombination=config.steroids_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            seed=config.seed,
            strategy=config.steroids_strategy
        )
    elif config.optimizer == 'cmaes':
        from cmaes import minimize
        # Use provided sigma0 or fall back to config value
        cmaes_sigma = sigma0 if sigma0 is not None else config.cmaes_sigma0
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            sigma0=cmaes_sigma,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            use_lhs=True,  # CMA-ES always uses LHS for better initialization
            CMA_rankmu=config.cmaes_rankmu,
            CMA_rankone=config.cmaes_rankone,
            seed=config.seed
        )
    elif config.optimizer == 'ipop_cmaes':
        from ipop_cmaes import minimize
        # Use IPOP-specific parameters if provided, otherwise fall back to CMA-ES defaults
        ipop_sigma = sigma0 if sigma0 is not None else (config.ipop_sigma0 if config.ipop_sigma0 is not None else config.cmaes_sigma0)
        ipop_rankmu = config.ipop_rankmu if config.ipop_rankmu is not None else config.cmaes_rankmu
        ipop_rankone = config.ipop_rankone if config.ipop_rankone is not None else config.cmaes_rankone
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=config.ipop_initial_popsize,  # Can be None (uses CMA-ES default)
            sigma0=ipop_sigma,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            restarts=config.ipop_restarts,
            incpopsize=config.ipop_incpopsize,
            use_lhs=True,  # IPOP-CMA-ES always uses LHS for better initialization
            CMA_rankmu=ipop_rankmu,
            CMA_rankone=ipop_rankone,
            seed=config.seed
        )
    elif config.optimizer == 'bipop_cmaes':
        from bipop_cmaes import minimize
        # Use BIPOP-specific parameters if provided, otherwise fall back to CMA-ES defaults
        bipop_sigma = sigma0 if sigma0 is not None else (config.bipop_sigma0 if config.bipop_sigma0 is not None else config.cmaes_sigma0)
        bipop_rankmu = config.bipop_rankmu if config.bipop_rankmu is not None else config.cmaes_rankmu
        bipop_rankone = config.bipop_rankone if config.bipop_rankone is not None else config.cmaes_rankone
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=config.bipop_initial_popsize,  # Can be None (uses CMA-ES default)
            sigma0=bipop_sigma,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            restarts=config.bipop_restarts,
            incpopsize=config.bipop_incpopsize,
            use_lhs=True,  # BIPOP-CMA-ES always uses LHS for better initialization
            CMA_rankmu=bipop_rankmu,
            CMA_rankone=bipop_rankone,
            seed=config.seed
        )
    elif config.optimizer == 'scipy_de':
        from scipy_de import minimize
        # Prepare mutation parameter: if adaptive, use (low, high) tuple, otherwise use single value
        if config.scipy_use_adaptive_mutation:
            mutation_param = (config.scipy_mutation_low, config.scipy_mutation_high)
        else:
            mutation_param = config.de_mutate  # Fall back to standard DE mutation
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            strategy=config.scipy_strategy,
            mutation=mutation_param,
            recombination_scipy=config.de_recombine,  # Use de_recombine for scipy's recombination
            updating=config.scipy_updating,
            seed=config.seed
        )
    elif config.optimizer == 'evox_jade':
        from evox_jade import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            jade_c=config.jade_c,
            jade_num_diff_vectors=config.jade_num_diff_vectors,
            jade_mean=config.jade_mean,
            jade_stdev=config.jade_stdev,
            seed=config.seed
        )
    elif config.optimizer == 'pygmo_pso_gen':
        from pygmo_pso_gen import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            use_lhs=True,  # PSO always uses LHS for better initialization
            seed=config.seed
        )
    elif config.optimizer == 'evox_shade':
        from evox_shade import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            diff_padding_num=config.shade_diff_padding_num,
            seed=config.seed
        )
    elif config.optimizer == 'evox_sade':
        from evox_sade import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            diff_padding_num=config.sade_diff_padding_num,
            LP=config.sade_lp,
            seed=config.seed
        )
    elif config.optimizer == 'evox_code':
        from evox_code import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            diff_padding_num=config.code_diff_padding_num,
            replace=config.code_replace,
            seed=config.seed
        )
    elif config.optimizer == 'evox_ode':
        from evox_ode import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,  # Not used by ODE, but required by interface
            recombination=config.de_recombine,  # Not used by ODE, but required by interface
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            base_vector=config.ode_base_vector,
            num_difference_vectors=config.ode_num_difference_vectors,
            differential_weight=config.ode_differential_weight,
            cross_probability=config.ode_cross_probability,
            seed=config.seed
        )
    elif config.optimizer == 'ngopt':
        from ngopt import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,
            recombination=config.de_recombine,
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            use_lhs=True,  # NGOpt always uses LHS for better initialization
            seed=config.seed
        )
    elif config.optimizer == 'sobol_search':
        from sobol_search import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            mutate=config.de_mutate,  # Dummy parameter for interface compatibility
            recombination=config.de_recombine,  # Dummy parameter for interface compatibility
            maxiter=maxiter,
            maxtime=maxtime,
            maxevaluations=maxevaluations,
            scramble=config.sobol_scramble,
            seed=config.seed
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    solution = decode(np.array([result_tour] * batch_size), model, config, instance, cost_fn)[0][0].tolist()
    return result_cost, solution, convergence_history, time_history, timing_breakdown


def write_results_csv(all_results, output_file, has_solutions):
    """
    Write per-instance results and summary statistics to a CSV file.

    Args:
        all_results: Dictionary containing results for each optimizer/parameter
        output_file: Path to the output CSV file
        has_solutions: Boolean indicating if optimal solutions were provided
    """
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Section 1: Per-instance results
        if has_solutions:
            writer.writerow(['optimizer', 'instance_id', 'runtime', 'gap', 'cost', 'optimal_cost'])
        else:
            writer.writerow(['optimizer', 'instance_id', 'runtime', 'cost'])

        # Get list of optimizers/parameters
        optimizer_names = list(all_results.keys())

        # Determine number of instances from first optimizer
        num_instances = len(all_results[optimizer_names[0]]['runtime_values'])

        # Write per-instance rows
        for i in range(num_instances):
            for optimizer_name in optimizer_names:
                results = all_results[optimizer_name]
                runtime = results['runtime_values'][i]
                cost = results['cost_values'][i]
                instance_id = results['instance_ids'][i]

                if has_solutions:
                    gap = results['gap_values'][i]
                    optimal_cost = results['optimal_values'][i]
                    writer.writerow([optimizer_name, instance_id, f"{runtime:.2f}", f"{gap:.2f}", f"{cost:.4f}", f"{optimal_cost:.4f}"])
                else:
                    writer.writerow([optimizer_name, instance_id, f"{runtime:.2f}", f"{cost:.4f}"])

        # Blank line separator
        writer.writerow([])

        # Section 2: Summary statistics
        if has_solutions:
            writer.writerow(['optimizer', 'mean_runtime', 'std_runtime', 'mean_gap', 'std_gap', 'num_instances'])
        else:
            writer.writerow(['optimizer', 'mean_runtime', 'std_runtime', 'mean_cost', 'std_cost', 'num_instances'])

        for optimizer_name in optimizer_names:
            results = all_results[optimizer_name]
            mean_runtime = np.mean(results['runtime_values'])
            std_runtime = np.std(results['runtime_values'])
            num_inst = len(results['runtime_values'])

            if has_solutions:
                mean_gap = np.mean(results['gap_values'])
                std_gap = np.std(results['gap_values'])
                writer.writerow([optimizer_name, f"{mean_runtime:.2f}", f"{std_runtime:.2f}",
                               f"{mean_gap:.2f}", f"{std_gap:.2f}", num_inst])
            else:
                mean_cost = np.mean(results['cost_values'])
                std_cost = np.std(results['cost_values'])
                writer.writerow([optimizer_name, f"{mean_runtime:.2f}", f"{std_runtime:.2f}",
                               f"{mean_cost:.4f}", f"{std_cost:.4f}", num_inst])


def solve_instance_set(model, config, instances, solutions=None, verbose=True):
    global _last_batch_size
    model.eval()

    # Reset batch size tracking for clean state
    _last_batch_size = None

    if config.problem == "TSP":
        cost_fn = tsp.tours_length
    elif config.problem == "CVRP":
        cost_fn = cvrp.tours_length
        if solutions:
            solutions = [cvrp.solution_to_single_tour(solution) for solution in solutions]

    # Create search output directory if saving plots
    if config.save_plots:
        search_output_dir = os.path.join(config.output_path, "search")
        os.makedirs(search_output_dir, exist_ok=True)

        # Create subdirectories for per-instance and average plots
        instances_dir = os.path.join(search_output_dir, "instances")
        average_dir = os.path.join(search_output_dir, "average")
        os.makedirs(instances_dir, exist_ok=True)
        os.makedirs(average_dir, exist_ok=True)

    # Detect special modes
    optimizer_comparison_mode = config.compare_optimizers
    sigma_sweep_mode = config.cmaes_sigma_sweep is not None

    if optimizer_comparison_mode:
        # Optimizer comparison mode: run all optimizers with fixed batch size
        fixed_batch_size = config.batch_sizes[0]

        # Define mapping from short names to display names
        optimizer_name_mapping = {
            'de': 'DE',
            'de_vectorized': 'DE-Vectorized',
            'de_steroids': 'DE-Steroids',
            'cmaes': 'CMA-ES',
            'ipop_cmaes': 'IPOP-CMA-ES',
            'bipop_cmaes': 'BIPOP-CMA-ES',
            'jade': 'EvoX-JADE',
            'shade': 'EvoX-SHADE',
            'sade': 'EvoX-SaDE',
            'code': 'EvoX-CoDE',
            'ode': 'EvoX-ODE',
            'sobol': 'Sobol-Search'
        }

        # Determine which optimizers to run
        if config.compare_optimizer_list is not None:
            # Parse comma-separated list
            requested_optimizers = [opt.strip().lower() for opt in config.compare_optimizer_list.split(',')]
            # Validate optimizer names
            invalid_optimizers = [opt for opt in requested_optimizers if opt not in optimizer_name_mapping]
            if invalid_optimizers:
                raise ValueError(f"Invalid optimizer names: {invalid_optimizers}. "
                                f"Available: {', '.join(optimizer_name_mapping.keys())}")
            # Map to display names
            optimizers_to_run = [optimizer_name_mapping[opt] for opt in requested_optimizers]
        else:
            # Default: run all optimizers
            optimizers_to_run = ['DE', 'DE-Vectorized', 'DE-Steroids', 'CMA-ES', 'IPOP-CMA-ES',
                                 'BIPOP-CMA-ES', 'EvoX-JADE', 'EvoX-SHADE', 'EvoX-SaDE',
                                 'EvoX-CoDE', 'EvoX-ODE', 'Sobol-Search']

        logging.info(f"Running optimizer comparison mode with batch size {fixed_batch_size}")
        logging.info(f"Comparing: {', '.join(optimizers_to_run)}")

        # Store results for each optimizer (dynamic based on selected optimizers)
        all_results = {opt_name: {'gap_values': [], 'cost_values': [], 'runtime_values': [],
                                   'ask_times': [], 'eval_times': [], 'tell_times': [],
                                   'instance_ids': [], 'optimal_values': []}
                       for opt_name in optimizers_to_run}

        # Accumulator for averaging convergence data across instances
        all_instances_data = {opt_name: {'convergence': [], 'time': []}
                              for opt_name in optimizers_to_run}
    elif sigma_sweep_mode:
        # Sigma sweep mode: loop over sigma values with fixed batch size
        sweep_values = config.cmaes_sigma_sweep
        fixed_batch_size = config.batch_sizes[0]
        logging.info(f"Running CMA-ES sigma sweep mode with batch size {fixed_batch_size}")
        logging.info(f"Sigma values: {sweep_values}")

        # Store results for each sigma value
        all_results = {sigma: {'gap_values': [], 'cost_values': [], 'runtime_values': [],
                               'ask_times': [], 'eval_times': [], 'tell_times': []}
                       for sigma in sweep_values}

        # Accumulator for averaging convergence data across instances
        all_instances_data = {sigma: {'convergence': [], 'time': []}
                              for sigma in sweep_values}
    else:
        # Normal mode: loop over batch sizes
        # Store results for each batch size
        all_results = {bs: {'gap_values': [], 'cost_values': [], 'runtime_values': [],
                            'ask_times': [], 'eval_times': [], 'tell_times': []}
                       for bs in config.batch_sizes}

        # Accumulator for averaging convergence data across instances
        all_instances_data = {bs: {'convergence': [], 'time': []}
                              for bs in config.batch_sizes}

    for i, instance in enumerate(instances):
        logging.info(f"Solving instance {i + 1}/{len(instances)}")
        convergence_data = {}

        if optimizer_comparison_mode:
            # Run both optimizers: DE and CMA-ES
            de_runtime = None  # Will store DE runtime for time-matching mode

            # Calculate optimal value if solutions provided (used for plotting)
            if solutions:
                optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                        torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
            else:
                optimal_value = None

            for optimizer_name in optimizers_to_run:
                logging.info(f"  Optimizer: {optimizer_name}")
                start_time = time.time()

                # Temporarily change config.optimizer for solve_instance
                original_optimizer = config.optimizer
                if optimizer_name == 'DE':
                    config.optimizer = 'de'
                elif optimizer_name == 'DE-Vectorized':
                    config.optimizer = 'de_vectorized'
                elif optimizer_name == 'DE-Steroids':
                    config.optimizer = 'de_on_steroids'
                elif optimizer_name == 'CMA-ES':
                    config.optimizer = 'cmaes'
                elif optimizer_name == 'IPOP-CMA-ES':
                    config.optimizer = 'ipop_cmaes'
                elif optimizer_name == 'BIPOP-CMA-ES':
                    config.optimizer = 'bipop_cmaes'
                elif optimizer_name == 'Pygmo-DE':
                    config.optimizer = 'pygmo_de'
                elif optimizer_name == 'EvoX-JADE':
                    config.optimizer = 'evox_jade'
                elif optimizer_name == 'EvoX-SHADE':
                    config.optimizer = 'evox_shade'
                elif optimizer_name == 'EvoX-SaDE':
                    config.optimizer = 'evox_sade'
                elif optimizer_name == 'EvoX-CoDE':
                    config.optimizer = 'evox_code'
                elif optimizer_name == 'EvoX-ODE':
                    config.optimizer = 'evox_ode'
                elif optimizer_name == 'Sobol-Search':
                    config.optimizer = 'sobol_search'

                # Determine stopping criteria based on mode
                override_maxiter = None
                override_maxtime = None

                if config.stopping_criteria == 'time_of_de':
                    if optimizer_name == 'DE':
                        # DE runs for 300 iterations
                        override_maxiter = 300
                        override_maxtime = None  # No time limit for DE
                    else:
                        # Other optimizers match DE's runtime
                        override_maxiter = 999999  # Effectively no iteration limit (much larger than time allows)
                        override_maxtime = de_runtime
                        logging.info(f"    Using time-matching: maxtime={de_runtime:.2f}s (matching DE runtime)")
                else:
                    # Default mode: DE runs with config iterations, others match DE's runtime
                    if optimizer_name == 'DE':
                        # DE runs with the specified iterations from config
                        override_maxiter = None  # Use config.search_iterations
                        override_maxtime = None  # No time limit for DE
                    else:
                        # Other optimizers match DE's runtime (time-based stopping)
                        override_maxiter = 999999  # Effectively no iteration limit (much larger than time allows)
                        override_maxtime = de_runtime
                        logging.info(f"    Using time-matching: maxtime={de_runtime:.2f}s (matching DE runtime)")

                objective_value, solution, convergence_history, time_history, timing_breakdown = solve_instance(
                    model, instance, config, cost_fn, fixed_batch_size,
                    override_maxiter=override_maxiter, override_maxtime=override_maxtime)

                # Restore original optimizer
                config.optimizer = original_optimizer
                runtime = time.time() - start_time

                # Store DE runtime for time-matching (both time_of_de and default modes)
                if optimizer_name == 'DE':
                    de_runtime = runtime
                    logging.info(f"    DE runtime captured: {de_runtime:.2f}s (will be used for other optimizers)")

                # Store convergence history and time history for comparison plots
                convergence_data[optimizer_name] = (convergence_history, time_history)

                # Accumulate data for averaging across instances
                all_instances_data[optimizer_name]['convergence'].append(convergence_history)
                all_instances_data[optimizer_name]['time'].append(time_history)

                # Calculate gap if solutions provided
                if solutions:
                    optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                            torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
                    gap = (objective_value / optimal_value - 1) * 100
                    all_results[optimizer_name]['gap_values'].append(gap)
                    all_results[optimizer_name]['optimal_values'].append(optimal_value)
                    logging.info(f"    Objective: {objective_value:.4f}, Optimal: {optimal_value:.4f}, Gap: {gap:.2f}%")
                else:
                    all_results[optimizer_name]['gap_values'].append(0)
                    all_results[optimizer_name]['optimal_values'].append(0)
                    logging.info(f"    Objective: {objective_value:.4f}")

                all_results[optimizer_name]['cost_values'].append(objective_value)
                all_results[optimizer_name]['runtime_values'].append(runtime)
                all_results[optimizer_name]['instance_ids'].append(i)
                logging.info(f"    Runtime: {runtime:.2f}s")
                logging.info(f"    Iterations: {timing_breakdown['iterations']}")
                # Log restarts for IPOP and BIPOP
                if optimizer_name in ['IPOP-CMA-ES', 'BIPOP-CMA-ES']:
                    logging.info(f"    Restarts: {timing_breakdown['restarts']}")

                # Store and log timing breakdown
                all_results[optimizer_name]['ask_times'].append(timing_breakdown['ask_time'])
                all_results[optimizer_name]['eval_times'].append(timing_breakdown['eval_time'])
                all_results[optimizer_name]['tell_times'].append(timing_breakdown['tell_time'])
                logging.info(f"    Timing breakdown:")
                logging.info(f"      ASK:  {timing_breakdown['ask_time']:.2f}s ({timing_breakdown['ask_time']/runtime*100:.1f}%)")
                logging.info(f"      EVAL: {timing_breakdown['eval_time']:.2f}s ({timing_breakdown['eval_time']/runtime*100:.1f}%)")
                logging.info(f"      TELL: {timing_breakdown['tell_time']:.2f}s ({timing_breakdown['tell_time']/runtime*100:.1f}%)")

            # Generate per-instance optimizer comparison plots if in per_instance mode
            if config.save_plots and config.plot_mode == 'per_instance':
                plot_optimizer_comparison_iterations_pct_per_instance(
                    convergence_data, search_output_dir, config.search_iterations,
                    fixed_batch_size, i, optimal_value, config.problem, config.problem_size, config.model_type)
                plot_optimizer_comparison_evaluations_pct_per_instance(
                    convergence_data, search_output_dir, config.search_iterations,
                    fixed_batch_size, i, optimal_value, config.problem, config.problem_size, config.model_type)
                plot_optimizer_comparison_time_pct_per_instance(
                    convergence_data, search_output_dir, config.search_iterations,
                    fixed_batch_size, i, optimal_value, config.problem, config.problem_size, config.model_type)
        elif sigma_sweep_mode:
            # Run search for each sigma value with fixed batch size
            for sigma_value in sweep_values:
                logging.info(f"  Sigma: {sigma_value}")
                start_time = time.time()
                objective_value, solution, convergence_history, time_history, timing_breakdown = solve_instance(
                    model, instance, config, cost_fn, fixed_batch_size, sigma0=sigma_value)
                runtime = time.time() - start_time

                # Store convergence history and time history for comparison plots
                convergence_data[sigma_value] = (convergence_history, time_history)

                # Accumulate data for averaging across instances
                all_instances_data[sigma_value]['convergence'].append(convergence_history)
                all_instances_data[sigma_value]['time'].append(time_history)

                # Calculate gap if solutions provided
                if solutions:
                    optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                            torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
                    gap = (objective_value / optimal_value - 1) * 100
                    all_results[sigma_value]['gap_values'].append(gap)
                    logging.info(f"    Objective: {objective_value:.4f}, Optimal: {optimal_value:.4f}, Gap: {gap:.2f}%")
                else:
                    all_results[sigma_value]['gap_values'].append(0)
                    logging.info(f"    Objective: {objective_value:.4f}")

                all_results[sigma_value]['cost_values'].append(objective_value)
                all_results[sigma_value]['runtime_values'].append(runtime)
                logging.info(f"    Runtime: {runtime:.2f}s")
                logging.info(f"    Iterations: {timing_breakdown['iterations']}")

                # Store and log timing breakdown
                all_results[sigma_value]['ask_times'].append(timing_breakdown['ask_time'])
                all_results[sigma_value]['eval_times'].append(timing_breakdown['eval_time'])
                all_results[sigma_value]['tell_times'].append(timing_breakdown['tell_time'])
                logging.info(f"    Timing breakdown:")
                logging.info(f"      ASK:  {timing_breakdown['ask_time']:.2f}s ({timing_breakdown['ask_time']/runtime*100:.1f}%)")
                logging.info(f"      EVAL: {timing_breakdown['eval_time']:.2f}s ({timing_breakdown['eval_time']/runtime*100:.1f}%)")
                logging.info(f"      TELL: {timing_breakdown['tell_time']:.2f}s ({timing_breakdown['tell_time']/runtime*100:.1f}%)")
        else:
            # Run search for each batch size
            for batch_size in config.batch_sizes:
                logging.info(f"  Batch size: {batch_size}")
                start_time = time.time()
                objective_value, solution, convergence_history, time_history, timing_breakdown = solve_instance(model, instance, config, cost_fn, batch_size)
                runtime = time.time() - start_time

                # Store convergence history and time history for comparison plots
                convergence_data[batch_size] = (convergence_history, time_history)

                # Accumulate data for averaging across instances
                all_instances_data[batch_size]['convergence'].append(convergence_history)
                all_instances_data[batch_size]['time'].append(time_history)

                # Calculate gap if solutions provided
                if solutions:
                    optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                            torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
                    gap = (objective_value / optimal_value - 1) * 100
                    all_results[batch_size]['gap_values'].append(gap)
                    logging.info(f"    Objective: {objective_value:.4f}, Optimal: {optimal_value:.4f}, Gap: {gap:.2f}%")
                else:
                    all_results[batch_size]['gap_values'].append(0)
                    logging.info(f"    Objective: {objective_value:.4f}")

                all_results[batch_size]['cost_values'].append(objective_value)
                all_results[batch_size]['runtime_values'].append(runtime)
                logging.info(f"    Runtime: {runtime:.2f}s")
                logging.info(f"    Iterations: {timing_breakdown['iterations']}")

                # Store and log timing breakdown
                all_results[batch_size]['ask_times'].append(timing_breakdown['ask_time'])
                all_results[batch_size]['eval_times'].append(timing_breakdown['eval_time'])
                all_results[batch_size]['tell_times'].append(timing_breakdown['tell_time'])
                logging.info(f"    Timing breakdown:")
                logging.info(f"      ASK:  {timing_breakdown['ask_time']:.2f}s ({timing_breakdown['ask_time']/runtime*100:.1f}%)")
                logging.info(f"      EVAL: {timing_breakdown['eval_time']:.2f}s ({timing_breakdown['eval_time']/runtime*100:.1f}%)")
                logging.info(f"      TELL: {timing_breakdown['tell_time']:.2f}s ({timing_breakdown['tell_time']/runtime*100:.1f}%)")

        # Save convergence comparison plots if enabled (only for per-instance mode)
        if config.save_plots and config.plot_mode == 'per_instance' and not optimizer_comparison_mode:
            # Format optimizer name for display
            optimizer_name_map = {
                'de': 'DE',
                'de_vectorized': 'DE-Vectorized',
                'de_on_steroids': 'DE-Steroids',
                'cmaes': 'CMA-ES',
                'ipop_cmaes': 'IPOP-CMA-ES',
                'bipop_cmaes': 'BIPOP-CMA-ES',
                'evox_jade': 'EvoX-JADE',
                'pygmo_pso_gen': 'Pygmo-PSO-Gen',
                'evox_shade': 'EvoX-SHADE',
                'evox_sade': 'EvoX-SaDE',
                'evox_code': 'EvoX-CoDE',
                'evox_ode': 'EvoX-ODE',
                'ngopt': 'NGOpt',
                'sobol_search': 'Sobol-Search'
            }
            optimizer_name = optimizer_name_map.get(config.optimizer, config.optimizer.upper())

            # Extract optimal value for gap-based plots
            if solutions:
                optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                        torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
            else:
                optimal_value = None

            # Create percentage-based comparison plots (all batch sizes on same graph)
            if len(config.batch_sizes) > 1:
                plot_convergence_comparison_iterations_pct(i, convergence_data, instances_dir, config.search_iterations, optimal_value, optimizer_name, config.model_type)
                plot_convergence_comparison_pct(i, convergence_data, instances_dir, config.search_iterations, optimal_value, optimizer_name, config.model_type)
                plot_convergence_comparison_time_pct(i, convergence_data, instances_dir, config.search_iterations, optimal_value, optimizer_name, config.model_type)
            else:
                # If single batch size, still create plots but they'll only have one curve
                plot_convergence_comparison_iterations_pct(i, convergence_data, instances_dir, config.search_iterations, optimal_value, optimizer_name, config.model_type)
                plot_convergence_comparison_pct(i, convergence_data, instances_dir, config.search_iterations, optimal_value, optimizer_name, config.model_type)
                plot_convergence_comparison_time_pct(i, convergence_data, instances_dir, config.search_iterations, optimal_value, optimizer_name, config.model_type)

    # Generate averaged plots
    # For optimizer_comparison_mode and sigma_sweep_mode, always generate averaged plots regardless of plot_mode
    # For normal mode, only generate averaged plots if plot_mode == 'average'
    if config.save_plots:
        # Extract optimal values for all instances (needed for gap-based plots)
        if solutions:
            optimal_values = []
            for i, instance in enumerate(instances):
                optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                        torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
                optimal_values.append(optimal_value)
        else:
            optimal_values = None

        if optimizer_comparison_mode:
            # Optimizer comparison mode: generate optimizer comparison plots (always, regardless of plot_mode)
            logging.info("Computing averaged convergence data across all instances for optimizer comparison...")
            averaged_data = compute_averaged_convergence(all_instances_data, optimizers_to_run, optimal_values)

            # Generate optimizer comparison plots
            plot_optimizer_comparison_iterations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size, config.model_type)
            plot_optimizer_comparison_evaluations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size, config.model_type)
            plot_optimizer_comparison_time_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size, config.model_type)
        elif sigma_sweep_mode:
            # Sigma sweep mode: generate sigma comparison plots (always, regardless of plot_mode)
            logging.info("Computing averaged convergence data across all instances for sigma sweep...")
            averaged_data = compute_averaged_convergence(all_instances_data, sweep_values, optimal_values)

            # Generate sigma comparison plots
            plot_sigma_comparison_iterations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size, config.model_type)
            plot_sigma_comparison_evaluations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size, config.model_type)
            plot_sigma_comparison_time_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size, config.model_type)
        elif config.plot_mode == 'average' and len(config.batch_sizes) > 1:
            # Normal mode with average plot_mode: generate batch size comparison plots
            logging.info("Computing averaged convergence data across all instances...")
            averaged_data = compute_averaged_convergence(all_instances_data, config.batch_sizes, optimal_values)

            # Format optimizer name for display
            optimizer_name_map = {
                'de': 'DE',
                'de_vectorized': 'DE-Vectorized',
                'de_on_steroids': 'DE-Steroids',
                'cmaes': 'CMA-ES',
                'ipop_cmaes': 'IPOP-CMA-ES',
                'bipop_cmaes': 'BIPOP-CMA-ES',
                'evox_jade': 'EvoX-JADE',
                'pygmo_pso_gen': 'Pygmo-PSO-Gen',
                'evox_shade': 'EvoX-SHADE',
                'evox_sade': 'EvoX-SaDE',
                'evox_code': 'EvoX-CoDE',
                'evox_ode': 'EvoX-ODE',
                'ngopt': 'NGOpt',
                'sobol_search': 'Sobol-Search'
            }
            optimizer_name = optimizer_name_map.get(config.optimizer, config.optimizer.upper())

            # Generate averaged percentage plots
            plot_average_convergence_iterations_pct(averaged_data, average_dir, config.search_iterations, len(instances), optimizer_name, config.model_type)
            plot_average_convergence_evaluations_pct(averaged_data, average_dir, config.search_iterations, len(instances), optimizer_name, config.model_type)
            plot_average_convergence_time_pct(averaged_data, average_dir, config.search_iterations, len(instances), optimizer_name, config.model_type)

    # Log final results
    logging.info("=" * 60)
    logging.info("Final search results:")

    if optimizer_comparison_mode:
        # Log results for each optimizer
        for optimizer_name in optimizers_to_run:
            results = all_results[optimizer_name]
            logging.info(f"\n{optimizer_name}:")
            logging.info(f"  Mean cost: {np.mean(results['cost_values']):.4f}")
            logging.info(f"  Mean runtime: {np.mean(results['runtime_values']):.2f}s")
            if solutions:
                logging.info(f"  Mean gap: {np.mean(results['gap_values']):.2f}%")
                logging.info(f"  Std gap: {np.std(results['gap_values']):.2f}%")
            # Log timing breakdown
            mean_runtime = np.mean(results['runtime_values'])
            mean_ask = np.mean(results['ask_times'])
            mean_eval = np.mean(results['eval_times'])
            mean_tell = np.mean(results['tell_times'])
            logging.info(f"  Mean timing breakdown:")
            logging.info(f"    ASK:  {mean_ask:.2f}s ({mean_ask/mean_runtime*100:.1f}%)")
            logging.info(f"    EVAL: {mean_eval:.2f}s ({mean_eval/mean_runtime*100:.1f}%)")
            logging.info(f"    TELL: {mean_tell:.2f}s ({mean_tell/mean_runtime*100:.1f}%)")

        # Write results to CSV file
        csv_output_file = os.path.join(config.output_path, "search", "optimizer_comparison_results.csv")
        write_results_csv(all_results, csv_output_file, solutions is not None)
        logging.info(f"\nResults saved to: {csv_output_file}")
    elif sigma_sweep_mode:
        # Log results for each sigma value
        for sigma_value in sweep_values:
            results = all_results[sigma_value]
            logging.info(f"\nSigma: {sigma_value}:")
            logging.info(f"  Mean cost: {np.mean(results['cost_values']):.4f}")
            logging.info(f"  Mean runtime: {np.mean(results['runtime_values']):.2f}s")
            if solutions:
                logging.info(f"  Mean gap: {np.mean(results['gap_values']):.2f}%")
                logging.info(f"  Std gap: {np.std(results['gap_values']):.2f}%")
            # Log timing breakdown
            mean_runtime = np.mean(results['runtime_values'])
            mean_ask = np.mean(results['ask_times'])
            mean_eval = np.mean(results['eval_times'])
            mean_tell = np.mean(results['tell_times'])
            logging.info(f"  Mean timing breakdown:")
            logging.info(f"    ASK:  {mean_ask:.2f}s ({mean_ask/mean_runtime*100:.1f}%)")
            logging.info(f"    EVAL: {mean_eval:.2f}s ({mean_eval/mean_runtime*100:.1f}%)")
            logging.info(f"    TELL: {mean_tell:.2f}s ({mean_tell/mean_runtime*100:.1f}%)")
    else:
        # Log results for each batch size
        for batch_size in config.batch_sizes:
            results = all_results[batch_size]
            logging.info(f"\nBatch size {batch_size}:")
            logging.info(f"  Mean cost: {np.mean(results['cost_values']):.4f}")
            logging.info(f"  Mean runtime: {np.mean(results['runtime_values']):.2f}s")
            if solutions:
                logging.info(f"  Mean gap: {np.mean(results['gap_values']):.2f}%")
                logging.info(f"  Std gap: {np.std(results['gap_values']):.2f}%")
            # Log timing breakdown
            mean_runtime = np.mean(results['runtime_values'])
            mean_ask = np.mean(results['ask_times'])
            mean_eval = np.mean(results['eval_times'])
            mean_tell = np.mean(results['tell_times'])
            logging.info(f"  Mean timing breakdown:")
            logging.info(f"    ASK:  {mean_ask:.2f}s ({mean_ask/mean_runtime*100:.1f}%)")
            logging.info(f"    EVAL: {mean_eval:.2f}s ({mean_eval/mean_runtime*100:.1f}%)")
            logging.info(f"    TELL: {mean_tell:.2f}s ({mean_tell/mean_runtime*100:.1f}%)")

            # Save results to file
            if verbose and not solutions:
                output_file = os.path.join(config.output_path, "search", f'results_bs{batch_size}.txt')
                results_array = np.array(list(zip(results['cost_values'], results['runtime_values'])))
                np.savetxt(output_file, results_array, delimiter=',', fmt=['%s', '%s'],
                           header="cost, runtime")

    # Return results (for backward compatibility)
    if optimizer_comparison_mode:
        # Return results for DE (first optimizer)
        return (np.mean(all_results['DE']['gap_values']),
                np.mean(all_results['DE']['runtime_values']),
                all_results['DE']['cost_values'])
    elif sigma_sweep_mode:
        # Return results for the first sigma value
        first_sigma = sweep_values[0]
        return (np.mean(all_results[first_sigma]['gap_values']),
                np.mean(all_results[first_sigma]['runtime_values']),
                all_results[first_sigma]['cost_values'])
    else:
        # Return results for the first batch size
        first_batch_size = config.batch_sizes[0]
        return (np.mean(all_results[first_batch_size]['gap_values']),
                np.mean(all_results[first_batch_size]['runtime_values']),
                all_results[first_batch_size]['cost_values'])
