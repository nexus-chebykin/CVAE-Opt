import torch
import tsp, cvrp
import numpy as np
import time
import logging
import os
from plotting import *


def decode(Z, model, config, instance, cost_fn):
    Z = torch.Tensor(Z).to(config.device)
    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance, Z, config)
    costs = cost_fn(instance, tour_idx)
    return tour_idx, costs.tolist()


def evaluate(Z, model, config, instance, cost_fn):
    Z = torch.Tensor(Z).to(config.device)
    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance, Z, config)
    costs = cost_fn(instance, tour_idx)
    return costs.tolist()


def solve_instance(model, instance, config, cost_fn, batch_size, sigma0=None):
    """
    Solve a single instance using the configured optimizer (DE or CMA-ES).

    Args:
        model: Neural network model
        instance: Problem instance to solve
        config: Configuration object with optimizer settings
        cost_fn: Cost function for evaluating tours
        batch_size: Population size for the optimizer
        sigma0: Optional sigma0 for CMA-ES (overrides config.cmaes_sigma0 if provided)

    Returns:
        result_cost: Best objective value found
        solution: Best solution (tour)
        convergence_history: History of best fitness per iteration
        time_history: History of elapsed time per iteration
    """
    instance = torch.Tensor(instance)
    instance = instance.unsqueeze(0).expand(batch_size, -1, -1)
    instance = instance.to(config.device)
    model.reset_decoder(batch_size, config)

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
            maxiter=config.search_iterations,
            maxtime=config.search_timelimit,
            maxevaluations=config.search_evaluations
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
            maxiter=config.search_iterations,
            maxtime=config.search_timelimit,
            maxevaluations=config.search_evaluations
        )
    elif config.optimizer == 'portfolio':
        from portfolio import minimize
        result_cost, result_tour, convergence_history, time_history, timing_breakdown = minimize(
            decode,
            (model, config, instance, cost_fn),
            config.search_space_bound,
            config.search_space_size,
            popsize=batch_size,
            maxiter=config.search_iterations,
            maxtime=config.search_timelimit,
            maxevaluations=config.search_evaluations
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    solution = decode(np.array([result_tour] * batch_size), model, config, instance, cost_fn)[0][0].tolist()
    return result_cost, solution, convergence_history, time_history, timing_breakdown


def solve_instance_set(model, config, instances, solutions=None, verbose=True):
    model.eval()

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

    # Detect special modes
    optimizer_comparison_mode = config.compare_optimizers
    sigma_sweep_mode = config.cmaes_sigma_sweep is not None

    if optimizer_comparison_mode:
        # Optimizer comparison mode: run DE, CMA-ES, and Portfolio with fixed batch size
        fixed_batch_size = config.batch_sizes[0]
        logging.info(f"Running optimizer comparison mode with batch size {fixed_batch_size}")
        logging.info(f"Comparing DE vs CMA-ES (sigma={config.cmaes_sigma0}) vs Portfolio")

        # Store results for each optimizer
        all_results = {'DE': {'gap_values': [], 'cost_values': [], 'runtime_values': [],
                              'ask_times': [], 'eval_times': [], 'tell_times': []},
                       'CMA-ES': {'gap_values': [], 'cost_values': [], 'runtime_values': [],
                                  'ask_times': [], 'eval_times': [], 'tell_times': []},
                       'Portfolio': {'gap_values': [], 'cost_values': [], 'runtime_values': [],
                                     'ask_times': [], 'eval_times': [], 'tell_times': []}}

        # Accumulator for averaging convergence data across instances
        all_instances_data = {'DE': {'convergence': [], 'time': []},
                              'CMA-ES': {'convergence': [], 'time': []},
                              'Portfolio': {'convergence': [], 'time': []}}
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
            # Run all three optimizers: DE, CMA-ES, and Portfolio
            for optimizer_name in ['DE', 'CMA-ES', 'Portfolio']:
                logging.info(f"  Optimizer: {optimizer_name}")
                start_time = time.time()

                # Temporarily change config.optimizer for solve_instance
                original_optimizer = config.optimizer
                if optimizer_name == 'DE':
                    config.optimizer = 'de'
                elif optimizer_name == 'CMA-ES':
                    config.optimizer = 'cmaes'
                elif optimizer_name == 'Portfolio':
                    config.optimizer = 'portfolio'

                objective_value, solution, convergence_history, time_history, timing_breakdown = solve_instance(
                    model, instance, config, cost_fn, fixed_batch_size)

                # Restore original optimizer
                config.optimizer = original_optimizer
                runtime = time.time() - start_time

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
                    logging.info(f"    Objective: {objective_value:.4f}, Optimal: {optimal_value:.4f}, Gap: {gap:.2f}%")
                else:
                    all_results[optimizer_name]['gap_values'].append(0)
                    logging.info(f"    Objective: {objective_value:.4f}")

                all_results[optimizer_name]['cost_values'].append(objective_value)
                all_results[optimizer_name]['runtime_values'].append(runtime)
                logging.info(f"    Runtime: {runtime:.2f}s")

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
                    fixed_batch_size, i, config.problem, config.problem_size)
                plot_optimizer_comparison_evaluations_pct_per_instance(
                    convergence_data, search_output_dir, config.search_iterations,
                    fixed_batch_size, i, config.problem, config.problem_size)
                plot_optimizer_comparison_time_pct_per_instance(
                    convergence_data, search_output_dir, config.search_iterations,
                    fixed_batch_size, i, config.problem, config.problem_size)
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

                # Store and log timing breakdown
                all_results[batch_size]['ask_times'].append(timing_breakdown['ask_time'])
                all_results[batch_size]['eval_times'].append(timing_breakdown['eval_time'])
                all_results[batch_size]['tell_times'].append(timing_breakdown['tell_time'])
                logging.info(f"    Timing breakdown:")
                logging.info(f"      ASK:  {timing_breakdown['ask_time']:.2f}s ({timing_breakdown['ask_time']/runtime*100:.1f}%)")
                logging.info(f"      EVAL: {timing_breakdown['eval_time']:.2f}s ({timing_breakdown['eval_time']/runtime*100:.1f}%)")
                logging.info(f"      TELL: {timing_breakdown['tell_time']:.2f}s ({timing_breakdown['tell_time']/runtime*100:.1f}%)")

        # Save convergence comparison plots if enabled (only for per-instance mode)
        if config.save_plots and config.plot_mode == 'per_instance':
            # Format optimizer name for display
            optimizer_name = 'CMA-ES' if config.optimizer == 'cmaes' else 'DE'

            # Create absolute value comparison plots (all batch sizes on same graph)
            if len(config.batch_sizes) > 1:
                plot_convergence_comparison_iterations(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_time(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                # Create percentage-based comparison plots
                plot_convergence_comparison_iterations_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_time_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
            else:
                # If single batch size, still create plots but they'll only have one curve
                plot_convergence_comparison_iterations(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_time(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                # Create percentage-based plots
                plot_convergence_comparison_iterations_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)
                plot_convergence_comparison_time_pct(i, convergence_data, search_output_dir, config.search_iterations, optimizer_name)

    # Generate averaged plots
    # For optimizer_comparison_mode and sigma_sweep_mode, always generate averaged plots regardless of plot_mode
    # For normal mode, only generate averaged plots if plot_mode == 'average'
    if config.save_plots:
        if optimizer_comparison_mode:
            # Optimizer comparison mode: generate optimizer comparison plots (always, regardless of plot_mode)
            logging.info("Computing averaged convergence data across all instances for optimizer comparison...")
            averaged_data = compute_averaged_convergence(all_instances_data, ['CMA-ES', 'DE', 'Portfolio'])

            # Generate optimizer comparison plots
            plot_optimizer_comparison_iterations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
            plot_optimizer_comparison_evaluations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
            plot_optimizer_comparison_time_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
        elif sigma_sweep_mode:
            # Sigma sweep mode: generate sigma comparison plots (always, regardless of plot_mode)
            logging.info("Computing averaged convergence data across all instances for sigma sweep...")
            averaged_data = compute_averaged_convergence(all_instances_data, sweep_values)

            # Generate sigma comparison plots
            plot_sigma_comparison_iterations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
            plot_sigma_comparison_evaluations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
            plot_sigma_comparison_time_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), fixed_batch_size)
        elif config.plot_mode == 'average' and len(config.batch_sizes) > 1:
            # Normal mode with average plot_mode: generate batch size comparison plots
            logging.info("Computing averaged convergence data across all instances...")
            averaged_data = compute_averaged_convergence(all_instances_data, config.batch_sizes)

            # Format optimizer name for display
            optimizer_name = 'CMA-ES' if config.optimizer == 'cmaes' else 'DE'

            # Generate averaged percentage plots
            plot_average_convergence_iterations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), optimizer_name)
            plot_average_convergence_evaluations_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), optimizer_name)
            plot_average_convergence_time_pct(averaged_data, search_output_dir, config.search_iterations, len(instances), optimizer_name)

    # Log final results
    logging.info("=" * 60)
    logging.info("Final search results:")

    if optimizer_comparison_mode:
        # Log results for each optimizer
        for optimizer_name in ['DE', 'CMA-ES', 'Portfolio']:
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
