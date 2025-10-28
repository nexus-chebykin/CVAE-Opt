import torch
import tsp, cvrp
import numpy as np
import time
from de import minimize
import logging
import os
import matplotlib.pyplot as plt


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


def plot_convergence(instance_idx, convergence_history, output_path):
    """
    Plot convergence history for a single instance (both linear and log scale).

    Args:
        instance_idx: Index of the instance
        convergence_history: List of best objective values per iteration
        output_path: Directory to save plots
    """
    iterations = list(range(1, len(convergence_history) + 1))

    # Create linear scale plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, convergence_history, linewidth=2, color='#2E86AB')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Objective Value', fontsize=12)
    plt.title(f'Convergence - Instance {instance_idx} (Linear Scale)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    linear_path = os.path.join(output_path, f'instance_{instance_idx}_convergence.png')
    plt.savefig(linear_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Create log scale plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, convergence_history, linewidth=2, color='#A23B72')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Objective Value (log scale)', fontsize=12)
    plt.title(f'Convergence - Instance {instance_idx} (Log Scale)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    log_path = os.path.join(output_path, f'instance_{instance_idx}_convergence_log.png')
    plt.savefig(log_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved convergence plots for instance {instance_idx}")


def solve_instance_de(model, instance, config, cost_fn):
    batch_size = config.search_batch_size
    instance = torch.Tensor(instance)
    instance = instance.unsqueeze(0).expand(batch_size, -1, -1)
    instance = instance.to(config.device)
    model.reset_decoder(batch_size, config)

    result_cost, result_tour, convergence_history = minimize(decode, (model, config, instance, cost_fn), config.search_space_bound,
                                        config.search_space_size, popsize=batch_size,
                                        mutate=config.de_mutate, recombination=config.de_recombine,
                                        maxiter=config.search_iterations, maxtime=config.search_timelimit)
    solution = decode(np.array([result_tour] * batch_size), model, config, instance, cost_fn)[0][0].tolist()
    return result_cost, solution, convergence_history


def solve_instance_set(model, config, instances, solutions=None, verbose=True):
    model.eval()

    if config.problem == "TSP":
        cost_fn = tsp.tours_length
    elif config.problem == "CVRP":
        cost_fn = cvrp.tours_length
        if solutions:
            solutions = [cvrp.solution_to_single_tour(solution) for solution in solutions]

    gap_values = np.zeros((len(instances)))
    cost_values = []
    runtime_values = []

    # Create search output directory if saving plots
    if config.save_plots:
        search_output_dir = os.path.join(config.output_path, "search")
        os.makedirs(search_output_dir, exist_ok=True)

    for i, instance in enumerate(instances):
        start_time = time.time()
        objective_value, solution, convergence_history = solve_instance_de(model, instance, config, cost_fn)
        runtime = time.time() - start_time

        # Save convergence plots if enabled
        if config.save_plots:
            plot_convergence(i, convergence_history, search_output_dir)

        if solutions:
            optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                    torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
            print(objective_value, optimal_value)
            print("Opt " + str(solutions[i]))
            gap = (objective_value / optimal_value - 1) * 100
            print("Gap " + str(gap) + "%")
            gap_values[i] = gap
        cost_values.append(objective_value)
        print("Costs " + str(objective_value))
        runtime_values.append(runtime)

    if not solutions and verbose:
        results = np.array(list(zip(cost_values, runtime_values)))
        np.savetxt(os.path.join(config.output_path, "search", 'results.txt'), results, delimiter=',', fmt=['%s', '%s'],
                   header="cost, runtime")
        logging.info("Final search results:")
        logging.info(f"Mean costs: {np.mean(cost_values)}")
        logging.info("Mean std: {}".format(np.mean(np.std(gap_values))))
        logging.info(f"Mean runtime: {np.mean(runtime_values)}")

    return np.mean(gap_values), np.mean(runtime_values), cost_values
