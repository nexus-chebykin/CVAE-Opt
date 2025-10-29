import argparse
import torch



def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="CAVE-Opt Search")

    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model_path', type=str, default='', required=True)
    parser.add_argument('--problem', type=str, default=None)
    parser.add_argument("--problem_size", type=int, default=None)
    parser.add_argument('--search_batch_size', default=600, type=int)
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=None,
                        help='List of batch sizes to test (e.g., --batch_sizes 50 100 300 600). If not specified, uses search_batch_size.')
    parser.add_argument('--instances_path', type=str, default="")
    parser.add_argument('--search_timelimit', default=600, type=int)
    parser.add_argument('--search_space_size', default=100, type=int)  # Nb. dimensions of search space
    parser.add_argument('--search_iterations', default=300, type=int)
    parser.add_argument('--save_plots', default=False, action='store_true',
                        help='Save convergence plots for each instance')
    parser.add_argument('--plot_mode', type=str, default='per_instance',
                        choices=['per_instance', 'average'],
                        help='Plotting mode: "per_instance" creates all plots (absolute and percentage) for each instance, '
                             '"average" skips per-instance plots and creates only 3 averaged percentage plots across all instances.')

    # Optimizer selection
    parser.add_argument('--optimizer', type=str, default='de',
                        choices=['de', 'cmaes'],
                        help='Optimizer to use: "de" (Differential Evolution) or "cmaes" (CMA-ES)')

    # Differential Evolution parameters
    parser.add_argument('--de_mutate', default=0.3, type=float,
                        help='Mutation factor F for DE (default: 0.3)')
    parser.add_argument('--de_recombine', default=0.95, type=float,
                        help='Crossover rate CR for DE (default: 0.95)')

    # CMA-ES parameters
    parser.add_argument('--cmaes_sigma0', default=0.5, type=float,
                        help='Initial step size for CMA-ES (default: 0.5, typically 0.2-0.5 of search range)')

    config = parser.parse_args()
    config.device = torch.device(config.device)

    # If batch_sizes not specified, use search_batch_size as default
    if config.batch_sizes is None:
        config.batch_sizes = [config.search_batch_size]

    return config
