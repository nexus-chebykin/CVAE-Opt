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

    # Differential Evolution
    parser.add_argument('--de_mutate', default=0.3, type=float)
    parser.add_argument('--de_recombine', default=0.95, type=float)

    config = parser.parse_args()
    config.device = torch.device(config.device)

    # If batch_sizes not specified, use search_batch_size as default
    if config.batch_sizes is None:
        config.batch_sizes = [config.search_batch_size]

    return config
