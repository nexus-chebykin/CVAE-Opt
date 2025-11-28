import argparse
import sys
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
    parser.add_argument('--search_timelimit', default=None, type=int,
                        help='Maximum wall-clock time in seconds (default: None, no time limit)')
    parser.add_argument('--search_space_size', default=100, type=int)  # Nb. dimensions of search space
    parser.add_argument('--search_iterations', default=300, type=int,
                        help='Maximum number of iterations (default: 300 for DE, overrides time limit if set)')
    parser.add_argument('--search_evaluations', default=None, type=int,
                        help='Maximum number of objective function evaluations (None = no limit, overrides time limit if set)')
    parser.add_argument('--stopping_criteria', type=str, default='default',
                        choices=['default', 'time_of_de'],
                        help='Stopping criteria mode: "default" uses specified time/iteration/evaluation limits, '
                             '"time_of_de" runs DE for 300 iterations then matches that time for other optimizers (requires --compare_optimizers)')
    parser.add_argument('--save_plots', default=False, action='store_true',
                        help='Save convergence plots for each instance')
    parser.add_argument('--plot_mode', type=str, default='per_instance',
                        choices=['per_instance', 'average'],
                        help='Plotting mode: "per_instance" creates all plots (absolute and percentage) for each instance, '
                             '"average" skips per-instance plots and creates only 3 averaged percentage plots across all instances.')
    parser.add_argument('--description', type=str, default='',
                        help='Brief description of this experiment for logging purposes')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for reproducibility (default: 1234)')

    # Optimizer selection
    parser.add_argument('--optimizer', type=str, default='de',
                        choices=['de', 'de_vectorized', 'de_on_steroids', 'cmaes', 'ipop_cmaes', 'bipop_cmaes', 'scipy_de', 'evox_jade', 'pygmo_pso_gen', 'evox_shade', 'evox_sade', 'evox_code', 'evox_ode', 'ngopt', 'sobol_search'],
                        help='Optimizer to use: "de" (Differential Evolution), "de_vectorized" (Vectorized DE), '
                             '"de_on_steroids" (DE with multiple strategies), "cmaes" (CMA-ES), '
                             '"ipop_cmaes" (IPOP-CMA-ES), "bipop_cmaes" (BIPOP-CMA-ES), '
                             '"scipy_de" (SciPy DE with adaptive dithering), "evox_jade" (EvoX JADE), '
                             '"pygmo_pso_gen" (Pygmo PSO Generational), "evox_shade" (EvoX SHADE), '
                             '"evox_sade" (EvoX SaDE), "evox_code" (EvoX CoDE), "evox_ode" (EvoX ODE), '
                             '"ngopt" (Nevergrad NGOpt), or "sobol_search" (Sobol quasi-random search)')

    # Differential Evolution parameters
    parser.add_argument('--de_mutate', default=0.3, type=float,
                        help='Mutation factor F for DE (default: 0.3)')
    parser.add_argument('--de_recombine', default=0.95, type=float,
                        help='Crossover rate CR for DE (default: 0.95)')

    # DE on Steroids parameters
    parser.add_argument('--steroids_strategy', default='rand1bin', type=str,
                        choices=['rand1bin', 'rand2bin', 'best1bin', 'best2bin', 'currenttobest1bin', 'randtobest1bin'],
                        help='Mutation strategy for DE on Steroids (default: rand1bin)')
    parser.add_argument('--steroids_mutate', default=0.232165, type=float,
                        help='Mutation factor F for DE on Steroids (default: 0.232165)')
    parser.add_argument('--steroids_recombine', default=0.875693, type=float,
                        help='Crossover rate CR for DE on Steroids (default: 0.875693)')

    # Scipy DE parameters
    parser.add_argument('--scipy_strategy', default='best1bin', type=str,
                        help='Mutation strategy for Scipy DE (default: best1bin, e.g., best1bin, currenttobest1bin, rand1bin)')
    parser.add_argument('--scipy_use_adaptive_mutation', default=False, action='store_true',
                        help='Use adaptive mutation (dithering) for Scipy DE (default: False)')
    parser.add_argument('--scipy_mutation_low', default=0.5, type=float,
                        help='Lower bound for mutation factor in Scipy DE adaptive mode (default: 0.5)')
    parser.add_argument('--scipy_mutation_high', default=1.0, type=float,
                        help='Upper bound for mutation factor in Scipy DE adaptive mode (default: 1.0)')
    parser.add_argument('--scipy_updating', default='immediate', type=str, choices=['immediate', 'deferred'],
                        help='Update strategy for Scipy DE: immediate or deferred (default: immediate)')

    # JADE (EvoX) parameters
    parser.add_argument('--jade_c', default=0.1, type=float,
                        help='Learning rate for JADE adaptive parameters (default: 0.1, range: 0.01-0.5)')
    parser.add_argument('--jade_num_diff_vectors', default=1, type=int,
                        help='Number of difference vectors for JADE mutation (default: 1, options: 1 or 2)')
    parser.add_argument('--jade_mean', default=None, type=float,
                        help='Mean for JADE population initialization (default: None, uniform initialization)')
    parser.add_argument('--jade_stdev', default=None, type=float,
                        help='Standard deviation for JADE population initialization (default: None, uniform initialization)')

    # ODE (EvoX) parameters
    parser.add_argument('--ode_base_vector', default='rand', type=str, choices=['rand', 'best'],
                        help='Base vector strategy for ODE mutation (default: rand, choices: rand or best)')
    parser.add_argument('--ode_num_difference_vectors', default=1, type=int,
                        help='Number of difference vectors for ODE mutation (default: 1)')
    parser.add_argument('--ode_differential_weight', default=0.5, type=float,
                        help='Mutation scaling factor F for ODE (default: 0.5)')
    parser.add_argument('--ode_cross_probability', default=0.9, type=float,
                        help='Crossover probability CR for ODE (default: 0.9)')

    # CMA-ES parameters
    parser.add_argument('--cmaes_sigma0', default=0.5, type=float,
                        help='Initial step size for CMA-ES (default: 0.5, typically 0.2-0.5 of search range)')
    parser.add_argument('--cmaes_rankmu', default=1.0, type=float,
                        help='Learning rate for rank-mu update in CMA-ES (default: 1.0)')
    parser.add_argument('--cmaes_rankone', default=1.0, type=float,
                        help='Learning rate for rank-one update in CMA-ES (default: 1.0)')
    parser.add_argument('--cmaes_sigma_sweep', nargs='+', type=float, default=None,
                        help='List of sigma values to test for CMA-ES comparison (e.g., --cmaes_sigma_sweep 0.3 0.5 1.0 1.5). '
                             'When provided, runs CMA-ES with each sigma value using a single fixed batch size.')

    # IPOP-CMA-ES parameters
    parser.add_argument('--ipop_restarts', default=5, type=int,
                        help='Number of restarts for IPOP-CMA-ES (default: 5)')
    parser.add_argument('--ipop_incpopsize', default=2.0, type=float,
                        help='Population size multiplier for IPOP-CMA-ES restarts (default: 2.0)')
    parser.add_argument('--ipop_initial_popsize', default=None, type=int,
                        help='Initial population size for IPOP-CMA-ES (default: None, uses CMA-ES library default based on problem dimension)')
    parser.add_argument('--ipop_sigma0', default=None, type=float,
                        help='Initial step size for IPOP-CMA-ES (default: None, uses cmaes_sigma0)')
    parser.add_argument('--ipop_rankmu', default=None, type=float,
                        help='Learning rate for rank-mu update in IPOP-CMA-ES (default: None, uses cmaes_rankmu)')
    parser.add_argument('--ipop_rankone', default=None, type=float,
                        help='Learning rate for rank-one update in IPOP-CMA-ES (default: None, uses cmaes_rankone)')

    # BIPOP-CMA-ES parameters
    parser.add_argument('--bipop_restarts', default=5, type=int,
                        help='Number of restarts for BIPOP-CMA-ES (default: 5)')
    parser.add_argument('--bipop_incpopsize', default=2.0, type=float,
                        help='Population size multiplier for BIPOP-CMA-ES restarts (default: 2.0)')
    parser.add_argument('--bipop_initial_popsize', default=None, type=int,
                        help='Initial population size for BIPOP-CMA-ES (default: None, uses CMA-ES library default based on problem dimension)')
    parser.add_argument('--bipop_sigma0', default=None, type=float,
                        help='Initial step size for BIPOP-CMA-ES (default: None, uses cmaes_sigma0)')
    parser.add_argument('--bipop_rankmu', default=None, type=float,
                        help='Learning rate for rank-mu update in BIPOP-CMA-ES (default: None, uses cmaes_rankmu)')
    parser.add_argument('--bipop_rankone', default=None, type=float,
                        help='Learning rate for rank-one update in BIPOP-CMA-ES (default: None, uses cmaes_rankone)')

    # SaDE (EvoX) parameters
    parser.add_argument('--sade_diff_padding_num', default=7, type=int,
                        help='Number of padding difference vectors for SaDE (default: 7)')
    parser.add_argument('--sade_lp', default=50, type=int,
                        help='Learning period for SaDE strategy adaptation (default: 50)')

    # SHADE (EvoX) parameters
    parser.add_argument('--shade_diff_padding_num', default=7, type=int,
                        help='Number of padding difference vectors for SHADE (default: 7)')

    # CoDE (EvoX) parameters
    parser.add_argument('--code_diff_padding_num', default=7, type=int,
                        help='Number of padding difference vectors for CoDE (default: 7)')
    parser.add_argument('--code_replace', default=True, action='store_true',
                        help='Enable replace mode for CoDE (default: True)')

    # Sobol Search parameters
    parser.add_argument('--sobol_scramble', default=True, action='store_true',
                        help='Use Owen scrambling for Sobol sequences (improves high-dimensional performance, default: True)')

    # Optimizer comparison
    parser.add_argument('--compare_optimizers', default=False, action='store_true',
                        help='Compare all optimizers (DE, CMA-ES, IPOP-CMA-ES, BIPOP-CMA-ES, Pygmo-DE) with the same batch size. '
                             'Requires exactly one batch size.')
    parser.add_argument('--compare_optimizer_list', type=str, default=None,
                        help='Comma-separated list of optimizers to compare (e.g., "de,jade"). '
                             'Available: de, de_vectorized, de_steroids, cmaes, ipop_cmaes, bipop_cmaes, scipy_de, jade, shade, sade, code, ode, sobol. '
                             'If not specified, compares all optimizers. Requires --compare_optimizers.')

    # Initialization strategy
    parser.add_argument('--use_lhs_init', default=False, action='store_true',
                        help='Use Latin Hypercube Sampling (LHS) for population initialization. '
                             'Applicable to: CMA-ES, IPOP-CMA-ES, BIPOP-CMA-ES, Pygmo-PSO-Gen, NGOpt. '
                             'Other optimizers ignore this flag (default: False, uniform random initialization).')

    config = parser.parse_args()
    config.device = torch.device(config.device)

    # If batch_sizes not specified, use search_batch_size as default
    if config.batch_sizes is None:
        config.batch_sizes = [config.search_batch_size]

    # Validate sigma sweep mode
    if config.cmaes_sigma_sweep is not None:
        if config.optimizer != 'cmaes':
            parser.error("--cmaes_sigma_sweep can only be used with --optimizer cmaes")
        if len(config.batch_sizes) != 1:
            parser.error("--cmaes_sigma_sweep requires exactly one batch size (use --batch_sizes 600)")

    # Validate optimizer comparison mode
    if config.compare_optimizers:
        if len(config.batch_sizes) != 1:
            parser.error("--compare_optimizers requires exactly one batch size (use --batch_sizes 600)")
        if config.cmaes_sigma_sweep is not None:
            parser.error("--compare_optimizers cannot be used with --cmaes_sigma_sweep")

    # Validate optimizer list selection
    if config.compare_optimizer_list is not None:
        if not config.compare_optimizers:
            parser.error("--compare_optimizer_list requires --compare_optimizers")

    # Validate stopping criteria mode
    if config.stopping_criteria == 'time_of_de':
        if not config.compare_optimizers:
            parser.error("--stopping_criteria time_of_de requires --compare_optimizers (it needs multiple optimizers to compare)")
        # Check if user explicitly set search_iterations or search_timelimit
        # We check sys.argv to see if they were explicitly provided
        cmd_args = args if args is not None else sys.argv[1:]
        if '--search_iterations' in cmd_args:
            parser.error("--stopping_criteria time_of_de cannot be used with --search_iterations (DE iterations are fixed at 300)")
        if '--search_timelimit' in cmd_args:
            parser.error("--stopping_criteria time_of_de cannot be used with --search_timelimit (time is determined by DE runtime)")

    return config
