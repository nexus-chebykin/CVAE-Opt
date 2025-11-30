from config_search import get_config

import torch
import numpy as np
import random
import datetime
import os
import logging
import sys
import train
import search_control
from utils import read_instance_pkl
from VAE_8 import VAE_8
from VAE_8_newReg import VAE_8 as VAE_8_newReg


def validate_model_compatibility(config, model_data):
    """Validate that model_type matches checkpoint requirements."""
    if config.model_type == 'newreg':
        problem = model_data.get('problem', config.problem)
        problem_size = model_data.get('problem_size', config.problem_size)
        search_space_size = config.search_space_size

        errors = []
        if problem != 'TSP':
            errors.append(f"newreg model requires problem=TSP, got {problem}")
        if problem_size != 100:
            errors.append(f"newreg model requires problem_size=100, got {problem_size}")
        if search_space_size != 100:
            errors.append(f"newreg model requires latent_dim=100, got {search_space_size}")

        if errors:
            error_msg = "VAE_8_newReg compatibility check FAILED:\n  " + "\n  ".join(errors)
            error_msg += "\n\nThe 'newreg' model is specifically designed for TSP-100 with 100-dimensional latent space."
            error_msg += "\nFor other configurations, use --model_type original"
            raise ValueError(error_msg)

        logging.info("✓ VAE_8_newReg compatibility validated: TSP-100, latent_dim=100")


if __name__ == "__main__":
    now = datetime.datetime.now()
    run_id = f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"

    config = get_config()

    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if config.output_path == "":
        config.output_path = os.getcwd()
    config.output_path = os.path.join(config.output_path, "runs", "run_" + str(now.day) + "." + str(now.month) +
                                      "." + str(now.year) + "_" + str(run_id))

    os.makedirs(os.path.join(config.output_path, "search"))

    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')

    logging.info("Started Search Run")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("Version: {0}".format(train.VERSION))
    logging.info("Random seed: {0}".format(config.seed))
    if config.description:
        logging.info("Description: {0}".format(config.description))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")

    model_data = torch.load(config.model_path, config.device, weights_only=False)

    config.search_space_bound = model_data['Z_bound']
    logging.info(f"Setting search space bound to {config.search_space_bound}")

    if not config.problem:
        config.problem = model_data['problem']
    if not config.problem_size:
        config.problem_size = model_data['problem_size']

    # Validate compatibility for newreg model
    if config.model_type == 'newreg':
        validate_model_compatibility(config, model_data)

    # Instantiate the correct model architecture
    if config.model_type == 'newreg':
        model = VAE_8_newReg(config).to(config.device)
        logging.info("Using VAE_8_newReg (Transformer with cost regression)")
    else:
        model = VAE_8(config).to(config.device)
        logging.info("Using VAE_8 (Original GRU-based)")

    model.load_state_dict(model_data['parameters'])
    model.eval()

    instances, solutions = read_instance_pkl(config)

    _, avg_runtime, costs = search_control.solve_instance_set(model, config, instances, solutions)
