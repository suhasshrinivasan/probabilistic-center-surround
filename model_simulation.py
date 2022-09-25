from argparse import ArgumentParser

import numpy as np
import wandb

from models import HierarchicalModel


def main(config):
    # load gabor filters and orientation preferences
    gabor_filters = np.load(config["gabor_filters_fname"])
    orientation_preferences_fname = (
        config["gabor_filters_fname"][:-5] + "_orientations.npy"
    )
    orientation_preferences = np.load(orientation_preferences_fname)
    # init model
    n_neurons = gabor_filters.shape[0]
    model = HierarchicalModel(
        n_neurons=n_neurons,
        global_orientation_bounds=(
            config["global_orientation_lower_bound"],
            config["global_orientation_upper_bound"],
        ),
        gabor_filters=gabor_filters,
        orientation_preferences=orientation_preferences,
        stimulus_std=config["stimulus_std"],
        rate_code_signature=config["rate_code_signature"],
        vonmises_loc=config["vonmises_loc"],
        vonmises_kappa=config["vonmises_kappa"],
        baseline_firing_rate=config["baseline_firing_rate"],
    )

    # load center surround stimuli
    stimuli = np.load(config["stimuli_fname"])


if __name__ == "__main__":
    wandb.init()
    parser = ArgumentParser()
    parser.add_argument(
        "--gabor_filters_fname",
        type=str,
        default="/src/project/computed/center_masked_surround_gabors.npy",
    )
    parser.add_argument(
        "--stimuli_fname",
        type=str,
        default="/src/project/computed/center_masked_surround_stimuli.npy",
    )
    parser.add_argument("--global_orientation_lower_bound", type=float, default=0.0)
    parser.add_argument("--global_orientation_upper_bound", type=float, default=360.0)
    parser.add_argument("--rate_code_signature", type=str, default="base")
    parser.add_argument("--baseline_firing_rate", type=float, default=1.0)
    parser.add_argument("--vonmises_loc", type=float, default=1.0)
    parser.add_argument("--vonmises_kappa", type=float, default=1.0)
    parser.add_argument("--stimulus_std", type=float, default=0.1)
    parser.add_argument("--prior_sampling_draws", type=int, default=10)
    parser.add_argument("--posterior_sampling_draws", type=int, default=1000)
    parser.add_argument("--posterior_sampling_tunes", type=int, default=1000)
    parser.add_argument("--posterior_sampling_chains", type=int, default=4)
    parser.add_argument("--posterior_sampling_cores", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=42)
    main(vars(parser.parse_args()))
