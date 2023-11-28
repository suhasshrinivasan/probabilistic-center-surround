from pathlib import Path

import numpy as np
from insilico_stimuli.stimuli import CenterSurround
from sklearn.metrics.pairwise import cosine_similarity

from models.pattern_completion_model import PatternCompletionModel


def get_center_x_mean(idata, center_x_id):
    """
    Get the mean of the center x posterior distribution from an idata object

    Args:
        idata (arviz.InferenceData): idata object containing posterior samples

    Returns:
        center_x_mean (np.ndarray): mean of center x posterior distribution
            where mean is computed across samples and not across chains
            Hence center_x_mean.shape = (n_chains,)
    """
    x_data = idata["posterior"]["X"].data
    # x_data.shape = (n_chains, n_samples, n_x_dims)
    center_x_mean = x_data[:, :, center_x_id].mean(axis=-1)
    # center_x_mean.shape = (n_chains,)
    return center_x_mean


def get_g_mean(idata):
    """
    Get the mean of the g posterior distribution from an idata object

    Args:
        idata (arviz.InferenceData): idata object containing posterior samples

    Returns:
        g_mean (np.ndarray): mean of g posterior distribution
            where mean is computed across samples and not across chains
            Hence g_mean.shape = (n_chains,)
    """
    g_data = idata["posterior"]["G"].data
    # g_data.shape = (n_chains, n_samples, n_g_dims)
    g_mean = g_data.mean(axis=1)
    # g_mean.shape = (n_chains, n_g_dims)
    return g_mean


def center_surround_experiment(
    seed,
    patterns,
    g_dim,
    g_prob,
    x_sigma,
    i_sigma,
    patterns_offset,
    n_tune,
    n_draws,
    n_chains,
    n_cores,
    # stimuli=None,
    pymc_logging=False,
):
    """
    Experiment to test the pattern completion model with center-surround stimuli.
    Note that the model being considered is models.pattern_completion_model.PatternCompletionModel.

    Args:
        patterns (np.ndarray): patterns to compose
        g_prob (float): Bernoulli probability of g_i being 1
        x_sigma (float): Laplace distribution scale parameter for x
        i_sigma (float): Normal distribution scale parameter for i
        patterns_offset (float): offset to add to patterns to induce
            cosine similarity between patterns
        # stimuli (np.ndarray): stimuli to condition on of shape (N, np.sqrt(pattern_dim), np.sqrt(pattern_dim))
        n_samples (int): number of samples to draw per chain
        random_seed (int): random seed for reproducibility
        tune (int): number of tuning steps per chain
        chains (int): number of chains
        cores (int): number of cores to use

    Returns:
        model (PatternCompletionModel): model built and used for inference]
        all_idata (list): list of idata sets for each center neuron
            Length of all_idata is g_dim
            Each element of all_idata is a list of idatas for each stimulus (MEI, completing surround, disrupting surround1, disrupting surround2, ...)
        all_stimuli (list): list of all stimuli sets for each center neuron
            Length of all_stimuli is g_dim
            Each element of all_stimuli is a list of stimuli for each center-neuron (MEI, completing surround, disrupting surround1, disrupting surround2, ...)
        all_g_means (list): list of all g means for each stimuli for each center neuron
            Length of all_g_means is g_dim
            Each element of all_g_means is a list of g means for each stimulus (MEI, completing surround, disrupting surround1, disrupting surround2, ...)
        all_g_means_sde (list): list of all g means standard errors for each stimuli for each center neuron
            Length of all_g_means_sde is g_dim
            Each element of all_g_means_sde is a list of g means standard errors for each stimulus (MEI, completing surround, disrupting surround1, disrupting surround2, ...)
        all_center_x_means (list): list of all center x means for each center neuron
            Length of all_center_x_means is g_dim
            Each element of all_center_x_means is a list of center x means for each stimulus (MEI, completing surround, disrupting surround1, disrupting surround2, ...)
        all_center_x_means_sde (list): list of all center x means standard errors for each center neuron
            Length of all_center_x_means_sde is g_dim
            Each element of all_center_x_means_sde is a list of center x means standard errors for each stimulus (MEI, completing surround, disrupting surround1, disrupting surround2, ...)
        all_center_x_perc_change_means (list): list of all center x percent change means for each center neuron
            Length of all_center_x_perc_change_means is g_dim - 1
            Each element of all_center_x_perc_change_means is a list of center x percent change means for each stimulus (completing surround, disrupting surround1, disrupting surround2, ...)
        all_center_x_perc_change_means_sde (list): list of all center x percent change means standard errors for each center neuron
            Length of all_center_x_perc_change_means_sde is g_dim - 1
            Each element of all_center_x_perc_change_means_sde is a list of center x percent change means standard errors for each stimulus (completing surround, disrupting surround1, disrupting surround2, ...)
        Note that sde is computed over sampling chains
    """
    print("Running experiment with config:")
    print(
        f"seed={seed},"
        f" g_dim={g_dim},"
        f" g_prob={g_prob},"
        f" x_sigma={x_sigma},"
        f" i_sigma={i_sigma},"
        f" pattern_offset={patterns_offset},"
        f" n_tune={n_tune},"
        f" n_draws={n_draws},"
        f" n_chains={n_chains},"
        f" n_cores={n_cores}"
    )
    print("Building model ...")
    model = PatternCompletionModel(
        patterns=patterns,
        G_prob=g_prob,
        X_sigma=x_sigma,
        I_sigma=i_sigma,
        patterns_offset=patterns_offset,
    )
    print("Setting up experiment ...")
    all_idata = []
    all_stimuli = []
    all_g_means = []
    all_g_means_sde = []
    all_center_x_means = []
    all_center_x_means_sde = []
    all_center_x_perc_change_means = []
    all_center_x_perc_change_means_sde = []
    total_center_x_dims = g_dim
    center_x_ids = np.arange(4 * g_dim, 4 * g_dim + total_center_x_dims)
    print("Number of center x dims: ", total_center_x_dims)
    for idx, center_x_id in enumerate(center_x_ids):
        print(f"Running experiment for center x dim {idx}/{total_center_x_dims} ...")
        idatas = []
        stimuli = []
        g_means = []
        g_means_sde = []
        center_x_means = []
        center_x_means_sde = []
        center_x_perc_change_means = []
        center_x_perc_change_means_sde = []
        # first create MEI
        MEI = np.zeros((36, 36))
        MEI_center = model.pattern_crops[4][idx].reshape(12, 12).copy()
        MEI[12:24, 12:24] = MEI_center.copy()
        # append to stimuli
        stimuli.append(MEI)
        print("Sampling posterior for MEI ...")
        # sample posterior for MEI
        mei_idata = model.sample_posterior(
            image=MEI,
            n_samples=n_draws,
            random_seed=seed,
            tune=n_tune,
            chains=n_chains,
            cores=n_draws,
            pymc_logging=pymc_logging,
        )
        # append to idata
        idatas.append(mei_idata)
        # compute center stats
        mei_x_mean = get_center_x_mean(mei_idata, center_x_id)
        mei_x_mean_mean_chains = mei_x_mean.mean()
        center_x_means.append(mei_x_mean_mean_chains)
        mei_x_mean_sde_chains = mei_x_mean.std() / np.sqrt(n_chains)
        center_x_means_sde.append(mei_x_mean_sde_chains)
        mei_g_mean = get_g_mean(mei_idata)
        mei_g_mean_mean_chains = mei_g_mean.mean(axis=0)
        mei_g_mean_sde_chains = mei_g_mean.std(axis=0) / np.sqrt(n_chains)
        g_means.append(mei_g_mean_mean_chains)
        g_means_sde.append(mei_g_mean_sde_chains)

        print("Sampling posterior for completing surround ...")
        # now create completing surround
        completing_pattern = model.patterns[idx].copy()
        # append to stimuli
        stimuli.append(completing_pattern)
        # sample posterior for completing surround
        completing_idata = model.sample_posterior(
            image=completing_pattern,
            n_samples=n_draws,
            random_seed=seed,
            tune=n_tune,
            chains=n_chains,
            cores=n_draws,
            pymc_logging=pymc_logging,
        )
        # append to idata
        all_idata.append(completing_idata)
        # compute center stats
        completing_x_mean = get_center_x_mean(completing_idata, center_x_id)
        completing_x_mean_mean_chains = completing_x_mean.mean()
        completing_x_mean_sde_chains = completing_x_mean.std() / np.sqrt(n_chains)
        center_x_means.append(completing_x_mean_mean_chains)
        center_x_means_sde.append(completing_x_mean_sde_chains)
        completing_g_mean = get_g_mean(completing_idata)
        completing_g_mean_mean_chains = completing_g_mean.mean(axis=0)
        completing_g_mean_sde_chains = completing_g_mean.std(axis=0) / np.sqrt(n_chains)
        g_means.append(completing_g_mean_mean_chains)
        g_means_sde.append(completing_g_mean_sde_chains)
        completing_perc_change = (completing_x_mean - mei_x_mean) / mei_x_mean * 100
        completing_perc_change_mean_chains = completing_perc_change.mean()
        completing_perc_change_sde_chains = completing_perc_change.std() / np.sqrt(
            n_chains
        )
        center_x_perc_change_means.append(completing_perc_change_mean_chains)
        center_x_perc_change_means_sde.append(completing_perc_change_sde_chains)

        print(f"Constructing {g_dim - 1} disruptive surround(s) ...")
        # now create disrupting surrounds
        for disrupting_idx, disrupting_id in enumerate(
            [i for i in range(g_dim) if i != idx]
        ):
            print(
                f"Running experiment for disruptive surround {disrupting_idx}/{g_dim - 1} ..."
            )
            disrupting_pattern = model.patterns[disrupting_id].copy()
            disrupting_pattern[12:24, 12:24] = MEI_center.copy()
            # append to stimuli
            stimuli.append(disrupting_pattern)
            # sample posterior for disrupting surround
            disrupting_idata = model.sample_posterior(
                image=disrupting_pattern,
                n_samples=n_draws,
                random_seed=seed,
                tune=n_tune,
                chains=n_chains,
                cores=n_draws,
                pymc_logging=pymc_logging,
            )
            # append to idata
            all_idata.append(disrupting_idata)
            # compute center stats
            disrupting_x_mean = get_center_x_mean(disrupting_idata, center_x_id)
            disrupting_x_mean_mean_chains = disrupting_x_mean.mean()
            disrupting_x_mean_sde_chains = disrupting_x_mean.std() / np.sqrt(n_chains)
            disrupting_g_mean = get_g_mean(disrupting_idata)
            disrupting_g_mean_mean_chains = disrupting_g_mean.mean(axis=0)
            disrupting_g_mean_sde_chains = disrupting_g_mean.std(axis=0) / np.sqrt(
                n_chains
            )
            g_means.append(disrupting_g_mean_mean_chains)
            g_means_sde.append(disrupting_g_mean_sde_chains)
            center_x_means.append(disrupting_x_mean_mean_chains)
            center_x_means_sde.append(disrupting_x_mean_sde_chains)
            disrupting_perc_change = (disrupting_x_mean - mei_x_mean) / mei_x_mean * 100
            disrupting_perc_change_mean_chains = disrupting_perc_change.mean()
            disrupting_perc_change_sde_chains = disrupting_perc_change.std() / np.sqrt(
                n_chains
            )
            center_x_perc_change_means.append(disrupting_perc_change_mean_chains)
            center_x_perc_change_means_sde.append(disrupting_perc_change_sde_chains)

        all_idata.append(idatas)
        all_stimuli.append(stimuli)
        all_center_x_means.append(center_x_means)
        all_center_x_means_sde.append(center_x_means_sde)
        all_center_x_perc_change_means.append(center_x_perc_change_means)
        all_center_x_perc_change_means_sde.append(center_x_perc_change_means_sde)
        all_g_means.append(g_means)
        all_g_means_sde.append(g_means_sde)

    return (
        model,
        all_idata,
        all_stimuli,
        all_g_means,
        all_g_means_sde,
        all_center_x_means,
        all_center_x_means_sde,
        all_center_x_perc_change_means,
        all_center_x_perc_change_means_sde,
    )


def exc_dj_experiment(
    config_id,
    seed,
    g_dim,
    g_prob,
    x_sigma,
    i_sigma,
    patterns_offset,
    n_tune,
    n_draws,
    n_chains,
    n_cores,
):
    """
    DJ function to run center surround experiment.
    All this does is call center_surround_experiment by constructing the patterns from the experimental data.
    """
    # first load data
    # TODO: parameterize this
    exc_fname = Path("/src/project/data/experiment/exc_images_preprocessed.npy")
    exc_images = np.load(exc_fname)
    exc_images_mean = exc_images.mean(axis=0)
    exc_images = exc_images - exc_images_mean

    patterns = exc_images[:g_dim]
    return center_surround_experiment(
        seed=seed,
        patterns=patterns,
        g_dim=g_dim,
        g_prob=g_prob,
        x_sigma=x_sigma,
        i_sigma=i_sigma,
        patterns_offset=patterns_offset,
        n_tune=n_tune,
        n_draws=n_draws,
        n_chains=n_chains,
        n_cores=n_cores,
    )


def grating_dj_experiment(
    config_id,
    seed,
    g_dim,
    g_prob,
    x_sigma,
    i_sigma,
    patterns_offset,
    n_tune,
    n_draws,
    n_chains,
    n_cores,
):
    """
    DJ function to run center surround experiment.
    All this does is call center_surround_experiment by constructing grating patterns.
    """
    # create stimuli
    # set parameters
    canvas_size = [36, 36]
    locations = [[18, 18]]  # center position
    sizes_total = [36]  # total size (center + surround)
    sizes_center = [0.5]  # portion of radius used for center circle
    sizes_surround = [0.5]  # defines the starting portion of radius for surround
    contrasts_center = [1.0]  # define center contrast
    contrasts_surround = [1.0]  # surround contrast
    spatial_frequencies_center = [0.2]  # fixed spatial frequency
    phases_center = [np.pi]  # center phases
    grey_levels = [0.0]  # fixed grey level

    # set orientations of gratings
    orientations_center = list(
        np.linspace(-np.pi / 2, np.pi / 2, g_dim, endpoint=False)
    )
    orientations_surround = list(
        np.linspace(-np.pi / 2, np.pi / 2, g_dim, endpoint=False)
    )

    center_surround = CenterSurround(
        canvas_size=canvas_size,
        locations=locations,
        sizes_total=sizes_total,
        sizes_center=sizes_center,
        sizes_surround=sizes_surround,
        contrasts_center=contrasts_center,
        contrasts_surround=contrasts_surround,
        orientations_center=orientations_center,
        orientations_surround=orientations_surround,
        spatial_frequencies_center=spatial_frequencies_center,
        phases_center=phases_center,
        grey_levels=grey_levels,
    )

    # choose only those gratings where center and surround have the same orientation
    patterns = np.array(
        [
            center_surround.images()[idx]
            for idx in range(len(center_surround.images()))
            if center_surround.params_dict_from_idx(idx)["orientation_center"]
            == center_surround.params_dict_from_idx(idx)["orientation_surround"]
        ]
    )

    return center_surround_experiment(
        seed=seed,
        patterns=patterns,
        g_dim=g_dim,
        g_prob=g_prob,
        x_sigma=x_sigma,
        i_sigma=i_sigma,
        patterns_offset=patterns_offset,
        n_tune=n_tune,
        n_draws=n_draws,
        n_chains=n_chains,
        n_cores=n_cores,
    )
