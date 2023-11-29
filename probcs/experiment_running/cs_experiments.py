from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from models.models import HierarchicalPanelICA


def create_center_stimulus(hmodel, stimulus_type, contrast_scale):
    stimulus = np.zeros((hmodel.I_side, hmodel.I_side))
    local_X_I_mapping = np.array([model.mixing_ for model in hmodel.X_I_models])
    stimulus[
        hmodel.I_patch_side : 2 * hmodel.I_patch_side,
        hmodel.I_patch_side : 2 * hmodel.I_patch_side,
    ] = local_X_I_mapping[4, :, stimulus_type - 160].reshape(
        hmodel.I_patch_side, hmodel.I_patch_side
    )
    return stimulus * contrast_scale


def center_experiment(
    config_id,
    seed,
    g_dim,
    n_draws_per_chain,
    n_chains,
    cores,
    n_burnin,
    contrast_scale,
    x_sigma,
    i_sigma,
    stimulus_type,
):
    print("Running experiment with config:")
    print(
        f"seed={seed},"
        f" g_dim={g_dim},"
        f" n_draws_per_chain={n_draws_per_chain},"
        f" n_chains={n_chains},"
        f" cores={cores},"
        f" n_burnin={n_burnin},"
        f" contrast_scale={contrast_scale},"
        f" x_sigma={x_sigma},"
        f" i_sigma={i_sigma},"
        f" stimulus_type={stimulus_type}"
    )
    data_basepath = Path("/src/project/data/ica/repeat_9_crop_12_comp_40")
    G_X_model_path = data_basepath / f"layer3_models/layer3_{g_dim}.joblib"
    print("Loading G_X_model")
    G_X_model = joblib.load(G_X_model_path)
    print("G_X_model", G_X_model)

    print("Loading X_I_models")
    X_I_models_path = data_basepath / "layer2_models"
    X_I_models = []
    for entry in X_I_models_path.iterdir():
        # go through files and load joblib files
        if entry.is_file() and entry.suffix == ".joblib":
            model = joblib.load(entry)
            X_I_models.append(model)

    print("Creating hierarchical model")
    hmodel = HierarchicalPanelICA(
        G_X_model=G_X_model,
        X_sigma=x_sigma,
        X_I_models=X_I_models,
        I_sigma=i_sigma,
    )

    print("Preparing stimulus")
    if stimulus_type == -1:
        stimulus = np.zeros((hmodel.I_side, hmodel.I_side))
    else:
        stimulus = create_center_stimulus(hmodel, stimulus_type, contrast_scale)

    fig, ax = plt.subplots()
    ax.imshow(stimulus)
    fig.savefig("stimulus.pdf")

    print("Sampling posterior")
    posterior_samples = hmodel.sample_posterior(
        image=stimulus,
        n_samples=n_draws_per_chain,
        chains=n_chains,
        cores=cores,
        tune=n_burnin,
        random_seed=seed,
    )
    return posterior_samples
