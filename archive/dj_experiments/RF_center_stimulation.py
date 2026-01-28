# %%
import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from models.models import HierarchicalPanelICA

from cs_experiments import create_center_stimulus
from tables import CenterExperimentConfig, CenterExperimentResult

# %%
results = CenterExperimentConfig * CenterExperimentResult

# %%
results

# %%
data_basepath = Path("/src/project/data/ica/repeat_9_crop_12_comp_40")
G_X_model_path = data_basepath / f"layer3_models/layer3_340.joblib"
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
    X_sigma=0.1,
    X_I_models=X_I_models,
    I_sigma=0.1,
)

# %%
for row_id, row in enumerate(results):
    if row_id < 10:
        continue
    print(row_id)
    stimulus = create_center_stimulus(
        hmodel,
        row["stimulus_type"],
        row["contrast_scale"],
    )
    fig, ax = plt.subplots(dpi=150)
    ax.imshow(stimulus, cmap="gray")
    ax.text(
        0.1,
        0.9,
        str(row["stimulus_type"]),
        color="orange",
        fontsize=13,
        transform=ax.transAxes,
    )
    ax.axis("off")
    fig.savefig(f"RF_center_stimulus_{row['stimulus_type']}.png", bbox_inches="tight")
    chains = row["n_chains"]
    n_draws_per_chain = row["n_draws_per_chain"]
    X_dim = hmodel.X_dim
    samples_path = row["posterior_samples"]
    with open(samples_path, "rb") as f:
        idata = pickle.load(f)
    samples = idata["posterior"]["X"].data.reshape(chains * n_draws_per_chain, X_dim)
    samples_mean = samples.mean(axis=0)

    g_samples = idata["posterior"]["G"].data.reshape(
        chains * n_draws_per_chain, row["g_dim"]
    )
    g_samples_mean = g_samples.mean(axis=0)

    center_offset = 160
    fig, axs = plt.subplots(
        nrows=4, ncols=10, figsize=(2 * 10, 2 * 4), sharex=True, sharey=True
    )
    for i, ax in enumerate(axs.flat):
        neuron_id = i + center_offset
        sns.histplot(
            samples[:, neuron_id],
            ax=ax,
            stat="density",
            color="red",
            element="step",
            alpha=0.5,
        )
        ax.axvline(samples_mean[neuron_id], color="red", linestyle="--")
        ax.text(
            0.8,
            0.9,
            f"{samples_mean[neuron_id]*1000:.1f}e-3",
            size=10,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="red",
        )
        ax.text(
            0.1,
            0.9,
            str(neuron_id),
            color="orange",
            fontsize=13,
            transform=ax.transAxes,
        )
        ax.set_ylim([0, 200])
        ax.set_ylabel("p(X|I)")
        # sns.despine(ax=ax, trim=True)
    fig.savefig(f"RF_center_posterior_{row['stimulus_type']}.png", bbox_inches="tight")
    fig, ax = plt.subplots(dpi=150)
    sns.histplot(
        samples[:, neuron_id],
        ax=ax,
        stat="density",
        color="red",
        element="step",
        alpha=0.5,
    )
    ax.axvline(samples_mean[neuron_id], color="red", linestyle="--")
    ax.text(
        0.8,
        0.9,
        f"{samples_mean[neuron_id]*1000:.1f}e-3",
        size=10,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="red",
    )
    ax.text(
        0.1,
        0.9,
        str(neuron_id),
        color="orange",
        # fontsize=13,
        transform=ax.transAxes,
    )
    # ax.set_ylim([0, 200])
    ax.set_ylabel("p(X|I)")
    sns.despine(ax=ax, trim=True)
    fig.savefig(
        f"RF_center_posterior_{row['stimulus_type']}_neuron_{row['stimulus_type']}.png",
        bbox_inches="tight",
    )

    fig, axs = plt.subplots(
        nrows=4, ncols=10, figsize=(2 * 10, 2 * 4), sharex=True, sharey=True
    )
    for i, ax in enumerate(axs.flat):
        sns.histplot(
            g_samples[:, i],
            ax=ax,
            stat="density",
            color="red",
            element="step",
            alpha=0.5,
        )
        ax.axvline(g_samples_mean[i], color="red", linestyle="--")
        ax.text(
            0.8,
            0.9,
            f"{g_samples_mean[i]*1000:.1f}e-3",
            size=10,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="red",
        )
        ax.text(
            0.1,
            0.9,
            str(i),
            color="orange",
            fontsize=13,
            transform=ax.transAxes,
        )
        # ax.set_ylim([0, 200])
        ax.set_ylabel("p(G|I)")
    fig.savefig(
        f"RF_center_posterior_{row['stimulus_type']}_g.png", bbox_inches="tight"
    )
    plt.close("all")
