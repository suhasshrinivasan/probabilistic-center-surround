# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from probcs.datajoint.exc_tables import ExcExponentResult, ExcExponentConfig, schema
from pathlib import Path
from probcs.models.pattern_completion_model import PatternCompletionModel
from probcs.experiment_running.exc_experiment import create_stimuli


# %%
def plot_posterior_half(G_dim, disrupting_pattern_indices, all_idata):
    """
    Plots posterior distributions of latent variables G and X for MEI,
    Completing, and two Disrupting patterns.

    Parameters:
    - G_dim (int): Dimension of the latent variable G.
    - disrupting_pattern_indices (list): List of indices corresponding to disrupting patterns.
    - all_idata (list): List containing InferenceData objects for MEI, Completing, and Disrupting 1 and Disrupting 2 patterns.

    Returns:
    - fig (matplotlib.figure.Figure): The Matplotlib figure object containing the generated plot.
    - axs (numpy.ndarray): 2D array of Matplotlib axes objects representing subplots in the figure.

    Each row of subplots corresponds to a different latent variable, and each column represents
    a specific aspect of the analysis (e.g., marginal distributions, G variable distributions).

    The function visualizes the posterior distributions of latent variables for MEI, Completing,
    and Disrupting patterns, comparing their means and percentage differences. The plot includes
    histograms, bar plots, and descriptive statistics to aid in the interpretation of the model results.
    """

    fig, axs = plt.subplots(
        G_dim,
        2,
        figsize=((2 + 3) * 2, (G_dim + 3) * 2),
        tight_layout=True,
        sharex="col",
    )
    legend_size = 10
    possible_center_x_ids = [4 * G_dim + i for i in range(G_dim)]
    completing_perc_diff_list = []
    disrupting_1_perc_diff_list = []
    disrupting_2_perc_diff_list = []
    for idx, (ax_set, idata_set, disrupting_pattern_idx) in enumerate(
        zip(axs, all_idata, disrupting_pattern_indices)
    ):
        center_x_id = 4 * G_dim + idx
        # i for i in range(idx, G_dim)
        completing_surround_x_ids = [
            i for i in range(idx, 9 * G_dim, G_dim) if i != center_x_id
        ]
        # complement of completing_surround_x_ids
        not_completing_surround_x_ids = [
            i
            for i in range(9 * G_dim)
            if (i not in completing_surround_x_ids) and (i != center_x_id)
        ]
        disrupting_surround_x_ids = [
            i
            for i in range(disrupting_pattern_idx, 9 * G_dim, G_dim)
            if i not in possible_center_x_ids
        ]
        # complement of disrupting_surround_x_ids
        not_disrupting_surround_x_ids = [
            i
            for i in range(9 * G_dim)
            if (i not in disrupting_surround_x_ids) and (i != center_x_id)
        ]
        print(not_disrupting_surround_x_ids)

        # idata['posterior']['X'].data.shape = (n_chains, n_samples, n_x_dims)
        mei_idata = idata_set[0]
        completing_idata = idata_set[1]
        disrupting_1_idata = idata_set[2]
        disrupting_2_idata = idata_set[3]

        mei_x_samples = mei_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        mei_x_mean = mei_x_samples.mean()
        sns.histplot(
            mei_x_samples,
            ax=ax_set[0],
            stat="density",
            color="blue",
            element="step",
            alpha=0.3,
            label="MEI",
        )

        # mei_idata["posterior"]["G"].data.shape = (n_chains, n_samples, n_G_dims)
        mei_g_samples = mei_idata["posterior"]["G"].data.mean(axis=0).mean(axis=0)
        # mei_g_samples.shape = (n_G_dims,)

        completing_x_samples = (
            completing_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        completing_x_mean = completing_x_samples.mean()
        completing_perc_diff = (completing_x_mean - mei_x_mean) / mei_x_mean * 100
        completing_perc_diff_list.append(completing_perc_diff)
        sns.histplot(
            completing_x_samples,
            ax=ax_set[0],
            stat="density",
            color="green",
            element="step",
            alpha=0.3,
            label="Completing",
        )

        completing_g_samples = (
            completing_idata["posterior"]["G"].data.mean(axis=0).mean(axis=0)
        )

        disrupting_1_x_samples = (
            disrupting_1_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        disrupting_1_x_mean = disrupting_1_x_samples.mean()
        disrupting_1_perc_diff = (disrupting_1_x_mean - mei_x_mean) / mei_x_mean * 100
        disrupting_1_perc_diff_list.append(disrupting_1_perc_diff)
        sns.histplot(
            disrupting_1_x_samples,
            ax=ax_set[0],
            stat="density",
            color="brown",
            element="step",
            alpha=0.3,
            label="Disrupting 1",
        )

        disrupting_1_g_samples = (
            disrupting_1_idata["posterior"]["G"].data.mean(axis=0).mean(axis=0)
        )

        disrupting_2_x_samples = (
            disrupting_2_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        disrupting_2_x_mean = disrupting_2_x_samples.mean()
        disrupting_2_perc_diff = (disrupting_2_x_mean - mei_x_mean) / mei_x_mean * 100
        disrupting_2_perc_diff_list.append(disrupting_2_perc_diff)
        sns.histplot(
            disrupting_2_x_samples,
            ax=ax_set[0],
            stat="density",
            color="red",
            element="step",
            alpha=0.3,
            label="Disrupting 2",
        )

        disrupting_2_g_samples = (
            disrupting_2_idata["posterior"]["G"].data.mean(axis=0).mean(axis=0)
        )

        ax_set[1].bar(x=np.arange(G_dim), height=mei_g_samples, width=0.8, color="blue")
        ax_set[1].bar(
            x=np.arange(G_dim) + G_dim,
            height=completing_g_samples,
            width=0.8,
            color="green",
        )
        ax_set[1].bar(
            x=np.arange(G_dim) + G_dim * 2,
            height=disrupting_1_g_samples,
            width=0.8,
            color="brown",
        )
        ax_set[1].bar(
            x=np.arange(G_dim) + G_dim * 3,
            height=disrupting_2_g_samples,
            width=0.8,
            color="red",
        )
        ax_set[1].set_xticks(np.arange(G_dim * 4))
        ax_set[1].set_xticklabels(list(np.arange(G_dim)) * 4)

        ax_set[0].text(
            0.7,
            0.9,
            f"$\mu$={mei_x_mean:.4f}",
            size=8,
            ha="left",
            va="bottom",
            transform=ax_set[0].transAxes,
            color="blue",
        )
        ax_set[0].text(
            0.7,
            0.8,
            f"$\mu$={completing_x_mean:.4f}, $\Delta$={completing_perc_diff:.4f}%",
            size=8,
            ha="left",
            va="bottom",
            transform=ax_set[0].transAxes,
            color="green",
        )
        ax_set[0].text(
            0.7,
            0.7,
            f"$\mu$={disrupting_1_x_mean:.4f}, $\Delta$={disrupting_1_perc_diff:.4f}%",
            size=8,
            ha="left",
            va="bottom",
            transform=ax_set[0].transAxes,
            color="brown",
        )
        ax_set[0].text(
            0.7,
            0.6,
            f"$\mu$={disrupting_2_x_mean:.4f}, $\Delta$={disrupting_2_perc_diff:.4f}%",
            size=8,
            ha="left",
            va="bottom",
            transform=ax_set[0].transAxes,
            color="red",
        )

        ax_set[0].axvline(
            mei_x_mean, color="blue", linestyle="--", linewidth=1, label="MEI $\mu$"
        )
        ax_set[0].axvline(
            completing_x_mean,
            color="green",
            linestyle="--",
            linewidth=1,
            label="completing $\mu$",
        )
        ax_set[0].axvline(
            disrupting_1_x_mean,
            color="brown",
            linestyle="--",
            linewidth=1,
            label="disrupting 1 $\mu$",
        )
        ax_set[0].axvline(
            disrupting_2_x_mean,
            color="red",
            linestyle="--",
            linewidth=1,
            label="disrupting 2 $\mu$",
        )
        ax_set[0].legend(prop={"size": legend_size}, loc="upper left")

    completing_perc_diff_list = np.array(completing_perc_diff_list)
    disrupting_1_perc_diff_list = np.array(disrupting_1_perc_diff_list)
    disrupting_2_perc_diff_list = np.array(disrupting_2_perc_diff_list)

    return (
        fig,
        axs,
        completing_perc_diff_list,
        disrupting_1_perc_diff_list,
        disrupting_2_perc_diff_list,
    )


# %%
def plot_samples_relative_mei(
    G_dim, disrupting_pattern_indices, all_idata, xlim=(0.6, 1.3), xticks=[0.6, 1.3]
):
    colors = plt.cm.tab10(range(len(all_idata)))
    fig, axs = plt.subplots(nrows=1, ncols=3, dpi=300, sharey=True, tight_layout=True)
    for idx, (idata_set, disrupting_pattern_idx) in enumerate(
        zip(all_idata, disrupting_pattern_indices)
    ):
        center_x_id = 4 * G_dim + idx

        # idata['posterior']['X'].data.shape = (n_chains, n_samples, n_x_dims)
        mei_idata = idata_set[0]
        completing_idata = idata_set[1]
        disrupting_1_idata = idata_set[2]
        disrupting_2_idata = idata_set[3]

        mei_x_samples = mei_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        mei_x_mean = mei_x_samples.mean()

        completing_x_samples = (
            completing_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        completing_x_mean = completing_x_samples.mean()

        axs[0].scatter(
            mei_x_samples,
            completing_x_samples,
            color=colors[idx],
            marker=".",
            s=1,
            alpha=0.4,
        )

        disrupting_1_x_samples = (
            disrupting_1_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        disrupting_1_x_mean = disrupting_1_x_samples.mean()

        disrupting_2_x_samples = (
            disrupting_2_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        disrupting_2_x_mean = disrupting_2_x_samples.mean()
        axs[1].scatter(
            mei_x_samples,
            disrupting_1_x_samples,
            color=colors[idx],
            marker=".",
            s=1,
            alpha=0.4,
        )
        axs[2].scatter(
            mei_x_samples,
            disrupting_2_x_samples,
            color=colors[idx],
            marker=".",
            s=1,
            alpha=0.4,
        )

    for ax in axs:
        # draw a diagonal line
        ax.plot(
            xlim,
            xlim,
            color="gray",
            # linestyle="dashed",
            linewidth=1,
        )
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(xlim)
        ax.set_yticks(xticks)
        ax.set_aspect("equal", adjustable="box")
        sns.despine(ax=ax, trim=True)
        # break

    for idx, (idata_set, disrupting_pattern_idx) in enumerate(
        zip(all_idata, disrupting_pattern_indices)
    ):
        center_x_id = 4 * G_dim + idx

        # idata['posterior']['X'].data.shape = (n_chains, n_samples, n_x_dims)
        mei_idata = idata_set[0]
        completing_idata = idata_set[1]
        disrupting_1_idata = idata_set[2]
        disrupting_2_idata = idata_set[3]

        mei_x_samples = mei_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        mei_x_mean = mei_x_samples.mean()

        completing_x_samples = (
            completing_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        completing_x_mean = completing_x_samples.mean()

        axs[0].scatter(
            mei_x_mean,
            completing_x_mean,
            color=colors[idx],
            marker="*",
            s=30,
            alpha=1,
            edgecolor="black",
            linewidth=0.2,
        )
        axs[0].set_xlabel("Response to MEI", fontsize=8)

        disrupting_1_x_samples = (
            disrupting_1_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        disrupting_1_x_mean = disrupting_1_x_samples.mean()

        axs[1].scatter(
            mei_x_mean,
            disrupting_1_x_mean,
            color=colors[idx],
            marker="*",
            s=30,
            alpha=1,
            edgecolor="black",
            linewidth=0.2,
        )
        axs[1].set_xlabel("Response to MEI", fontsize=8)

        disrupting_2_x_samples = (
            disrupting_2_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        disrupting_2_x_mean = disrupting_2_x_samples.mean()
        axs[2].scatter(
            mei_x_mean,
            disrupting_2_x_mean,
            color=colors[idx],
            marker="*",
            s=30,
            alpha=1,
            edgecolor="black",
            linewidth=0.2,
        )
        axs[0].set_ylabel("Response to completing pattern", fontsize=8)
        axs[1].set_ylabel("Response to disrupting 1 pattern", fontsize=8)
        axs[2].set_ylabel("Response to disrupting 2 pattern", fontsize=8)

    return fig, axs


# %%
# from probcs.utils.plotting import plot_posterior, plot_samples_relative_mei

# %%
results = ExcExponentConfig() * ExcExponentResult()
# %%

key_results = results.fetch(
    "config_id",
    "average_exc",
    "average_inh_1",
    "average_inh_2",
    order_by="average_inh_1",
    as_dict=True,
)

for idx, result in enumerate(key_results):
    # highest_inh_1_config_id = key_results[0]["config_id"]
    # # '32645fb5a995f73c4aba131b1c598425'
    # # %%
    # highest_inh_1_results = (results & {"config_id": highest_inh_1_config_id}).fetch(
    #     download_path="/tmp", as_dict=True
    # )
    if idx > 0:
        break
    # with open(highest_inh_1_results[idx]["all_idata"], "rb") as f:
    #     all_idata = pickle.load(f)
    # print("config_id:", result["config_id"])
    result_data = (results & {"config_id": result["config_id"]}).fetch1()
    print(result_data)
    with open(result_data["all_idata"], "rb") as f:
        all_idata = pickle.load(f)
    # g_dim = highest_inh_1_results[idx]["g_dim"]
    g_dim = result_data["g_dim"]
    # disrupting_pattern_indices = highest_inh_1_results[idx]["disrupting_pattern_indices"]
    disrupting_pattern_indices = result_data["disrupting_pattern_indices"]
    (
        fig,
        axs,
        completing_perc_diff_list,
        disrupting_1_perc_diff_list,
        disrupting_2_perc_diff_list,
    ) = plot_posterior_half(
        all_idata=all_idata,
        disrupting_pattern_indices=disrupting_pattern_indices,
        G_dim=g_dim,
    )
    fig_samples, axs_samples = plot_samples_relative_mei(
        all_idata=all_idata,
        disrupting_pattern_indices=disrupting_pattern_indices,
        G_dim=g_dim,
        xlim=(0.3, 1.4),
        xticks=[0.3, 1.4],
    )

# %%
all_idata[0][0]["posterior"]["X"].data.shape
# %%
from probcs.utils.plotting import plot_posterior

plot_result = plot_posterior(
    G_dim=g_dim,
    disrupting_pattern_indices=disrupting_pattern_indices,
    all_idata=all_idata,
)
# %%
for idata, disrupting_pattern_idx in zip(all_idata, disrupting_pattern_indices):
    fig, ax = plt.subplots(dpi=300)
    neuron_id = disrupting_pattern_idx + 4 * g_dim
    print(
        "disrupting neuron mean for MEI:",
        idata[0]["posterior"]["X"].data[:, :, neuron_id].mean(),
    )
    sns.histplot(
        idata[0]["posterior"]["X"].data[:, :, neuron_id].flatten(),
        label="MEI",
        stat="density",
        color="blue",
        element="step",
        ax=ax,
        alpha=0.4,
    )
    print(
        "disrupting neuron mean for Completing:",
        idata[1]["posterior"]["X"].data[:, :, neuron_id].mean(),
    )
    sns.histplot(
        idata[1]["posterior"]["X"].data[:, :, neuron_id].flatten(),
        label="CI",
        stat="density",
        color="green",
        element="step",
        ax=ax,
        alpha=0.4,
    )
    print(
        "disrupting neuron mean for Disrupting 1:",
        idata[2]["posterior"]["X"].data[:, :, neuron_id].mean(),
    )
    sns.histplot(
        idata[2]["posterior"]["X"].data[:, :, neuron_id].flatten(),
        label="DI",
        stat="density",
        color="brown",
        element="step",
        ax=ax,
        alpha=0.4,
    )
    print("----------")
    ax.legend()
# %%
for idx, (idata, disrupting_pattern_idx) in enumerate(
    zip(all_idata, disrupting_pattern_indices)
):
    # print the distribution of neurons other than the right center neuron
    fig, ax = plt.subplots(dpi=300)
    right_center_neuron_id = 4 * g_dim + idx
    other_neuron_ids = [i for i in range(9 * g_dim) if i != right_center_neuron_id]
    print(
        "mean of right center neuron for MEI:",
        idata[0]["posterior"]["X"].data[:, :, right_center_neuron_id].mean(),
    )
    for neuron_id in other_neuron_ids:
        sns.histplot(
            idata[0]["posterior"]["X"].data[:, :, neuron_id].flatten(),
            label="MEI",
            stat="density",
            color="blue",
            element="step",
            ax=ax,
            alpha=0.4,
        )
# %%
exc_fname = "/src/project/data/experiment/exc_images_preprocessed.npy"
exc_image_ids = [0, 1, 3, 8, 9]
exc_images = np.load(exc_fname)
exc_images_mean = exc_images.mean(axis=0)
exc_images = exc_images - exc_images_mean
vmin = exc_images.min()
vmax = exc_images.max()
patterns = exc_images[exc_image_ids]
g_dim = len(patterns)
model = PatternCompletionModel(
    patterns=patterns,
    G_prob=result_data["g_prob"],
    X_sigma=result_data["x_sigma"],
    I_sigma=result_data["i_sigma"],
    patterns_offset=result_data["patterns_offset"],
    G_X_exponent=result_data["g_x_exponent"],
)
# %%
(
    MEIs,
    completing_patterns,
    disrupting_patterns_1,
    disrupting_patterns_2,
    disrupting_pattern_indices,
) = create_stimuli(patterns)
# %%
for MEI, completing_pattern, disrupting_pattern_1, disrupting_pattern_2 in zip(
    MEIs, completing_patterns, disrupting_patterns_1, disrupting_patterns_2
):
    fig, axs = plt.subplots(1, 4, dpi=300)
    axs[0].imshow(MEI, cmap="gray", vmin=vmin, vmax=vmax)
    axs[0].axis("off")
    axs[1].imshow(completing_pattern, cmap="gray", vmin=vmin, vmax=vmax)
    axs[1].axis("off")
    axs[2].imshow(disrupting_pattern_1, cmap="gray", vmin=vmin, vmax=vmax)
    axs[2].axis("off")
    axs[3].imshow(disrupting_pattern_2, cmap="gray", vmin=vmin, vmax=vmax)
    axs[3].axis("off")
# %%
high_contrast_MEI = np.array(MEIs) * 5
plt.imshow(high_contrast_MEI[0], cmap="gray", vmin=vmin, vmax=vmax)
# %%
post_samples_dict = model.sample_posterior(
    high_contrast_MEI[0], n_samples=500, random_seed=42, chains=4, cores=4
)
# %%
post_samples_dict["posterior"]["X"].data[:, :, g_dim * 4].mean()
# %%
post_samples_dict["posterior"]["G"].data[:, :, 3].sum()
# %%
zeros = np.zeros_like(MEI)
post_samples_dict_2 = model.sample_posterior(
    zeros, n_samples=500, random_seed=42, chains=4, cores=4
)

# %%
post_samples_dict_2["posterior"]["G"].data[:, :, 4].sum()
# %%
plt.imshow(completing_patterns[0], cmap="gray", vmin=vmin, vmax=vmax)
# %%
new_MEI = np.zeros_like(MEI)
new_MEI[12:36, 12:36] = completing_patterns[0][12:36, 12:36].copy()
plt.imshow(new_MEI, cmap="gray", vmin=vmin, vmax=vmax)
# %%
post_samples_dict_3 = model.sample_posterior(
    new_MEI, n_samples=500, random_seed=42, chains=4, cores=4
)

# %%
post_samples_dict_3["posterior"]["G"].data[:, :, 4].sum()

# %%
new_MEI_2 = np.zeros_like(MEI)
new_MEI_2[0:36, 12:24] = completing_patterns[0][0:36, 12:24].copy()
plt.imshow(new_MEI_2, cmap="gray", vmin=vmin, vmax=vmax)
# %%
post_samples_dict_3 = model.sample_posterior(
    new_MEI_2, n_samples=500, random_seed=42, chains=4, cores=4
)
# %%
new_MEI_3 = np.zeros_like(MEI)
new_MEI_3[0:36, 12:36] = completing_patterns[0][0:36, 12:36].copy()
plt.imshow(new_MEI_3, cmap="gray", vmin=vmin, vmax=vmax)
# %%
post_samples_dict_4 = model.sample_posterior(
    new_MEI_3, n_samples=500, random_seed=42, chains=4, cores=4
)


# %%
post_samples_dict_4["posterior"]["G"].data[:, :, 0].sum()

# %%
new_MEI_4 = np.zeros_like(MEI)
new_MEI_4[12:36, 12:36] = completing_patterns[0][12:36, 12:36].copy()
plt.imshow(new_MEI_4, cmap="gray", vmin=vmin, vmax=vmax)
# %%
post_samples_dict_5 = model.sample_posterior(
    new_MEI_4, n_samples=500, random_seed=42, chains=4, cores=4
)

# %%
post_samples_dict_5["posterior"]["G"].data[:, :, 0].sum()
# %%
