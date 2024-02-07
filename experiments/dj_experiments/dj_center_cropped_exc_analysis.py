# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from probcs.datajoint.exc_tables import ExcExponentCenterCropResult, ExcExponentCenterCropConfig, schema


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
results = ExcExponentCenterCropConfig() * ExcExponentCenterCropResult()
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
    if idx > 9:
        break
    # with open(highest_inh_1_results[idx]["all_idata"], "rb") as f:
    #     all_idata = pickle.load(f)
    # print("config_id:", result["config_id"])
    result_data = (results & {"config_id": result["config_id"]}).fetch1()
    if result_data['config_id'] != '562d7f9354a5cb2ae16ddb69e1cc5255':
        continue
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
