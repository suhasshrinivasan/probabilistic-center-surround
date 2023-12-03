import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_posterior(G_dim, disrupting_pattern_indices, all_idata):
    """
    Plots posterior distributions of latent variables for MEI (Main Effect of Interest),
    Completing, and Disrupting patterns.

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
        3, 8, figsize=(8 * 4, 3 * 4), tight_layout=True, sharex="col", sharey="row"
    )
    possible_center_x_ids = [4 * G_dim + i for i in range(G_dim)]
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
        # plot a bar plot of the Gs
        ax_set[1].bar(
            x=np.arange(G_dim), height=mei_g_samples, color="blue", label="MEI"
        )

        colors = plt.cm.tab10(range(len(completing_surround_x_ids)))
        for completing_surround_index, completing_surround_x_id in enumerate(
            completing_surround_x_ids
        ):
            completing_surround_x_samples = (
                completing_idata["posterior"]["X"]
                .data[:, :, completing_surround_x_id]
                .flatten()
            )
            sns.histplot(
                completing_surround_x_samples,
                ax=ax_set[2],
                stat="density",
                color=colors[completing_surround_index],
                element="step",
                alpha=0.3,
                label=f"{completing_surround_x_id}",
            )
            ax_set[2].legend(prop={"size": 10}, loc="upper left")

        colors = plt.cm.tab20(range(len(not_completing_surround_x_ids)))
        for not_completing_surround_index, not_completing_surround_x_id in enumerate(
            not_completing_surround_x_ids
        ):
            not_completing_surround_x_samples = (
                completing_idata["posterior"]["X"]
                .data[:, :, not_completing_surround_x_id]
                .flatten()
            )
            sns.histplot(
                not_completing_surround_x_samples,
                ax=ax_set[3],
                stat="density",
                color=colors[not_completing_surround_index],
                element="step",
                alpha=0.3,
                label=f"{not_completing_surround_x_id}",
            )
            ax_set[3].legend(prop={"size": 8}, loc="upper left")

        completing_x_samples = (
            completing_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        completing_x_mean = completing_x_samples.mean()
        completing_perc_diff = (completing_x_mean - mei_x_mean) / mei_x_mean * 100
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
        ax_set[1].bar(
            x=np.arange(G_dim) + 0.2,
            height=completing_g_samples,
            color="green",
            label="Completing",
        )

        disrupting_1_x_samples = (
            disrupting_1_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        disrupting_1_x_mean = disrupting_1_x_samples.mean()
        disrupting_1_perc_diff = (disrupting_1_x_mean - mei_x_mean) / mei_x_mean * 100
        sns.histplot(
            disrupting_1_x_samples,
            ax=ax_set[0],
            stat="density",
            color="orange",
            element="step",
            alpha=0.3,
            label="Disrupting 1",
        )

        disrupting_1_g_samples = (
            disrupting_1_idata["posterior"]["G"].data.mean(axis=0).mean(axis=0)
        )
        ax_set[1].bar(
            x=np.arange(G_dim) + 0.4,
            height=disrupting_1_g_samples,
            color="orange",
            label="Disrupting 1",
        )

        disrupting_2_x_samples = (
            disrupting_2_idata["posterior"]["X"].data[:, :, center_x_id].flatten()
        )
        disrupting_2_x_mean = disrupting_2_x_samples.mean()
        disrupting_2_perc_diff = (disrupting_2_x_mean - mei_x_mean) / mei_x_mean * 100
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
        ax_set[1].bar(
            x=np.arange(G_dim) + 0.6,
            height=disrupting_2_g_samples,
            color="red",
            label="Disrupting 2",
        )

        colors = plt.cm.tab10(range(len(disrupting_surround_x_ids)))
        for disrupting_surround_index, disrupting_surround_x_id in enumerate(
            disrupting_surround_x_ids
        ):
            disrupting_1_surround_x_samples = (
                disrupting_1_idata["posterior"]["X"]
                .data[:, :, disrupting_surround_x_id]
                .flatten()
            )
            sns.histplot(
                disrupting_1_surround_x_samples,
                ax=ax_set[4],
                stat="density",
                color=colors[disrupting_surround_index],
                element="step",
                alpha=0.3,
                label=f"{disrupting_surround_x_id}",
            )
            ax_set[4].legend(prop={"size": 10}, loc="upper left")
            disrupting_2_surround_x_samples = (
                disrupting_2_idata["posterior"]["X"]
                .data[:, :, disrupting_surround_x_id]
                .flatten()
            )
            sns.histplot(
                disrupting_2_surround_x_samples,
                ax=ax_set[6],
                stat="density",
                color=colors[disrupting_surround_index],
                element="step",
                alpha=0.3,
                label=f"{disrupting_surround_x_id}",
            )
            ax_set[6].legend(prop={"size": 8}, loc="upper left")

        colors = plt.cm.tab20(range(len(not_disrupting_surround_x_ids)))
        for not_disrupting_surround_index, not_disrupting_surround_x_id in enumerate(
            not_disrupting_surround_x_ids
        ):
            not_disrupting_1_surround_x_samples = (
                disrupting_1_idata["posterior"]["X"]
                .data[:, :, not_disrupting_surround_x_id]
                .flatten()
            )
            sns.histplot(
                not_disrupting_1_surround_x_samples,
                ax=ax_set[5],
                stat="density",
                color=colors[not_disrupting_surround_index],
                element="step",
                alpha=0.3,
                label=f"{not_disrupting_surround_x_id}",
            )
            ax_set[5].legend(prop={"size": 8}, loc="upper left")
            not_disrupting_2_surround_x_samples = (
                disrupting_2_idata["posterior"]["X"]
                .data[:, :, not_disrupting_surround_x_id]
                .flatten()
            )
            sns.histplot(
                not_disrupting_2_surround_x_samples,
                ax=ax_set[7],
                stat="density",
                color=colors[not_disrupting_surround_index],
                element="step",
                alpha=0.3,
                label=f"{not_disrupting_surround_x_id}",
            )
            ax_set[7].legend(prop={"size": 8}, loc="upper left")

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
            color="orange",
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
            color="orange",
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
        ax_set[0].legend(prop={"size": 10}, loc="upper left")
    return fig, axs
