import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def plot_posterior(G_dim, disrupting_pattern_indices, all_idata):
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
        G_dim, 8, figsize=(8 * 4, 3 * 4), tight_layout=True, sharex="col", sharey="row"
    )
    legend_size = 6
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
            # ax_set[2].legend(prop={'size': legend_size}, loc="upper left")

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
            # ax_set[3].legend(prop={'size': legend_size}, loc="upper left")

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
            color="orange",
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
            # ax_set[4].legend(prop={'size': legend_size}, loc="upper left")
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
            # ax_set[6].legend(prop={'size': legend_size}, loc="upper left")

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
            # ax_set[5].legend(prop={'size': legend_size}, loc="upper left")
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
            # ax_set[7].legend(prop={'size': legend_size}, loc="upper left")

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
            color="orange",
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


def plot_posterior_binary(G_dim, disrupting_pattern_indices, all_idata):
    """
    Modified function to plot bar plots of MEI, Completing, and Disrupting patterns for binary X
    in the same subplot for each latent variable G.

    Parameters:
    - G_dim (int): Dimension of the latent variable G.
    - disrupting_pattern_indices (list): List of indices corresponding to disrupting patterns.
    - all_idata (list): List containing InferenceData objects for MEI, Completing, and Disrupting 1 and Disrupting 2 patterns.

    Returns:
    - fig (matplotlib.figure.Figure): The Matplotlib figure object containing the generated plot.
    - axs (numpy.ndarray): 2D array of Matplotlib axes objects representing subplots in the figure.
    - completing_perc_diff_list (list): List of percentage differences for Completing.
    - disrupting_1_perc_diff_list (list): List of percentage differences for Disrupting 1.
    - disrupting_2_perc_diff_list (list): List of percentage differences for Disrupting 2.
    """

    fig, axs = plt.subplots(
        G_dim, 6, figsize=(6 * 8, 3 * 4), tight_layout=True, sharex="col", sharey="row"
    )
    legend_size = 6

    completing_perc_diff_list = []
    disrupting_1_perc_diff_list = []
    disrupting_2_perc_diff_list = []

    for idx, (ax_set, idata_set, disrupting_pattern_idx) in enumerate(
        zip(axs, all_idata, disrupting_pattern_indices)
    ):
        center_x_id = 4 * G_dim + idx

        # Collect probabilities for MEI, Completing, and Disrupting patterns
        center_probabilities = []
        all_probabilities = []
        for pattern_id, idata in enumerate(idata_set):
            x_prob = idata["posterior"]["X"].data[:, :, center_x_id].mean()
            center_probabilities.append(x_prob)
            all_x_prob = idata["posterior"]["X"].data.mean(axis=0).mean(axis=0)
            all_probabilities.append(all_x_prob)

            if pattern_id == 0:
                mei_x_prob = x_prob
            elif pattern_id > 0:
                perc_diff = (x_prob - mei_x_prob) / mei_x_prob * 100
                if pattern_id == 1:
                    completing_perc_diff_list.append(perc_diff)
                elif pattern_id == 2:
                    disrupting_1_perc_diff_list.append(perc_diff)
                elif pattern_id == 3:
                    disrupting_2_perc_diff_list.append(perc_diff)
            # if idx == 3:
            #     print(idata["posterior"]["X"].data[:, :, center_x_id])
            #     print(center_probabilities)

        # Plot center_probabilities in the same subplot for comparison
        ax_set[0].bar(
            ["MEI", "Completing", "Disrupting 1", "Disrupting 2"],
            center_probabilities,
            color=["blue", "green", "orange", "red"],
        )
        ax_set[0].set_ylim(0, 1)  # Probabilities range from 0 to 1

        # Plot all_probabilities in the same subplot for comparison
        for idx, probs in enumerate(all_probabilities):
            ax_set[2 + idx].bar(
                list(range(G_dim * 9)),
                probs,
                color=["red" if i == center_x_id else "blue" for i in range(G_dim * 9)],
            )
            ax_set[2 + idx].set_xticks(list(range(G_dim * 9)))
            ax_set[2 + idx].set_xticklabels(list(range(G_dim * 9)))

        # Plot G distributions (unchanged)
        mei_g_samples = idata_set[0]["posterior"]["G"].data.mean(axis=0).mean(axis=0)
        completing_g_samples = (
            idata_set[1]["posterior"]["G"].data.mean(axis=0).mean(axis=0)
        )
        disrupting_1_g_samples = (
            idata_set[2]["posterior"]["G"].data.mean(axis=0).mean(axis=0)
        )
        disrupting_2_g_samples = (
            idata_set[3]["posterior"]["G"].data.mean(axis=0).mean(axis=0)
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
            color="orange",
        )
        ax_set[1].bar(
            x=np.arange(G_dim) + G_dim * 3,
            height=disrupting_2_g_samples,
            width=0.8,
            color="red",
        )
        ax_set[1].set_xticks(np.arange(G_dim * 4))
        ax_set[1].set_xticklabels(list(np.arange(G_dim)) * 4)

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
