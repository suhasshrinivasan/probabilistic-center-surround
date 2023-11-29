from argparse import ArgumentParser

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from pandas import DataFrame as DF

from models import HierarchicalModel
from utils import rate_code

mpl.style.use("seaborn")
sns.set_palette("colorblind")
sns.set_context("paper")
sns.set_style("darkgrid")
mpl.style.use("seaborn")


def analyze_prior(prior_idata, gabor_filters, orientation_preferences, config):
    orientation_samples = prior_idata["prior"]["global_orientation"].data.squeeze()
    neuron_samples = prior_idata["prior"]["neurons"].data.squeeze()
    stimulus_samples = prior_idata["prior_predictive"]["stimulus"].data.squeeze()
    lambdas = np.array(
        [
            rate_code(
                baseline_firing_rate=config["baseline_firing_rate"],
                global_orientation_variable=orientation_sample,
                orientation_preferences=orientation_preferences,
                vonmises_loc=config["vonmises_loc"],
                vonmises_kappa=config["vonmises_kappa"],
                rate_code_signature=config["rate_code_signature"],
            )
            for orientation_sample in orientation_samples
        ]
    )
    fig_prior, axs_prior = plt.subplots(nrows=6, ncols=9, dpi=300, sharex="col")
    fig_prior.suptitle("Prior samples", fontsize=8)
    # name the first column of the figure as "stimulus" with small font size
    axs_prior[0, 0].set_title("Stimulus", fontsize=6)
    # name rest of columns as "neurons" i
    for i in range(1, 6):
        axs_prior[0, i].set_title(f"Neuron {i}", fontsize=6)
    # name the last column as "global orientation"
    axs_prior[0, 6].set_title("Sample", fontsize=6)
    axs_prior[0, 7].set_title("Mean rate", fontsize=6)
    axs_prior[0, 8].set_title("global_Orientation", fontsize=6)
    # axs_prior[0, 9].set_title("Correlation", fontsize=6)
    for i, ax in enumerate(axs_prior[:, 0]):
        ax.imshow(stimulus_samples[i], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    for i, axs_prior_features in enumerate(axs_prior[:, 1:-3]):
        for j, ax in enumerate(axs_prior_features):
            ax.imshow(
                gabor_filters[j] * neuron_samples[i][j], cmap="gray", vmin=-1, vmax=1
            )
            ax.axis("off")
    for i, ax in enumerate(axs_prior[:, -3]):
        ax.bar(np.arange(1, 6), neuron_samples[i], color="green")
        ax.tick_params(axis="both", which="major", labelsize=3)
        ax.yaxis.set_tick_params(pad=1.5)
        ax.xaxis.set_tick_params(pad=1.5)
    for i, ax in enumerate(axs_prior[:, -2]):
        ax.bar(np.arange(1, 6), 1 / lambdas[i], color="limegreen", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=3)
        ax.yaxis.set_tick_params(pad=1.5)
        ax.xaxis.set_tick_params(pad=1.5)
    for i, ax in enumerate(axs_prior[:, -1]):
        ax.text(
            0.02,
            0.90,
            f"{int(np.rad2deg(orientation_samples[i]))}$\degree$",
            transform=ax.transAxes,
            fontsize=3,
        )
        ax.quiver(
            0,
            0,
            np.cos(orientation_samples[i]),
            np.sin(orientation_samples[i]),
            scale=2,
            width=0.05,
            color="orange",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    log_dict = {
        "Prior Sample": wandb.Image(fig_prior),
    }
    corr = np.corrcoef(neuron_samples.T)
    fig_corr, ax_corr = plt.subplots(dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, cmap="seismic", ax=ax_corr)
    ax_corr.set_facecolor("white")
    ax_corr.set_aspect("equal")
    ax_corr.set_xticklabels(range(1, 6))
    ax_corr.set_yticklabels(range(1, 6))
    ax_corr.set_title("Prior correlation matrix of neuron samples", fontsize=8)
    log_dict.update({"Prior Correlation": wandb.Image(fig_corr)})
    total_horizontal_correlation = np.sum(corr[np.tril_indices(3, k=-1)])
    total_vertical_correlation = np.sum(corr[3:, 3:][np.tril_indices(2, k=-1)])
    log_dict.update(
        {
            "Total horizontal correlation": total_horizontal_correlation,
            "Total vertical correlation": total_vertical_correlation,
        }
    )
    # convert log_dict to a pandas dataframe
    log_df = DF.from_dict([log_dict])
    log_table = wandb.Table(dataframe=log_df)
    wandb.log({"Prior Samples": log_table})


def analyze_posterior(trace, gabor_filters, config):
    fig_samples, axs_samples = plt.subplots(
        nrows=2, ncols=5, dpi=300, sharex="row", sharey="row"
    )
    fig_samples.suptitle("Posterior samples", fontsize=8)
    for idx, ax in enumerate(axs_samples[0]):
        response = trace["neurons"][:, idx]
        response_mean = np.mean(trace["neurons"][:, idx].ravel())
        sns.histplot(
            response.ravel(), color="green", stat="probability", element="step", ax=ax
        )
        ax.text(0.05, 0.95, idx + 1, fontsize=6, transform=ax.transAxes)
        ax.text(0.7, 0.95, f"{response_mean:.2f}", fontsize=6, transform=ax.transAxes)
        ax.tick_params(axis="both", which="major", labelsize=5)
        # reduce the font size of the x and y labels
        ax.xaxis.label.set_size(5)
        ax.yaxis.label.set_size(5)
        # set a title for the subplot (the neuron number)
        ax.set_title(f"Neuron {idx+1}", fontsize=6)
    for idx, ax in enumerate(axs_samples[1]):
        ax.imshow(gabor_filters[idx], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    log_dict = {
        "Posterior Samples": wandb.Image(fig_samples),
    }

    corr = np.corrcoef(trace["neurons"].T)
    fig_corr, ax_corr = plt.subplots(dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, cmap="seismic", ax=ax_corr)
    ax_corr.set_facecolor("white")
    ax_corr.set_aspect("equal")
    ax_corr.set_xticklabels(range(1, 6))
    ax_corr.set_yticklabels(range(1, 6))
    ax_corr.set_title("Posterior correlation matrix of neuron samples", fontsize=8)
    log_dict.update({"Posterior Correlation": wandb.Image(fig_corr)})

    fig_orientation, ax_orientation = plt.subplots(dpi=300)
    sns.histplot(
        trace["global_orientation"].ravel(),
        color="orange",
        stat="probability",
        element="step",
        ax=ax_orientation,
    )
    ax_orientation.set_title("Posterior samples of orientation", fontsize=8)
    ax_orientation.set_xlabel("Orientation ($\degree$)", fontsize=6)
    ax_orientation.set_ylabel("Probability", fontsize=6)
    ax_orientation.tick_params(axis="both", which="major", labelsize=5)
    # convert xtick labels to degrees
    xticks = ax_orientation.get_xticks()
    xticklabels = [f"{int(np.rad2deg(xtick))}" for xtick in xticks]
    ax_orientation.set_xticklabels(xticklabels)
    ax_orientation.xaxis.label.set_size(5)
    ax_orientation.yaxis.label.set_size(5)
    mean_orientation = np.mean(np.rad2deg(trace["global_orientation"].ravel()))
    std_orientation = np.std(np.rad2deg(trace["global_orientation"].ravel()))
    log_dict.update(
        {
            "Posterior Orientation": wandb.Image(fig_orientation),
            "Mean Orientation": mean_orientation,
            "Std Orientation": std_orientation,
        }
    )
    return log_dict


def analyze_traces(center_cut_trace, congruent_trace, incongruent_trace):
    center_neuron_id = 1
    fig, ax = plt.subplots(dpi=300)
    sns.kdeplot(
        center_cut_trace["neurons"][:, center_neuron_id].ravel(),
        linewidth=2,
        shade=True,
        label="MEI",
    )
    sns.kdeplot(
        congruent_trace["neurons"][:, center_neuron_id].ravel(),
        linewidth=2,
        shade=True,
        label="Congruent",
    )
    sns.kdeplot(
        incongruent_trace["neurons"][:, center_neuron_id].ravel(),
        linewidth=2,
        shade=True,
        label="Incongruent",
    )
    ax.set_xlabel("Mean firing rate")
    ax.set_ylabel("Probability")
    ax.legend(prop={"size": 8}, title="Stimulus", title_fontsize=10, loc="upper center")
    ax.set_title("Posterior samples of center neuron", fontsize=8)
    log_dict = {"Posterior Center Neuron": wandb.Image(fig)}
    center_cut_mean = np.mean(center_cut_trace["neurons"][:, center_neuron_id].ravel())
    congruent_mean = np.mean(congruent_trace["neurons"][:, center_neuron_id].ravel())
    incongruent_mean = np.mean(
        incongruent_trace["neurons"][:, center_neuron_id].ravel()
    )
    log_dict.update(
        {
            "MEI_mean": center_cut_mean,
            "congruent_mean": congruent_mean,
            "incongruent_mean": incongruent_mean,
        }
    )
    diff_congruent = congruent_mean - center_cut_mean
    diff_incongruent = incongruent_mean - center_cut_mean
    sum_diffs = np.abs(diff_congruent) + np.abs(diff_incongruent)
    log_dict.update(
        {
            "diff_congruent": diff_congruent,
            "diff_incongruent": diff_incongruent,
            "sum_diffs": sum_diffs,
        }
    )
    wandb.log(log_dict)


def main(config):
    gabor_filters = np.load(config["gabor_filters_fname"])
    orientation_preferences_fname = (
        config["gabor_filters_fname"][:-5] + "_orientations.npy"
    )
    orientation_preferences = np.load(orientation_preferences_fname)
    n_neurons = gabor_filters.shape[0]
    model = HierarchicalModel(
        n_neurons=n_neurons,
        global_orientation_bounds=(
            np.deg2rad(config["global_orientation_lower_bound"]),
            np.deg2rad(config["global_orientation_upper_bound"]),
        ),
        gabor_filters=gabor_filters,
        orientation_preferences=orientation_preferences,
        stimulus_std=config["stimulus_std"],
        rate_code_signature=config["rate_code_signature"],
        vonmises_loc=config["vonmises_loc"],
        vonmises_kappa=config["vonmises_kappa"],
        baseline_firing_rate=config["baseline_firing_rate"],
    )
    print("Prior sampling")
    prior_idata = model.sample_prior(config["prior_sampling_draws"])
    analyze_prior(prior_idata, gabor_filters, orientation_preferences, config)
    center_cut_stimulus, congruent_stimulus, incongruent_stimulus = np.load(
        config["stimuli_fname"]
    )
    print("Posterior sampling")
    center_cut_trace, congruent_trace, incongruent_trace = [
        model.sample_posterior(
            observed_stimulus=stimulus,
            draws=config["posterior_sampling_draws"],
            tunes=config["posterior_sampling_tunes"],
            chains=config["posterior_sampling_chains"],
            cores=config["posterior_sampling_cores"],
            random_seed=config["random_seed"],
            return_inferencedata=False,
        )
        for stimulus in [center_cut_stimulus, congruent_stimulus, incongruent_stimulus]
    ]
    posterior_log_dicts = []
    for trace, stimulus in zip(
        [center_cut_trace, congruent_trace, incongruent_trace],
        [center_cut_stimulus, congruent_stimulus, incongruent_stimulus],
    ):
        posterior_log_dict = analyze_posterior(trace, gabor_filters, config)
        posterior_log_dict.update({"Stimulus": wandb.Image(stimulus)})
        posterior_log_dicts.append(posterior_log_dict)
    posterior_log_df = DF(posterior_log_dicts)
    log_table = wandb.Table(dataframe=posterior_log_df)
    wandb.log({"Posterior Samples": log_table})
    analyze_traces(center_cut_trace, congruent_trace, incongruent_trace)


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
    parser.add_argument("--global_orientation_upper_bound", type=float, default=180.0)
    parser.add_argument(
        "--rate_code_signature", type=str, default="inv_(base+vonmises)"
    )
    parser.add_argument("--baseline_firing_rate", type=float, default=1.0)
    parser.add_argument("--vonmises_loc", type=float, default=0.0)
    parser.add_argument("--vonmises_kappa", type=float, default=1.0)
    parser.add_argument("--stimulus_std", type=float, default=0.1)
    parser.add_argument("--prior_sampling_draws", type=int, default=10)
    parser.add_argument("--posterior_sampling_draws", type=int, default=1000)
    parser.add_argument("--posterior_sampling_tunes", type=int, default=1000)
    parser.add_argument("--posterior_sampling_chains", type=int, default=4)
    parser.add_argument("--posterior_sampling_cores", type=int, default=4)
    parser.add_argument("--posterior_sampling_tries", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=42)
    main(vars(parser.parse_args()))
