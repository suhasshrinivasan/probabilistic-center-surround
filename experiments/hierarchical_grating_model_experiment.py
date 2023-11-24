# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from insilico_stimuli.stimuli import CenterSurround, GaborSet
from sklearn.metrics.pairwise import cosine_similarity

from models.hierarchical_grating_model import HierarchicalGratingModel

canvas_size = [36, 36]
locations = [[18, 18]]  # center position
sizes_total = [36]  # total size (center + surround)
sizes_center = [0.5]  # portion of radius used for center circle
sizes_surround = [0.5]  # defines the starting portion of radius for surround
contrasts_center = [1.0]  # try 2 center contrasts
contrasts_surround = [1.0]  # surround contrast
orientations_center = [0.0, np.pi / 2, np.pi / 4]  # variable orientation
orientations_surround = [0.0, np.pi / 2, np.pi / 4]  # center only
spatial_frequencies_center = [0.2]  # fixed spatial frequency
phases_center = [np.pi]  # center phases
grey_levels = [0.0]  # fixed grey level
# spatial_frequencies_surround = [0.1, 0.3]  # optional parameter, default: same as spatial_frequencies_center
# phases_surround = [np.pi/4]                # optional parameter, default: same as phases_center

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

# gratings_offset = 0.5
gratings = np.array(
    [
        center_surround.images()[idx]
        for idx in range(len(center_surround.images()))
        if center_surround.params_dict_from_idx(idx)["orientation_center"]
        == center_surround.params_dict_from_idx(idx)["orientation_surround"]
    ]
)
plt.figure(figsize=(10, 5))
for i, img in enumerate(gratings):
    plt.subplot(4, 8, i + 1)
    plt.imshow(img, cmap="gray", vmin=-1, vmax=1)
    plt.axis("off")

# %%
G_prob = 1 / 3
x_sigma = 0.1
I_sigma = 1

x_posterior_samples = []
mei_means_list = []
completing_means_list = []
completing_deltas = []
disrupting_means_list = []
disrupting_deltas = []

gratings_offsets = np.linspace(0, 1, 11)
for gratings_offset in gratings_offsets:
    model = HierarchicalGratingModel(
        gratings=gratings,
        G_prob=G_prob,
        X_sigma=x_sigma,
        I_sigma=I_sigma,
        gratings_offset=gratings_offset,
    )
    print(f"gratings_offset: {gratings_offset}")
    print("visualizing learned G and X")
    model.visualize_learned_G(nrows=1, ncols=3)
    model.visualize_learned_X(nrows=3, ncols=9)
    empty_image = np.zeros((36, 36))
    MEI_center = model.grating_crops[4][0].reshape(12, 12)
    empty_image[12:24, 12:24] = MEI_center
    MEI = empty_image
    print("visualizing MEI")
    plt.imshow(MEI, cmap="gray")
    plt.axis("off")
    plt.show()
    print("visualizing completing image")
    completing_image = model.gratings[0]
    plt.imshow(completing_image, cmap="gray")
    plt.axis("off")
    print("visualizing disrupting image")
    disrupting_image_1 = model.gratings[1]
    disrupting_image_1[12:24, 12:24] = MEI_center
    plt.imshow(disrupting_image_1, cmap="gray")
    plt.axis("off")

    images = [MEI, completing_image, disrupting_image_1]
    n_samples = 1000
    n_tune = 1000
    all_samples = []
    for image in images:
        post_samples = model.sample_posterior(
            image=image,
            n_samples=n_samples,
            random_seed=42,
            tune=n_tune,
            chains=4,
            cores=4,
        )
        all_samples.append(post_samples)
    print("visualizing posterior samples of neuron 12")
    n_neurons = 27
    all_x_samples_reshaped = []
    for samples in all_samples:
        samples = samples["posterior"]["X"].data.reshape(4 * n_samples, n_neurons)
        all_x_samples_reshaped.append(samples)
    x_posterior_samples.append(all_x_samples_reshaped)
    fig, ax = plt.subplots(dpi=300)
    neuron_id = 12
    sns.histplot(
        all_x_samples_reshaped[0][:, neuron_id],
        ax=ax,
        stat="density",
        color="blue",
        element="step",
        alpha=0.3,
        label="MEI",
    )
    sns.histplot(
        all_x_samples_reshaped[1][:, neuron_id],
        ax=ax,
        stat="density",
        color="green",
        element="step",
        alpha=0.3,
        label="completing",
    )
    sns.histplot(
        all_x_samples_reshaped[2][:, neuron_id],
        ax=ax,
        stat="density",
        color="red",
        element="step",
        alpha=0.3,
        label="disrupting",
    )

    MEI_mean = all_x_samples_reshaped[0][:, neuron_id].mean()
    mei_means_list.append(MEI_mean)
    completing_mean = all_x_samples_reshaped[1][:, neuron_id].mean()
    completing_means_list.append(completing_mean)
    completing_perc_change = (completing_mean - MEI_mean) / MEI_mean
    completing_deltas.append(completing_perc_change)
    disrupting_mean = all_x_samples_reshaped[2][:, neuron_id].mean()
    disrupting_means_list.append(disrupting_mean)
    disrupting_perc_change = (disrupting_mean - MEI_mean) / MEI_mean
    disrupting_deltas.append(disrupting_perc_change)

    ax.text(
        0.7,
        0.9,
        f"$\mu$={MEI_mean:.4f}",
        size=10,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        color="blue",
    )
    ax.text(
        0.7,
        0.8,
        f"$\mu$={completing_mean:.4f}, $\Delta$={completing_perc_change*100:.1f}%",
        size=10,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        color="green",
    )
    ax.text(
        0.7,
        0.7,
        f"$\mu$={disrupting_mean:.4f}, $\Delta$={disrupting_perc_change*100:.1f}%",
        size=10,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        color="red",
    )

    ax.axvline(MEI_mean, color="blue", linestyle="--", linewidth=0.5, label="MEI $\mu$")
    ax.axvline(
        completing_mean,
        color="green",
        linestyle="--",
        linewidth=0.5,
        label="completing $\mu$",
    )
    ax.axvline(
        disrupting_mean,
        color="red",
        linestyle="--",
        linewidth=0.5,
        label="disrupting $\mu$",
    )

    ax.set_ylabel("p(X|I)")
    ax.legend(loc="upper left")


# %%
x_posterior_samples = np.array(x_posterior_samples)
mei_means_list = np.array(mei_means_list)
completing_means_list = np.array(completing_means_list)
completing_deltas = np.array(completing_deltas)
disrupting_means_list = np.array(disrupting_means_list)
disrupting_deltas = np.array(disrupting_deltas)
np.save("x_posterior_samples.npy", x_posterior_samples)
np.save("mei_means_list.npy", mei_means_list)
np.save("completing_means_list.npy", completing_means_list)
np.save("completing_deltas.npy", completing_deltas)
np.save("disrupting_means_list.npy", disrupting_means_list)
np.save("disrupting_deltas.npy", disrupting_deltas)
