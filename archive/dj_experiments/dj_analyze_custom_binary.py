# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from probcs.datajoint.exc_tables import CustomBinaryConfig, CustomBinaryResult, schema
from pathlib import Path
from probcs.models.pattern_completion_model import PatternCompletionModel
from probcs.experiment_running.exc_experiment import create_stimuli
from probcs.utils.plotting import plot_posterior_binary


# %%
# from probcs.utils.plotting import plot_posterior, plot_samples_relative_mei

# %%
results = CustomBinaryConfig() * CustomBinaryResult()
# %%

key_results = results.fetch(
    "config_id",
    as_dict=True,
)

for idx, result in enumerate(key_results):
    # highest_inh_1_config_id = key_results[0]["config_id"]
    # # '32645fb5a995f73c4aba131b1c598425'
    # # %%
    # highest_inh_1_results = (results & {"config_id": highest_inh_1_config_id}).fetch(
    #     download_path="/tmp", as_dict=True
    # )
    # if idx > 0:
    #     break
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
    ) = plot_posterior_binary(g_dim, disrupting_pattern_indices, all_idata)

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
