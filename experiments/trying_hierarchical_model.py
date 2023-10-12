# %%
import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from insilico_stimuli.stimuli import CenterSurround, GaborSet

from models.models import HierarchicalPanelICA

seed = 42


data_basepath = Path("/src/project/data/ica/repeat_9_crop_12_comp_40")
G_X_model_path = data_basepath / "layer3_models/layer3_340.joblib"
X_I_models_path = data_basepath / "layer2_models"

G_X_model = joblib.load(G_X_model_path)
print("G_X_model", G_X_model)
print("G n_components", G_X_model.n_components)
G_X_mapping = G_X_model.mixing_

X_I_models = []
print("Loading layer 2 models")
for entry in X_I_models_path.iterdir():
    # go through files and load joblib files
    if entry.is_file() and entry.suffix == ".joblib":
        model = joblib.load(entry)
        X_I_models.append(model)
        # print(model)

X_patch_dim = model.components_.shape[0]
print("X_patch_dim", X_patch_dim)
I_patch_dim = model.components_.shape[1]
print("I_patch_dim", I_patch_dim)


hmodel = HierarchicalPanelICA(
    G_X_model=G_X_model,
    X_sigma=0.1,
    X_I_models=X_I_models,
    I_sigma=0.1,
)


# hmodel.visualize_learned_G()


# prior_sample_size = 10
# prior_samples = hmodel.sample_prior_predictive(
#     n_samples=prior_sample_size,
#     random_seed=seed,
# )

# with open("prior_samples.pkl", "wb") as f:
#     pickle.dump(prior_samples, f)

# %%
center_surround_params_center_only = {
    "canvas_size": [36, 36],
    "spatial_frequencies_center": [1 / 5],
    "spatial_frequencies_surround": [1 / 5],
    "phases_center": [np.pi / 2],
    "phases_surround": [np.pi / 2],
    "grey_levels": [0.0],
    "orientations_center": [0, np.pi / 2],
    "orientations_surround": [0.0, np.pi / 2],
    "locations": [[18, 18]],
    "sizes_total": [36],
    "sizes_center": [20 / 4 / 36],
    "sizes_surround": [20 / 4 / 36],
    "contrasts_center": [1.0],
    "contrasts_surround": [0],
}

center_surround_gabors_center_only = CenterSurround(
    **center_surround_params_center_only
)


blank = {
    "canvas_size": [36, 36],
    "spatial_frequencies_center": [1 / 5],
    "spatial_frequencies_surround": [1 / 5],
    "phases_center": [np.pi / 2],
    "phases_surround": [np.pi / 2],
    "grey_levels": [0.0],
    "orientations_center": [0],
    "orientations_surround": [0],
    "locations": [[18, 18]],
    "sizes_total": [36],
    "sizes_center": [20 / 4 / 36],
    "sizes_surround": [20 / 4 / 36],
    "contrasts_center": [0],
    "contrasts_surround": [0],
}
blank_stim = CenterSurround(**blank)

all_gabors = np.concatenate(
    [center_surround_gabors_center_only.images(), blank_stim.images()], axis=0
)


# %%
# center_surround_params_hv = {
#     "canvas_size": [36, 36],
#     "spatial_frequencies_center": [1 / 5],
#     "spatial_frequencies_surround": [1 / 5],
#     "phases_center": [np.pi / 2],
#     "phases_surround": [np.pi / 2],
#     "grey_levels": [0.0],
#     "orientations_center": [0, np.pi / 2],
#     "orientations_surround": [0.0, np.pi / 2],
#     "locations": [[18, 18]],
#     "sizes_total": [36],
#     "sizes_center": [20 / 4 / 36],
#     "sizes_surround": [20 / 4 / 36],
#     "contrasts_center": [1.0],
#     "contrasts_surround": [1.0],
# }

# center_surround_gabors_hv = CenterSurround(**center_surround_params_hv)
# # fig, axs = plt.subplots(2, 2, dpi=100)
# # for i, ax in enumerate(axs.flatten()):
# #     if i < center_surround_gabors_hv.images().shape[0]:
# #         ax.imshow(center_surround_gabors_hv.images()[i], cmap="gray", vmin=-1, vmax=1)
# #         # write the label on the left side of the subplot
# #         # ax.text(-0.3, 0.5, f"{i}", size=5, ha="center", va="center", transform=ax.transAxes)
# #     ax.axis("off")

# # %%
# center_surround_params_sl = {
#     "canvas_size": [36, 36],
#     "spatial_frequencies_center": [1 / 5],
#     "spatial_frequencies_surround": [1 / 5],
#     "phases_center": [np.pi / 2],
#     "phases_surround": [np.pi / 2],
#     "grey_levels": [0.0],
#     "orientations_center": [-np.pi / 4, np.pi / 4],
#     "orientations_surround": [-np.pi / 4, np.pi / 4],
#     "locations": [[18, 18]],
#     "sizes_total": [36],
#     "sizes_center": [20 / 4 / 36],
#     "sizes_surround": [20 / 4 / 36],
#     "contrasts_center": [1.0],
#     "contrasts_surround": [1.0],
# }

# center_surround_gabors_sl = CenterSurround(**center_surround_params_sl)
# # fig, axs = plt.subplots(2, 2, dpi=100)
# # for i, ax in enumerate(axs.flatten()):
# #     if i < center_surround_gabors_sl.images().shape[0]:
# #         ax.imshow(center_surround_gabors_sl.images()[i], cmap="gray", vmin=-1, vmax=1)
# #         # write the label on the left side of the subplot
# #         # ax.text(-0.3, 0.5, f"{i}", size=5, ha="center", va="center", transform=ax.transAxes)
# #     ax.axis("off")

# %%
# all_gabors = np.concatenate(
#     [center_surround_gabors_hv.images(), center_surround_gabors_sl.images()], axis=0
# )

# gabors_fname = "all_gabors.npy"
# np.save(gabors_fname, all_gabors)
# %%

n_post_samples = 1000
tune = 1000
chains = 4
cores = 4

for idx, image in enumerate(all_gabors):
    post_samples = hmodel.sample_posterior(
        image=image,
        n_samples=n_post_samples,
        tune=tune,
        chains=chains,
        cores=cores,
        random_seed=seed,
    )
    with open(f"post_samples_center_and_blank_{idx}.pkl", "wb") as f:
        pickle.dump(post_samples, f)
