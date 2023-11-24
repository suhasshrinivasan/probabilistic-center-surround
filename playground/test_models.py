# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from insilico_stimuli.stimuli import CenterSurround, GaborSet

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

gratings = np.array(
    [
        center_surround.images()[idx]
        for idx in range(len(center_surround.images()))
        if center_surround.params_dict_from_idx(idx)["orientation_center"]
        == center_surround.params_dict_from_idx(idx)["orientation_surround"]
    ]
)

x_sigma = 1
I_sigma = 1

model = HierarchicalGratingModel(
    gratings=gratings,
    G_prob=1 / 3,
    X_sigma=x_sigma,
    I_sigma=I_sigma,
)

# %%
h, w = gratings.shape[1:]
grating_crops = np.array(
    [
        gratings[:, i * h // 3 : (i + 1) * h // 3, j * w // 3 : (j + 1) * w // 3]
        for i in range(3)
        for j in range(3)
    ]
)
grating_crops = grating_crops.reshape((*grating_crops.shape[:-2], -1))
# %%
overlapping_crops = np.array(
    [
        gratings[
            :,
            (i * h // 3) + h // 6 : ((i + 1) * h // 3) + h // 6,
            (j * w // 3) + w // 6 : ((j + 1) * w // 3) + w // 6,
        ]
        for i in range(2)
        for j in range(2)
    ]
)
overlapping_crops = overlapping_crops.reshape((*overlapping_crops.shape[:-2], -1))
# %%
I_dim = h * w
I_patch_dim = h // 3 * w // 3
X_dim = 9 * gratings.shape[0] + 4 * gratings.shape[0]
X_patch_dim = gratings.shape[0]
X_I_mapping = np.zeros((I_dim, X_dim))

for idx, crop in enumerate(grating_crops.transpose(0, 2, 1)):
    X_I_mapping[
        idx * I_patch_dim : (idx + 1) * I_patch_dim,
        idx * X_patch_dim : (idx + 1) * X_patch_dim,
    ] = crop

# X_I_overlap_mapping = np.zeros((2 * h // 3 * 2 * w // 3, 4 * gratings.shape[0]))
# for idx, crop in enumerate(overlapping_crops.transpose(0, 2, 1)):
#     X_I_overlap_mapping[
#         idx * I_patch_dim : (idx + 1) * I_patch_dim,
#         idx * X_patch_dim : (idx + 1) * X_patch_dim,
#     ] = crop
# %%

for idx, crop in enumerate(overlapping_crops.transpose(0, 2, 1)):
    print(
        idx,
        crop.shape,
        ((idx + h // 6) * I_patch_dim, (idx + h // 6 + 1) * I_patch_dim),
        ((idx + w // 6) * X_patch_dim, (idx + w // 6 + 1) * X_patch_dim),
    )
    X_I_mapping[
        (idx + h // 6) * I_patch_dim : (idx + h // 6 + 1) * I_patch_dim,
        (idx + w // 6) * X_patch_dim : (idx + w // 6 + 1) * X_patch_dim,
    ] = crop
    print(
        X_I_mapping[
            (idx + h // 6) * I_patch_dim : (idx + h // 6 + 1) * I_patch_dim,
            (idx + w // 6) * X_patch_dim : (idx + w // 6 + 1) * X_patch_dim,
        ].shape
    )
# %%
