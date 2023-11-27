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

# plot the generated images
plt.figure(figsize=(10, 5))
for i, img in enumerate(center_surround.images()):
    plt.subplot(4, 8, i + 1)
    plt.imshow(img, cmap="gray", vmin=-1, vmax=1)
    plt.axis("off")
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
x_sigma = 1
I_sigma = 1

model = HierarchicalGratingModel(
    gratings=gratings,
    G_prob=1 / 3,
    X_sigma=x_sigma,
    I_sigma=I_sigma,
)

# %%
model.visualize_learned_G()

# %%
