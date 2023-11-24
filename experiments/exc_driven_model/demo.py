# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from insilico_stimuli.stimuli import GaborSet

from models.exc_based_model import ExcitatoryModel

# %%
params = {
    "canvas_size": [36, 36],
    "sizes": [30],
    "spatial_frequencies": [1 / 3],
    "contrasts": [1.0],
    "grey_levels": [0.0],
    "eccentricities": [0.0],
    "locations": [[18, 18]],
    "phases": [np.pi / 2],
    "relative_sf": False,
    "orientations": [np.pi, np.pi / 2],
}

gabor_images = GaborSet(**params)
ncols = len(gabor_images.images())
fig, axs = plt.subplots(1, ncols=ncols, figsize=(ncols * 2, 2))
for idx, ax in enumerate(axs.flatten()):
    ax.imshow(gabor_images.images()[idx], cmap="gray", vmin=-1, vmax=1)
    ax.axis("off")

# %%
data_basepath = Path("/src/project/data/ica/repeat_9_crop_12_comp_40")
x_sigma = 0.1
I_sigma = 0.1

print("Loading X_I_models")
X_I_models_path = data_basepath / "layer2_models"
X_I_models = []
for entry in X_I_models_path.iterdir():
    # go through files and load joblib files
    if entry.is_file() and entry.suffix == ".joblib":
        model = joblib.load(entry)
        X_I_models.append(model)

print("Creating hierarchical model")
hmodel = ExcitatoryModel(
    excitatory_stimuli=gabor_images.images(),
    G_prob=0.5,
    X_sigma=x_sigma,
    X_I_models=X_I_models,
    I_sigma=I_sigma,
)

# %%
