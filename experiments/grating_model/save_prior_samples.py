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


prior_sample_size = 5000
prior_samples = hmodel.sample_prior_predictive(
    n_samples=prior_sample_size,
    random_seed=seed,
)

with open("prior_samples.pkl", "wb") as f:
    pickle.dump(prior_samples, f)
