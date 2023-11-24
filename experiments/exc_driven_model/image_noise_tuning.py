import pickle
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from models.hierarchical_grating_model import HierarchicalGratingModel

seed = 42
rng = np.random.default_rng(seed)

# define all constants here
G_dim = 3
G_prob = 1 / G_dim
x_sigma = 0.1
offset = 0
G_visual_nrows = 3
G_visual_ncols = 1
x_visual_nrows = 3
x_visual_ncols = 9
n_tune = 10000
n_draws = 10000
n_chains = 10
n_cores = 4

exc_fname = Path("/src/project/data/experiment/exc_images_preprocessed.npy")
exc_images = np.load(exc_fname)

exc_images_mean = exc_images.mean(axis=0)
exc_images = exc_images - exc_images_mean

samples_across_I_sigma = []
I_sigmas = np.arange(start=1, stop=100, step=10)
model_images = exc_images[-G_dim:]
for I_sigma in I_sigmas:
    model = HierarchicalGratingModel(
        gratings=model_images,
        G_prob=G_prob,
        X_sigma=x_sigma,
        I_sigma=I_sigma,
        gratings_offset=offset,
    )

    empty_image = np.zeros((36, 36))
    empty_image[12:24, 12:24] = model.grating_crops[4][2].reshape(12, 12).copy()
    MEI_center = model.grating_crops[4][2].reshape(12, 12).copy()
    MEI = empty_image

    completing = model.gratings[2].copy()

    disrupting = model.gratings[1].copy()
    disrupting[12:24, 12:24] = MEI_center.copy()

    images = [MEI, completing, disrupting]
    all_samples = []
    for image in images:
        post_samples = model.sample_posterior(
            image=image,
            n_samples=n_draws,
            random_seed=42,
            tune=n_tune,
            chains=n_chains,
            cores=n_draws,
        )
        all_samples.append(post_samples)
    samples_across_I_sigma.append(all_samples)

# save the samples as a pickle file
samples_across_I_sigma_fname = Path(
    "/src/project/data/experiment/samples_across_I_sigma.npy"
)
with open(samples_across_I_sigma_fname, "wb") as f:
    pickle.dump(samples_across_I_sigma, f)
