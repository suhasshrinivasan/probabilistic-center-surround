
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from probcs.models.pattern_completion_model import PatternCompletionModel, BinaryPatternCompletionModel
from probcs.experiment_running.exc_experiment import center_surround_experiment, create_stimuli
from probcs.utils.utils import load_images, get_center_patches
from torchvision import transforms
import arviz as az
from sklearn.metrics.pairwise import cosine_similarity
from probcs.utils.utils import select_focus_images, get_excitatory_images
from probcs.utils.plotting import plot_posterior_binary
import pickle

seed = 42
rng = np.random.default_rng(seed)


# - Run ~1000 small natural image patches in the normative model
# - Pick ~200 most activating patches per neuron
# - Show full field version and analyze response modulation
# - Post results in channel to discuss whether to include in paper


natimgs = load_images(
    crop_size=36,
    n_samples=1_000,
    transform=transforms.Compose(
        [
            transforms.Resize(36),
        ]
    ),
    shuffle=True,
)
natimgs = natimgs - natimgs.mean(axis=0)


exc_images = get_excitatory_images()
vmin = exc_images.min()
vmax = exc_images.max()


patterns = exc_images[:10]

G_dim = len(patterns)
G_prob = 0.5
I_sigma = 100
patterns_offset = 0
cross_g_x_feedback = 0.05
direct_g_x_feedback = 0.8

offset_x_1 = 0
offset_x_2 = 1
scale_x = 0.1
exponent_x = 3
zero_threshold_x = 0.00

G_visual_nrows = G_dim
G_visual_ncols = 1
x_visual_nrows = G_dim
x_visual_ncols = 9
n_tune = 300
n_draws = 300
n_chains = 4
n_cores = 4


mapping = np.zeros((G_dim * 9, G_dim)) + cross_g_x_feedback
for j in range(0, G_dim * 9, G_dim):
    for i in range(G_dim):
        mapping[i + j, i] = direct_g_x_feedback


model = BinaryPatternCompletionModel(
    patterns=patterns,
    G_prob=G_prob,
    I_sigma=I_sigma,
    patterns_offset=patterns_offset,
    offset_x_1=offset_x_1,
    offset_x_2=offset_x_2,
    scale_x=scale_x,
    exponent_x=exponent_x,
    zero_threshold_x=zero_threshold_x,
    g_x_mapping=mapping,
)



natimgs_normed = natimgs / exc_images.std()
print(f"natimgs_normed mean: {natimgs_normed.mean()}, std: {natimgs_normed.std()}")



for idx, big_stimulus in enumerate(natimgs_normed):
    print(f"Processing stimulus {idx + 1}/{len(natimgs_normed)}")
    
    idata = model(
        image=big_stimulus,
        n_samples=n_draws,
        tune=n_tune,
        chains=n_chains,
        cores=n_cores,
        random_seed=seed,
    )

    with open(f"results/coen_cagli_big_stimulus_results_{idx}.pkl", "wb") as f:
        pickle.dump(idata, f)
    
    # also save the stimulus
    np.save(f"results/coen_cagli_presented_big_stimulus_{idx}.npy", big_stimulus)
    



