import pathlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from insilico_stimuli.stimuli import CenterSurround, GaborSet
from sklearn.decomposition import FastICA

from utils.utils import load_images

seed = 42
rng = np.random.default_rng(seed)


class HierarchicalPanelICA:
    def __init__(
        self,
        G_X_model,
        X_sigma,
        X_I_models,
        I_sigma,
    ):
        """
        Initialize the generative model.

        The generative model is G -> X -> I, where G is a global "shape" variable,
        X is a local "feature" variable, and I is the image.
        """
        self.G_X_model = G_X_model
        self.G_dim = G_X_model.n_components
        self.G_X_mapping = G_X_model.mixing_

        self.X_sigma = X_sigma
        self.X_I_models = X_I_models
        (
            self.X_I_mapping,
            self.I_patch_dim,
            self.X_dim,
            self.X_patch_dim,
        ) = self._construct_X_I_mapping()
        self.I_patch_side = int(np.sqrt(self.I_patch_dim))

        self.I_sigma = I_sigma
        self.I_dim = self.X_I_mapping.shape[0]
        self.I_side = int(np.sqrt(self.I_dim))

        self.prob_model = pm.Model()
        with self.prob_model:
            G = pm.Laplace("G", mu=0, b=1, shape=(self.G_dim))

            X_mu = pm.Deterministic("X_mu", self.G_X_mapping @ G)
            X_sigma = pm.Deterministic("X_sigma", pt.as_tensor_variable([0.1]))
            X = pm.Normal("X", mu=X_mu, sigma=X_sigma)

            I_mu = pm.Deterministic("I_mu", self.X_I_mapping @ X)
            I_sigma = pm.Deterministic("I_sigma", pt.as_tensor_variable([0.1]))
            obs = pm.MutableData("obs", np.zeros(self.I_dim))
            I = pm.Normal(
                "I",
                mu=I_mu,
                sigma=I_sigma,
                observed=obs,
            )

    def _construct_X_I_mapping(self):
        """
        Construct the mapping from X to I.
        """
        X_patch_dim = self.X_I_models[0].mixing_.shape[1]
        X_dim = X_patch_dim * len(self.X_I_models)
        I_patch_dim = self.X_I_models[0].mixing_.shape[0]
        I_dim = I_patch_dim * len(self.X_I_models)

        X_I_mapping = np.zeros((I_dim, X_dim))
        for idx, model in enumerate(self.X_I_models):
            X_I_mapping[
                idx * I_patch_dim : (idx + 1) * I_patch_dim,
                idx * X_patch_dim : (idx + 1) * X_patch_dim,
            ] = model.mixing_

        # alternatively, tensorize
        # X_I_mapping = np.array([model.mixing_ for model in self.X_I_models])

        return (
            X_I_mapping,
            I_patch_dim,
            X_dim,
            X_patch_dim,
        )

    def __call__(
        self,
        image,
        n_samples,
        random_seed,
        tune=1000,
        chains=None,
        cores=None,
        return_inferencedata=True,
    ):
        """
        Sample from the posterior distribution.
        See self.sample_posterior for details.
        """
        return self.sample_posterior(
            image=image,
            n_samples=n_samples,
            random_seed=random_seed,
            tune=tune,
            chains=chains,
            cores=cores,
            return_inferencedata=return_inferencedata,
        )

    def sample_prior_predictive(
        self,
        n_samples,
        random_seed,
    ):
        """
        Sample from the prior predictive distribution.

        Args:
            n_samples (int): number of samples to draw
            random_seed (int): random seed for reproducibility

        Returns:
            samples as a dict containing the samples for all variables
        """
        with self.prob_model:
            prior_samples_dict = pm.sample_prior_predictive(
                n_samples,
                random_seed=random_seed,
                return_inferencedata=False,
            )
        # reshape images
        images = prior_samples_dict["I"]
        reshaped_images = []
        # first reshape each image patch into a square
        for image in images:
            reshaped_image = np.array(
                [
                    image[i * self.I_patch_dim : (i + 1) * self.I_patch_dim].reshape(
                        self.I_patch_side, self.I_patch_side
                    )
                    for i in range(len(self.X_I_models))
                ]
            )
            reshaped_images.append(reshaped_image)
        # then reshape the square patches into larger squares
        reshaped_images = np.array(
            [
                np.vstack(
                    [np.hstack(reshaped_image[i * 3 : (i + 1) * 3]) for i in range(3)]
                )
                for reshaped_image in reshaped_images
            ]
        )
        prior_samples_dict["I"] = reshaped_images
        return prior_samples_dict

    def sample_posterior(
        self,
        image,
        n_samples,
        random_seed,
        tune=1000,
        chains=None,
        cores=None,
        return_inferencedata=True,
    ):
        """
        Sample from the posterior distribution.

        Args:
            image (np.ndarray): image to condition on of shape self.I_dim (flattened image)
            n_samples (int): number of samples to draw per chain
            random_seed (int): random seed for reproducibility
            tune (int): number of tuning steps per chain
            chains (int): number of chains
            cores (int): number of cores to use

        Returns:
            samples as a dict with keys containing samples of latent variables
        """
        reshaped_image = []
        for i in range(3):
            for j in range(3):
                reshaped_stim = image[
                    i * self.I_patch_side : (i + 1) * self.I_patch_side,
                    j * self.I_patch_side : (j + 1) * self.I_patch_side,
                ].reshape(self.I_patch_side**2)
                reshaped_image.append(reshaped_stim)
        reshaped_image = np.array(reshaped_image).flatten()
        print("reshaped_image", reshaped_image.shape)
        with self.prob_model:
            pm.set_data({"obs": reshaped_image})
            post_samples_dict = pm.sample(
                draws=n_samples,
                random_seed=random_seed,
                return_inferencedata=return_inferencedata,
                tune=tune,
                chains=chains,
                cores=cores,
            )
        return post_samples_dict

    def visualize_learned_G(self):
        """
        Visualize learned G via one-hot encoding G and plotting resultant I.
        """
        # One-hot encoded Gs
        # and generate corresponding Is
        generated_Is = []
        I_patch_side = int(np.sqrt(self.I_patch_dim))
        for dim in range(self.G_dim):
            G = np.zeros(self.G_dim)
            G[dim] = 1
            I = self.X_I_mapping @ self.G_X_mapping @ G
            I = np.array(
                [
                    I[i * self.I_patch_dim : (i + 1) * self.I_patch_dim].reshape(
                        (I_patch_side, I_patch_side)
                    )
                    for i in range(len(self.X_I_models))
                ]
            )
            generated_Is.append(I)
        generated_Is = np.array(generated_Is)
        generated_Is = np.array(
            [
                np.vstack([np.hstack(I[i * 3 : (i + 1) * 3]) for i in range(3)])
                for I in generated_Is
            ]
        )
        # compute min and max for normalization
        # vmin = -np.max(np.abs(generated_Is))
        # vmax = np.max(np.abs(generated_Is))   # normalization causing very faint images
        # plot generated Is
        nrows = int(self.G_dim / 10)
        ncols = 10
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
        for idx, ax in enumerate(axs.flatten()):
            ax.imshow(generated_Is[idx], cmap="gray")
            # write idx on the top right corner in red
            ax.text(
                0.9,
                0.1,
                str(idx),
                color="orange",
                fontsize=10,
                horizontalalignment="left",
                verticalalignment="top",
            )
            ax.axis("off")

    def visualize_learned_X(self):
        """
        Visualize learned X via feature visualization.
        """
        # break apart into center and surround panels
        panel_ids = {
            "top-left": 0,
            "top-center": 1,
            "top-right": 2,
            "center-left": 3,
            "center-center": 4,
            "center-right": 5,
            "bottom-left": 6,
            "bottom-center": 7,
            "bottom-right": 8,
        }
        for model_id, model in enumerate(self.X_I_models):
            # get the learned mixing matrix
            mixing = model.mixing_
            # visualize the mixing matrix
            nrows = int(self.X_patch_dim / 10)
            ncols = 10
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
            vmin = -np.max(np.abs(mixing))
            vmax = np.max(np.abs(mixing))
            for ax_id, ax in enumerate(axs.flatten()):
                ax.imshow(
                    mixing[:, ax_id].reshape((self.I_patch_dim, self.I_patch_dim)),
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.text(
                    0.9,
                    0.1,
                    str(ax_id + model_id * self.X_patch_dim),
                    color="red",
                    fontsize=20,
                    horizontalalignment="left",
                    verticalalignment="top",
                )
                ax.axis("off")
            fig.suptitle(panel_ids[model_id])

    def visualize_center_X_mappings(self, center_model_id):
        center_X_I_model = self.X_I_models[center_model_id]
        center_X_I_mapping = center_X_I_model.mixing_
        center_latent_offset = center_model_id * self.X_patch_dim
        ncols = 10
        nrows = int(self.X_patch_dim / ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
        for idx, ax in enumerate(axs.flatten()):
            mapping_image = center_X_I_mapping[:, idx].reshape(
                (self.I_patch_side, self.I_patch_side)
            )
            ax.imshow(mapping_image, cmap="gray")
            latent_id = idx + center_latent_offset
            ax.text(
                0.1,
                0.9,
                str(latent_id),
                color="orange",
                fontsize=10,
                transform=ax.transAxes,
            )
            ax.axis("off")
