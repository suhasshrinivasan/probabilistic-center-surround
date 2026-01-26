import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from sklearn.metrics.pairwise import cosine_similarity

seed = 42
rng = np.random.default_rng(seed)


class PatternCompletionModel:
    """
    A probabilistic model that composes image patterns from its crops.
    The model is G -> X -> I, where G is a latent variable that controls
    the pattern to be composed, X is a latent variable that controls the
    composition of the pattern, and I is the image (pattern) that is composed.
    The model is programmed in PyMC3.
    """

    def __init__(
        self,
        patterns,
        G_prob,
        X_sigma,
        I_sigma,
        patterns_offset=0,
        G_X_exponent=1,
    ):
        """
        Args:
            patterns (np.ndarray): patterns to compose
            G_prob (float): Bernoulli probability of G_i being 1
            X_sigma (float): Laplace distribution scale parameter for X
            I_sigma (float): Normal distribution scale parameter for I
            patterns_offset (float): offset to add to patterns to induce
                cosine similarity between patterns
        """
        self.patterns_offset = patterns_offset
        # add offset to each of the patterns
        self.patterns = np.array([image + patterns_offset for image in patterns])
        self.G_prob = G_prob
        self.X_sigma = X_sigma
        self.I_sigma = I_sigma
        # construct mapping from X to I
        (
            self.X_I_mapping,
            self.I_patch_dim,
            self.X_dim,
            self.X_patch_dim,
            self.pattern_crops,
        ) = self._construct_X_I_mapping()
        self.I_patch_side = int(np.sqrt(self.I_patch_dim))
        self.I_dim = self.X_I_mapping.shape[0]
        self.I_side = int(np.sqrt(self.I_dim))
        self.G_X_exponent = G_X_exponent
        # construct mapping from G to X
        self.G_X_mapping = self._construct_G_X_mapping()
        self.G_dim = self.patterns.shape[0]
        # construct the model
        self.prob_model = pm.Model()


        g_p = pt.as_tensor_variable([G_prob] * self.G_dim)
        g_x_mapping = pt.as_tensor_variable(self.G_X_mapping)
        x_i_mapping = pt.as_tensor_variable(self.X_I_mapping)
        x_sigma = pt.as_tensor_variable([X_sigma])
        i_sigma = pt.as_tensor_variable([I_sigma])
        with self.prob_model:
            G = pm.Bernoulli("G", p=g_p, shape=self.G_dim)

            X_mu = pm.Deterministic("X_mu", g_x_mapping @ G)
            X_sigma = pm.Deterministic("X_sigma", x_sigma)
            X = pm.Laplace("X", mu=X_mu, b=X_sigma)
            I_mu = pm.Deterministic("I_mu", x_i_mapping @ X)
            I_sigma = pm.Deterministic("I_sigma", i_sigma)
            # obs is a placeholder for the observed image
            obs = pm.MutableData("obs", np.zeros(self.I_dim))
            I = pm.Normal(
                "I",
                mu=I_mu,
                sigma=i_sigma,
                observed=obs,
            )

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        """
        Sample from the posterior distribution.
        See self.sample_posterior for details.
        """
        return self.sample_posterior(
            *args,
            **kwargs,
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
        # the images that are sampled are flattened
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
                    for i in range(9)
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
        pymc_logging=False,
    ):
        """
        Sample from the posterior distribution over X and G given an image: p(X, G | I)

        Args:
            image (np.ndarray): image to condition on of shape (np.sqrt(I_dim), np.sqrt(I_dim))
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

    def _images_to_consecutive_crops(self, images):
        """
        Convert images to consecutive, non-overlapping crops.
        Args:
            images (np.ndarray): images to convert to consecutive crops
        Returns:
            crops (np.ndarray): consecutive crops of the images
        """
        h, w = images.shape[1:]
        crops = np.array(
            [
                self.patterns[
                    :, i * h // 3 : (i + 1) * h // 3, j * w // 3 : (j + 1) * w // 3
                ]
                for i in range(3)
                for j in range(3)
            ]
        )
        return crops

    def _construct_X_I_mapping(self):
        """
        Construct the mapping from X to I.
        Args:
            None
        Returns:
            X_I_mapping (np.ndarray): mapping from X to I of shape (I_dim, X_dim)
            I_patch_dim (int): number of pixels in each patch of I
            X_dim (int): number of latent variables in X
            X_patch_dim (int): number of latent variables corresponding to each patch of I
            pattern_crops (np.ndarray): consecutive crops of the patterns (projective fields of X)
        Notes:
            Given a set of patterns (self.patterns), this function breaks
            each pattern down to a 3*3 grid of crops. Each crop represents the
            projective field of a latent variable (dimension) in X.
            If there are N patterns, then there are 3*3*N crops and hence
            3*3*N latent variables (dimensions) in X.
            The mapping using 3*3*N crops is constructed each latent variable
            (dimension) in X maps to a corresponding patch in I via its projective field.
        """
        h, w = self.patterns.shape[1:]
        # first convert the (N, h, w) patterns into
        # (9, N, h//3*w//3) consecutive crops
        pattern_crops = self._images_to_consecutive_crops(self.patterns)
        pattern_crops = pattern_crops.reshape((*pattern_crops.shape[:-2], -1))

        I_dim = h * w  # number of pixels in the image
        I_patch_dim = h // 3 * w // 3  # number of pixels in each image patch
        X_dim = 9 * self.patterns.shape[0]  # number of latent variables in X
        X_patch_dim = self.patterns.shape[
            0
        ]  # number of latent variables corresponding to each patch of I
        X_I_mapping = np.zeros((I_dim, X_dim))  # mapping matrix from X to I
        # fill in the mapping matrix
        # construct the mapping from X to I via crops
        # each crop represents the projective field of a latent variable in X.
        # the mapping is constructed such that each latent variable in X
        # maps to a corresponding patch in I via its projective field
        # via a diagonal matrix (each diagonal block is a crop)
        # the diagonal blocks are arranged in the order of the crops
        for idx, crop in enumerate(pattern_crops.transpose(0, 2, 1)):
            X_I_mapping[
                idx * I_patch_dim : (idx + 1) * I_patch_dim,
                idx * X_patch_dim : (idx + 1) * X_patch_dim,
            ] = crop

        return X_I_mapping, I_patch_dim, X_dim, X_patch_dim, pattern_crops

    def _construct_G_X_mapping(self):
        """
        Construct the mapping from G to X via cosine similarity.
        Args:
            None
        Returns:
            G_X_mapping (np.ndarray): mapping from G to X of shape (X_dim, G_dim)
        Notes:
            The mapping is constructed such that turning a latent variable (dimension)
            in G produces a pattern that is close to the associated pattern provided
            as input in self.patterns.
            If there are N self.patterns, then there are N latent variables (dimensions) in G.
            The pattern is produced when activating a latent variable in G by activating
            latent variables in X (with crops as projective fields).
            Each latent variable (dimension) in G maps all the latent variables in X
            via cosine similarity its pattern and the projective field of each latent
            variable in X.
            The cosine similarity is computed between crops of the patterns and projective
            fields of the latent variables in X (which are also crops of the pattern).
        """
        # first convert the (N, h, w) stimuli into (9, N, h//3*w//3) crops
        pattern_crops = self._images_to_consecutive_crops(self.patterns)
        pattern_crops = pattern_crops.reshape((*pattern_crops.shape[:-2], -1))
        # compute the cosine similarity between each projective field and each crop
        # to get a (9, N_x, N) cosine similarity matrix
        cos_sim_matrix = []
        for PF, crop in zip(self.pattern_crops, pattern_crops):
            cos_sim_matrix.append(cosine_similarity(PF, crop))
        cos_sim_matrix = np.array(cos_sim_matrix)
        # convert the (9, N_x, N) cosine similarity matrix into (9 * N_x, N) matrix
        G_X_mapping = cos_sim_matrix.reshape((-1, cos_sim_matrix.shape[-1]))
        return np.sign(G_X_mapping) * np.abs(G_X_mapping) ** self.G_X_exponent

    def visualize_learned_G(self, nrows=None, ncols=None):
        """
        Visualize learned G via one-hot encoding G and plotting resultant I.
        Args:
            nrows (int): number of rows in the plot
            ncols (int): number of columns in the plot
        Returns:
            generated_Is (np.ndarray): generated images via one-hot encoded G
        Notes:
            The one-hot encoding is done by setting each dimension in G to 1 and
            setting the rest to 0.
            The resultant I is computed by multiplying the one-hot encoded G with
            the mapping from G to X and then multiplying the result with the mapping
            from X to I.
            There is no stochasticity in this process.
        """
        # One-hot encoded Gs
        # and generate corresponding Is
        generated_Is = []
        I_patch_side = int(np.sqrt(self.I_patch_dim))
        # cycle through each dimension in G
        for dim in range(self.G_dim):
            # construct one-hot encoded G
            G = np.zeros(self.G_dim)
            G[dim] = 1
            # compute I
            I = self.X_I_mapping @ self.G_X_mapping @ G
            # reshape I into a 3*3 grid of patches
            I = np.array(
                [
                    I[i * self.I_patch_dim : (i + 1) * self.I_patch_dim].reshape(
                        (I_patch_side, I_patch_side)
                    )
                    for i in range(9)
                ]
            )
            generated_Is.append(I)
        generated_Is = np.array(generated_Is)
        # reshape each 3*3 grid of patches into a full image
        generated_Is = np.array(
            [
                np.vstack([np.hstack(I[i * 3 : (i + 1) * 3]) for i in range(3)])
                for I in generated_Is
            ]
        )
        # plot generated Is
        if nrows is None:
            nrows = int(self.G_dim / 10)
        if ncols is None:
            ncols = 10
        # plot generated Is
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
        for idx, ax in enumerate(axs.flatten()):
            vmin = np.min(generated_Is[idx])
            vmax = np.max(generated_Is[idx])
            ax.imshow(generated_Is[idx], vmin=vmin, vmax=vmax, cmap="gray")
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
        fig.suptitle("G pfs")

        return generated_Is

    def visualize_learned_X(self, nrows, ncols):
        """
        Visualize learned X via one-hot encoding X and plotting resultant I.
        Args:
            nrows (int): number of rows in the plot
            ncols (int): number of columns in the plot
        Returns:
            generated_Is (np.ndarray): generated images via one-hot encoded X
        Notes:
            The one-hot encoding is done by setting each dimension in X to 1 and
            setting the rest to 0.
            The resultant I is computed by multiplying the one-hot encoded X with
            the mapping from X to I.
            There is no stochasticity in this process.
            There is also no G in this process.
        """
        generated_Is = []
        I_patch_side = int(np.sqrt(self.I_patch_dim))
        # cycle through each dimension in X
        for hot_idx in range(self.X_I_mapping.shape[-1]):
            # construct one-hot encoded X
            X = np.zeros(self.X_I_mapping.shape[-1])
            X[hot_idx] = 1
            # compute I
            I = self.X_I_mapping @ X
            # reshape I into a 3*3 grid of patches
            I = np.array(
                [
                    I[i * self.I_patch_dim : (i + 1) * self.I_patch_dim].reshape(
                        (I_patch_side, I_patch_side)
                    )
                    for i in range(9)
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
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, dpi=300)
        vmin = np.min(self.patterns)
        vmax = np.max(self.patterns)
        for idx, ax in enumerate(axs.flatten()):
            ax.imshow(generated_Is[idx], cmap="gray", vmin=vmin, vmax=vmax)
            ax.text(
                0.1,
                0.9,
                f"{idx}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="yellow",
                fontsize=5,
            )
            ax.axis("off")
        # write a figure title as "X pfs"
        fig.suptitle("X pfs")

        return generated_Is


class ExpPatternCompletionModel(PatternCompletionModel):
    """
    Same as PatternCompletionModel but with an exponential distribution for X.
    """

    def __init__(
        self,
        patterns,
        G_prob,
        X_sigma,
        I_sigma,
        patterns_offset=0,
        G_X_exponent=1,
        scale_offset=2,
        scale_exponent=1,
    ):
        """
        Same as PatternCompletionModel but with one additional parameter:
            scale_offset (float): offset to add to the mapping from G to X
                This offset is added to the mapping from G to X to induce
                a positive scale parameter for the exponential distribution
                of X.
        """
        super().__init__(
            patterns,
            G_prob,
            X_sigma,
            I_sigma,
            patterns_offset=patterns_offset,
            G_X_exponent=G_X_exponent,
        )
        self.scale_offset = scale_offset
        self.scale_exponent = scale_exponent
        self.prob_model = pm.Model()
        g_p = pt.as_tensor_variable([G_prob] * self.G_dim)
        g_x_mapping = pt.as_tensor_variable(self.G_X_mapping)
        x_i_mapping = pt.as_tensor_variable(self.X_I_mapping)
        i_sigma = pt.as_tensor_variable([I_sigma])
        with self.prob_model:
            G = pm.Bernoulli("G", p=g_p, shape=self.G_dim)
            X_scale = pm.Deterministic(
                "X_mu", (self.scale_offset + (g_x_mapping @ G)) ** self.scale_exponent
            )
            X = pm.Exponential("X", scale=X_scale)
            I_mu = pm.Deterministic("I_mu", x_i_mapping @ X)
            I_sigma = pm.Deterministic("I_sigma", i_sigma)
            # obs is a placeholder for the observed image
            obs = pm.MutableData("obs", np.zeros(self.I_dim))
            I = pm.Normal(
                "I",
                mu=I_mu,
                sigma=i_sigma,
                observed=obs,
            )


class BinaryPatternCompletionModel(PatternCompletionModel):
    """
    Same as PatternCompletionModel but with a Bernoulli distribution for X.
    """

    def __init__(
        self,
        patterns,
        G_prob,
        I_sigma,
        patterns_offset=0,
        offset_x_1=0,
        offset_x_2=0,
        scale_x=1,
        exponent_x=1,
        zero_threshold_x=0,
        g_x_mapping=None,
    ):
        """
        Same as PatternCompletionModel but with the following additional parameter that transforms
        the mapping from G to X to be positive via the formula:
            G_X_mapping = offset_x_1  + scale_x * (G_X_mapping + offset_x_2) ** exponent_x

        Args:
            offset_x_1 (float): offset to add to the mapping from G to X
            offset_x_2 (float): offset to add to the mapping from G to X
            scale_x (float): scale to multiply the mapping from G to X
            exponent_x (float): exponent to raise the mapping from G to X
            zero_threshold_x (float): set minimum value for probabilities in X
            g_x_mapping (np.ndarray): mapping from G to X of shape (X_dim, G_dim)
        """
        self.patterns_offset = patterns_offset
        # add offset to each of the patterns
        self.patterns = np.array([image + patterns_offset for image in patterns])
        self.G_prob = G_prob
        self.G_dim = self.patterns.shape[0]  # number of latent variables in G
        # construct mapping from X to I
        (
            self.X_I_mapping,
            self.I_patch_dim,
            self.X_dim,
            self.X_patch_dim,
            self.pattern_crops,
        ) = self._construct_X_I_mapping()
        self.I_patch_side = int(np.sqrt(self.I_patch_dim))
        self.I_dim = self.X_I_mapping.shape[0]
        self.I_side = int(np.sqrt(self.I_dim))

        self.offset_x_1 = offset_x_1
        self.offset_x_2 = offset_x_2
        self.scale_x = scale_x
        self.exponent_x = exponent_x
        self.zero_threshold_x = zero_threshold_x
        self.G_X_mapping = g_x_mapping
        if self.G_X_mapping is None:
            self.G_X_mapping = self._construct_G_X_mapping()
        zero_threshold_x_column = np.full((self.X_dim, 1), self.zero_threshold_x)
        self.prob_model = pm.Model()
        g_p = pt.as_tensor_variable([G_prob] * self.G_dim)
        g_x_mapping = pt.as_tensor_variable(self.G_X_mapping)
        x_i_mapping = pt.as_tensor_variable(self.X_I_mapping)
        i_sigma = pt.as_tensor_variable([I_sigma])

        with self.prob_model:
            G = pm.Bernoulli("G", p=g_p, shape=self.G_dim)
            X_p = pm.Deterministic(
                "X_p",
                pt.max(
                    pt.concatenate([g_x_mapping * G, zero_threshold_x_column], axis=1),
                    axis=1,
                ),
            )
            X = pm.Bernoulli("X", p=X_p, shape=(self.X_dim,))
            I_mu = pm.Deterministic("I_mu", x_i_mapping @ X)
            I_sigma = pm.Deterministic("I_sigma", i_sigma)
            # obs is a placeholder for the observed image
            obs = pm.MutableData("obs", np.zeros(self.I_dim))
            I = pm.Normal(
                "I",
                mu=I_mu,
                sigma=I_sigma,
                observed=obs,
            )

    def _construct_G_X_mapping(self):
        """
        Construct the mapping from G to X via cosine similarity,
        and then transform the mapping to be positive via the formula:
            G_X_mapping = offset_x_1  + scale_x * (G_X_mapping + offset_x_2) ** exponent_x

        Args:
            None
        Returns:
            G_X_mapping (np.ndarray): mapping from G to X of shape (X_dim, G_dim)
        Notes:
            *** This is called only if a mapping is already not passed ***
            The mapping is constructed such that turning a latent variable (dimension)
            in G produces a pattern that is close to the associated pattern provided
            as input in self.patterns.
            If there are N self.patterns, then there are N latent variables (dimensions) in G.
            The pattern is produced when activating a latent variable in G by activating
            latent variables in X (with crops as projective fields).
            Each latent variable (dimension) in G maps all the latent variables in X
            via cosine similarity its pattern and the projective field of each latent
            variable in X.
            The cosine similarity is computed between crops of the patterns and projective
            fields of the latent variables in X (which are also crops of the pattern).
        """
        # first convert the (N, h, w) stimuli into (9, N, h//3*w//3) crops
        pattern_crops = self._images_to_consecutive_crops(self.patterns)
        pattern_crops = pattern_crops.reshape((*pattern_crops.shape[:-2], -1))
        # compute the cosine similarity between each projective field and each crop
        # to get a (9, N_x, N) cosine similarity matrix
        cos_sim_matrix = []
        for PF, crop in zip(self.pattern_crops, pattern_crops):
            cos_sim_matrix.append(cosine_similarity(PF, crop))
        cos_sim_matrix = np.array(cos_sim_matrix)
        # convert the (9, N_x, N) cosine similarity matrix into (9 * N_x, N) matrix
        G_X_mapping = cos_sim_matrix.reshape((-1, cos_sim_matrix.shape[-1]))
        # apply the transformation
        return (
            self.offset_x_1
            + self.scale_x * (G_X_mapping + self.offset_x_2) ** self.exponent_x
        )
