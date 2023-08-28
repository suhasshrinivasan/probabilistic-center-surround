import numpy as np


class RepeatICA:
    """
    Container class for housing and applying multiple ICA models.
    """

    def __init__(self, ica_models):
        self.ica_models = ica_models

    def transform(self, image_set):
        """
        Transform a set of images into their ICA latent representations.
        Args:
            image_set (np.ndarray): Set of flattened images to transform of shape
                (n_sets, n_images, n_pixels)
        Returns:
            np.ndarray (np.ndarray): Set of latent representations of shape
                (n_sets, n_images, n_components)
        Remarks:
            Assumes that the ICA models have identical dimensinality.
        """
        zs = []
        for images, model in zip(image_set, self.ica_models):
            images = images.reshape(images.shape[0], -1)
            z = model.transform(images)
            zs.append(z)
        return np.array(zs)

    def inverse_transform(self, z_set):
        """
        Transform a set of ICA latent representations into their image representations.
        Args:
            z_set (np.ndarray): Set of latent representations of shape
                (n_sets, n_images, n_components)
        Returns:
            np.ndarray (np.ndarray): Set of images of shape
                (n_sets, n_images, n_pixels)
        Remarks:
            Assumes that the ICA models have identical dimensinality.
        """
        images = []
        for z, model in zip(z_set, self.ica_models):
            image = model.inverse_transform(z)
            images.append(image)
        return np.array(images)

    def get_components(self):
        """
        Get the components of each ICA model.
        Returns:
            np.ndarray (np.ndarray): Set of components of shape (n_sets, n_components, n_pixels)
        """
        components = []
        for model in self.ica_models:
            components.append(model.components_)
        return np.array(components)

    def get_mixing(self):
        """
        Get the mixing matrices of each ICA model.
        Returns:
            np.ndarray (np.ndarray): Set of mixing matrices of shape (n_sets, n_components, n_components)
        """
        mixing = []
        for model in self.ica_models:
            mixing.append(model.mixing_)
        return np.array(mixing)

    def fit(self, image_set):
        """
        Fit each ICA model to a set of images.
        Remarks:
            This method is not implemented yet. See ica.py.
        """
        return NotImplementedError("This method is not implemented yet.")
