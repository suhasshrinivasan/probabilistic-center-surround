import numpy as np
import torch


def generate_gabor(theta, sigma, Lambda, psi, gamma, center, image_size):
    """
    Gabor generator function.
    Params:
        theta (float): Orientation of the sinusoid (in radians).
        sigma (float): std deviation of the Gaussian.
        Lambda (float): Sinusoid wavelengh (1/frequency).
        psi (float): Phase of the sinusoid.
        gamma (float): The ratio between sigma in x-dim over sigma in y-dim (acts
            like an aspect ratio of the Gaussian).
        center (tuple of integers): The position of the filter.
        image_size (tuple of integers): Image height and width.
    Returns:
        2D torch.tensor: A gabor filter.
    """
    sigma_x = sigma
    sigma_y = sigma / gamma

    ymax, xmax = image_size
    xmax, ymax = (xmax - 1) / 3, (ymax - 1) / 3
    xmin = -xmax
    ymin = -ymax
    (y, x) = torch.meshgrid(torch.arange(ymin, ymax + 1), torch.arange(xmin, xmax + 1))

    # Rotation
    x_theta = (x - center[0]) * np.cos(theta) + (y - center[1]) * np.sin(theta)
    y_theta = -(x - center[0]) * np.sin(theta) + (y - center[1]) * np.cos(theta)

    gb = np.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * np.cos(2 * torch.pi / Lambda * x_theta + psi)

    return gb
