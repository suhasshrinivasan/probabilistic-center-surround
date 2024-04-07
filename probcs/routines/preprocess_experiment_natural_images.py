from pathlib import Path

import numpy as np

from probcs.utils.utils import sqcrop_and_resize, sqcrop_and_resize_center_of_mass


def call_sqcrop_and_resize(
    images_fname,
    allow_pickle=True,
    size=(36, 36),
    anti_aliasing=False,
    center_of_mass=False,
):
    images = np.load(images_fname, allow_pickle=allow_pickle)
    images = np.array([np.array(x) for x in images]).astype(np.float64)
    if center_of_mass:
        images = sqcrop_and_resize_center_of_mass(
            images, size=size, anti_aliasing=anti_aliasing
        )
    else:
        images = sqcrop_and_resize(images, size=size, anti_aliasing=anti_aliasing)
    return images


basepath = Path("/src/project/data/experiment")
images_fname = basepath / "natural_images.npy"
images_save_fname = basepath / "natural_images_preprocessed_36x36.npy"
images = call_sqcrop_and_resize(images_fname)
np.save(images_save_fname, images)
