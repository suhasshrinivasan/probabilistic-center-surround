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
exc_fname = basepath / "exc_images.npy"
exc_save_fname = basepath / "exc_images_preprocessed_24x24.npy"
meis_fname = basepath / "meis.npy"
meis_save_fname = basepath / "meis_preprocessed_24x24.npy"

exc_images = call_sqcrop_and_resize(exc_fname, size=(24, 24), center_of_mass=False)
np.save(exc_save_fname, exc_images)
meis_images = call_sqcrop_and_resize(meis_fname, size=(24, 24), center_of_mass=False)
np.save(meis_save_fname, meis_images)
