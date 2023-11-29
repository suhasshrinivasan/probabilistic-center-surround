import hashlib
from collections import Iterable, Mapping, OrderedDict
from hashlib import md5

import numpy as np
import torch
from skimage import transform as sk_transform
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder


def load_images(
    dataset_folder_path="/src/project/data/natimgs/",
    crop_size=25,
    n_samples=20_000,
    seed=0,
):
    torch.manual_seed(seed)
    transform = transforms.RandomCrop(crop_size)
    dataset = DatasetFolder(
        root=dataset_folder_path,
        loader=lambda path: torch.from_numpy(np.load(path)),
        extensions=".npy",
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    images, labels = next(iter(dataloader))
    return images.numpy()[:n_samples]


def sqcrop_and_resize(images, size=(36, 36), anti_aliasing=False):
    final_images = []
    for image in images:
        h, w = image.shape
        cropped = image[:, int((w - h) / 2) : -int((w - h) / 2)]
        resized = sk_transform.resize(cropped, size, anti_aliasing=anti_aliasing)
        final_images.append(resized)
    return np.array(final_images)


# def turn_images_into_cropsets(images, n_sets=9):
#     n_pixels = images.shape[-1] * images.shape[-2]
#     crop_dim = int(np.sqrt(n_pixels / n_sets))


# from nnfabrik:
# https://github.com/sinzlab/nnfabrik/blob/ea4f5148c943741e45d937fe7ee681978b4224f7/nnfabrik/utility/dj_helpers.py#L58-L97
def make_hash(obj):
    """
    Given a Python object, returns a 32 character hash string to uniquely identify
    the content of the object. The object can be arbitrary nested (i.e. dictionary
    of dictionary of list etc), and hashing is applied recursively to uniquely
    identify the content.
    For dictionaries (at any level), the key order is ignored when hashing
    so that {"a":5, "b": 3, "c": 4} and {"b": 3, "a": 5, "c": 4} will both
    give rise to the same hash. Exception to this rule is when an OrderedDict
    is passed, in which case difference in key order is respected. To keep
    compatible with previous versions of Python and the assumed general
    intentions, key order will be ignored even in Python 3.7+ where the
    default dictionary is officially an ordered dictionary.
    Args:
        obj - A (potentially nested) Python object
    Returns:
        hash: str - a 32 charcter long hash string to uniquely identify the object.
    """
    hashed = hashlib.md5()

    if isinstance(obj, str):
        hashed.update(obj.encode())
    elif isinstance(obj, OrderedDict):
        for k, v in obj.items():
            hashed.update(str(k).encode())
            hashed.update(make_hash(v).encode())
    elif isinstance(obj, Mapping):
        for k in sorted(obj, key=str):
            hashed.update(str(k).encode())
            hashed.update(make_hash(obj[k]).encode())
    elif isinstance(obj, Iterable):
        for v in obj:
            hashed.update(make_hash(v).encode())
    else:
        hashed.update(str(obj).encode())

    return hashed.hexdigest()
