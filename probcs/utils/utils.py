import hashlib
from collections import Iterable, Mapping, OrderedDict
from hashlib import md5
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from skimage import transform as sk_transform
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from scipy.ndimage import center_of_mass


def get_excitatory_images(
    path="/src/project/data/experiment/exc_images_preprocessed.npy",
):
    exc_images = np.load(path)
    exc_images_mean = exc_images.mean(axis=0)
    exc_images = exc_images - exc_images_mean
    return exc_images


def select_focus_images(
    image_dataset, threshold_low, threshold_high, num_focus_images=10, seed=None
):
    """
    Select a set of focus images from an image dataset based on cosine similarity thresholds.

    Parameters:
    - image_dataset (list or numpy.ndarray): A collection of images to select focus images from.
    - threshold_low (float): Lower threshold for cosine similarity between focus images.
    - threshold_high (float): Upper threshold for cosine similarity between focus images.
    - num_focus_images (int): Number of focus images to select (default is 10).
    - seed (int or None): Seed for the random number generator (default is None).

    Returns:
    - focus_images (list of numpy.ndarray): List of selected focus images.
    - final_cossim_matrix (numpy.ndarray): The final cosine similarity matrix for the selected focus images.

    Note:
    - The function uses cosine similarity to measure similarity between focus images.
    - The initial focus image is randomly sampled from the input image dataset.
    - Subsequent focus images are selected randomly from the dataset based on similarity thresholds.
    - The function ensures that the cosine similarity between all selected focus images
        falls within the specified range (threshold_low, threshold_high).
    - The final cosine similarity matrix is computed for the selected focus images.

    Example:
    >>> image_dataset = [np.random.rand(28, 28) for _ in range(100)]  # Assuming a list of 100 images
    >>> focus_images, final_cossim_matrix = select_focus_images(image_dataset, 0.5, 0.9, num_focus_images=5, seed=42)
    """
    rng = np.random.default_rng(seed)

    # Randomly sample the first focus image
    initial_focus_image = image_dataset[rng.integers(len(image_dataset))]
    focus_images = [initial_focus_image]

    while len(focus_images) < num_focus_images:
        random_index = rng.integers(len(image_dataset))
        current_image = image_dataset[random_index]

        # Reshape the arrays to 2D
        focus_images_matrix = np.vstack(focus_images).reshape(len(focus_images), -1)
        current_image_matrix = current_image.reshape(1, -1)

        cossim_matrix = cosine_similarity(focus_images_matrix, current_image_matrix)

        if np.all((threshold_low < cossim_matrix) & (cossim_matrix < threshold_high)):
            focus_images.append(current_image)

    # Compute the final cosine similarity matrix
    focus_images_matrix = np.vstack(focus_images).reshape(len(focus_images), -1)
    final_cossim_matrix = cosine_similarity(focus_images_matrix)

    return focus_images, final_cossim_matrix


def load_images(
    dataset_folder_path="/src/project/data/natimgs/",
    crop_size=25,
    n_samples=20_000,
    seed=0,
    transform=transforms.RandomCrop(25),
):
    torch.manual_seed(seed)
    # transform = transforms.RandomCrop(crop_size)

    dataset = DatasetFolder(
        root=dataset_folder_path,
        loader=lambda path: torch.from_numpy(np.load(path)).unsqueeze(0),
        extensions=".npy",
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    images, labels = next(iter(dataloader))
    return images.squeeze(1).numpy()[:n_samples]


def sqcrop_and_resize(images, size=(36, 36), anti_aliasing=False):
    final_images = []
    for image in images:
        h, w = image.shape
        cropped = image[:, int((w - h) / 2) : -int((w - h) / 2)]
        resized = sk_transform.resize(cropped, size, anti_aliasing=anti_aliasing)
        final_images.append(resized)
    return np.array(final_images)


def sqcrop_and_resize_center_of_mass(
    images, size=(30, 30), anti_aliasing=False, edge_threshold=None
):
    final_images = []
    for image in images:
        h, w = image.shape
        center_y, center_x = center_of_mass(image)

        # Determine if the center of mass is too close to an edge
        half_size_y, half_size_x = size[0] // 2, size[1] // 2
        if edge_threshold is None:
            edge_threshold = min(half_size_y, half_size_x)

        if (
            center_x < edge_threshold
            or center_x > w - edge_threshold
            or center_y < edge_threshold
            or center_y > h - edge_threshold
        ):
            continue  # Skip this image

        # Calculate cropping coordinates ensuring they are within the image bounds
        start_x = max(int(center_x - half_size_x), 0)
        end_x = min(start_x + size[1], w)
        start_y = max(int(center_y - half_size_y), 0)
        end_y = min(start_y + size[0], h)

        # Crop and resize
        cropped = image[start_y:end_y, start_x:end_x]
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


def dj_error_msg(table, key_hash):
    return (table & f"key_hash='{key_hash}'").fetch1("error_message")


def dj_error_stack(table, key_hash):
    return (table & f"key_hash='{key_hash}'").fetch1("error_stack")
