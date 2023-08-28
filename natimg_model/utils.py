import numpy as np
import torch
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


def turn_images_into_cropsets(images, n_sets=9):
    n_pixels = images.shape[-1] * images.shape[-2]
    crop_dim = int(np.sqrt(n_pixels / n_sets))
