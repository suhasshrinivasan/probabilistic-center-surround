import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder

import numpy as np


def load_images(
    dataset_folder_path="/src/project/data/natimgs/",
    transform=transforms.RandomCrop(25),
):
    dataset = DatasetFolder(
        root=dataset_folder_path,
        loader=lambda path: torch.from_numpy(np.load(path)),
        extensions=".npy",
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    return next(iter(dataloader))