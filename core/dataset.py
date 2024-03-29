import torch
import cv2
import os
import random
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .utils import file as file_utils


class DatasetBase(Dataset):
    """Dataset Base Class reads basic image data from a directory.
    """

    def __init__(self, dir: str, suffix: List[str] = [".jpg"], img_size: int = 28, transform: Optional[transforms.Compose] = None, size: int = -1, subdir: Optional[str] = None) -> None:

        if subdir is not None:
            dir = os.path.join(dir, subdir)

        self.input_dir = dir

        self.path_list = file_utils.get_files(self.input_dir, ext=suffix)[:size]
        random.shuffle(self.path_list)

        self.aug_transforms = transform

        self.to_tensor = transforms.ToTensor()

        self.sample_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        ])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx: int):
        file_path = self.path_list[idx]

        label = file_path.split(os.sep)[-2]

        np_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        img = self.to_tensor(np_img)

        if self.aug_transforms is not None:
            img = self.aug_transforms(img)

        img: torch.Tensor = self.sample_transform(img)

        return img, label


def dataset_statistic(dataset: Dataset):

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    mean = 0.
    std = 0.
    size = 0
    for images in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        size += batch_samples

    mean /= size
    std /= size

    return mean, std
