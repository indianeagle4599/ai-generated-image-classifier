# data/datasets.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets

import numpy as np
from PIL import Image

from .preprocess import dft_magnitude_spectrum, dct_features


class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class AIImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, features=[]):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

        self.features = features

        self.feature_function_dict = {
            "FFT": {"method": dft_magnitude_spectrum, "need_transform": True},
            "DCT": {"method": dct_features, "need_transform": False},
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        feature_list = []

        if self.transform:
            np_image = np.array(image)
            for feature in self.features:
                feature_set = self.feature_function_dict[feature]["method"](np_image)
                if self.feature_function_dict[feature]["need_transform"]:
                    feature_list.append(
                        self.transform(Image.fromarray(feature_set).convert("RGB"))
                    )
                else:
                    feature_list.append(torch.from_numpy(feature_set))
            image = self.transform(image)

        label = self.dataframe.iloc[idx, 1]
        label = torch.Tensor([label, 1 - label])
        return (image, feature_list), label


class TestAIImageDataset(Dataset):
    def __init__(self, file_list, transform=None, features=[]):
        self.file_list = file_list
        self.transform = transform

        self.features = features

        self.feature_function_dict = {
            "FFT": {"method": dft_magnitude_spectrum, "need_transform": True},
            "DCT": {"method": dct_features, "need_transform": False},
        }

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = Image.open(img_name).convert("RGB")

        feature_list = []

        if self.transform:
            np_image = np.array(image)
            for feature in self.features:
                feature_set = self.feature_function_dict[feature]["method"](np_image)
                if self.feature_function_dict[feature]["need_transform"]:
                    feature_list.append(
                        self.transform(Image.fromarray(feature_set).convert("RGB"))
                    )
                else:
                    feature_list.append(torch.from_numpy(feature_set))
            image = self.transform(image)

        return (image, feature_list), os.path.basename(
            img_name
        )  # Return image and filename
