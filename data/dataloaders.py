# data/dataloaders.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from .datasets import AIImageDataset, TestAIImageDataset
from .transforms import get_transforms


def get_dataloaders(
    data_path, train_csv, batch_size, num_workers, image_size, use_augmentation
):

    # Load the training and test datasets
    train = pd.read_csv(os.path.join(data_path, train_csv))

    # Preprocess column names for consistency
    train = train[["file_name", "label"]]
    train.columns = ["id", "label"]

    # Split the training data into training and validation sets
    train_df, val_df = train_test_split(train, test_size=0.20, random_state=42)

    train_transform, val_transform = get_transforms(image_size, use_augmentation)

    train_dataset = AIImageDataset(
        dataframe=train_df, root_dir=os.path.join(data_path), transform=train_transform
    )
    val_dataset = AIImageDataset(
        dataframe=val_df, root_dir=os.path.join(data_path), transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(
    data_path, batch_size, num_workers, image_size, use_augmentation
):

    # Load the training and test datasets
    test = pd.read_csv(os.path.join(data_path, "test.csv"))

    _, val_transform = get_transforms(image_size, use_augmentation)

    # For testing, create a list of file paths
    test_file_list = [os.path.join(data_path, fname) for fname in test["id"]]
    test_dataset = TestAIImageDataset(file_list=test_file_list, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return test, test_loader
