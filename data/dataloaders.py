# data/dataloaders.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from .datasets import AIImageDataset, TestAIImageDataset
from .transforms import get_transforms


def get_dataloaders(
    data_path,
    train_csv,
    batch_size,
    num_workers,
    image_size,
    use_augmentation,
    features,
    val_csv=None,
    val_size=0.2,
):
    # Load the training dataset
    train = pd.read_csv(os.path.join(data_path, train_csv))

    # Preprocess column names for consistency
    train = train[["file_name", "label"]]
    train.columns = ["id", "label"]

    if val_csv:
        # If a separate validation CSV is provided, load it
        val = pd.read_csv(os.path.join(data_path, val_csv))
        val = val[["file_name", "label"]]
        val.columns = ["id", "label"]
        train_df = train
        val_df = val
    else:
        # If no separate validation CSV, split the training data
        train_df, val_df = train_test_split(train, test_size=val_size, random_state=42)

    train_transform, val_transform = get_transforms(image_size, use_augmentation)

    train_dataset = AIImageDataset(
        dataframe=train_df,
        root_dir=os.path.join(data_path),
        transform=train_transform,
        features=features,
    )
    val_dataset = AIImageDataset(
        dataframe=val_df,
        root_dir=os.path.join(data_path),
        transform=val_transform,
        features=features,
    )

    # Define DataLoader parameters based on num_workers
    dataloader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    if num_workers > 0:
        dataloader_params.update(
            {
                "persistent_workers": True,
                "pin_memory": True,
            }
        )

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_params)

    return train_loader, val_loader


def get_test_dataloader(
    data_path, batch_size, num_workers, image_size, use_augmentation, features
):

    # Load the training and test datasets
    test = pd.read_csv(os.path.join(data_path, "test.csv"))

    _, val_transform = get_transforms(image_size, use_augmentation)

    # For testing, create a list of file paths
    test_file_list = [os.path.join(data_path, fname) for fname in test["file_name"]]
    test_dataset = TestAIImageDataset(
        file_list=test_file_list,
        transform=val_transform,
        features=features,
    )

    # Define DataLoader parameters based on num_workers
    dataloader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    if num_workers > 0:
        dataloader_params.update(
            {
                "persistent_workers": True,
                "pin_memory": True,
            }
        )

    test_loader = DataLoader(test_dataset, shuffle=True, **dataloader_params)

    return test, test_loader
