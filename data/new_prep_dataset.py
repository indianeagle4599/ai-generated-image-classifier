import json

import os
import shutil
from datetime import datetime

import pandas as pd
import numpy as np

from tqdm import tqdm

from augment import Augmentor

DEFAULT_CONFIG_PATH = "assets/"


def load_config_with_fallback(dataset_path, config_name, output_path):
    """Load config with fallback to default assets"""
    config_path = os.path.join(dataset_path, f"{config_name}.json")
    if not os.path.exists(config_path):
        default_config = os.path.join(DEFAULT_CONFIG_PATH, f"{config_name}.json")
        shutil.copy(default_config, config_path)
    shutil.copy(config_path, output_path)

    with open(config_path) as f:
        return json.load(f)


def validate_exclusive_group(splits):
    """Validate exclusive group ratios sum to 1"""
    ratios = [s["size_value"] for s in splits if s["size_type"] == "ratio"]
    if not np.isclose(sum(ratios), 1.0, atol=1e-3):
        raise ValueError(f"Exclusive group ratios sum {sum(ratios):.3f} != 1.0")


def update_dataframe(df, path_column, new_folder_name):
    """
    Updates the 'file_name' column so that the folder prefix is set to new_folder_name.
    Assumes original file names are formatted as "<old_folder>/<file_name>".
    """
    df[path_column] = df[path_column].apply(
        lambda x: os.path.join(new_folder_name, os.path.basename(x))
    )
    return df


def format_path(base, *args):
    """Improved path formatting with depth limit"""
    max_depth = 4
    full_path = os.path.join(base, *args)
    parts = os.path.normpath(full_path).split(os.sep)
    if len(parts) > max_depth:
        parts = ["..."] + parts[-max_depth:]
    return f'"{os.sep.join(parts)}"'


def stratified_sample(df, n_samples, stratify_col, random_state=None):
    groups = df.groupby(stratify_col)
    group_sizes = (groups.size() / len(df) * n_samples).round().astype(int)

    sampled = groups.apply(
        lambda x: x.sample(n=group_sizes[x.name], random_state=random_state)
    )
    return sampled.reset_index(drop=True)


def load_or_create_split(split_config, dataset_path, source_df, label_as_probs):
    """Handle split creation with enhanced size handling"""
    split_source = split_config.get("source")
    split_size = split_config.get("split_size", 1.0)
    shuffle = split_config.get("shuffle", False)

    # Handle existing CSV
    if split_source and os.path.exists(os.path.join(dataset_path, split_source)):
        df = pd.read_csv(os.path.join(dataset_path, split_source))
        print(f"🔁 Using existing source: {format_path(dataset_path, split_source)}")
    elif source_df is not None:
        df = source_df.copy()
    else:
        raise ValueError(f"No source available for {split_config['name']} split")

    # Apply column mapping
    if "column_mapping" in split_config:
        df = df.rename(columns=split_config["column_mapping"])

    # Handle split sizing
    if split_size != 1.0:
        original_size = len(df)

        if split_size < 0:  # Subtract samples
            n_samples = max(0, original_size + split_size)
        elif 0 < split_size <= 1:  # Ratio
            n_samples = int(original_size * split_size)
        else:  # Absolute count
            n_samples = min(split_size, original_size)

        # Stratification handling
        stratify = None
        if label_as_probs and "label" in df.columns:
            bins = np.linspace(0, 1, 10)
            stratify = np.digitize(df["label"], bins)
            df = stratified_sample(
                df,
                n_samples=n_samples,
                stratify_col=stratify,
                random_state=42 if shuffle else None,
            )
        else:
            df = df.sample(
                n=n_samples,
                replace=False,
                random_state=42 if shuffle else None,
            )

    return df


def process_split(split_config, df, dataset_path, output_root, output_folder):
    """Updated split processing with proper parameter handling"""
    print(f"\n📂 Processing {split_config['name']} split:")

    # Initialize augmentor with current split's configuration
    augmentor = Augmentor(
        config_path=os.path.join(dataset_path, "aug_config.json"),
        df=df,
        image_folder=dataset_path,
        output_folder=os.path.join(output_root, output_folder),
        output_csv=os.path.join(output_root, split_config["csv_path"]),
        path_column=split_config["path_column"],
    )

    # Create output directory
    os.makedirs(augmentor.output_folder, exist_ok=True)

    # Process images
    if split_config.get("preprocess", False):
        print("🖼️  Preprocessing images...")
        augmentor.process_original_images()

    # Apply augmentations
    if split_config.get("augment", False):
        print("🎨 Applying augmentations...")
        df = augmentor.augment_images()

    df = update_dataframe(
        df, split_config["path_column"], split_config["output_folder"]
    )
    # Save metadata
    df.to_csv(augmentor.output_csv, index=False)
    print(
        f"✅ Saved {split_config['name']} metadata: {format_path(output_root, split_config['csv_path'])}"
    )


def main():
    # Allow hardcoded path for notebook users
    dataset_path = "dataset/example"  # Set this as needed

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_suffix = os.path.basename(dataset_path)
    output_path = os.path.join(
        os.path.dirname(dataset_path), f"{timestamp}_{dir_suffix}"
    )
    os.makedirs(output_path, exist_ok=True)

    # Load configurations with fallback
    data_config = load_config_with_fallback(dataset_path, "data_config", output_path)
    aug_config = load_config_with_fallback(dataset_path, "aug_config", output_path)

    # Load source data if needed
    source_df = None
    if any(not s.get("source") for s in data_config["splits"]):
        print("📖 Loading source data...")
        source_df = pd.read_csv(os.path.join(dataset_path, data_config["source_csv"]))

    # Process splits
    for split_config in data_config["splits"]:
        # Get label handling mode
        label_as_probs = split_config.get("label_as_probs", False)

        # Load/create split data
        split_df = load_or_create_split(
            split_config=split_config,
            dataset_path=dataset_path,
            source_df=source_df,
            label_as_probs=label_as_probs,
        )

        # Handle output folder naming
        output_folder = split_config.get("output_folder", split_config["name"])

        # Process split with updated parameters
        process_split(
            split_config=split_config,
            df=split_df,
            dataset_path=dataset_path,
            output_root=output_path,
            output_folder=output_folder,
        )


if __name__ == "__main__":
    main()
