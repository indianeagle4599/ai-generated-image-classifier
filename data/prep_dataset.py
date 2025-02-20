import os
import shutil
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import json


def format_path(path, depth=4):
    """
    Returns only the first few parts of a path to reduce verbosity.
    """
    parts = os.path.normpath(path).split(os.sep)
    return os.sep.join(parts[-1 * min(depth, len(parts)) :])


def split_data(df, split_value, seed=None):
    """
    Shuffles and splits a DataFrame.

    Parameters:
      - df: The DataFrame to split.
      - split_value: If a float (0 < value < 1), the fraction for the split;
                     if >= 1, the exact number of samples.
      - seed: The seed for random shuffling.

    Returns:
      - (split_df, remaining_df)
    """
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if split_value >= 1:
        split_size = int(split_value)
    else:
        split_size = int(len(df_shuffled) * split_value)
    split_df = df_shuffled.iloc[:split_size].copy()
    remaining_df = df_shuffled.iloc[split_size:].copy()
    return split_df, remaining_df


def update_dataframe(df, new_folder_name):
    """
    Updates the 'file_name' column so that the folder prefix is set to new_folder_name.
    Assumes original file names are formatted as "<old_folder>/<file_name>".
    """
    df["file_name"] = df["file_name"].apply(
        lambda x: os.path.join(new_folder_name, os.path.basename(x))
    )
    return df


def copy_images_batch(csv_paths, src_folders, dest_folders):
    """
    Copies images to new directories based on CSV file lists.
    Uses tqdm for progress reporting.
    """
    total_images = 0
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            total_images += len(df)

    with tqdm(total=total_images, desc="Copying images") as pbar:
        for csv_path, src_folder, dest_folder in zip(
            csv_paths, src_folders, dest_folders
        ):
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    file_name = os.path.basename(row["file_name"])
                    src_img = os.path.join(src_folder, file_name)
                    dest_img = os.path.join(dest_folder, file_name)
                    if os.path.exists(src_img):
                        shutil.copy(src_img, dest_img)
                        pbar.update(1)
                    else:
                        print(f"Warning: Image not found: {src_img}")

    # Verify copied images
    for csv_path, dest_folder in zip(csv_paths, dest_folders):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            expected_count = len(df)
            actual_count = len(
                [
                    f
                    for f in os.listdir(dest_folder)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            print(f"Folder: {dest_folder}")
            print(f"  Expected images: {expected_count}")
            print(f"  Actual images: {actual_count}")
            if expected_count != actual_count:
                print(f"  Warning: Mismatch in image count for {dest_folder}")


def setup_directories(new_base_dir, splits):
    """
    Creates new directories for each split.

    Parameters:
      - new_base_dir: The base output directory.
      - splits: A list of split names (e.g. train, val, dev, test).

    Returns:
      - A dict mapping split name to new directory path.
    """
    new_dirs = {}
    for split in splits:
        new_path = os.path.join(new_base_dir, split)
        os.makedirs(new_path, exist_ok=True)
        new_dirs[split] = new_path
    return new_dirs


def main(
    root_dir_path="dataset/example",
    train_dir="train",
    val_dir="val",
    dev_dir="dev",
    test_dir="test",
    train_csv="train.csv",
    val_csv="val.csv",
    dev_csv="dev.csv",
    test_csv="test.csv",
    val_split=0.2,  # fraction of training data allocated to validation if CSV not present
    dev_split=0.1,  # fraction from final train used for dev (overlapping with train)
    test_split=0.1,  # fraction of training data allocated to test if CSV not present
    seed=42,
    preprocess_images=False,
    augment=False,
    augment_config_json=None,
):
    """
    Organizes a dataset into train, validation, development, and test splits.

    Behavior:
      - If a CSV file for a split already exists, it is used (after updating file names).
      - Otherwise, splits are created from the master train CSV.
      - Data is randomly shuffled using the provided seed.
      - The 'dev' split is sampled from train and overlaps with training.
      - File names are updated to reflect the new folder names.
      - If augment is True, then preprocess_images is enabled and augmentation is applied only to train images.
    """
    # If augmentation is enabled then enable preprocessing
    if augment:
        preprocess_images = True

    print("-" * 50)
    print("Starting dataset preparation with settings:")
    print(f"  Preprocess Images: {preprocess_images}")
    print(f"  Augment: {augment}")
    print(f"  Seed: {seed}")
    print("-" * 50)

    # Create new output base directory with timestamp and optional augment suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_suffix = "_aug" if augment else ""
    new_base_dir = os.path.join(
        os.path.dirname(root_dir_path), f"{timestamp}{dir_suffix}"
    )
    os.makedirs(new_base_dir, exist_ok=True)
    print(f"Output base directory: {format_path(new_base_dir)}")
    print("-" * 50)

    # Set up original directories and CSV paths
    orig_dirs = {
        "train": os.path.join(root_dir_path, train_dir),
        "val": os.path.join(root_dir_path, val_dir),
        "dev": os.path.join(root_dir_path, dev_dir),
        "test": os.path.join(root_dir_path, test_dir),
    }
    csv_paths = {
        "train": os.path.join(root_dir_path, train_csv),
        "val": os.path.join(root_dir_path, val_csv),
        "dev": os.path.join(root_dir_path, dev_csv),
        "test": os.path.join(root_dir_path, test_csv),
    }
    dfs = {
        "train": None,
        "val": None,
        "dev": None,
        "test": None,
    }

    # Create new directories for each split
    splits = ["train", "val", "dev", "test"]
    new_dirs = setup_directories(new_base_dir, splits)

    # Load master training CSV (required)
    if os.path.exists(csv_paths["train"]):
        full_df = pd.read_csv(csv_paths["train"])
        full_df = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"Loaded training CSV: {format_path(csv_paths['train'])}")
    else:
        print(f"Training CSV not found at: {format_path(csv_paths['train'])}")
        return

    original_size = len(full_df)

    # --- Split Test Set ---
    if not os.path.exists(csv_paths["test"]) and test_split > 0:
        test_df, remaining_df = split_data(
            full_df,
            test_split if test_split >= 1 else int(original_size * test_split),
            seed,
        )
        test_df = update_dataframe(test_df, test_dir)
        new_test_csv = os.path.join(new_base_dir, test_csv)
        test_df.to_csv(new_test_csv, index=False)
        print(
            f"Test set created at: {format_path(new_test_csv)} with {len(test_df)} samples"
        )
        dfs["test"] = test_df
    elif os.path.exists(csv_paths["test"]):
        test_df = pd.read_csv(csv_paths["test"])
        test_df = update_dataframe(test_df, test_dir)
        new_test_csv = os.path.join(new_base_dir, test_csv)
        test_df.to_csv(new_test_csv, index=False)
        print(
            f"Copied existing test CSV to: {format_path(new_test_csv)} with {len(test_df)} samples"
        )
        dfs["test"] = test_df
    else:
        new_test_csv = os.path.join(new_base_dir, test_csv)
        open(new_test_csv, "w").close()
        print(f"Created placeholder test CSV at: {format_path(new_test_csv)}")

    # --- Split Validation Set ---
    if not os.path.exists(csv_paths["val"]) and val_split > 0:
        val_df, _ = split_data(
            full_df,
            val_split if val_split >= 1 else int(original_size * val_split),
            seed,
        )
        val_df = update_dataframe(val_df, val_dir)
        new_val_csv = os.path.join(new_base_dir, val_csv)
        val_df.to_csv(new_val_csv, index=False)
        print(
            f"Validation set created at: {format_path(new_val_csv)} with {len(val_df)} samples"
        )
        dfs["val"] = val_df
    elif os.path.exists(csv_paths["val"]):
        val_df = pd.read_csv(csv_paths["val"])
        val_df = update_dataframe(val_df, val_dir)
        new_val_csv = os.path.join(new_base_dir, val_csv)
        val_df.to_csv(new_val_csv, index=False)
        print(
            f"Copied existing validation CSV to: {format_path(new_val_csv)} with {len(val_df)} samples"
        )
        dfs["val"] = val_df
    else:
        new_val_csv = os.path.join(new_base_dir, val_csv)
        open(new_val_csv, "w").close()
        print(f"Created placeholder validation CSV at: {format_path(new_val_csv)}")

    # --- Create Development Set (overlap with train) ---
    if not os.path.exists(csv_paths["dev"]) and dev_split > 0:
        dev_df, _ = split_data(
            full_df,
            dev_split if dev_split >= 1 else int(original_size * dev_split),
            seed,
        )
        dev_df = update_dataframe(dev_df, dev_dir)
        new_dev_csv = os.path.join(new_base_dir, dev_csv)
        dev_df.to_csv(new_dev_csv, index=False)
        print(
            f"Development set created at: {format_path(new_dev_csv)} with {len(dev_df)} samples"
        )
        dfs["dev"] = dev_df
    elif os.path.exists(csv_paths["dev"]):
        dev_df = pd.read_csv(csv_paths["dev"])
        dev_df = update_dataframe(dev_df, dev_dir)
        new_dev_csv = os.path.join(new_base_dir, dev_csv)
        dev_df.to_csv(new_dev_csv, index=False)
        print(
            f"Copied existing development CSV to: {format_path(new_dev_csv)} with {len(dev_df)} samples"
        )
        dfs["dev"] = dev_df
    else:
        new_dev_csv = os.path.join(new_base_dir, dev_csv)
        open(new_dev_csv, "w").close()
        print(f"Created placeholder development CSV at: {format_path(new_dev_csv)}")

    # --- Final Training Set ---
    train_df = full_df[
        ~full_df.index.isin(test_df.index) & ~full_df.index.isin(val_df.index)
    ]
    train_df = update_dataframe(train_df.copy(), train_dir)
    new_train_csv = os.path.join(new_base_dir, train_csv)
    train_df.to_csv(new_train_csv, index=False)
    print(
        f"Final training set CSV at: {format_path(new_train_csv)} with {len(train_df)} samples"
    )
    dfs["train"] = train_df

    print("-" * 50)
    print("Copying images to new split directories...")

    csvs = {
        "train": new_train_csv,
        "val": new_val_csv,
        "dev": new_dev_csv,
        "test": new_test_csv,
    }

    # Copy images for each split
    copy_images_batch(
        csv_paths=[new_train_csv, new_val_csv, new_dev_csv, new_test_csv],
        src_folders=[
            orig_dirs["train"],
            (
                orig_dirs["val"]
                if os.path.exists(orig_dirs["val"])
                else orig_dirs["train"]
            ),
            (
                orig_dirs["dev"]
                if os.path.exists(orig_dirs["dev"])
                else orig_dirs["train"]
            ),
            (
                orig_dirs["test"]
                if os.path.exists(orig_dirs["test"])
                else orig_dirs["train"]
            ),
        ],
        dest_folders=[
            new_dirs["train"],
            new_dirs["val"],
            new_dirs["dev"],
            new_dirs["test"],
        ],
    )
    print("-" * 50)
    print("Dataset splitting and copying complete.")

    # --- Preprocessing / Augmentation ---
    if preprocess_images or augment:
        from augment import Augmentor  # Ensure this module is available

        if augment_config_json:
            augment_config_path = os.path.join(root_dir_path, augment_config_json)
        else:
            augment_config_path = os.path.join(root_dir_path, "aug_config.json")

        print("-" * 50)
        print("Preprocessing images for all splits...")
        # For each split, call your preprocessing logic here.
        # This is a placeholder loop; replace it with your actual image preprocessing function.
        for split in splits:
            if os.path.exists(csvs[split]) and os.path.getsize(csvs[split]) > 0:
                print(
                    f"Preprocessing images in '{split}' folder: {format_path(new_dirs[split])}"
                )
                # E.g., preprocess_folder(folder_path, csv_file, target_shape=(256,256))
                augmentor = Augmentor(
                    config_path=augment_config_path,
                    df=dfs[split],
                    image_folder=os.path.dirname(new_dirs[split]),
                    output_folder=os.path.dirname(new_dirs[split]),
                    output_csv=csvs[split],
                )
                # Perform augmentation only on training images.
                augmentor.process_original_images()
            else:
                print(f"No CSV data for '{split}', skipping preprocessing.")

    if augment:
        print("-" * 50)
        print("Performing augmentation on train images only...")
        try:
            print("Initializing Augmentor...")
            # Initialize the augmentor.
            augmentor = Augmentor(
                config_path=augment_config_path,
                df=train_df,
                image_folder=os.path.dirname(new_dirs["train"]),
                output_folder=os.path.dirname(new_dirs["train"]),
                output_csv=new_train_csv,
            )
            # Perform augmentation only on training images.
            augmentor.augment_images()
            augmentor.save_augmentations()
            print(
                f"Augmentation complete for train folder: {format_path(new_dirs['train'])}"
            )
        except ImportError:
            print(
                "Augmentor module not found. Please ensure the augmentation module is available."
            )
        except Exception as e:
            print(f"An error occurred during augmentation: {e}")

    print("-" * 50)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main(
        root_dir_path="C:/Users/hites/OneDrive/Desktop/AI6102 - ML Project/dataset/example",
        val_split=0.2,
        dev_split=0.3,
        augment=True,
    )
