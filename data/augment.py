import numpy as np
import cv2
import scipy
from scipy.ndimage import gaussian_filter

import json
import csv
import os
import random

from tqdm import tqdm


class Augmentor:

    def __init__(self, config_path, df, image_folder, output_folder, output_csv):
        self.config = self.load_config(config_path)
        self.df = df

        # Set random seed for reproducibility
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

        self.augmentations = self.setup_augmentations()
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.output_csv = output_csv

        self.aug_dict = {image_name: 0 for image_name in self.df["file_name"]}

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            return json.load(f)

    def setup_augmentations(self):
        return {
            "random_flip": self.random_flip,
            "random_rotation": self.random_rotation,
            "color_jitter": self.color_jitter,
            "random_perspective": self.random_perspective,
            "random_affine": self.random_affine,
            "gaussian_blur": self.gaussian_blur,
            "random_erasing": self.random_erasing,
            "grid_distortion": self.grid_distortion,
            "elastic_transform": self.elastic_transform,
            "add_rain": self.add_rain,
            "add_glare": self.add_glare,
            "contrast_shift": self.contrast_shift,
            "random_crop": self.random_crop,
            "salt_pepper": self.salt_pepper_noise,
            "mixup": self.mixup,
            "mosaic": self.mosaic,
        }

    def letterbox(
        self,
        im,
        new_shape=(224, 224),
        color=(114, 114, 114),
        auto=True,
        scaleup=True,
        stride=32,
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        # Divide padding into 2 sides
        dw /= 2
        dh /= 2

        # Resize
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        # Ensure the final shape is exactly as specified
        im = cv2.resize(im, new_shape, interpolation=cv2.INTER_LINEAR)

        return im, r, (dw, dh)

    def salt_pepper_noise(self, image, amount=0.05):
        row, col, ch = image.shape
        s_vs_p = 0.5
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords[0], coords[1], :] = 1
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords[0], coords[1], :] = 0
        return out

    def mixup(self, images, labels, alpha=0.35):
        lam = np.random.beta(alpha, alpha)
        lam = lam if (lam > 1 - lam) else (1 - lam)
        mixed_image = (lam * images[0] + (1 - lam) * images[1]).astype(np.uint8)
        mixed_label = lam * labels[0] + (1 - lam) * labels[1]
        return mixed_image, mixed_label

    def mosaic(self, images, labels, target_shape=(640, 640)):
        mosaic_img = np.full((*target_shape, 3), 114, dtype=np.uint8)
        s = target_shape[0] // 2
        combined_label = np.zeros_like(labels[0])
        for i, (img, label) in enumerate(zip(images[:4], labels[:4])):
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = 0, 0, s, s
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = s, 0, target_shape[1], s
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = 0, s, s, target_shape[0]
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = s, s, target_shape[1], target_shape[0]
            img_resized, _, _ = self.letterbox(img, new_shape=(s, s))
            mosaic_img[y1a:y2a, x1a:x2a] = img_resized
            combined_label += label
        combined_label /= 4  # Average the labels
        return mosaic_img, combined_label

    def random_flip(self, image, p_horizontal=0.5, p_vertical=0.5):
        if np.random.random() < p_horizontal:
            image = cv2.flip(image, 1)
        if np.random.random() < p_vertical:
            image = cv2.flip(image, 0)
        return image

    def random_rotation(self, image, max_angle=30):
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(
            image,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )

    def color_jitter(
        self, image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)

        v = v.astype(np.float32)
        v *= np.random.uniform(1 - brightness, 1 + brightness)
        v = np.clip(v, 0, 255).astype(np.uint8)

        s = s.astype(np.float32)
        s *= np.random.uniform(1 - saturation, 1 + saturation)
        s = np.clip(s, 0, 255).astype(np.uint8)

        h = h.astype(np.float32)
        h += np.random.uniform(-hue * 180, hue * 180)
        h = np.mod(h, 180).astype(np.uint8)

        image = cv2.merge([h, s, v])
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        image = image.astype(np.float32)
        image *= np.random.uniform(1 - contrast, 1 + contrast)
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def random_perspective(self, image, scale=0.05):
        h, w = image.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32(
            [
                [
                    np.random.uniform(-scale * w, scale * w),
                    np.random.uniform(-scale * h, scale * h),
                ]
                for _ in range(4)
            ]
        )
        M = cv2.getPerspectiveTransform(pts1, pts1 + pts2)
        return cv2.warpPerspective(image, M, (w, h))

    def random_affine(self, image, scale=(0.95, 1.05), translate=(0.05, 0.05), shear=5):
        h, w = image.shape[:2]

        # Center of the image
        center = (w // 2, h // 2)

        # Random scale
        s = np.random.uniform(scale[0], scale[1])

        # Random translation (reduced effect)
        t = (
            np.random.uniform(-translate[0], translate[0]) * w * 0.5,
            np.random.uniform(-translate[1], translate[1]) * h * 0.5,
        )

        # Random shear (converted to radians and reduced effect)
        sh = np.deg2rad(np.random.uniform(-shear, shear) * 0.5)

        # Affine transformation matrix
        M = cv2.getRotationMatrix2D(center, 0, s)
        M[0, 2] += t[0]
        M[1, 2] += t[1]

        # Apply shear
        M = np.array(
            [[M[0, 0], M[0, 1] * np.tan(sh), M[0, 2]], [M[1, 0], M[1, 1], M[1, 2]]]
        )

        # Apply affine transformation
        return cv2.warpAffine(
            image,
            M,
            (w, h),
            borderMode=cv2.BORDER_REFLECT,
            borderValue=(114, 114, 114),
        )

    def gaussian_blur(self, image, max_sigma=3):
        sigma = np.random.uniform(0, max_sigma)
        return cv2.GaussianBlur(image, (0, 0), sigma)

    def random_erasing(self, image, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1 / 0.3):
        if np.random.random() < p:
            h, w = image.shape[:2]
            S = h * w

            for attempt in range(100):  # Limit the number of attempts
                Se = np.random.uniform(sl, sh) * S
                re = np.random.uniform(r1, r2)
                He = int(np.sqrt(Se * re))
                We = int(np.sqrt(Se / re))

                if (
                    He < h and We < w
                ):  # Check if the erasing rectangle fits within the image
                    xe = np.random.randint(0, w - We + 1)
                    ye = np.random.randint(0, h - He + 1)

                    image[ye : ye + He, xe : xe + We] = np.random.randint(
                        0, 256, (He, We, 3)
                    )
                    break

        return image

    def grid_distortion(self, image, num_steps=5, distort_limit=0.3):
        h, w = image.shape[:2]

        x_steps = np.linspace(0, w, num_steps)
        y_steps = np.linspace(0, h, num_steps)

        x_grid, y_grid = np.meshgrid(x_steps, y_steps)

        # Create full-size grid
        full_x_grid = np.linspace(0, w, w)
        full_y_grid = np.linspace(0, h, h)
        full_x, full_y = np.meshgrid(full_x_grid, full_y_grid)

        # Distort the grid
        distort_x = (
            x_grid
            + np.random.uniform(-distort_limit, distort_limit, x_grid.shape)
            * w
            / num_steps
        )
        distort_y = (
            y_grid
            + np.random.uniform(-distort_limit, distort_limit, y_grid.shape)
            * h
            / num_steps
        )

        # Interpolate to full size
        interp_x = scipy.interpolate.griddata(
            (x_grid.flatten(), y_grid.flatten()),
            distort_x.flatten(),
            (full_x, full_y),
            method="linear",
        )
        interp_y = scipy.interpolate.griddata(
            (x_grid.flatten(), y_grid.flatten()),
            distort_y.flatten(),
            (full_x, full_y),
            method="linear",
        )

        # Handle NaN values
        interp_x = np.nan_to_num(interp_x, nan=full_x)
        interp_y = np.nan_to_num(interp_y, nan=full_y)

        # Apply distortion
        return cv2.remap(
            image,
            interp_x.astype(np.float32),
            interp_y.astype(np.float32),
            cv2.INTER_LINEAR,
        )

    def elastic_transform(self, image, alpha=40, sigma=4, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape[:2]
        dx = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )
        dy = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # Ensure the distorted coordinates stay within the image boundaries
        distorted_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
        distorted_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)

        return cv2.remap(image, distorted_x, distorted_y, cv2.INTER_LINEAR)

    def add_rain(
        self,
        image,
        slant=-1,
        drop_length=20,
        drop_width=1,
        drop_color=(200, 200, 200),
        num_drops=100,
    ):
        imshape = image.shape
        rain = np.zeros_like(image)
        for _ in range(num_drops):
            x = np.random.randint(0, imshape[1])
            y = np.random.randint(0, max(1, imshape[0] - drop_length))
            cv2.line(rain, (x, y), (x + slant, y + drop_length), drop_color, drop_width)
        rain_blend = cv2.addWeighted(image, 1, rain, 0.7, 0)
        return rain_blend

    def add_glare(self, image, num_circles=3, max_radius=100):
        glare = np.zeros_like(image)
        h, w = image.shape[:2]
        for _ in range(num_circles):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            radius = np.random.randint(20, max_radius)
            cv2.circle(glare, (x, y), radius, (255, 255, 255), -1)
        glare = cv2.GaussianBlur(glare, (25, 25), 0)
        return cv2.addWeighted(image, 1, glare, 0.5, 0)

    def contrast_shift(self, image, factor=1.5):
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

    def random_crop(self, image, min_crop_size=(192, 192), max_crop_size=(224, 224)):
        h, w = image.shape[:2]
        ch = min(h, np.random.randint(min_crop_size[0], max_crop_size[0] + 1))
        cw = min(w, np.random.randint(min_crop_size[1], max_crop_size[1] + 1))

        if h > ch and w > cw:
            x = np.random.randint(0, w - cw + 1)
            y = np.random.randint(0, h - ch + 1)
            image = image[y : y + ch, x : x + cw]
        else:
            image = cv2.resize(image, (cw, ch))

        return cv2.resize(image, max_crop_size)

    def get_augmentation_combination(self):
        combination = {}
        current_noise = 0
        for aug, config in self.config["augmentations"].items():
            if (
                random.random() < config["probability"]
                and current_noise + config["noise_rating"]
                <= self.config["noise_threshold"]
            ):
                combination[aug] = config["params"]
                current_noise += config["noise_rating"]
        return combination

    def apply_augmentations(self, images, labels):
        augmented_images = []
        augmented_labels = []
        augmentation_lists = []
        target_shape = self.config["target_shape"]
        for img, label in zip(images, labels):
            img_resized, _, _ = self.letterbox(img, new_shape=target_shape)
            aug_combination = self.get_augmentation_combination()
            applied_augs = []
            for aug, params in aug_combination.items():
                if aug == "mixup" and len(images) >= 2:
                    idx = random.randint(0, len(images) - 1)
                    img2, _, _ = self.letterbox(images[idx], new_shape=target_shape)
                    img_resized, label = self.augmentations[aug](
                        images=[img_resized, img2],
                        labels=[label, labels[idx]],
                        **params,
                    )
                    applied_augs.append(aug)
                elif aug == "mosaic" and len(images) >= 4:
                    indices = random.sample(range(len(images)), 4)
                    img_resized, label = self.augmentations[aug](
                        [images[i] for i in indices],
                        [labels[i] for i in indices],
                        target_shape=target_shape,
                    )
                    applied_augs.append(aug)
                elif aug != "mixup" and aug != "mosaic":
                    img_resized = self.augmentations[aug](img_resized, **params)
                    applied_augs.append(aug)
            augmented_images.append(img_resized)
            augmented_labels.append(label)
            augmentation_lists.append("_".join(applied_augs))
        return augmented_images, augmented_labels, augmentation_lists

    def preprocess_image(self, image_path, output_path, target_shape):
        image = cv2.imread(image_path)
        image, _, _ = self.letterbox(im=image, new_shape=target_shape)
        cv2.imwrite(output_path, image)

    def process_original_images(self):
        for index, row in tqdm(self.df.iterrows(), desc="Processing original images"):
            image_path = os.path.join(self.image_folder, row["file_name"])
            new_file_path = os.path.join(self.output_folder, row["file_name"])
            self.preprocess_image(
                image_path, new_file_path, self.config["target_shape"]
            )

    def augment_images(self):
        batch_size = self.config.get("batch_size", 16)
        total_batches = (len(self.df) + batch_size - 1) // batch_size

        results = []
        for m in range(self.config["multiplier"]):
            # Shuffle the DataFrame before creating batches
            shuffled_df = self.df.sample(frac=1).reset_index(drop=True)

            for i in tqdm(
                range(total_batches),
                desc=f"Processing batches (Epoch {m+1}/{self.config['multiplier']})",
            ):
                batch = shuffled_df.iloc[i * batch_size : (i + 1) * batch_size]
                batch_results = self.process_augmentations(
                    batch, self.config, self.image_folder, self.config["target_shape"]
                )
                results.extend(batch_results)

        self.save_augmentations(results)

    def process_augmentations(self, batch, config, image_folder, target_shape):
        images = []
        labels = []
        file_names = []
        for index, row in batch.iterrows():
            image_path = os.path.join(image_folder, row["file_name"])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image, _, _ = self.letterbox(image, new_shape=target_shape)
            label = float(row["label"])
            images.append(image)
            labels.append(label)
            file_names.append(row["file_name"])
            self.aug_dict[row["file_name"]] += 1

        augmented_images, augmented_labels, augmentation_lists = (
            self.apply_augmentations(images, labels)
        )

        results = []
        for i, (aug_img, aug_label, aug_list) in enumerate(
            zip(augmented_images, augmented_labels, augmentation_lists)
        ):

            new_filename = f"{os.path.splitext(file_names[i])[0]}_aug_{self.aug_dict[file_names[i]]}.jpg"
            results.append((aug_img, aug_label, new_filename, aug_list))
        return results

    def save_augmentations(self, results):
        with open(self.output_csv, "a", newline="") as csvfile:
            fieldnames = ["file_name", "label", "augmentations"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()

            for img, label, filename, aug_list in results:
                cv2.imwrite(
                    os.path.join(self.output_folder, filename),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
                writer.writerow(
                    {
                        "file_name": filename,
                        "label": label,
                        "augmentations": aug_list,
                    }
                )


def main(config_path, df, output_folder, output_csv, num_workers):
    augmentor = Augmentor(config_path, df, output_folder, output_csv)
    augmentor.process_images(num_workers)
    print(f"Augmented images saved in: {augmentor.output_folder}")
    print(f"Augmented CSV file: {augmentor.output_csv}")


if __name__ == "__main__":
    import pandas as pd

    config_path = "aug_config.json"
    df = pd.read_csv("dataset/train.csv")
    output_folder = "path/to/output/folder"
    output_csv = "path/to/output/csv.csv"
    main(config_path, df, output_folder, output_csv, num_workers=4)
