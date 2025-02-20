import json


class Config:
    # Data configuration
    DATA_PATH = "dataset"
    TRAIN_CSV = "augmented_20250214_2324.csv"
    TEST_CSV = "test.csv"
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    IMAGE_SIZE = (112, 112)
    NUM_CLASSES = 2

    # Model configuration
    MODEL_NAME = "mobilenet_v2"  # Options: "resnet50", "vgg16", "mobilenet_v2", "inception_v3", "convnext_tiny", "resnext50_32x4d", "vit_b_16", "vit_l_32", "swin_v2_b", "swin_v2_t"
    PRETRAINED = False

    # Training configuration
    EPOCHS = 25
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001

    # Augmentation configuration
    USE_AUGMENTATION = False
    AUGMENTATION_OPTIONS = {
        "random_crop": True,
        "horizontal_flip": True,
        "color_jitter": True,
        "rotation": 10,
    }

    @classmethod
    def to_dict(self):
        return {
            key.lower(): value
            for key, value in vars(self).items()
            if not key.startswith("__") and not callable(value)
        }

    @classmethod
    def create_json_config(self, filename="config.json"):
        config_dict = self.to_dict()
        structured_config = {
            "data": {
                "data_path": config_dict.get("data_path"),
                "train_csv": config_dict.get("train_csv"),
                "test_csv": config_dict.get("test_csv"),
                "batch_size": config_dict.get("batch_size"),
                "num_workers": config_dict.get("num_workers"),
                "image_size": config_dict.get("image_size"),
                "num_classes": config_dict.get("num_classes"),
            },
            "model": {
                "model_name": config_dict.get("model_name"),
                "pretrained": config_dict.get("pretrained"),
            },
            "training": {
                "epochs": config_dict.get("epochs"),
                "learning_rate": config_dict.get("learning_rate"),
                "momentum": config_dict.get("momentum"),
                "weight_decay": config_dict.get("weight_decay"),
            },
            "augmentation": {
                "use_augmentation": config_dict.get("use_augmentation"),
                "options": config_dict.get("augmentation_options"),
            },
        }

        with open(filename, "w") as f:
            json.dump(structured_config, f, indent=4)

        print(f"Configuration saved to {filename}")

    @classmethod
    def load_json_config(self, filename="config.json"):
        with open(filename, "r") as f:
            config_data = json.load(f)

        # Update class attributes
        for category, values in config_data.items():
            for key, value in values.items():
                setattr(self, key.upper(), value)

        print(f"Configuration loaded from {filename}")
