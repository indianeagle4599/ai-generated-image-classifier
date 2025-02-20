# evaluate.py

import torch
from torch.utils.data import DataLoader
from data.dataloaders import get_dataloaders
from models.model import CustomModel
from config import Config


def evaluate():
    config = Config()

    _, val_loader = get_dataloaders(
        config.DATA_PATH,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.IMAGE_SIZE,
        config.USE_AUGMENTATION,
    )

    model = CustomModel(config.MODEL_NAME, config.PRETRAINED, config.NUM_CLASSES)
    model.load_state_dict(torch.load("model.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")


if __name__ == "__main__":
    evaluate()
