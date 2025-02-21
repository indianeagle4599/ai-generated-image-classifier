import os
from tqdm import tqdm

import torch
from data.dataloaders import get_test_dataloader
from models.model import CustomModel

from config import Config


def predict():
    config = Config()

    run_folder = "runs/20250215_2147"
    model_name = "best.pt"

    config.load_json_config(os.path.join(run_folder, "config.json"))

    test, test_loader = get_test_dataloader(
        config.DATA_PATH,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.IMAGE_SIZE,
        False,
    )

    model = CustomModel(
        config.MODEL_NAME, config.PRETRAINED, num_classes=config.NUM_CLASSES
    )
    model.load_state_dict(
        torch.load(os.path.join(run_folder, model_name))["model_state_dict"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    test_pred_classes = []

    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Generating Test Predictions"):
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1)  # Get predicted class (0 or 1)
            test_pred_classes.extend(preds.cpu().numpy())

    # Add predictions to the test DataFrame
    test["label"] = test_pred_classes

    # Save predictions to a CSV file
    test[["id", "label"]].to_csv(
        os.path.join(run_folder, "submission.csv"), index=False
    )
    print("Test predictions saved to 'submission.csv'")


if __name__ == "__main__":
    predict()
