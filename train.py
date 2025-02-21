# train.py

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassF1Score


from data.dataloaders import get_dataloaders, get_test_dataloader

from models.model import CustomModel
from config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import os
from datetime import datetime


def create_run_folder():
    base_folder = "runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_folder = os.path.join(base_folder, timestamp)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def plot_confusion_matrices(
    y_true_train, y_pred_train, y_true_val, y_pred_val, classes, save_path
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Train Confusion Matrix
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    sns.heatmap(
        cm_train,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax1,
    )
    ax1.set_title("Train Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    # Validation Confusion Matrix
    cm_val = confusion_matrix(y_true_val, y_pred_val)
    sns.heatmap(
        cm_val,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax2,
    )
    ax2.set_title("Validation Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    train_f1_scores,
    val_f1_scores,
    run_folder,
):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")

    plt.subplot(1, 3, 3)
    plt.plot(train_f1_scores, label="Train F1 Score")
    plt.plot(val_f1_scores, label="Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("F1 Score Curves")

    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "curves.png"))
    plt.close()


def train():
    config = Config()
    run_folder = create_run_folder()

    # Save configuration
    config.create_json_config(os.path.join(run_folder, "config.json"))

    train_loader, val_loader = get_dataloaders(
        config.DATA_PATH,
        config.TRAIN_CSV,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.IMAGE_SIZE,
        config.USE_AUGMENTATION,
        config.FEATURES,
    )

    if config.TEST_CSV:
        test, test_loader = get_test_dataloader(
            config.DATA_PATH,
            config.BATCH_SIZE,
            config.NUM_WORKERS,
            config.IMAGE_SIZE,
            False,
            config.FEATURES,
        )

    model = CustomModel(
        config.MODEL_NAME,
        config.PRETRAINED,
        config.NUM_CLASSES,
        config.FEATURES,
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)

    if config.MODEL_NAME and config.PRETRAINED:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.parameters(),
                    "lr": 1e-4,
                },  # Higher LR for classifier
                {
                    "params": model.backbone.parameters(),
                    "lr": 1e-5,
                },  # Lower LR for backbone
            ]
        )

    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize F1 Score metric
    f1_metric = MulticlassF1Score(
        num_classes=config.NUM_CLASSES, average="weighted"
    ).to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    best_val_f1 = -1
    run_str = ""

    for epoch in range(config.EPOCHS):
        torch.cuda.empty_cache()
        # Training loop
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_f1 = 0

        all_train_preds, all_train_labels = [], []
        for x, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(x, device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds, labs = outputs.argmax(dim=1), labels.argmax(dim=1)
            acc = (preds == labs).float().mean().item()
            train_accuracy += acc

            all_train_preds.extend(list(preds.detach().cpu()))
            all_train_labels.extend(list(labs.detach().cpu()))

        all_train_preds, all_train_labels = (
            torch.Tensor(all_train_preds).flatten().to(device),
            torch.Tensor(all_train_labels).flatten().to(device),
        )

        train_true = all_train_labels.cpu().numpy()
        train_pred = all_train_preds.cpu().numpy()

        train_f1 = f1_metric(all_train_preds, all_train_labels).item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_f1 = 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for x, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                labels = labels.to(device)

                outputs = model(x, device)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds, labs = outputs.argmax(dim=1), labels.argmax(dim=1)
                acc = (preds == labs).float().mean().item()
                val_accuracy += acc
                all_val_preds.extend(list(preds.detach().cpu()))
                all_val_labels.extend(list(labs.detach().cpu()))

        all_val_preds, all_val_labels = (
            torch.Tensor(all_val_preds).flatten().to(device),
            torch.Tensor(all_val_labels).flatten().to(device),
        )

        val_true = all_val_labels.cpu().numpy()
        val_pred = all_val_preds.cpu().numpy()

        val_f1 = f1_metric(all_val_preds, all_val_labels).item()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        eval_str = (
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f} | "
            + f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}"
        )
        run_str += eval_str + "\n"
        print(eval_str)

        with open(os.path.join(run_folder, "run.txt"), "w") as f:
            f.write(run_str)

        # Save model checkpoints
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_f1": train_f1,
            "val_f1": val_f1,
        }

        # Save last.pt
        torch.save(checkpoint, os.path.join(run_folder, "last.pt"))

        # Save best.pt (now based on F1 score)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(checkpoint, os.path.join(run_folder, "best.pt"))
            plot_confusion_matrices(
                train_true,
                train_pred,
                val_true,
                val_pred,
                classes=range(config.NUM_CLASSES),
                save_path=os.path.join(run_folder, f"confusion_matrix_best.png"),
            )

            # Predict on test.csv and create submission.csv
            if config.TEST_CSV:
                test_pred_classes = []
                with torch.no_grad():
                    for data, _ in tqdm(
                        test_loader, desc="Generating Test Predictions"
                    ):
                        output = model(data, device)
                        preds = output.argmax(dim=1)  # Get predicted class (0 or 1)
                        test_pred_classes.extend(preds.cpu().numpy())

                # Add predictions to the test DataFrame
                test["label"] = test_pred_classes

                test["id"] = test["file_name"]
                # Save predictions to a CSV file
                test[["id", "label"]].to_csv(
                    os.path.join(run_folder, "submission.csv"), index=False
                )
                print("Test predictions saved to 'submission.csv'")

        # Plot and save curves (including F1 scores)
        plot_curves(
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            train_f1_scores,
            val_f1_scores,
            run_folder,
        )

        # Plot confusion matrices
        plot_confusion_matrices(
            train_true,
            train_pred,
            val_true,
            val_pred,
            classes=range(config.NUM_CLASSES),
            save_path=os.path.join(run_folder, f"confusion_matrix_last.png"),
        )


if __name__ == "__main__":
    train()
