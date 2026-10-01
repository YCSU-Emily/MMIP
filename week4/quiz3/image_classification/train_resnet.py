import os
import json
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ============================================================
# Configuration
# ============================================================

SEED = 3407

DATA_DIR = "./data"
MODEL_DIR = "./models"
RESULT_DIR = "./results/resnet18"

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# Utility
# ============================================================

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate(model, loader, criterion, class_names):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    inference_time = time.time() - start_time

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)

    avg_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    # Macro-AUC
    y_true_bin = label_binarize(
        all_labels,
        classes=np.arange(len(class_names))
    )

    macro_auc = roc_auc_score(
        y_true_bin,
        all_probs,
        multi_class="ovr",
        average="macro",
    )

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "macro_auc": float(macro_auc),
        "inference_time_sec": float(inference_time),
        "inference_time_per_image_ms": float(
            inference_time / len(loader.dataset) * 1000
        ),
    }

    return metrics, all_labels, all_preds, all_probs


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 70)
    print("Week 4 Quiz 3 - ResNet18 Image Classification")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    weights = ResNet18_Weights.DEFAULT

    imagenet_mean = weights.transforms().mean
    imagenet_std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std,
        ),
    ])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform=eval_transform,
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print()
    print("Dataset")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")
    print(f"Classes: {num_classes}")
    print(f"Class names: {class_names}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading pretrained ResNet18")

    model = resnet18(weights=weights)

    # Replace the original 1000-class classifier
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes,
    )

    model = model.to(DEVICE)

    total_params = count_parameters(model)
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --------------------------------------------------------
    # Loss / Optimizer / Scheduler
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_val_auc = -1.0
    best_epoch = 0

    model_path = os.path.join(
        MODEL_DIR,
        "best_resnet18.pth",
    )

    total_training_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            preds = torch.argmax(
                outputs,
                dim=1,
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / len(
            train_loader.dataset
        )

        train_acc = correct / total

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            class_names,
        )

        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]

        history_item = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_macro_auc": val_metrics["macro_auc"],
            "learning_rate": float(current_lr),
            "epoch_time_sec": float(epoch_time),
        }

        history.append(history_item)

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc * 100:.2f}% "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% "
            f"Val F1: {val_metrics['f1_macro']:.4f} "
            f"Val AUC: {val_metrics['macro_auc']:.4f} "
            f"Time: {epoch_time:.2f}s"
        )

        # ----------------------------------------------------
        # Save best model based on validation Macro-AUC
        # ----------------------------------------------------

        if val_metrics["macro_auc"] > best_val_auc:

            best_val_auc = val_metrics["macro_auc"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_auc": best_val_auc,
                    "class_names": class_names,
                },
                model_path,
            )

    total_training_time = (
        time.time() - total_training_start
    )

    # --------------------------------------------------------
    # Load Best Model
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_metrics, test_labels, test_preds, test_probs = evaluate(
        model,
        test_loader,
        criterion,
        class_names,
    )

    # --------------------------------------------------------
    # Save Test Predictions
    # --------------------------------------------------------

    prediction_data = {
        "true_label": test_labels,
        "predicted_label": test_preds,
    }

    for i, class_name in enumerate(class_names):
        prediction_data[
            f"prob_{class_name}"
        ] = test_probs[:, i]

    predictions_df = pd.DataFrame(
        prediction_data
    )

    predictions_path = os.path.join(
        RESULT_DIR,
        "test_predictions.csv",
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save History
    # --------------------------------------------------------

    history_path = os.path.join(
        RESULT_DIR,
        "history.json",
    )

    save_json(
        history_path,
        history,
    )

    # --------------------------------------------------------
    # Save Metrics
    # --------------------------------------------------------

    final_metrics = {
        "model": "ResNet18",
        "num_classes": num_classes,
        "class_names": class_names,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "best_epoch": best_epoch,
        "best_val_macro_auc": float(best_val_auc),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_macro_auc": test_metrics["macro_auc"],
        "training_time_sec": total_training_time,
        "test_inference_time_sec": test_metrics[
            "inference_time_sec"
        ],
        "test_inference_time_per_image_ms": test_metrics[
            "inference_time_per_image_ms"
        ],
    }

    metrics_path = os.path.join(
        RESULT_DIR,
        "metrics.json",
    )

    save_json(
        metrics_path,
        final_metrics,
    )

    # --------------------------------------------------------
    # Save Configuration
    # --------------------------------------------------------

    config = {
        "seed": SEED,
        "model": "resnet18",
        "pretrained": True,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "num_workers": NUM_WORKERS,
        "device": str(DEVICE),
    }

    config_path = os.path.join(
        RESULT_DIR,
        "config.json",
    )

    save_json(
        config_path,
        config,
    )

    # --------------------------------------------------------
    # Print Final Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Test Results")
    print("=" * 70)

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Test Precision Macro: "
        f"{test_metrics['precision_macro']:.4f}"
    )

    print(
        f"Test Recall Macro: "
        f"{test_metrics['recall_macro']:.4f}"
    )

    print(
        f"Test F1 Macro: "
        f"{test_metrics['f1_macro']:.4f}"
    )

    print(
        f"Test Macro-AUC: "
        f"{test_metrics['macro_auc']:.4f}"
    )

    print(
        f"Training Time: "
        f"{total_training_time:.2f} sec"
    )

    print(
        f"Test Inference Time: "
        f"{test_metrics['inference_time_sec']:.2f} sec"
    )

    print(
        f"Test Inference Time/Image: "
        f"{test_metrics['inference_time_per_image_ms']:.4f} ms"
    )

    print(
        f"Best Validation Macro-AUC: "
        f"{best_val_auc:.4f}"
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Best checkpoint: {model_path}"
    )

    print(
        f"Results directory: {RESULT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
