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
from torchvision.models import vit_b_16, ViT_B_16_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# Configuration
# ============================================================

SEED = 3407

DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

RESULT_DIR = "./results/vit"
MODEL_DIR = "./models"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
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
# Dataset
# ============================================================

weights = ViT_B_16_Weights.DEFAULT

mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("=" * 70)
print("Dataset")
print("=" * 70)

print(f"Train: {len(train_dataset)}")
print(f"Val:   {len(val_dataset)}")
print(f"Test:  {len(test_dataset)}")
print(f"Classes: {num_classes}")
print(f"Class names: {class_names}")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# ============================================================
# Model
# ============================================================

print("=" * 70)
print("Loading pretrained ViT-B/16")
print("=" * 70)

model = vit_b_16(weights=weights)

# Replace ImageNet 1000-class classification head
in_features = model.heads.head.in_features

model.heads.head = nn.Linear(
    in_features,
    num_classes
)

model = model.to(DEVICE)


# ============================================================
# Model information
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Device: {DEVICE}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ============================================================
# Loss / Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# Training function
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ============================================================
# Evaluation function
# ============================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = outputs.argmax(
                dim=1
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    try:
        macro_auc = roc_auc_score(
            all_labels,
            all_probabilities,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        macro_auc = float("nan")

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "macro_auc": macro_auc,
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }


# ============================================================
# Training
# ============================================================

print("=" * 70)
print("Training")
print("=" * 70)

history = []

best_val_auc = -1.0

training_start = time.perf_counter()

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.perf_counter()

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    val_metrics = evaluate(
        model,
        val_loader
    )

    scheduler.step()

    epoch_time = time.perf_counter() - epoch_start

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc * 100:.2f}% "
        f"Val Loss: {val_metrics['loss']:.4f} "
        f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% "
        f"Val F1: {val_metrics['f1_macro']:.4f} "
        f"Val AUC: {val_metrics['macro_auc']:.4f} "
        f"Time: {epoch_time:.2f}s"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision_macro": val_metrics["precision_macro"],
        "val_recall_macro": val_metrics["recall_macro"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_macro_auc": val_metrics["macro_auc"],
        "epoch_time_sec": epoch_time,
    })

    if val_metrics["macro_auc"] > best_val_auc:

        best_val_auc = val_metrics["macro_auc"]

        checkpoint_path = os.path.join(
            MODEL_DIR,
            "best_vit_b16.pth"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "best_val_macro_auc": best_val_auc,
                "epoch": epoch,
            },
            checkpoint_path
        )

        print(
            f"  -> Saved best model: "
            f"{checkpoint_path}"
        )


training_time = (
    time.perf_counter()
    - training_start
)


# ============================================================
# Load best model
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_vit_b16.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# Test inference
# ============================================================

print("=" * 70)
print("Test Evaluation")
print("=" * 70)

inference_start = time.perf_counter()

test_metrics = evaluate(
    model,
    test_loader
)

inference_time = (
    time.perf_counter()
    - inference_start
)


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
    f"{training_time:.2f} sec"
)

print(
    f"Test Inference Time: "
    f"{inference_time:.2f} sec"
)

print(
    f"Test Inference Time/Image: "
    f"{inference_time / len(test_dataset) * 1000:.4f} ms"
)


# ============================================================
# Save metrics
# ============================================================

metrics = {
    "model": "ViT-B/16",
    "dataset": "Wafer Defect",
    "num_classes": num_classes,
    "class_names": class_names,
    "test_samples": len(test_dataset),
    "accuracy": test_metrics["accuracy"],
    "precision_macro": test_metrics["precision_macro"],
    "recall_macro": test_metrics["recall_macro"],
    "f1_macro": test_metrics["f1_macro"],
    "macro_auc": test_metrics["macro_auc"],
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "training_time_sec": training_time,
    "inference_time_sec": inference_time,
    "inference_time_per_image_ms": (
        inference_time / len(test_dataset) * 1000
    ),
}

with open(
    os.path.join(RESULT_DIR, "metrics.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
        ensure_ascii=False
    )


with open(
    os.path.join(RESULT_DIR, "history.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        history,
        f,
        indent=2,
        ensure_ascii=False
    )


config = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "num_workers": NUM_WORKERS,
    "image_size": 224,
    "model": "ViT-B/16",
    "pretrained": True,
    "device": str(DEVICE),
}

with open(
    os.path.join(RESULT_DIR, "config.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# Save test predictions
# ============================================================

prediction_data = {
    "true_label": test_metrics["labels"],
    "predicted_label": test_metrics["predictions"],
}

for idx, class_name in enumerate(class_names):
    prediction_data[
        f"prob_{class_name}"
    ] = test_metrics["probabilities"][:, idx]

predictions_df = pd.DataFrame(
    prediction_data
)

predictions_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "test_predictions.csv"
    ),
    index=False
)


print("=" * 70)
print("Finished")
print("=" * 70)
print(
    f"Best checkpoint: {checkpoint_path}"
)
print(
    f"Results directory: {RESULT_DIR}"
)
