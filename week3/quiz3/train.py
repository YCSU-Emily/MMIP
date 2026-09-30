import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from models import (
    create_model,
    count_parameters,
    count_trainable_parameters,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_topk(output, target, topk=(1, 5)):
    maxk = max(topk)

    batch_size = target.size(0)

    _, pred = output.topk(
        maxk,
        dim=1,
        largest=True,
        sorted=True
    )

    pred = pred.t()

    correct = pred.eq(
        target.view(1, -1).expand_as(pred)
    )

    results = []

    for k in topk:
        correct_k = (
            correct[:k]
            .reshape(-1)
            .float()
            .sum(0)
        )

        results.append(
            (correct_k / batch_size * 100.0).item()
        )

    return results


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0
    top1_total = 0.0
    top5_total = 0.0
    total = 0

    progress = tqdm(
        loader,
        desc="Train",
        leave=False
    )

    for images, labels in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        top1, top5 = accuracy_topk(
            outputs,
            labels,
            topk=(1, 5)
        )

        running_loss += (
            loss.item() * batch_size
        )

        top1_total += (
            top1 * batch_size
        )

        top5_total += (
            top5 * batch_size
        )

        total += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            top1=f"{top1:.2f}%"
        )

    return (
        running_loss / total,
        top1_total / total,
        top5_total / total
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    top1_total = 0.0
    top5_total = 0.0
    total = 0

    progress = tqdm(
        loader,
        desc="Val",
        leave=False
    )

    for images, labels in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        batch_size = labels.size(0)

        top1, top5 = accuracy_topk(
            outputs,
            labels,
            topk=(1, 5)
        )

        running_loss += (
            loss.item() * batch_size
        )

        top1_total += (
            top1 * batch_size
        )

        top5_total += (
            top5 * batch_size
        )

        total += batch_size

    return (
        running_loss / total,
        top1_total / total,
        top5_total / total
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=["plain", "resnet18"],
        required=True
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="../dataset/wafer"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results"
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    # Quiz 3:
    # Enable training data augmentation
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation for training"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            args.device
        )

    print("=" * 60)
    print("Week 3 Quiz 3 CNN Training")
    print("=" * 60)

    print(f"Model        : {args.model}")
    print(f"Device       : {device}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Augmentation : {args.augment}")

    # Different LR for the two models
    if args.lr is None:
        if args.model == "plain":
            lr = 1e-3
        else:
            lr = 1e-4
    else:
        lr = args.lr

    print(f"Learning rate: {lr}")
    print("=" * 60)

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Val samples  : {len(val_dataset)}"
    )

    print(
        f"Test samples : {len(test_dataset)}"
    )

    print(
        f"Classes      : {train_dataset.classes}"
    )

    # Model
    model = create_model(
        args.model,
        pretrained=not args.no_pretrained
    )

    model = model.to(device)

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)

    print(
        f"Total parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: {trainable_params:,}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_top1 = 0.0

    history = []

    checkpoint_path = os.path.join(
        args.output_dir,
        "best_model.pth"
    )

    for epoch in range(1, args.epochs + 1):

        print(
            f"\nEpoch [{epoch}/{args.epochs}]"
        )

        train_loss, train_top1, train_top5 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_top1, val_top5 = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Train Top-1: {train_top1:.2f}%"
        )

        print(
            f"Train Top-5: {train_top5:.2f}%"
        )

        print(
            f"Val Loss   : {val_loss:.4f}"
        )

        print(
            f"Val Top-1  : {val_top1:.2f}%"
        )

        print(
            f"Val Top-5  : {val_top5:.2f}%"
        )

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "train_top5": train_top5,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "lr": current_lr
        }

        history.append(record)

        # Save best model
        if val_top1 > best_top1:

            best_top1 = val_top1

            checkpoint = {
                "model": model.state_dict(),
                "model_name": args.model,
                "classes": train_dataset.classes,
                "epoch": epoch,
                "val_top1": val_top1,
                "val_top5": val_top5,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "augmentation": args.augment
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print(
                f"Saved best model -> {checkpoint_path}"
            )

    # Save history
    history_path = os.path.join(
        args.output_dir,
        "history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            history,
            f,
            indent=4
        )

    # Save config
    config = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "seed": args.seed,
        "augmentation": args.augment,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "best_val_top1": best_top1
    }

    config_path = os.path.join(
        args.output_dir,
        "config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=4
        )

    print("\nTraining finished.")

    print(
        f"Best validation Top-1: {best_top1:.2f}%"
    )

    print(
        f"Augmentation: {args.augment}"
    )


if __name__ == "__main__":
    main()
