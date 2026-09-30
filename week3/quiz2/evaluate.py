import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_curve,
)
from torch.utils.data import DataLoader

from dataset import get_datasets
from models import (
    create_model,
    count_parameters,
    count_trainable_parameters,
)


def accuracy_topk(output, target, topk=(1, 5)):
    maxk = max(topk)

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

    result = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()

        result.append(
            correct_k.item() / target.size(0) * 100.0
        )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="../../dataset/wafer"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
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
        default="evaluation"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

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

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model_name = checkpoint["model_name"]

    classes = checkpoint["classes"]

    model = create_model(
        model_name,
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model = model.to(device)
    model.eval()

    print("=" * 60)
    print("Quiz 2 Evaluation")
    print("=" * 60)

    print(f"Model : {model_name}")
    print(f"Device: {device}")

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)

    print(
        f"Parameters: {total_params:,}"
    )

    print(
        f"Trainable: {trainable_params:,}"
    )

    _, _, test_dataset = get_datasets(
        args.data_root
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []

    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            top1, top5 = accuracy_topk(
                outputs,
                labels,
                topk=(1, 5)
            )

            batch_size = labels.size(0)

            total_top1 += (
                top1 * batch_size
            )

            total_top5 += (
                top5 * batch_size
            )

            total_samples += batch_size

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(
        all_probabilities
    )

    test_top1 = total_top1 / total_samples
    test_top5 = total_top5 / total_samples

    print(
        f"\nTest Top-1 Accuracy: {test_top1:.2f}%"
    )

    print(
        f"Test Top-5 Accuracy: {test_top5:.2f}%"
    )

    # -------------------------------------------------
    # Predictions CSV
    # -------------------------------------------------

    prediction_path = os.path.join(
        args.output_dir,
        "test_predictions.csv"
    )

    test_samples = test_dataset.samples

    with open(
        prediction_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        header = [
            "image",
            "true_label",
            "predicted_label",
            "correct"
        ]

        for class_name in classes:
            header.append(
                f"prob_{class_name}"
            )

        writer.writerow(header)

        for i, (image_path, true_index) in enumerate(
            test_samples
        ):

            pred_index = all_predictions[i]

            row = [
                image_path,
                classes[true_index],
                classes[pred_index],
                bool(true_index == pred_index)
            ]

            row.extend(
                all_probabilities[i].tolist()
            )

            writer.writerow(row)

    print(
        f"Saved predictions -> {prediction_path}"
    )

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    im = ax.imshow(cm)

    ax.set_xticks(
        np.arange(len(classes))
    )

    ax.set_yticks(
        np.arange(len(classes))
    )

    ax.set_xticklabels(
        classes,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    fig.colorbar(im)

    fig.tight_layout()

    cm_path = os.path.join(
        args.output_dir,
        "confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved confusion matrix -> {cm_path}"
    )

    # -------------------------------------------------
    # ROC / AUC
    # -------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    auc_values = []

    for class_index, class_name in enumerate(classes):

        binary_labels = (
            all_labels == class_index
        ).astype(int)

        fpr, tpr, _ = roc_curve(
            binary_labels,
            all_probabilities[:, class_index]
        )

        class_auc = auc(
            fpr,
            tpr
        )

        auc_values.append(
            class_auc
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC={class_auc:.4f})"
        )

    macro_auc = float(
        np.mean(auc_values)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"{model_name} ROC Curves\n"
        f"Macro-AUC = {macro_auc:.4f}"
    )

    plt.legend(
        loc="lower right",
        fontsize=8
    )

    plt.grid(True)

    roc_path = os.path.join(
        args.output_dir,
        "roc_curves.png"
    )

    plt.savefig(
        roc_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved ROC curves -> {roc_path}"
    )

    # -------------------------------------------------
    # Metrics JSON
    # -------------------------------------------------

    metrics = {
        "model": model_name,
        "test_top1": test_top1,
        "test_top5": test_top5,
        "macro_auc": macro_auc,
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "per_class_auc": {
            classes[i]: float(auc_values[i])
            for i in range(len(classes))
        }
    }

    metrics_path = os.path.join(
        args.output_dir,
        "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"Saved metrics -> {metrics_path}"
    )

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()
