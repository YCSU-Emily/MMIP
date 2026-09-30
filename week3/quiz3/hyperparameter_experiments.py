import os
import json
import csv
import subprocess
import sys


# ============================================================
# Quiz 2 Hyperparameter Experiment
# ============================================================

DATA_ROOT = "../dataset/wafer"
NUM_EPOCHS = 1
BATCH_SIZE = 32
NUM_WORKERS = 2
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"

# ------------------------------------------------------------
# 實驗設計
#
# Plain CNN:
#   LR = 1e-2, 1e-3, 1e-4
#
# ResNet18:
#   LR = 1e-3, 1e-4, 1e-5
#
# 其他參數全部固定
# ------------------------------------------------------------

EXPERIMENTS = [
    # Plain CNN
    {
        "model": "plain",
        "lr": 1e-2,
        "pretrained": False,
    },
    {
        "model": "plain",
        "lr": 1e-3,
        "pretrained": False,
    },
    {
        "model": "plain",
        "lr": 1e-4,
        "pretrained": False,
    },

    # ResNet18
    {
        "model": "resnet18",
        "lr": 1e-3,
        "pretrained": True,
    },
    {
        "model": "resnet18",
        "lr": 1e-4,
        "pretrained": True,
    },
    {
        "model": "resnet18",
        "lr": 1e-5,
        "pretrained": True,
    },
]


def run_command(cmd):
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nExperiment failed!")
        return False

    return True


def load_metrics(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    os.makedirs("results/hyperparameter", exist_ok=True)

    all_results = []

    for idx, exp in enumerate(EXPERIMENTS, start=1):

        model = exp["model"]
        lr = exp["lr"]
        pretrained = exp["pretrained"]

        exp_name = f"{model}_lr_{lr:.0e}"

        output_dir = os.path.join(
            "results",
            "hyperparameter",
            exp_name
        )

        evaluation_dir = os.path.join(
            output_dir,
            "evaluation"
        )

        print("\n")
        print("#" * 80)
        print(f"Experiment {idx}/{len(EXPERIMENTS)}")
        print(f"Model       : {model}")
        print(f"Learning rate: {lr}")
        print(f"Pretrained  : {pretrained}")
        print(f"Epochs      : {NUM_EPOCHS}")
        print(f"Batch size  : {BATCH_SIZE}")
        print("#" * 80)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_cmd = [
            sys.executable,
            "train.py",

            "--model",
            model,

            "--data-root",
            DATA_ROOT,

            "--epochs",
            str(NUM_EPOCHS),

            "--batch-size",
            str(BATCH_SIZE),

            "--num-workers",
            str(NUM_WORKERS),

            "--device",
            DEVICE,

            "--lr",
            str(lr),

            "--weight-decay",
            str(WEIGHT_DECAY),

            "--output-dir",
            output_dir,
        ]

        if not pretrained:
            train_cmd.append("--no-pretrained")

        success = run_command(train_cmd)

        if not success:
            continue

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        checkpoint = os.path.join(
            output_dir,
            "best_model.pth"
        )

        eval_cmd = [
            sys.executable,
            "evaluate.py",

            "--checkpoint",
            checkpoint,

            "--data-root",
            DATA_ROOT,

            "--batch-size",
            str(BATCH_SIZE),

            "--num-workers",
            str(NUM_WORKERS),

            "--device",
            DEVICE,

            "--output-dir",
            evaluation_dir,
        ]

        success = run_command(eval_cmd)

        if not success:
            continue

        # ----------------------------------------------------
        # Load evaluation result
        # ----------------------------------------------------

        metrics_path = os.path.join(
            evaluation_dir,
            "metrics.json"
        )

        if not os.path.exists(metrics_path):
            print(f"Cannot find metrics: {metrics_path}")
            continue

        metrics = load_metrics(metrics_path)

        result = {
            "experiment": exp_name,
            "model": model,
            "learning_rate": lr,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": NUM_EPOCHS,
            "pretrained": pretrained,

            "parameter_count": metrics["parameter_count"],
            "test_top1": metrics["test_top1"],
            "test_top5": metrics["test_top5"],
            "macro_auc": metrics["macro_auc"],
        }

        all_results.append(result)

        print("\nExperiment result:")
        print(f"Top-1   : {result['test_top1']:.2f}%")
        print(f"Top-5   : {result['test_top5']:.2f}%")
        print(f"Macro-AUC: {result['macro_auc']:.4f}")
        print(f"Params  : {result['parameter_count']:,}")

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = "results/hyperparameter/hyperparameter_results.csv"

    if all_results:

        fieldnames = [
            "experiment",
            "model",
            "learning_rate",
            "batch_size",
            "weight_decay",
            "epochs",
            "pretrained",
            "parameter_count",
            "test_top1",
            "test_top5",
            "macro_auc",
        ]

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(all_results)

        print("\n")
        print("=" * 80)
        print("All experiments completed.")
        print(f"Results saved to:")
        print(csv_path)
        print("=" * 80)

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print("\nSummary")
        print("-" * 80)

        for r in all_results:
            print(
                f"{r['model']:8s} | "
                f"LR={r['learning_rate']:.0e} | "
                f"Top-1={r['test_top1']:.2f}% | "
                f"Top-5={r['test_top5']:.2f}% | "
                f"AUC={r['macro_auc']:.4f}"
            )

    else:
        print("\nNo successful experiments.")


if __name__ == "__main__":
    main()
