import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from test_multimodal_loader import COCOMultimodalDataset
from train_multimodal import MultimodalClassifier, MODEL_NAME, NUM_CLASSES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "baseline" / "best_model.pth"
OUTPUT_PATH = ROOT / "results" / "baseline" / "test_metrics.json"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到模型：{MODEL_PATH}")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"輸出檔已存在，為避免覆蓋而停止：{OUTPUT_PATH}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("載入 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("載入 test dataset...")
    dataset = COCOMultimodalDataset("test", tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    print("Test 筆數:", len(dataset))

    print("建立模型並載入最佳權重...")
    model = MultimodalClassifier()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_seen = 0
    all_targets = []
    all_predictions = []

    print("開始測試集評估...")
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=True
            )
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, labels)

            count = image.size(0)
            total_loss += loss.item() * count
            total_seen += count

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).to(torch.int32)

            all_targets.append(labels.cpu().numpy().astype(np.int32))
            all_predictions.append(predictions.cpu().numpy())

            if step % 10 == 0 or step == len(loader):
                print(f"Batch {step}/{len(loader)}")

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)

    metrics = {
        "split": "test",
        "num_samples": int(total_seen),
        "num_classes": NUM_CLASSES,
        "prediction_threshold": 0.5,
        "test_loss": float(total_loss / total_seen),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "micro_precision": float(
            precision_score(
                y_true, y_pred, average="micro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
    }

    # 使用 exclusive mode，若檔案已存在便拒絕寫入。
    with OUTPUT_PATH.open("x", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("Test Set 評估完成")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("\n結果已保存至:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
