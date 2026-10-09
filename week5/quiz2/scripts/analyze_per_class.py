import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from test_multimodal_loader import COCOMultimodalDataset
from train_multimodal import MultimodalClassifier, MODEL_NAME, NUM_CLASSES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "baseline" / "best_model.pth"
LABEL_MAP_PATH = ROOT / "data" / "processed" / "label_map.json"
OUTPUT_PATH = ROOT / "results" / "baseline" / "per_class_metrics.csv"

THRESHOLD = 0.5
BATCH_SIZE = 16


def main():
    for path in (MODEL_PATH, LABEL_MAP_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"找不到必要檔案：{path}")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"輸出檔已存在，為避免覆蓋而停止：{OUTPUT_PATH}"
        )

    with LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    label_names = [label_map[str(i)] for i in range(NUM_CLASSES)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = COCOMultimodalDataset("test", tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = MultimodalClassifier()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    all_targets = []
    all_predictions = []

    print("開始對 test set 進行推論...")
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=True
            )

            logits = model(images, input_ids, attention_mask)
            predictions = (torch.sigmoid(logits) >= THRESHOLD).to(torch.int32)

            all_targets.append(batch["labels"].numpy().astype(np.int32))
            all_predictions.append(predictions.cpu().numpy())

            if step % 10 == 0 or step == len(loader):
                print(f"Batch {step}/{len(loader)}")

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    rows = []
    for i, name in enumerate(label_names):
        rows.append({
            "class_id": i,
            "class_name": name,
            "support": int(support[i]),
            "predicted_positives": int(y_pred[:, i].sum()),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        })

    # exclusive mode：不覆蓋既有報表
    with OUTPUT_PATH.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n完成！共輸出", len(rows), "個類別")
    print("報表位置：", OUTPUT_PATH)

    print("\nF1 最低的 10 個類別：")
    for row in sorted(rows, key=lambda r: r["f1"])[:10]:
        print(
            f'{row["class_name"]:20s} '
            f'F1={row["f1"]:.4f} '
            f'P={row["precision"]:.4f} '
            f'R={row["recall"]:.4f} '
            f'support={row["support"]}'
        )

    print("\nF1 最高的 10 個類別：")
    for row in sorted(rows, key=lambda r: r["f1"], reverse=True)[:10]:
        print(
            f'{row["class_name"]:20s} '
            f'F1={row["f1"]:.4f} '
            f'P={row["precision"]:.4f} '
            f'R={row["recall"]:.4f} '
            f'support={row["support"]}'
        )


if __name__ == "__main__":
    main()
