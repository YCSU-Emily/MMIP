import csv
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
OUTPUT_PATH = ROOT / "results" / "baseline" / "threshold_analysis.csv"

THRESHOLDS = [0.5, 0.3, 0.2, 0.1]


def main():
    for path in (MODEL_PATH, LABEL_MAP_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"找不到必要檔案：{path}")

    if OUTPUT_PATH.exists():
        raise FileExistsError(f"輸出檔已存在，拒絕覆蓋：{OUTPUT_PATH}")

    import json
    with LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    names = [label_map[str(i)] for i in range(NUM_CLASSES)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = COCOMultimodalDataset("test", tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = MultimodalClassifier()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    targets = []
    probabilities = []

    print("推論 test set...")
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=True
            )

            logits = model(images, input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            targets.append(batch["labels"].numpy().astype(np.int32))
            probabilities.append(probs.cpu().numpy())

            if step % 10 == 0 or step == len(loader):
                print(f"Batch {step}/{len(loader)}")

    y_true = np.concatenate(targets, axis=0)
    y_prob = np.concatenate(probabilities, axis=0)

    fields = [
        "threshold", "class_id", "class_name", "support",
        "predicted_positives", "precision", "recall", "f1"
    ]

    with OUTPUT_PATH.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        summary = []
        for threshold in THRESHOLDS:
            y_pred = (y_prob >= threshold).astype(np.int32)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            macro_f1 = float(np.mean(f1))
            micro_p, micro_r, micro_f1, _ = (
                precision_recall_fscore_support(
                    y_true, y_pred, average="micro", zero_division=0
                )
            )
            summary.append({
                "threshold": threshold,
                "macro_f1": macro_f1,
                "micro_f1": float(micro_f1),
                "micro_precision": float(micro_p),
                "micro_recall": float(micro_r),
                "predicted_labels": int(y_pred.sum()),
                "zero_f1_classes": int(np.sum(f1 == 0)),
            })

            for i, name in enumerate(names):
                writer.writerow({
                    "threshold": threshold,
                    "class_id": i,
                    "class_name": name,
                    "support": int(support[i]),
                    "predicted_positives": int(y_pred[:, i].sum()),
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                })

    print("\n===== 各閾值整體比較 =====")
    print(
        f'{"閾值":>6} {"Macro-F1":>10} {"Micro-F1":>10} '
        f'{"Micro-P":>10} {"Micro-R":>10} {"預測正類數":>12} {"零分類別":>10}'
    )
    for s in summary:
        print(
            f'{s["threshold"]:6.2f} {s["macro_f1"]:10.4f} '
            f'{s["micro_f1"]:10.4f} {s["micro_precision"]:10.4f} '
            f'{s["micro_recall"]:10.4f} {s["predicted_labels"]:12d} '
            f'{s["zero_f1_classes"]:10d}'
        )

    print("\n===== 低分類別在不同閾值的結果 =====")
    focus = {
        "bicycle", "donut", "potted plant", "hot dog",
        "parking meter", "snowboard", "toothbrush"
    }

    with OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for name in sorted(focus):
        print(f"\n{name}")
        for row in rows:
            if row["class_name"] == name:
                print(
                    f'  threshold={float(row["threshold"]):.1f} '
                    f'pred={row["predicted_positives"]:>3} '
                    f'P={float(row["precision"]):.3f} '
                    f'R={float(row["recall"]):.3f} '
                    f'F1={float(row["f1"]):.3f}'
                )

    print("\n完整報表：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
