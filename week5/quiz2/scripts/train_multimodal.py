import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel, AutoTokenizer

from test_multimodal_loader import COCOMultimodalDataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "distilbert/distilbert-base-uncased"
NUM_CLASSES = 80
SEED = 42


class MultimodalClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # 預訓練圖片骨幹
        self.image_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_encoder.fc = nn.Identity()

        # 預訓練文字骨幹
        self.text_encoder = AutoModel.from_pretrained(MODEL_NAME)

        # 第一版先凍結兩個骨幹，只訓練融合分類器
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.image_encoder.eval()
        self.text_encoder.eval()

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def train(self, mode=True):
        # 即使分類器進入 train mode，凍結的骨幹仍維持 eval mode
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self

    def forward(self, image, input_ids, attention_mask):
        with torch.no_grad():
            image_features = self.image_encoder(image)
            text_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # DistilBERT 第一個 token 的隱藏狀態作為文字特徵
            text_features = text_output.last_hidden_state[:, 0, :]

        fused = torch.cat([image_features, text_features], dim=1)
        logits = self.classifier(fused)
        return logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(batch, device):
    return (
        batch["image"].to(device),
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_set = COCOMultimodalDataset("train", tokenizer)
    val_set = COCOMultimodalDataset("val", tokenizer)

    batch_size = 4 if args.smoke_test else args.batch_size
    workers = 0 if args.smoke_test else args.num_workers

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=not args.smoke_test,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MultimodalClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # Smoke test：只測一個 batch，不儲存訓練結果
    if args.smoke_test:
        model.train()
        batch = next(iter(train_loader))
        image, input_ids, attention_mask, labels = make_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in model.parameters())

        print("Image batch:", tuple(image.shape))
        print("Token batch:", tuple(input_ids.shape))
        print("Label batch:", tuple(labels.shape))
        print("Logits:", tuple(logits.shape))
        print("Loss:", round(loss.item(), 6))
        print("Trainable parameters:", trainable)
        print("Total parameters:", total)
        assert logits.shape == (image.shape[0], NUM_CLASSES)
        assert torch.isfinite(loss)
        assert any(
            p.grad is not None
            for p in model.classifier.parameters()
        )
        print("PASS：前向傳播、損失計算及分類器反向傳播正常。")
        return

    if args.epochs < 1:
        raise SystemExit("--epochs 必須大於 0")

    results_dir = ROOT / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        results_dir / "best_model.pth",
        results_dir / "history.json",
        results_dir / "test_metrics.json",
    ]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise SystemExit(
            "為避免覆蓋既有結果，停止訓練：\n" + "\n".join(existing)
        )

    history = []
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            image, input_ids, attention_mask, labels = make_batch(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            count = image.size(0)
            running_loss += loss.item() * count
            seen += count

            if step % 50 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        train_loss = running_loss / max(seen, 1)

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch in val_loader:
                image, input_ids, attention_mask, labels = make_batch(
                    batch, device
                )
                logits = model(image, input_ids, attention_mask)
                loss = criterion(logits, labels)

                count = image.size(0)
                val_loss_sum += loss.item() * count
                val_seen += count

                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).to(torch.int32)
                all_targets.append(labels.cpu().numpy().astype(np.int32))
                all_predictions.append(predictions.cpu().numpy())

        y_true = np.concatenate(all_targets, axis=0)
        y_pred = np.concatenate(all_predictions, axis=0)

        from sklearn.metrics import f1_score
        macro_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        micro_f1 = float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        )
        val_loss = val_loss_sum / max(val_seen, 1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": macro_f1,
            "val_micro_f1": micro_f1,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("沒有取得有效的最佳模型權重。")

    # 只在訓練完成後寫出檔案，並使用 exclusive mode 防止覆蓋
    with outputs[0].open("xb") as f:
        torch.save(best_state, f)

    with outputs[1].open("x", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("訓練完成。最佳驗證 Macro-F1:", best_f1)
    print("模型與訓練紀錄位置:", results_dir)


if __name__ == "__main__":
    main()
