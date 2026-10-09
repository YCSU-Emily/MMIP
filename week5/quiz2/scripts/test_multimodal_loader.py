import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
IMAGE_DIR = ROOT / "data" / "raw" / "val2017" / "val2017"
MODEL_NAME = "distilbert/distilbert-base-uncased"
NUM_CLASSES = 80


class COCOMultimodalDataset(Dataset):
    def __init__(self, split, tokenizer):
        jsonl_path = DATA_DIR / f"{split}.jsonl"
        with jsonl_path.open(encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = ROOT / record["image"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        tokens = self.tokenizer(
            record["caption"],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        labels[record["labels"]] = 1.0

        return {
            "image": image_tensor,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
            "image_id": record["image_id"],
            "caption": record["caption"],
        }


def main():
    print("載入 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = COCOMultimodalDataset("train", tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("=" * 60)
    print("多模態 DataLoader 測試")
    print("=" * 60)
    print("Dataset 筆數:", len(dataset))
    print("Image shape:", tuple(batch["image"].shape))
    print("Input IDs shape:", tuple(batch["input_ids"].shape))
    print("Attention mask shape:", tuple(batch["attention_mask"].shape))
    print("Labels shape:", tuple(batch["labels"].shape))
    print("Label dtype:", batch["labels"].dtype)
    print("每張圖片的正類數:", batch["labels"].sum(dim=1).tolist())
    print("Image IDs:", batch["image_id"])
    print("第一筆 caption:", batch["caption"][0])

    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["input_ids"].shape == (4, 64)
    assert batch["attention_mask"].shape == (4, 64)
    assert batch["labels"].shape == (4, NUM_CLASSES)
    assert batch["labels"].dtype == torch.float32
    assert torch.all((batch["labels"] == 0) | (batch["labels"] == 1))

    print("\nPASS：圖片、文字 token 與多標籤已成功組成 batch。")


if __name__ == "__main__":
    main()
