import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

from train_clip_caption import (
    ClipLSTMCaptioner,
    DATA_ROOT,
    OUT_DIR,
    MODEL_NAME,
)


class TestImageDataset(Dataset):
    def __init__(self, records, image_dir, processor):
        self.records = records
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.image_dir / record["file_name"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixels = self.processor(
                images=image,
                return_tensors="pt"
            )["pixel_values"].squeeze(0)

        return pixels, record["id"], record["file_name"]


def collate_fn(batch):
    pixels, image_ids, filenames = zip(*batch)
    return torch.stack(pixels), list(image_ids), list(filenames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(OUT_DIR / "models/best.pth"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    vocab = checkpoint["vocab"]
    id_to_token = {index: token for token, index in vocab.items()}

    print("Loading CLIP...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip = CLIPModel.from_pretrained(MODEL_NAME)

    model = ClipLSTMCaptioner(
        clip=clip,
        vocab_size=len(vocab),
        pad_id=vocab["<pad>"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    # A compact checkpoint intentionally omits the frozen CLIP weights.
    unexpected_missing = [
        name for name in incompatible.missing_keys
        if not name.startswith("clip.")
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    model.eval()

    captions_path = DATA_ROOT / "captions" / f"{args.split}.json"
    with captions_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    image_dir = DATA_ROOT / "images" / args.split
    records = coco["images"][:args.limit]
    dataset = TestImageDataset(records, image_dir, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    results = []
    for batch_index, (pixels, image_ids, filenames) in enumerate(loader, 1):
        pixels = pixels.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            generated = model.generate(
                pixels,
                bos_id=vocab["<bos>"],
                eos_id=vocab["<eos>"],
                max_length=checkpoint["max_length"],
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        for i, token_ids in enumerate(generated.tolist()):
            tokens = []
            for token_id in token_ids:
                token = id_to_token.get(token_id, "<unk>")
                if token in ("<eos>", "<pad>"):
                    break
                if token not in ("<bos>", "<unk>"):
                    tokens.append(token)

            caption = " ".join(tokens)
            results.append({
                "image_id": image_ids[i],
                "file_name": filenames[i],
                "generated_caption": caption,
                "batch_elapsed_sec": elapsed,
                "approx_image_inference_sec": elapsed / len(image_ids),
            })

        print(
            f"Batch {batch_index}/{len(loader)} complete; "
            f"processed {len(results)}/{len(dataset)} images",
            flush=True,
        )

    output_path = OUT_DIR / "results" / f"clip_lstm_{args.split}_captions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nSaved:", output_path)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Number of captions:", len(results))
    print("\nSample captions:")
    for item in results[:10]:
        print(f'{item["file_name"]}: {item["generated_caption"]}')


if __name__ == "__main__":
    main()
