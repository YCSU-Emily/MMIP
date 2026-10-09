import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "week4/quiz4/image_captioning/data"
OUT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "openai/clip-vit-base-patch32"

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def tokenize(text):
    return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())


def build_vocab(json_path, min_freq=2):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = Counter()
    for ann in data["annotations"]:
        counts.update(tokenize(ann["caption"]))

    vocab = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
    for token, count in counts.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def load_coco(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    captions_by_id = {}
    for ann in annotations:
        captions_by_id.setdefault(ann["image_id"], []).append(ann["caption"])

    records = []
    for item in images:
        captions = captions_by_id.get(item["id"], [])
        if captions:
            records.append({
                "id": item["id"],
                "file_name": item["file_name"],
                "captions": captions,
            })
    return records


class CocoCaptionDataset(Dataset):
    def __init__(self, split, vocab, max_length=30):
        self.split = split
        self.vocab = vocab
        self.max_length = max_length
        self.records = load_coco(DATA_ROOT / "captions" / f"{split}.json")
        self.image_dir = DATA_ROOT / "images" / split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.image_dir / record["file_name"]

        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.split == "train":
            caption = random.choice(record["captions"])
        else:
            caption = record["captions"][0]

        tokens = tokenize(caption)[: self.max_length - 2]
        ids = [self.vocab[BOS]]
        ids.extend(self.vocab.get(t, self.vocab[UNK]) for t in tokens)
        ids.append(self.vocab[EOS])

        return image, torch.tensor(ids, dtype=torch.long), record["id"]


class CaptionCollator:
    def __init__(self, processor, pad_id):
        self.processor = processor
        self.pad_id = pad_id

    def __call__(self, batch):
        images, sequences, image_ids = zip(*batch)
        pixels = self.processor(
            images=list(images),
            return_tensors="pt"
        )["pixel_values"]

        max_len = max(len(s) for s in sequences)
        captions = torch.full(
            (len(sequences), max_len),
            self.pad_id,
            dtype=torch.long
        )
        for i, seq in enumerate(sequences):
            captions[i, :len(seq)] = seq

        return pixels, captions, list(image_ids)


class ClipLSTMCaptioner(nn.Module):
    def __init__(self, clip, vocab_size, pad_id, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.clip = clip
        for p in self.clip.parameters():
            p.requires_grad = False

        vision_dim = clip.config.projection_dim
        self.image_projection = nn.Linear(vision_dim, hidden_dim)
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_id
        )
        self.decoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode_image(self, pixel_values):
        with torch.no_grad():
            vision_output = self.clip.vision_model(
                pixel_values=pixel_values
            )
            features = self.clip.visual_projection(
                vision_output.pooler_output
            )
        return features

    def forward(self, pixel_values, captions):
        features = self.encode_image(pixel_values)
        h0 = torch.tanh(self.image_projection(features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        # Decoder input: BOS through the token before EOS.
        decoder_inputs = captions[:, :-1]
        embedded = self.embedding(decoder_inputs)
        outputs, _ = self.decoder(embedded, (h0, c0))
        return self.output(outputs)

    @torch.no_grad()
    def generate(self, pixel_values, bos_id, eos_id, max_length=30):
        features = self.encode_image(pixel_values)
        h = torch.tanh(self.image_projection(features)).unsqueeze(0)
        c = torch.zeros_like(h)

        batch_size = pixel_values.size(0)
        current = torch.full(
            (batch_size, 1), bos_id,
            dtype=torch.long, device=pixel_values.device
        )
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=pixel_values.device
        )
        generated = []

        for _ in range(max_length):
            embedded = self.embedding(current[:, -1:])
            out, (h, c) = self.decoder(embedded, (h, c))
            logits = self.output(out[:, -1])
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_id),
                next_token
            )
            generated.append(next_token)
            finished |= next_token.eq(eos_id)
            current = torch.cat([current, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return torch.stack(generated, dim=1)


def make_loader(split, vocab, processor, batch_size, workers, max_length):
    dataset = CocoCaptionDataset(split, vocab, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=CaptionCollator(processor, vocab[PAD])
    )
    return dataset, loader


def run_epoch(model, loader, optimizer, device, pad_id, train):
    # Switch the decoder to train/eval mode correctly.
    model.train(train)
    # The CLIP encoder remains frozen and always in eval mode.
    model.clip.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss = 0.0
    total_tokens = 0

    for step, (pixels, captions, _) in enumerate(loader, start=1):
        pixels = pixels.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(pixels, captions)
            targets = captions[:, 1:]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                optimizer.step()

        valid_tokens = targets.ne(pad_id).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

        if step % 50 == 0:
            print(
                f"{'Train' if train else 'Val'} "
                f"[{step}/{len(loader)}] loss={loss.item():.4f}",
                flush=True
            )

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--limit-train", type=int, default=0,
                        help="Debug only: use first N training images; 0 means all.")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_json = DATA_ROOT / "captions/train.json"
    vocab = build_vocab(train_json, args.min_freq)
    vocab_path = OUT_DIR / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("Vocabulary size:", len(vocab))

    print("Loading CLIP model:", MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip = CLIPModel.from_pretrained(MODEL_NAME)

    model = ClipLSTMCaptioner(
        clip=clip,
        vocab_size=len(vocab),
        pad_id=vocab[PAD]
    ).to(device)

    train_ds, train_loader = make_loader(
        "train", vocab, processor, args.batch_size,
        args.workers, args.max_length
    )
    val_ds, val_loader = make_loader(
        "val", vocab, processor, args.batch_size,
        args.workers, args.max_length
    )

    if args.limit_train > 0:
        from torch.utils.data import Subset
        indices = list(range(min(args.limit_train, len(train_ds))))
        train_loader = DataLoader(
            Subset(train_ds, indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=CaptionCollator(processor, vocab[PAD])
        )
        print("DEBUG train subset:", len(indices))

    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in trainable))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    model_dir = OUT_DIR / "models"
    result_dir = OUT_DIR / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss = run_epoch(
            model, train_loader, optimizer, device, vocab[PAD], train=True
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, device, vocab[PAD], train=False
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": args.lr
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        # CLIP is frozen, so save only trainable captioning parameters.
        trainable_state = {
            name: parameter.detach().cpu()
            for name, parameter in model.state_dict().items()
            if not name.startswith("clip.")
        }

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainable_state,
            "vocab": vocab,
            "model_name": MODEL_NAME,
            "embed_dim": 256,
            "hidden_dim": 512,
            "max_length": args.max_length,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "checkpoint_format": "trainable_parameters_only"
        }

        torch.save(checkpoint, model_dir / "last.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, model_dir / "best.pth")
            print("Saved new best checkpoint.")

        with (result_dir / "training_history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print("\nTraining complete.")
    print("Best validation loss:", best_val)
    print("Best checkpoint:", model_dir / "best.pth")


if __name__ == "__main__":
    main()
