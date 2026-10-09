import os
import json
import re
import time
import random
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# Configuration
# ============================================================

SEED = 3407

DATA_DIR = "./data"
MODEL_DIR = "./models"
RESULT_DIR = "./results"

TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "images/train")
VAL_IMAGE_DIR = os.path.join(DATA_DIR, "images/val")

TRAIN_CAPTION_FILE = os.path.join(
    DATA_DIR, "captions/train.json"
)

VAL_CAPTION_FILE = os.path.join(
    DATA_DIR, "captions/val.json"
)

IMAGE_SIZE = 224

BATCH_SIZE = 64

EMBED_DIM = 256
HIDDEN_DIM = 512

NUM_EPOCHS = 5

LEARNING_RATE = 1e-3

MIN_WORD_FREQ = 5

MAX_LENGTH = 30

NUM_WORKERS = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# Vocabulary
# ============================================================

class Vocabulary:

    def __init__(self, min_freq=5):

        self.min_freq = min_freq

        self.word2idx = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3,
        }

        self.idx2word = {
            0: "<PAD>",
            1: "<START>",
            2: "<END>",
            3: "<UNK>",
        }

    def tokenize(self, text):

        text = text.lower()

        tokens = re.findall(
            r"[a-z]+(?:'[a-z]+)?",
            text
        )

        return tokens

    def build(self, captions):

        counter = Counter()

        for caption in captions:

            tokens = self.tokenize(caption)

            counter.update(tokens)

        for word, freq in counter.items():

            if freq >= self.min_freq:

                idx = len(self.word2idx)

                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, caption):

        tokens = self.tokenize(caption)

        tokens = (
            ["<START>"]
            + tokens
            + ["<END>"]
        )

        ids = []

        for token in tokens:

            ids.append(
                self.word2idx.get(
                    token,
                    self.word2idx["<UNK>"]
                )
            )

        return ids

    def decode(self, ids):

        words = []

        for idx in ids:

            word = self.idx2word.get(
                int(idx),
                "<UNK>"
            )

            if word == "<START>":
                continue

            if word == "<END>":
                break

            if word == "<PAD>":
                continue

            words.append(word)

        return " ".join(words)

    def __len__(self):

        return len(self.word2idx)


# ============================================================
# COCO Caption Loader
# ============================================================

def load_coco_annotations(json_file):

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    image_id_to_filename = {}

    for image in data["images"]:

        image_id_to_filename[
            image["id"]
        ] = image["file_name"]

    image_to_captions = {}

    for annotation in data["annotations"]:

        image_id = annotation["image_id"]

        caption = annotation["caption"]

        if image_id not in image_to_captions:

            image_to_captions[image_id] = []

        image_to_captions[
            image_id
        ].append(caption)

    samples = []

    for image_id, captions in image_to_captions.items():

        filename = image_id_to_filename[image_id]

        for caption in captions:

            samples.append(
                (
                    filename,
                    caption
                )
            )

    return samples


# ============================================================
# Dataset
# ============================================================

class CaptionDataset(Dataset):

    def __init__(
        self,
        samples,
        image_dir,
        vocabulary,
        transform=None,
        max_length=30
    ):

        self.samples = samples

        self.image_dir = image_dir

        self.vocabulary = vocabulary

        self.transform = transform

        self.max_length = max_length

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, caption = self.samples[index]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(image)

        caption_ids = self.vocabulary.encode(
            caption
        )

        # Truncate
        caption_ids = caption_ids[
            :self.max_length
        ]

        # Pad
        if len(caption_ids) < self.max_length:

            caption_ids += [
                self.vocabulary.word2idx["<PAD>"]
            ] * (
                self.max_length
                - len(caption_ids)
            )

        return (
            image,
            torch.tensor(
                caption_ids,
                dtype=torch.long
            )
        )


# ============================================================
# Encoder
# ============================================================

class EncoderCNN(nn.Module):

    def __init__(self, embed_dim):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        resnet = resnet18(
            weights=weights
        )

        # Remove final classification layer
        modules = list(
            resnet.children()
        )[:-1]

        self.resnet = nn.Sequential(
            *modules
        )

        # Freeze BatchNorm statistics but keep
        # convolutional features frozen.
        for parameter in self.resnet.parameters():

            parameter.requires_grad = False

        self.fc = nn.Linear(
            512,
            embed_dim
        )

    def forward(self, images):

        with torch.no_grad():

            features = self.resnet(images)

        features = features.flatten(1)

        features = self.fc(features)

        return features


# ============================================================
# Decoder
# ============================================================

class DecoderLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.init_h = nn.Linear(
            embed_dim,
            hidden_dim
        )

        self.init_c = nn.Linear(
            embed_dim,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(
        self,
        image_features,
        captions
    ):

        # Remove final <END> target
        inputs = captions[:, :-1]

        embeddings = self.embedding(
            inputs
        )

        h0 = self.init_h(
            image_features
        ).unsqueeze(0)

        c0 = self.init_c(
            image_features
        ).unsqueeze(0)

        outputs, _ = self.lstm(
            embeddings,
            (h0, c0)
        )

        outputs = self.fc(
            outputs
        )

        return outputs


# ============================================================
# Collate
# ============================================================

def collate_fn(batch):

    images = []

    captions = []

    for image, caption in batch:

        images.append(image)

        captions.append(caption)

    images = torch.stack(images)

    captions = torch.stack(captions)

    return images, captions


# ============================================================
# Training
# ============================================================

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("Week 4 Quiz 4 - Image Captioning")
    print("=" * 70)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    print()
    print("Loading COCO annotations...")

    train_samples = load_coco_annotations(
        TRAIN_CAPTION_FILE
    )

    val_samples = load_coco_annotations(
        VAL_CAPTION_FILE
    )

    print(
        f"Train caption samples: "
        f"{len(train_samples)}"
    )

    print(
        f"Validation caption samples: "
        f"{len(val_samples)}"
    )

    # --------------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------------

    print()
    print("Building vocabulary...")

    train_captions = [
        caption
        for _, caption
        in train_samples
    ]

    vocabulary = Vocabulary(
        min_freq=MIN_WORD_FREQ
    )

    vocabulary.build(
        train_captions
    )

    print(
        f"Vocabulary size: "
        f"{len(vocabulary):,}"
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    weights = ResNet18_Weights.DEFAULT

    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = CaptionDataset(
        train_samples,
        TRAIN_IMAGE_DIR,
        vocabulary,
        train_transform,
        MAX_LENGTH
    )

    val_dataset = CaptionDataset(
        val_samples,
        VAL_IMAGE_DIR,
        vocabulary,
        val_transform,
        MAX_LENGTH
    )

    print()
    print(
        f"Train dataset: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation dataset: "
        f"{len(val_dataset):,}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    encoder = EncoderCNN(
        EMBED_DIM
    ).to(DEVICE)

    decoder = DecoderLSTM(
        len(vocabulary),
        EMBED_DIM,
        HIDDEN_DIM
    ).to(DEVICE)

    encoder_params = sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )

    decoder_params = sum(
        p.numel()
        for p in decoder.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable encoder parameters: "
        f"{encoder_params:,}"
    )

    print(
        f"Trainable decoder parameters: "
        f"{decoder_params:,}"
    )

    print(
        f"Total trainable parameters: "
        f"{encoder_params + decoder_params:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    # Only train decoder + projection
    trainable_parameters = list(
        encoder.fc.parameters()
    ) + list(
        decoder.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    model_path = os.path.join(
        MODEL_DIR,
        "best_caption_model.pth"
    )

    history = []

    total_start = time.time()

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.time()

        encoder.train()
        decoder.train()

        train_loss_sum = 0.0

        for images, captions in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            captions = captions.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            image_features = encoder(
                images
            )

            outputs = decoder(
                image_features,
                captions
            )

            targets = captions[:, 1:]

            loss = criterion(
                outputs.reshape(
                    -1,
                    outputs.size(-1)
                ),
                targets.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                5.0
            )

            optimizer.step()

            train_loss_sum += (
                loss.item()
                * images.size(0)
            )

        train_loss = (
            train_loss_sum
            / len(train_dataset)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        encoder.eval()
        decoder.eval()

        val_loss_sum = 0.0

        with torch.no_grad():

            for images, captions in val_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                captions = captions.to(
                    DEVICE,
                    non_blocking=True
                )

                image_features = encoder(
                    images
                )

                outputs = decoder(
                    image_features,
                    captions
                )

                targets = captions[:, 1:]

                loss = criterion(
                    outputs.reshape(
                        -1,
                        outputs.size(-1)
                    ),
                    targets.reshape(-1)
                )

                val_loss_sum += (
                    loss.item()
                    * images.size(0)
                )

        val_loss = (
            val_loss_sum
            / len(val_dataset)
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": epoch_time
        })

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Time: {epoch_time:.2f}s"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "encoder_state_dict":
                        encoder.state_dict(),

                    "decoder_state_dict":
                        decoder.state_dict(),

                    "vocab_word2idx":
                        vocabulary.word2idx,

                    "vocab_idx2word":
                        vocabulary.idx2word,

                    "embed_dim":
                        EMBED_DIM,

                    "hidden_dim":
                        HIDDEN_DIM,

                    "max_length":
                        MAX_LENGTH,

                    "epoch":
                        epoch,

                    "val_loss":
                        val_loss
                },
                model_path
            )

    total_time = (
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = os.path.join(
        RESULT_DIR,
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
            indent=2
        )

    # --------------------------------------------------------
    # Save config
    # --------------------------------------------------------

    config = {
        "model": "ResNet18 Encoder + LSTM Decoder",
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "min_word_freq": MIN_WORD_FREQ,
        "max_length": MAX_LENGTH,
        "vocabulary_size": len(vocabulary),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_loss": best_val_loss,
        "training_time_sec": total_time
    }

    config_path = os.path.join(
        RESULT_DIR,
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
            indent=2
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Training Complete")
    print("=" * 70)

    print(
        f"Best Validation Loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Training Time: "
        f"{total_time:.2f} sec"
    )

    print(
        f"Best checkpoint: "
        f"{model_path}"
    )

    print(
        f"Results directory: "
        f"{RESULT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
