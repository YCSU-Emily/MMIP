import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights

import train_caption as base


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------
SOURCE_CHECKPOINT = "./models/best_caption_model.pth"
EXP_DIR = "./models/experiments"
RESULT_DIR = "./results/experiments"

EXP_NAME = "continuation_v1"
CONTINUE_EPOCHS = 5

# Lower learning rate than the original 1e-3.
LEARNING_RATE = 1e-4

BATCH_SIZE = base.BATCH_SIZE
DEVICE = base.DEVICE

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

BEST_PATH = os.path.join(EXP_DIR, f"{EXP_NAME}_best.pth")
LATEST_PATH = os.path.join(EXP_DIR, f"{EXP_NAME}_latest.pth")
HISTORY_PATH = os.path.join(RESULT_DIR, f"{EXP_NAME}_history.json")
CONFIG_PATH = os.path.join(RESULT_DIR, f"{EXP_NAME}_config.json")


def save_checkpoint(path, encoder, decoder, vocab, epoch, val_loss):
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "vocab_word2idx": vocab.word2idx,
        "vocab_idx2word": vocab.idx2word,
        "embed_dim": base.EMBED_DIM,
        "hidden_dim": base.HIDDEN_DIM,
        "max_length": base.MAX_LENGTH,
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def main():
    base.set_seed(base.SEED)

    print("=" * 72)
    print("Image Captioning - Continuation Experiment v1")
    print("=" * 72)
    print("Device:", DEVICE)

    # Load checkpoint without changing the original.
    checkpoint = torch.load(
        SOURCE_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    start_epoch = int(checkpoint["epoch"])
    best_val_loss = float(checkpoint["val_loss"])

    # Reuse the exact vocabulary from the checkpoint.
    vocab = base.Vocabulary(min_freq=base.MIN_WORD_FREQ)
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = checkpoint["vocab_idx2word"]

    if len(vocab.word2idx) != len(vocab.idx2word):
        raise ValueError("Vocabulary mapping sizes do not match.")

    if not all(
        vocab.idx2word.get(idx) == word
        for word, idx in vocab.word2idx.items()
    ):
        raise ValueError("Vocabulary index mapping is inconsistent.")

    print("Source epoch:", start_epoch)
    print("Source validation loss:", best_val_loss)
    print("Vocabulary size:", len(vocab))

    # Load the same COCO annotations as the original experiment.
    train_samples = base.load_coco_annotations(base.TRAIN_CAPTION_FILE)
    val_samples = base.load_coco_annotations(base.VAL_CAPTION_FILE)

    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.Resize((base.IMAGE_SIZE, base.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((base.IMAGE_SIZE, base.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = base.CaptionDataset(
        train_samples,
        base.TRAIN_IMAGE_DIR,
        vocab,
        train_transform,
        base.MAX_LENGTH,
    )

    val_dataset = base.CaptionDataset(
        val_samples,
        base.VAL_IMAGE_DIR,
        vocab,
        val_transform,
        base.MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=base.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(base.NUM_WORKERS > 0),
        collate_fn=base.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=base.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(base.NUM_WORKERS > 0),
        collate_fn=base.collate_fn,
    )

    # Construct model architecture and restore learned weights.
    encoder = base.EncoderCNN(base.EMBED_DIM).to(DEVICE)
    decoder = base.DecoderLSTM(
        len(vocab), base.EMBED_DIM, base.HIDDEN_DIM
    ).to(DEVICE)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    # Keep the original training strategy: frozen ResNet,
    # train the projection layer and LSTM decoder.
    for p in encoder.resnet.parameters():
        p.requires_grad = False

    parameters = list(encoder.fc.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Seed the experiment history with the baseline checkpoint.
    history = [{
        "epoch": start_epoch,
        "phase": "baseline_checkpoint",
        "train_loss": None,
        "val_loss": best_val_loss,
    }]

    # Preserve the starting weights in the experiment directory.
    save_checkpoint(
        BEST_PATH, encoder, decoder, vocab, start_epoch, best_val_loss
    )
    save_checkpoint(
        LATEST_PATH, encoder, decoder, vocab, start_epoch, best_val_loss
    )

    total_start = time.time()

    for epoch in range(start_epoch + 1, start_epoch + CONTINUE_EPOCHS + 1):
        epoch_start = time.time()
        encoder.train()
        decoder.train()

        train_loss_sum = 0.0

        for images, captions in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            captions = captions.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            image_features = encoder(images)
            outputs = decoder(image_features, captions)
            targets = captions[:, 1:]

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

        train_loss = train_loss_sum / len(train_dataset)

        encoder.eval()
        decoder.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                captions = captions.to(DEVICE, non_blocking=True)

                image_features = encoder(images)
                outputs = decoder(image_features, captions)
                targets = captions[:, 1:]

                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1),
                )
                val_loss_sum += loss.item() * images.size(0)

        val_loss = val_loss_sum / len(val_dataset)
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "phase": "continued_training",
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": elapsed,
            "learning_rate": LEARNING_RATE,
        }
        history.append(row)

        # Save latest weights after every completed epoch.
        save_checkpoint(
            LATEST_PATH, encoder, decoder, vocab, epoch, val_loss
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                BEST_PATH, encoder, decoder, vocab, epoch, val_loss
            )
            print("New best checkpoint saved.")

        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch [{epoch}/{start_epoch + CONTINUE_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Time: {elapsed:.1f}s "
            f"Best Val Loss: {best_val_loss:.4f}"
        )

    config = {
        "experiment": EXP_NAME,
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_epoch": start_epoch,
        "continued_epochs": CONTINUE_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "vocabulary_size": len(vocab),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_loss": best_val_loss,
        "best_checkpoint": BEST_PATH,
        "latest_checkpoint": LATEST_PATH,
        "total_time_sec": time.time() - total_start,
        "optimizer_state_resumed": False,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("=" * 72)
    print("Continuation experiment completed.")
    print("Best model:", BEST_PATH)
    print("Latest model:", LATEST_PATH)
    print("History:", HISTORY_PATH)
    print("Config:", CONFIG_PATH)
    print("Original model and original results were not overwritten.")
    print("=" * 72)


if __name__ == "__main__":
    main()
