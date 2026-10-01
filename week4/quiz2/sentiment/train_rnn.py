import os
import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# Configuration
# ============================================================

SEED = 3407

DATA_DIR = "data"
MODEL_DIR = "models"
RESULT_DIR = "results/rnn"

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

BEST_MODEL = os.path.join(MODEL_DIR, "best_rnn.pth")
HISTORY_FILE = os.path.join(RESULT_DIR, "history.json")
METRICS_FILE = os.path.join(RESULT_DIR, "metrics.json")
PREDICTION_FILE = os.path.join(RESULT_DIR, "test_predictions.csv")
CONFIG_FILE = os.path.join(RESULT_DIR, "config.json")

VOCAB_SIZE = 20000
MAX_LEN = 200

EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3

VAL_RATIO = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Simple tokenizer
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
UNK_IDX = 1


def tokenize(text):
    return text.lower().split()


def build_vocab(texts, max_vocab_size):
    word_count = {}

    for text in texts:
        for word in tokenize(text):
            word_count[word] = word_count.get(word, 0) + 1

    sorted_words = sorted(
        word_count.items(),
        key=lambda x: x[1],
        reverse=True
    )

    vocab = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for word, _ in sorted_words[:max_vocab_size - 2]:
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len):
    tokens = tokenize(text)

    ids = [
        vocab.get(token, UNK_IDX)
        for token in tokens[:max_len]
    ]

    if len(ids) < max_len:
        ids += [PAD_IDX] * (max_len - len(ids))

    return ids


# ============================================================
# Dataset
# ============================================================

class SentimentDataset(Dataset):

    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        input_ids = encode_text(
            self.texts[idx],
            self.vocab,
            self.max_len
        )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ============================================================
# RNN Model
# ============================================================

class RNNClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers=1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=PAD_IDX
        )

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            2
        )

    def forward(self, x):

        x = self.embedding(x)

        output, hidden = self.rnn(x)

        # hidden:
        # [num_layers, batch, hidden_dim]

        last_hidden = hidden[-1]

        logits = self.fc(last_hidden)

        return logits


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item() * inputs.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    avg_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 70)
    print("Week 4 Quiz 2 - IMDb Sentiment Analysis")
    print("Basic: RNN")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples : {len(test_df)}")

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=train_df["label"]
    )

    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print(f"Train split : {len(train_texts)}")
    print(f"Validation  : {len(val_texts)}")
    print(f"Test        : {len(test_texts)}")

    # --------------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------------

    print("\nBuilding vocabulary...")

    vocab = build_vocab(
        train_texts,
        VOCAB_SIZE
    )

    print(f"Vocabulary size: {len(vocab)}")

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    train_dataset = SentimentDataset(
        train_texts,
        train_labels,
        vocab,
        MAX_LEN
    )

    val_dataset = SentimentDataset(
        val_texts,
        val_labels,
        vocab,
        MAX_LEN
    )

    test_dataset = SentimentDataset(
        test_texts,
        test_labels,
        vocab,
        MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel:")
    print(model)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "seed": SEED,
        "vocab_size": VOCAB_SIZE,
        "actual_vocab_size": len(vocab),
        "max_len": MAX_LEN,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "val_ratio": VAL_RATIO,
        "device": str(DEVICE),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Training RNN")
    print("=" * 70)

    history = []

    best_val_f1 = -1.0

    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0

        train_labels_epoch = []
        train_predictions_epoch = []

        for inputs, labels in train_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * inputs.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            train_labels_epoch.extend(
                labels.detach().cpu().numpy()
            )

            train_predictions_epoch.extend(
                predictions.detach().cpu().numpy()
            )

        train_loss = (
            running_loss /
            len(train_loader.dataset)
        )

        train_accuracy = accuracy_score(
            train_labels_epoch,
            train_predictions_epoch
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion
        )

        epoch_time = time.time() - epoch_start

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "epoch_time_sec": epoch_time,
        }

        history.append(epoch_result)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_metrics["f1"] > best_val_f1:

            best_val_f1 = val_metrics["f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": config,
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                },
                BEST_MODEL
            )

            print(
                f"  -> Saved best model "
                f"(Val F1 = {best_val_f1:.4f})"
            )

    total_training_time = (
        time.time() - training_start
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print("\nLoading best RNN model...")

    checkpoint = torch.load(
        BEST_MODEL,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Test Evaluation")
    print("=" * 70)

    inference_start = time.time()

    test_metrics = evaluate(
        model,
        test_loader,
        criterion
    )

    inference_time = (
        time.time() - inference_start
    )

    print(
        f"Test Loss      : {test_metrics['loss']:.4f}"
    )

    print(
        f"Test Accuracy   : {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Precision  : {test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall     : {test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1-score   : {test_metrics['f1']:.4f}"
    )

    print(
        f"Training time   : {total_training_time:.2f} sec"
    )

    print(
        f"Inference time  : {inference_time:.2f} sec"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    model.eval()

    predictions = []

    with torch.no_grad():

        for inputs, _ in test_loader:

            inputs = inputs.to(DEVICE)

            logits = model(inputs)

            probs = torch.softmax(
                logits,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

    prediction_df = pd.DataFrame({
        "text": test_texts,
        "label": test_labels,
        "prediction": predictions
    })

    prediction_df.to_csv(
        PREDICTION_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "model": "RNN",
        "test_loss": test_metrics["loss"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "training_time_sec": total_training_time,
        "inference_time_sec": inference_time,
        "best_val_f1": best_val_f1,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("RNN training completed!")
    print("=" * 70)

    print("\nOutput files:")

    print(f"  Model      : {BEST_MODEL}")
    print(f"  History    : {HISTORY_FILE}")
    print(f"  Metrics    : {METRICS_FILE}")
    print(f"  Predictions: {PREDICTION_FILE}")
    print(f"  Config     : {CONFIG_FILE}")


if __name__ == "__main__":
    main()
