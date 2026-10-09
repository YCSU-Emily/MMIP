from pathlib import Path
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


# --------------------------------------------------
# Configuration
# --------------------------------------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-4
MARGIN = 0.3
VAL_RATIO = 0.1
NUM_WORKERS = 4

PROJECT_DIR = Path(__file__).resolve().parents[1]
GALLERY_DIR = PROJECT_DIR / "data" / "gallery"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class TripletGalleryDataset(Dataset):
    """Load gallery images and return image tensors with class labels."""

    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(
            root=str(root),
            transform=transform,
        )
        self.samples = self.dataset.samples
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label


class ResNet18Embedding(nn.Module):
    """ResNet18 backbone that outputs L2-normalized 512-D embeddings."""

    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

    def forward(self, images):
        features = self.backbone(images)
        features = torch.flatten(features, start_dim=1)
        return F.normalize(features, p=2, dim=1)


def batch_hard_triplet_loss(embeddings, labels, margin):
    """
    Batch-hard triplet loss:
    hardest positive = farthest same-class example in the batch
    hardest negative = nearest different-class example in the batch
    """
    distances = torch.cdist(embeddings, embeddings, p=2)

    same_class = labels[:, None].eq(labels[None, :])
    different_class = ~same_class

    # Exclude each sample from its own positive set.
    same_class.fill_diagonal_(False)

    valid_anchors = same_class.any(dim=1) & different_class.any(dim=1)
    if not valid_anchors.any():
        return None

    positive_distances = distances.masked_fill(~same_class, float("-inf"))
    hardest_positive = positive_distances.max(dim=1).values

    negative_distances = distances.masked_fill(
        ~different_class, float("inf")
    )
    hardest_negative = negative_distances.min(dim=1).values

    losses = F.relu(
        hardest_positive[valid_anchors]
        - hardest_negative[valid_anchors]
        + margin
    )
    return losses.mean()


def main():
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = TripletGalleryDataset(GALLERY_DIR, transform=transform)
    print("Gallery images:", len(dataset))
    print("Classes:", dataset.classes)

    # Split only the gallery dataset into train/validation subsets.
    indices = list(range(len(dataset)))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    val_size = int(len(indices) * VAL_RATIO)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = ResNet18Embedding().to(device)

    # Fine-tune the last ResNet block and embedding output.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    for parameter in model.backbone[-2].parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    checkpoint_path = MODEL_DIR / "resnet18_triplet_best.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            loss = batch_hard_triplet_loss(
                embeddings, labels, MARGIN
            )

            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

            if step % 20 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        if train_batches == 0:
            raise RuntimeError(
                "No valid triplets found. Check batch composition."
            )

        train_loss = train_loss_sum / train_batches

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                embeddings = model(images)
                loss = batch_hard_triplet_loss(
                    embeddings, labels, MARGIN
                )

                if loss is not None:
                    val_loss_sum += loss.item()
                    val_batches += 1

        val_loss = (
            val_loss_sum / val_batches
            if val_batches
            else float("inf")
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS} summary | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}\n"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "classes": dataset.classes,
                    "feature_dim": 512,
                    "margin": MARGIN,
                },
                checkpoint_path,
            )
            print("Saved best checkpoint:", checkpoint_path)

    print("\nTraining completed.")
    print("Best validation loss:", best_val_loss)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
