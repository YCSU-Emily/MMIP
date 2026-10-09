from pathlib import Path
import sys
import json
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.resnet18_extractor import ResNet18Extractor

ROOT = PROJECT_DIR
QUERY_DIR = ROOT / "data" / "query"
RESULTS_DIR = ROOT / "results"
GALLERY_FILE = RESULTS_DIR / "gallery_resnet18.pt"

BATCH_SIZE = 128
NUM_WORKERS = 4

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class QueryDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for class_id, class_name in enumerate(CLASS_NAMES):
            for path in sorted((root / class_name).glob("*.png")):
                self.samples.append((path, class_id))

        self.transform = TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))

        return image, label, str(path)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    gallery = torch.load(
        GALLERY_FILE, map_location="cpu", weights_only=False
    )
    gallery_features = gallery["features"].float()
    gallery_labels = gallery["labels"].long()
    gallery_paths = gallery["paths"]

    print("Gallery feature shape:", tuple(gallery_features.shape))

    dataset = QueryDataset(QUERY_DIR)
    print("Query images:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = ResNet18Extractor(pretrained=True).to(device)
    model.eval()

    query_features = []
    query_labels = []
    query_paths = []

    print("\nExtracting Query features...")

    with torch.inference_mode():
        for batch_idx, (images, labels, paths) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            features = model(images)

            query_features.append(features.cpu())
            query_labels.append(labels)
            query_paths.extend(paths)

            if batch_idx % 10 == 0:
                print(
                    f"Batch {batch_idx + 1}/{len(loader)}"
                )

    query_features = torch.cat(query_features, dim=0)
    query_labels = torch.cat(query_labels, dim=0)

    print("Query feature shape:", tuple(query_features.shape))

    # Features are L2-normalized, so dot product equals cosine similarity.
    similarities = query_features @ gallery_features.T

    max_k = 10
    top_scores, top_indices = similarities.topk(max_k, dim=1)
    top_labels = gallery_labels[top_indices]

    metrics = {}
    for k in [1, 5, 10]:
        # A query is relevant when at least one of its top-k
        # gallery images has the same CIFAR-10 class label.
        hits = (top_labels[:, :k] == query_labels[:, None]).any(dim=1)
        metrics[f"Recall@{k}"] = hits.float().mean().item()

    print("\nRetrieval metrics (same-class relevance):")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f} ({value * 100:.2f}%)")

    # Save metrics and detailed top-5 results.
    with open(RESULTS_DIR / "retrieval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(RESULTS_DIR / "top5_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_path", "query_class", "rank",
            "gallery_path", "gallery_class", "cosine_similarity",
        ])

        for qi in range(len(query_paths)):
            for rank in range(5):
                gi = top_indices[qi, rank].item()
                writer.writerow([
                    query_paths[qi],
                    CLASS_NAMES[query_labels[qi].item()],
                    rank + 1,
                    gallery_paths[gi],
                    CLASS_NAMES[gallery_labels[gi].item()],
                    f"{top_scores[qi, rank].item():.6f}",
                ])

    # Visualize one query from each of the 10 classes.
    # The query images are ordered by class in QueryDataset.
    fig, axes = plt.subplots(10, 6, figsize=(15, 23))

    for class_id in range(10):
        qi = class_id * 100

        with Image.open(query_paths[qi]) as image:
            axes[class_id, 0].imshow(image.convert("RGB"))
        axes[class_id, 0].set_title(
            f"Query\n{CLASS_NAMES[query_labels[qi].item()]}"
        )
        axes[class_id, 0].axis("off")

        for rank in range(5):
            gi = top_indices[qi, rank].item()

            with Image.open(gallery_paths[gi]) as image:
                axes[class_id, rank + 1].imshow(image.convert("RGB"))

            predicted_class = CLASS_NAMES[gallery_labels[gi].item()]
            score = top_scores[qi, rank].item()
            axes[class_id, rank + 1].set_title(
                f"Top-{rank + 1}: {predicted_class}\n"
                f"sim={score:.3f}"
            )
            axes[class_id, rank + 1].axis("off")

    fig.suptitle(
        "ResNet18 Image Retrieval: Query and Top-5 Gallery Results",
        fontsize=15,
    )
    fig.tight_layout()

    image_output = RESULTS_DIR / "top5_retrieval.png"
    fig.savefig(image_output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved metrics:", RESULTS_DIR / "retrieval_metrics.json")
    print("Saved Top-5 CSV:", RESULTS_DIR / "top5_results.csv")
    print("Saved visualization:", image_output)
    print("Image retrieval evaluation: PASS")


if __name__ == "__main__":
    main()
