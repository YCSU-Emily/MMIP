"""Evaluate ViT-B/16 image retrieval without overwriting CNN results."""

import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.vit_b16_extractor import ViTB16Extractor

ROOT = PROJECT_DIR
QUERY_DIR = ROOT / "data" / "query"
RESULTS_DIR = ROOT / "results"
GALLERY_FILE = RESULTS_DIR / "gallery_vit_b16.pt"

BATCH_SIZE = 32
NUM_WORKERS = 4


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_file = RESULTS_DIR / "vit_retrieval_metrics.json"
    csv_file = RESULTS_DIR / "vit_top5_results.csv"
    image_file = RESULTS_DIR / "vit_top5_retrieval.png"

    outputs = [metrics_file, csv_file, image_file]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise FileExistsError(
            "ViT output files already exist; refusing to overwrite:\n"
            + "\n".join(existing)
            + "\nBack up or rename them before rerunning."
        )

    if not GALLERY_FILE.is_file():
        raise FileNotFoundError(f"Gallery file not found: {GALLERY_FILE}")

    if not QUERY_DIR.is_dir():
        raise FileNotFoundError(f"Query directory not found: {QUERY_DIR}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    gallery = torch.load(
        GALLERY_FILE, map_location="cpu", weights_only=True
    )
    gallery_features = gallery["features"].float()
    gallery_labels = gallery["labels"].long()
    gallery_paths = gallery["paths"]
    class_names = gallery["class_names"]

    if gallery_features.shape != (len(gallery_paths), 768):
        raise ValueError(
            f"Unexpected gallery feature shape: {gallery_features.shape}"
        )

    print("Backbone: ViT-B/16")
    print("Gallery feature shape:", tuple(gallery_features.shape))
    print("Gallery images:", len(gallery_paths))
    print("Classes:", class_names)

    extractor = ViTB16Extractor(device=device)

    query_dataset = ImageFolder(
        root=str(QUERY_DIR),
        transform=extractor.transform,
    )

    if query_dataset.classes != class_names:
        raise ValueError(
            "Query and Gallery class order differs.\n"
            f"Query: {query_dataset.classes}\n"
            f"Gallery: {class_names}"
        )

    if len(query_dataset) == 0:
        raise RuntimeError("Query dataset is empty.")

    loader = DataLoader(
        query_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    query_feature_batches = []
    query_label_batches = []

    print("\nExtracting ViT Query features...")
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            features = extractor.extract_batch(images)
            query_feature_batches.append(features.cpu())
            query_label_batches.append(labels.cpu())

            if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
                processed = min(
                    (batch_idx + 1) * BATCH_SIZE,
                    len(query_dataset),
                )
                print(f"Progress: {processed}/{len(query_dataset)}")

    query_features = torch.cat(query_feature_batches, dim=0)
    query_labels = torch.cat(query_label_batches, dim=0).long()
    query_paths = [
        str(Path(path).resolve())
        for path, _ in query_dataset.samples
    ]

    print("Query feature shape:", tuple(query_features.shape))

    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError("Query and Gallery feature dimensions do not match.")

    # Both sets are L2-normalized; dot product is cosine similarity.
    similarities = query_features @ gallery_features.T
    max_k = 10
    top_scores, top_indices = similarities.topk(max_k, dim=1)
    top_labels = gallery_labels[top_indices]

    metrics = {}
    for k in [1, 5, 10]:
        hits = (
            top_labels[:, :k] == query_labels[:, None]
        ).any(dim=1)
        metrics[f"Recall@{k}"] = hits.float().mean().item()

    print("\nViT-B/16 Retrieval metrics (same-class relevance):")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f} ({value * 100:.2f}%)")

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
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
                    class_names[query_labels[qi].item()],
                    rank + 1,
                    gallery_paths[gi],
                    class_names[gallery_labels[gi].item()],
                    f"{top_scores[qi, rank].item():.6f}",
                ])

    # Pick the first Query image from each class for visualization.
    first_query_index = {}
    for index, (_, label) in enumerate(query_dataset.samples):
        first_query_index.setdefault(label, index)

    fig, axes = plt.subplots(10, 6, figsize=(15, 23))

    for class_id, class_name in enumerate(class_names):
        qi = first_query_index[class_id]

        with Image.open(query_paths[qi]) as image:
            axes[class_id, 0].imshow(image.convert("RGB"))

        axes[class_id, 0].set_title(f"Query\n{class_name}")
        axes[class_id, 0].axis("off")

        for rank in range(5):
            gi = top_indices[qi, rank].item()

            with Image.open(gallery_paths[gi]) as image:
                axes[class_id, rank + 1].imshow(image.convert("RGB"))

            predicted_class = class_names[gallery_labels[gi].item()]
            score = top_scores[qi, rank].item()
            axes[class_id, rank + 1].set_title(
                f"Top-{rank + 1}: {predicted_class}\n"
                f"sim={score:.3f}"
            )
            axes[class_id, rank + 1].axis("off")

    fig.suptitle(
        "ViT-B/16 Image Retrieval: Query and Top-5 Gallery Results",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(image_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n========== ViT retrieval completed ==========")
    print("Saved metrics:", metrics_file)
    print("Saved Top-5 CSV:", csv_file)
    print("Saved visualization:", image_file)
    print("Query count:", len(query_dataset))
    print("Gallery count:", len(gallery_paths))
    print("PASS: ViT image retrieval evaluation completed.")


if __name__ == "__main__":
    main()
