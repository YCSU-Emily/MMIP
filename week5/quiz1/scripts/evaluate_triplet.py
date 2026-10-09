from pathlib import Path
import sys
import json
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.resnet18_extractor import ResNet18Extractor

ROOT = PROJECT_DIR
QUERY_DIR = ROOT / "data" / "query"
RESULTS_DIR = ROOT / "results"
CHECKPOINT = ROOT / "models" / "resnet18_triplet_best.pth"
BASELINE_GALLERY = RESULTS_DIR / "gallery_resnet18.pt"

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


class ImagePathDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = [str(p) for p in paths]
        self.labels = [int(x) for x in labels]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = TRANSFORM(image.convert("RGB"))
        return image, self.labels[index]


def extract_features(model, paths, labels, device):
    dataset = ImagePathDataset(paths, labels)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    features = []
    with torch.inference_mode():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            features.append(model(images).cpu())

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}")

    return torch.cat(features, dim=0)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    baseline = torch.load(
        BASELINE_GALLERY, map_location="cpu", weights_only=False
    )
    gallery_paths = baseline["paths"]
    gallery_labels = baseline["labels"].long()

    print("Gallery images:", len(gallery_paths))

    query_paths = []
    query_labels_list = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        paths = sorted((QUERY_DIR / class_name).glob("*.png"))
        query_paths.extend(str(p) for p in paths)
        query_labels_list.extend([class_id] * len(paths))

    query_labels = torch.tensor(query_labels_list, dtype=torch.long)
    print("Query images:", len(query_paths))

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )

    model = ResNet18Extractor(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    print("Loaded checkpoint epoch:", checkpoint["epoch"])
    print("Validation loss:", checkpoint["val_loss"])

    print("\nExtracting Triplet Gallery features...")
    gallery_features = extract_features(
        model, gallery_paths, gallery_labels, device
    )

    gallery_output = RESULTS_DIR / "gallery_resnet18_triplet.pt"
    torch.save({
        "features": gallery_features,
        "labels": gallery_labels,
        "paths": gallery_paths,
    }, gallery_output)

    print("Saved:", gallery_output)
    print("Gallery feature shape:", tuple(gallery_features.shape))

    print("\nExtracting Triplet Query features...")
    query_features = extract_features(
        model, query_paths, query_labels, device
    )
    print("Query feature shape:", tuple(query_features.shape))

    similarities = query_features @ gallery_features.T
    top_scores, top_indices = similarities.topk(10, dim=1)
    top_labels = gallery_labels[top_indices]

    metrics = {}
    for k in [1, 5, 10]:
        hits = (
            top_labels[:, :k] == query_labels[:, None]
        ).any(dim=1)
        metrics[f"Recall@{k}"] = hits.float().mean().item()

    print("\nTriplet Loss retrieval metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f} ({value * 100:.2f}%)")

    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    metrics["train_loss"] = float(checkpoint["train_loss"])
    metrics["val_loss"] = float(checkpoint["val_loss"])

    metrics_path = RESULTS_DIR / "triplet_retrieval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = RESULTS_DIR / "triplet_top5_results.csv"
    with open(csv_path, "w", newline="") as f:
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

    # Compare with the existing Baseline metrics without overwriting them.
    baseline_path = RESULTS_DIR / "retrieval_metrics.json"
    with open(baseline_path) as f:
        baseline_metrics = json.load(f)

    comparison = {}
    print("\n===== Baseline vs Triplet Loss =====")
    for k in [1, 5, 10]:
        key = f"Recall@{k}"
        before = float(baseline_metrics[key])
        after = float(metrics[key])
        delta = after - before

        comparison[key] = {
            "baseline": before,
            "triplet": after,
            "delta_percentage_points": delta * 100,
        }

        print(
            f"{key}: Baseline={before * 100:.2f}% | "
            f"Triplet={after * 100:.2f}% | "
            f"Change={delta * 100:+.2f} percentage points"
        )

    comparison_path = RESULTS_DIR / "retrieval_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nSaved metrics:", metrics_path)
    print("Saved Top-5 CSV:", csv_path)
    print("Saved comparison:", comparison_path)
    print("Triplet Loss evaluation: PASS")


if __name__ == "__main__":
    main()
