from pathlib import Path
import csv
import sys

import torch
from PIL import Image
from torchvision import transforms

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.resnet18_extractor import ResNet18Extractor

RESULTS = PROJECT_DIR / "results"
CHECKPOINT = PROJECT_DIR / "models" / "resnet18_triplet_best.pth"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

TARGETS = [
    "truck/00034.png",
    "truck/00095.png",
    "deer/00097.png",
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pt(path):
    try:
        return torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_one(model, path):
    with Image.open(path) as img:
        x = TRANSFORM(img.convert("RGB")).unsqueeze(0)
    x = x.to(DEVICE)

    with torch.inference_mode():
        feature = model(x).cpu().squeeze(0)

    return feature


def main():
    print("Device:", DEVICE)

    baseline_gallery = load_pt(
        RESULTS / "gallery_resnet18.pt"
    )
    triplet_gallery = load_pt(
        RESULTS / "gallery_resnet18_triplet.pt"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    baseline_model = ResNet18Extractor(pretrained=True).to(DEVICE)
    baseline_model.eval()

    triplet_model = ResNet18Extractor(pretrained=False)
    triplet_model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    triplet_model = triplet_model.to(DEVICE)
    triplet_model.eval()

    models = [
        ("Baseline", baseline_model, baseline_gallery),
        ("Triplet", triplet_model, triplet_gallery),
    ]

    output_rows = []

    for rel_path in TARGETS:
        query_path = PROJECT_DIR / "data" / "query" / rel_path
        true_class = query_path.parent.name

        print("\n" + "=" * 90)
        print("QUERY:", rel_path)
        print("True class:", true_class)

        for model_name, model, gallery in models:
            gallery_features = gallery["features"].float()
            gallery_labels = gallery["labels"].long()
            gallery_paths = gallery["paths"]

            query_feature = extract_one(model, query_path)

            # Features are L2-normalized; dot product = cosine similarity.
            similarities = query_feature @ gallery_features.T
            scores, indices = similarities.topk(10)

            same_class_indices = (
                gallery_labels[indices] ==
                CLASS_NAMES.index(true_class)
            ).nonzero(as_tuple=True)[0]

            first_correct_rank = (
                int(same_class_indices[0].item()) + 1
                if len(same_class_indices) else None
            )

            best_same = similarities[
                gallery_labels == CLASS_NAMES.index(true_class)
            ].max().item()

            best_wrong = similarities[
                gallery_labels != CLASS_NAMES.index(true_class)
            ].max().item()

            print(f"\n--- {model_name} ---")
            print("Best same-class rank:", first_correct_rank)
            print(f"Best same-class similarity: {best_same:.6f}")
            print(f"Best wrong-class similarity: {best_wrong:.6f}")
            print(f"Margin (same - wrong): {best_same - best_wrong:+.6f}")
            print("Top-10:")

            for rank, (score, idx) in enumerate(
                zip(scores.tolist(), indices.tolist()), start=1
            ):
                gallery_class = CLASS_NAMES[
                    int(gallery_labels[idx].item())
                ]
                gallery_path = gallery_paths[idx]

                print(
                    f"{rank:2d}. {gallery_class:10s} "
                    f"sim={score:.6f}  {gallery_path}"
                )

                output_rows.append({
                    "query_path": str(query_path),
                    "query_class": true_class,
                    "model": model_name,
                    "rank": rank,
                    "gallery_path": str(gallery_path),
                    "gallery_class": gallery_class,
                    "cosine_similarity": f"{score:.6f}",
                    "best_same_class_rank": first_correct_rank,
                    "best_same_similarity": f"{best_same:.6f}",
                    "best_wrong_similarity": f"{best_wrong:.6f}",
                    "margin_same_minus_wrong":
                        f"{best_same - best_wrong:.6f}",
                })

    output = RESULTS / "triplet_failure_top10.csv"
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(output_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nSaved:", output.resolve())
    print("Done. No training or existing result files overwritten.")


if __name__ == "__main__":
    main()
