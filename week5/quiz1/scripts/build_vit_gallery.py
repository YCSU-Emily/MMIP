"""Build and save ViT-B/16 features for the image retrieval gallery."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.vit_b16_extractor import ViTB16Extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    gallery_dir = PROJECT_ROOT / "data" / "gallery"
    output_path = PROJECT_ROOT / "results" / "gallery_vit_b16.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"{output_path} already exists. "
            "Use --force only if you intentionally want to replace it."
        )

    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")

    extractor = ViTB16Extractor()
    dataset = ImageFolder(
        root=str(gallery_dir),
        transform=extractor.transform,
    )

    if len(dataset) == 0:
        raise RuntimeError("Gallery dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(extractor.device.type == "cuda"),
    )

    feature_batches = []
    label_batches = []

    print("Device:", extractor.device)
    print("Backbone: ViT-B/16")
    print("Gallery images:", len(dataset))
    print("Classes:", dataset.classes)
    print("Batch size:", args.batch_size)

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            features = extractor.extract_batch(images)

            feature_batches.append(features.cpu())
            label_batches.append(labels.cpu())

            processed = min(
                (batch_idx + 1) * args.batch_size, len(dataset)
            )
            if batch_idx == 0 or (batch_idx + 1) % 20 == 0:
                print(f"Progress: {processed}/{len(dataset)}")

    features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0).long()
    paths = [str(Path(path).resolve()) for path, _ in dataset.samples]

    result = {
        "features": features,
        "labels": labels,
        "paths": paths,
        "class_names": dataset.classes,
        "feature_dim": int(features.shape[1]),
        "backbone": "vit_b_16",
    }

    assert features.shape == (len(dataset), 768), features.shape
    assert labels.shape == (len(dataset),), labels.shape
    assert len(paths) == len(dataset)
    assert torch.isfinite(features).all()
    assert torch.allclose(
        features.norm(dim=1),
        torch.ones(len(dataset)),
        atol=1e-4,
    ), "Some feature vectors are not L2-normalized"

    torch.save(result, output_path)

    print("\n========== Gallery 建立完成 ==========")
    print("Saved to:", output_path)
    print("Feature shape:", tuple(features.shape))
    print("Label shape:", tuple(labels.shape))
    print("Feature dimension:", result["feature_dim"])
    print("Backbone:", result["backbone"])
    print("All features finite:", bool(torch.isfinite(features).all()))
    print("Feature norms valid: PASS")
    print("PASS: ViT Gallery features saved.")


if __name__ == "__main__":
    main()
