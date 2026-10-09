import os
import shutil
from pathlib import Path

from torchvision.datasets import CIFAR10


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

GALLERY_DIR = DATA_DIR / "gallery"
QUERY_DIR = DATA_DIR / "query"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def save_dataset(dataset, output_dir, samples_per_class):
    output_dir.mkdir(parents=True, exist_ok=True)

    counters = {class_id: 0 for class_id in range(len(CLASS_NAMES))}

    for image, label in dataset:
        if counters[label] >= samples_per_class:
            continue

        class_dir = output_dir / CLASS_NAMES[label]
        class_dir.mkdir(parents=True, exist_ok=True)

        image_path = class_dir / f"{counters[label]:05d}.png"
        image.save(image_path)

        counters[label] += 1

        if all(
            counters[class_id] >= samples_per_class
            for class_id in range(len(CLASS_NAMES))
        ):
            break


def count_images(root):
    counts = {}

    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        counts[class_name] = len(list(class_dir.glob("*.png")))

    return counts


def main():
    print("=" * 60)
    print("Week 5 Quiz 1 - Image Retrieval Dataset")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Downloading CIFAR-10...")

    train_dataset = CIFAR10(
        root=str(RAW_DIR),
        train=True,
        download=True,
    )

    test_dataset = CIFAR10(
        root=str(RAW_DIR),
        train=False,
        download=True,
    )

    print("\n[2] Preparing Gallery...")
    print("    1,000 images per class")

    save_dataset(
        train_dataset,
        GALLERY_DIR,
        samples_per_class=1000,
    )

    print("\n[3] Preparing Query...")
    print("    100 images per class")

    save_dataset(
        test_dataset,
        QUERY_DIR,
        samples_per_class=100,
    )

    print("\n[4] Dataset summary")

    gallery_counts = count_images(GALLERY_DIR)
    query_counts = count_images(QUERY_DIR)

    print("\nGallery:")
    for class_name, count in gallery_counts.items():
        print(f"  {class_name:12s}: {count}")

    print(f"  Total: {sum(gallery_counts.values())}")

    print("\nQuery:")
    for class_name, count in query_counts.items():
        print(f"  {class_name:12s}: {count}")

    print(f"  Total: {sum(query_counts.values())}")

    print("\n[5] Dataset location")
    print(f"  Gallery: {GALLERY_DIR}")
    print(f"  Query:   {QUERY_DIR}")

    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()
