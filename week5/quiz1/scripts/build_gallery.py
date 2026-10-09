from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.resnet18_extractor import ResNet18Extractor


ROOT = Path(__file__).resolve().parents[1]
GALLERY_DIR = ROOT / "data" / "gallery"
RESULTS_DIR = ROOT / "results"

BATCH_SIZE = 128
NUM_WORKERS = 4

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class GalleryDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = root / class_name

            for path in sorted(class_dir.glob("*.png")):
                self.samples.append((path, class_id))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label, str(path)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = GalleryDataset(GALLERY_DIR)
    print("Gallery images:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = ResNet18Extractor(pretrained=True).to(device)
    model.eval()

    all_features = []
    all_labels = []
    all_paths = []

    print("\nExtracting Gallery features...")

    with torch.inference_mode():
        for batch_index, (images, labels, paths) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            features = model(images)

            all_features.append(features.cpu())
            all_labels.append(labels)
            all_paths.extend(paths)

            if batch_index % 10 == 0:
                print(
                    f"Batch {batch_index + 1}/{len(loader)} "
                    f"| processed {(batch_index + 1) * images.size(0)} "
                    f"images"
                )

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    output_path = RESULTS_DIR / "gallery_resnet18.pt"

    torch.save({
        "features": features,
        "labels": labels,
        "paths": all_paths,
        "class_names": CLASS_NAMES,
        "feature_dim": 512,
        "backbone": "ResNet18_ImageNet_pretrained",
    }, output_path)

    print("\nGallery feature extraction completed.")
    print("Feature shape:", tuple(features.shape))
    print("Label shape:", tuple(labels.shape))
    print("Number of paths:", len(all_paths))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
