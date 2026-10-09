"""Pretrained ViT-B/16 feature extractor for image retrieval."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models


class ViTB16Extractor:
    """Extract 768-dimensional, L2-normalized ViT-B/16 image features."""

    def __init__(self, device=None):
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.weights = models.ViT_B_16_Weights.DEFAULT
        self.transform = self.weights.transforms()

        self.model = models.vit_b_16(weights=self.weights)
        # Replace the 1000-class classifier with an identity layer.
        self.model.heads = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract_batch(self, images):
        """Extract features from a tensor shaped [B, 3, H, W]."""
        images = images.to(self.device)
        features = self.model(images)
        return F.normalize(features, p=2, dim=1)

    @torch.inference_mode()
    def extract_image(self, image_path):
        """Extract features from a single image path."""
        image_path = Path(image_path)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image).unsqueeze(0)

        return self.extract_batch(tensor).squeeze(0).cpu()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    image_path = next(
        (root / "data" / "query" / "airplane").glob("*")
    )

    print("Test image:", image_path)
    extractor = ViTB16Extractor()

    features = extractor.extract_image(image_path)

    print("Device:", extractor.device)
    print("Feature shape:", tuple(features.shape))
    print("Feature norm:", features.norm().item())
    print("All finite:", bool(torch.isfinite(features).all()))

    assert features.shape == (768,), "Expected 768-dimensional features"
    assert torch.isfinite(features).all(), "Features contain NaN or Inf"
    assert torch.isclose(
        features.norm(), torch.tensor(1.0), atol=1e-5
    ), "Feature vector is not L2-normalized"

    print("PASS: ViT-B/16 feature extraction works.")
