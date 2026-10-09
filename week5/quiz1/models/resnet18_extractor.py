import torch
import torch.nn as nn
from torchvision import models


class ResNet18Extractor(nn.Module):
    """Extract 512-dimensional image features using pretrained ResNet18."""

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = models.resnet18(weights=weights)

        # Remove the final classification layer.
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.feature_dim = 512

    def forward(self, images):
        # Output: [batch_size, 512, 1, 1]
        features = self.backbone(images)

        # Output: [batch_size, 512]
        features = torch.flatten(features, start_dim=1)

        # Normalize each feature vector to unit length.
        features = nn.functional.normalize(
            features, p=2, dim=1
        )

        return features


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet18Extractor(pretrained=True).to(device)
    model.eval()

    dummy_images = torch.randn(4, 3, 224, 224).to(device)

    with torch.inference_mode():
        features = model(dummy_images)

    print("Device:", device)
    print("Feature shape:", tuple(features.shape))
    print("Feature dimension:", model.feature_dim)
    print("Feature norms:", features.norm(dim=1).tolist())
    print("ResNet18 feature extractor test: PASS")
