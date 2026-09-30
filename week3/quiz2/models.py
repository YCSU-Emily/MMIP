import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 9


class PlainCNN(nn.Module):
    """
    Self-designed Plain CNN

    Input:
        3 x 224 x 224

    Architecture:
        Conv 3 -> 32
        Conv 32 -> 64
        Conv 64 -> 128
        Conv 128 -> 256
        Adaptive Average Pooling
        Fully Connected -> 9 classes
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def create_resnet18(
    num_classes=NUM_CLASSES,
    pretrained=True
):
    """
    Classic CNN Backbone:
    ResNet18

    pretrained=True:
        使用 ImageNet 預訓練權重。

    pretrained=False:
        不下載權重，適合測試環境。
    """

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model


def create_model(
    model_name,
    pretrained=True
):
    if model_name == "plain":
        return PlainCNN()

    elif model_name == "resnet18":
        return create_resnet18(
            pretrained=pretrained
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )


def count_parameters(model):
    """
    計算模型總參數量。
    """
    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_trainable_parameters(model):
    """
    計算可訓練參數量。
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
