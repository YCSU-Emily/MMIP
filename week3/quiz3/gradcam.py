import os
import csv

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import transforms

from models import create_model


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = "results/resnet18_aug/best_model.pth"
DATA_ROOT = "../dataset/wafer"
OUTPUT_DIR = "results/resnet18_aug/gradcam"

IMAGE_SIZE = 224

CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Local",
    "Near-Full",
    "Normal",
    "Random",
    "Scratch",
]


# ============================================================
# Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load model
# ============================================================

print("=" * 70)
print("Grad-CAM")
print("=" * 70)

print(f"Checkpoint : {CHECKPOINT}")
print(f"Device     : {device}")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model = create_model(
    "resnet18",
    pretrained=False
)

model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Grad-CAM target layer
# ============================================================

target_layer = model.layer4[-1]

activations = []
gradients = []


def forward_hook(module, input, output):
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# ============================================================
# Image preprocessing
# ============================================================

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    ),
])


# ============================================================
# Denormalize image
# ============================================================

def denormalize(tensor):
    mean = torch.tensor(
        MEAN,
        device=tensor.device
    ).view(3, 1, 1)

    std = torch.tensor(
        STD,
        device=tensor.device
    ).view(3, 1, 1)

    image = tensor * std + mean

    return torch.clamp(image, 0, 1)


# ============================================================
# Grad-CAM function
# ============================================================

def generate_gradcam(input_tensor, class_index):

    activations.clear()
    gradients.clear()

    output = model(input_tensor)

    score = output[0, class_index]

    model.zero_grad()

    score.backward()

    activation = activations[0]
    gradient = gradients[0]

    # Global Average Pooling over spatial dimensions
    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Weighted sum of feature maps
    cam = (weights * activation).sum(
        dim=1,
        keepdim=True
    )

    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    cam = cam[0, 0]

    # Normalize to 0~1
    cam_min = cam.min()
    cam_max = cam.max()

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min + 1e-8
    )

    return cam.detach().cpu().numpy()


# ============================================================
# Find test images
# ============================================================

test_dir = os.path.join(
    DATA_ROOT,
    "test"
)

print(f"Test directory: {test_dir}")

samples = []

for class_index, class_name in enumerate(CLASSES):

    class_dir = os.path.join(
        test_dir,
        class_name
    )

    if not os.path.isdir(class_dir):
        continue

    files = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    if len(files) == 0:
        continue

    # Take the first image of each class
    image_path = os.path.join(
        class_dir,
        files[0]
    )

    samples.append(
        (
            image_path,
            class_index,
            class_name
        )
    )


print(f"Selected {len(samples)} samples.")


# ============================================================
# CSV summary
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_results.csv"
)

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "image",
    "true_class",
    "predicted_class",
    "confidence"
])


# ============================================================
# Process images
# ============================================================

for sample_id, (
    image_path,
    true_index,
    true_class
) in enumerate(samples):

    print()
    print("-" * 70)
    print(f"Sample {sample_id}")
    print(f"Image : {image_path}")
    print(f"True  : {true_class}")

    # Load original image
    original = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        original
    ).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        output = model(input_tensor)

        probability = torch.softmax(
            output,
            dim=1
        )

        predicted_index = (
            probability.argmax(
                dim=1
            ).item()
        )

        confidence = probability[
            0,
            predicted_index
        ].item()

    predicted_class = CLASSES[
        predicted_index
    ]

    print(
        f"Predicted : {predicted_class}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    # Generate Grad-CAM
    cam = generate_gradcam(
        input_tensor,
        predicted_index
    )

    # Resize original image
    original_resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    original_np = np.asarray(
        original_resized
    ) / 255.0

    # --------------------------------------------------------
    # Save original
    # --------------------------------------------------------

    original_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_id:02d}_original.png"
    )

    Image.fromarray(
        (original_np * 255).astype(
            np.uint8
        )
    ).save(original_path)

    # --------------------------------------------------------
    # Save Grad-CAM heatmap
    # --------------------------------------------------------

    heatmap_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_id:02d}_heatmap.png"
    )

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(
        cam,
        cmap="jet"
    )

    plt.axis("off")

    plt.title(
        f"{predicted_class} Grad-CAM"
    )

    plt.tight_layout()

    plt.savefig(
        heatmap_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    overlay_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_id:02d}_overlay.png"
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        original_np
    )

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    plt.axis("off")

    plt.title(
        f"True: {true_class} | "
        f"Pred: {predicted_class} "
        f"({confidence:.2%})"
    )

    plt.tight_layout()

    plt.savefig(
        overlay_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    writer.writerow([
        image_path,
        true_class,
        predicted_class,
        f"{confidence:.6f}"
    ])


csv_file.close()


print()
print("=" * 70)
print("Grad-CAM finished.")
print("=" * 70)
print(f"Output directory : {OUTPUT_DIR}")
print(f"CSV              : {csv_path}")
