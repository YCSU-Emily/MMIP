import os
import numpy as np
from PIL import Image


CAM_DIR = "results/resnet18_aug/gradcam"

IMAGE_SIZE = 224

print("=" * 70)
print("Grad-CAM Numerical Analysis")
print("=" * 70)


for i in range(9):

    heatmap_path = os.path.join(
        CAM_DIR,
        f"sample_{i:02d}_heatmap.png"
    )

    if not os.path.exists(heatmap_path):
        print(f"Missing: {heatmap_path}")
        continue

    img = Image.open(
        heatmap_path
    ).convert("RGB")

    img = img.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    arr = np.asarray(img).astype(
        np.float32
    ) / 255.0

    # Grad-CAM 使用 jet colormap。
    # 取紅色通道作為高活化區域的近似指標。
    score = arr[:, :, 0]

    # Normalize
    score = (
        score - score.min()
    ) / (
        score.max() - score.min() + 1e-8
    )

    # High-attention threshold
    threshold = 0.70

    mask = score >= threshold

    ys, xs = np.where(mask)

    print()
    print("-" * 70)
    print(f"Sample {i}")

    if len(xs) == 0:
        print("No high-attention region found.")
        continue

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    print(
        f"High-attention pixels : {len(xs)}"
    )

    print(
        f"Bounding box           : "
        f"x={x_min}~{x_max}, "
        f"y={y_min}~{y_max}"
    )

    # Center of high-attention region
    center_x = xs.mean()
    center_y = ys.mean()

    print(
        f"Attention center       : "
        f"({center_x:.1f}, {center_y:.1f})"
    )

    # Determine quadrant
    if center_x < 112 and center_y < 112:
        quadrant = "Upper-Left"
    elif center_x >= 112 and center_y < 112:
        quadrant = "Upper-Right"
    elif center_x < 112 and center_y >= 112:
        quadrant = "Lower-Left"
    else:
        quadrant = "Lower-Right"

    print(
        f"Main attention region  : {quadrant}"
    )

    # Divide image into 3x3 regions
    grid_scores = []

    for gy in range(3):
        row = []

        for gx in range(3):

            y0 = gy * 74
            y1 = (
                (gy + 1) * 74
                if gy < 2
                else 224
            )

            x0 = gx * 74
            x1 = (
                (gx + 1) * 74
                if gx < 2
                else 224
            )

            region_score = score[
                y0:y1,
                x0:x1
            ].mean()

            row.append(region_score)

        grid_scores.append(row)

    print("3x3 attention scores:")

    for row in grid_scores:
        print(
            "  ".join(
                f"{v:.3f}"
                for v in row
            )
        )


print()
print("=" * 70)
print("Analysis finished.")
print("=" * 70)
