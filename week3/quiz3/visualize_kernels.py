import os
import torch
import matplotlib.pyplot as plt

from models import PlainCNN


# =========================
# 設定
# =========================
CHECKPOINT = "results/plain_aug/best_model.pth"
OUTPUT_DIR = "results/plain_aug"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Load model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PlainCNN()
checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

# train.py 儲存的是 checkpoint dictionary
model.load_state_dict(checkpoint["model"])

model.to(device)
model.eval()

print("=" * 60)
print("Kernel Visualization")
print("=" * 60)
print(f"Checkpoint : {CHECKPOINT}")
print(f"Device     : {device}")


# =========================
# 取得第一層 Conv2d
# =========================
first_conv = model.features[0]

weights = first_conv.weight.detach().cpu()

print(f"Kernel shape: {weights.shape}")

# PlainCNN 第一層：
# Conv2d(3, 32, kernel_size=3)
#
# shape = [32, 3, 3, 3]
#
# 32   = 32 個 filters
# 3    = RGB 三個 channel
# 3x3  = kernel size


# =========================
# 選擇兩個 Kernel
# =========================
kernel_indices = [0, 1]


for idx in kernel_indices:

    kernel = weights[idx]

    # 將 RGB 三個 channel 做平均，
    # 方便把 3-channel kernel 視覺化成 2D heatmap
    kernel_mean = kernel.mean(dim=0)

    plt.figure(figsize=(5, 5))

    plt.imshow(
        kernel_mean.numpy(),
        cmap="gray",
        interpolation="nearest"
    )

    plt.colorbar()

    plt.title(f"Plain CNN - Kernel {idx}")

    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"kernel_{idx}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved -> {output_path}")


# =========================
# 兩個 Kernel 放在一起
# =========================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, idx in zip(axes, kernel_indices):

    kernel = weights[idx]
    kernel_mean = kernel.mean(dim=0)

    im = ax.imshow(
        kernel_mean.numpy(),
        cmap="gray",
        interpolation="nearest"
    )

    ax.set_title(f"Kernel {idx}")

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


fig.suptitle(
    "Plain CNN First-Layer Kernel Visualization",
    fontsize=14
)

plt.tight_layout()

comparison_path = os.path.join(
    OUTPUT_DIR,
    "kernels_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved -> {comparison_path}")

print("=" * 60)
print("Visualization finished.")
print("=" * 60)
