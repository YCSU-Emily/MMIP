from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# =========================
# Configuration
# =========================

SOURCE_DIR = Path("../dataset/wafer/original")
OUTPUT_DIR = Path("../dataset/wafer")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# =========================
# Check ratios
# =========================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

# =========================
# Find classes
# =========================

classes = sorted([
    d.name for d in SOURCE_DIR.iterdir()
    if d.is_dir()
])

print("=" * 60)
print("Wafer Defect Dataset Split")
print("=" * 60)

print(f"Source directory: {SOURCE_DIR.resolve()}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
print(f"Classes: {len(classes)}")
print()

for cls in classes:
    print(f"  {cls}")

# =========================
# Create output directories
# =========================

for split in ["train", "val", "test"]:
    for cls in classes:
        (OUTPUT_DIR / split / cls).mkdir(
            parents=True,
            exist_ok=True
        )

# =========================
# Split each class
# =========================

summary = {}

for cls in classes:

    class_dir = SOURCE_DIR / cls

    images = sorted([
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".jpg"
    ])

    # 70% Train / 30% temporary
    train_images, temp_images = train_test_split(
        images,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Temporary -> 15% Validation / 15% Test
    val_images, test_images = train_test_split(
        temp_images,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    summary[cls] = {}

    for split_name, split_images in splits.items():

        summary[cls][split_name] = len(split_images)

        destination = OUTPUT_DIR / split_name / cls

        for image_path in split_images:

            target = destination / image_path.name

            shutil.copy2(image_path, target)

    print(
        f"{cls:10s} | "
        f"Train: {len(train_images):3d} | "
        f"Val: {len(val_images):3d} | "
        f"Test: {len(test_images):3d}"
    )

# =========================
# Overall statistics
# =========================

print()
print("=" * 60)
print("Overall Statistics")
print("=" * 60)

total_train = sum(summary[c]["train"] for c in classes)
total_val = sum(summary[c]["val"] for c in classes)
total_test = sum(summary[c]["test"] for c in classes)

total = total_train + total_val + total_test

print(f"Train      : {total_train}")
print(f"Validation : {total_val}")
print(f"Test       : {total_test}")
print(f"Total      : {total}")

print()
print("Split ratio:")
print(f"Train      : {total_train / total * 100:.2f}%")
print(f"Validation : {total_val / total * 100:.2f}%")
print(f"Test       : {total_test / total * 100:.2f}%")

print()
print("Dataset created at:")
print(OUTPUT_DIR.resolve())

print("=" * 60)
print("Done!")
print("=" * 60)
