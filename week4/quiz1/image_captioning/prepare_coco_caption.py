import json
import os
import shutil
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

SOURCE_ROOT = Path.home() / "Desktop" / "ViECap" / "coco"
SOURCE_ANN = Path.home() / "Desktop" / "ViECap" / "annotations" / "coco"

OUTPUT_ROOT = Path("data")

TRAIN_SIZE = 10000
VAL_SIZE = 1000
TEST_SIZE = 1000

# ============================================================
# Paths
# ============================================================

TRAIN_IMAGE_DIR = SOURCE_ROOT / "images" / "train2017"
VAL_IMAGE_DIR = SOURCE_ROOT / "images" / "val2017"

TRAIN_CAPTION_FILE = SOURCE_ANN / "train_captions.json"
VAL_CAPTION_FILE = SOURCE_ANN / "val_captions.json"

# Output
OUTPUT_IMAGE_DIR = OUTPUT_ROOT / "images"
OUTPUT_CAPTION_DIR = OUTPUT_ROOT / "captions"

OUTPUT_TRAIN_IMAGE_DIR = OUTPUT_IMAGE_DIR / "train"
OUTPUT_VAL_IMAGE_DIR = OUTPUT_IMAGE_DIR / "val"
OUTPUT_TEST_IMAGE_DIR = OUTPUT_IMAGE_DIR / "test"

OUTPUT_TRAIN_CAPTION = OUTPUT_CAPTION_DIR / "train.json"
OUTPUT_VAL_CAPTION = OUTPUT_CAPTION_DIR / "val.json"
OUTPUT_TEST_CAPTION = OUTPUT_CAPTION_DIR / "test.json"


# ============================================================
# Utility functions
# ============================================================

def load_json(path):
    print(f"Loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src, dst):
    """
    Use hard link first to avoid duplicating image data.
    If hard link fails, fall back to normal copy.
    """
    if dst.exists():
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_caption_data(
    source_json,
    selected_images,
    output_json,
    image_dir
):
    """
    Build a COCO-format JSON containing only selected images
    and their corresponding annotations.
    """

    selected_ids = {img["id"] for img in selected_images}

    annotations = [
        ann
        for ann in source_json["annotations"]
        if ann["image_id"] in selected_ids
    ]

    output_data = {
        "info": source_json.get("info", {}),
        "licenses": source_json.get("licenses", []),
        "images": selected_images,
        "annotations": annotations,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved captions: {output_json}")
    print(f"  Images      : {len(selected_images)}")
    print(f"  Annotations : {len(annotations)}")

    # Copy/link images
    for idx, image_info in enumerate(selected_images, start=1):

        file_name = image_info["file_name"]
        src = image_dir / file_name
        dst = output_json.parent.parent / "images" / (
            "train" if output_json.name == "train.json"
            else "val" if output_json.name == "val.json"
            else "test"
        ) / file_name

        if not src.exists():
            print(f"[WARNING] Missing image: {src}")
            continue

        link_or_copy(src, dst)

        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(selected_images)} images")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("COCO Image Captioning Dataset Preparation")
    print("=" * 60)

    # Create directories
    create_dir(OUTPUT_TRAIN_IMAGE_DIR)
    create_dir(OUTPUT_VAL_IMAGE_DIR)
    create_dir(OUTPUT_TEST_IMAGE_DIR)
    create_dir(OUTPUT_CAPTION_DIR)

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    train_data = load_json(TRAIN_CAPTION_FILE)
    val_data = load_json(VAL_CAPTION_FILE)

    print()
    print("Original dataset:")
    print(f"  Train images: {len(train_data['images'])}")
    print(f"  Train annotations: {len(train_data['annotations'])}")
    print(f"  Val images: {len(val_data['images'])}")
    print(f"  Val annotations: {len(val_data['annotations'])}")

    # --------------------------------------------------------
    # Select images
    # --------------------------------------------------------

    train_images = train_data["images"][:TRAIN_SIZE]

    # Use official COCO validation set.
    # First 1000 -> validation
    # Next 1000  -> test
    val_images = val_data["images"][:VAL_SIZE]
    test_images = val_data["images"][
        VAL_SIZE:VAL_SIZE + TEST_SIZE
    ]

    print()
    print("Selected dataset:")
    print(f"  Train: {len(train_images)}")
    print(f"  Val  : {len(val_images)}")
    print(f"  Test : {len(test_images)}")

    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Preparing TRAIN")
    print("=" * 60)

    build_caption_data(
        train_data,
        train_images,
        OUTPUT_TRAIN_CAPTION,
        TRAIN_IMAGE_DIR
    )

    print()
    print("=" * 60)
    print("Preparing VALIDATION")
    print("=" * 60)

    build_caption_data(
        val_data,
        val_images,
        OUTPUT_VAL_CAPTION,
        VAL_IMAGE_DIR
    )

    print()
    print("=" * 60)
    print("Preparing TEST")
    print("=" * 60)

    build_caption_data(
        val_data,
        test_images,
        OUTPUT_TEST_CAPTION,
        VAL_IMAGE_DIR
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)

    print()
    print("Output:")
    print(f"  {OUTPUT_TRAIN_CAPTION}")
    print(f"  {OUTPUT_VAL_CAPTION}")
    print(f"  {OUTPUT_TEST_CAPTION}")

    print()
    print("Image directories:")
    print(f"  {OUTPUT_TRAIN_IMAGE_DIR}")
    print(f"  {OUTPUT_VAL_IMAGE_DIR}")
    print(f"  {OUTPUT_TEST_IMAGE_DIR}")


if __name__ == "__main__":
    main()
