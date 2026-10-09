import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

IMAGE_DIR = RAW / "val2017" / "val2017"
ANN_DIR = RAW / "annotations" / "annotations"
CAPTIONS_FILE = ANN_DIR / "captions_val2017.json"
INSTANCES_FILE = ANN_DIR / "instances_val2017.json"

SEED = 42

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    output_files = [
        OUT / "train.jsonl",
        OUT / "val.jsonl",
        OUT / "test.jsonl",
        OUT / "label_map.json",
        OUT / "dataset_summary.json",
    ]
    existing = [str(p) for p in output_files if p.exists()]
    if existing:
        raise FileExistsError(
            "為避免覆蓋既有檔案，停止執行：\n" + "\n".join(existing)
        )

    for path in [CAPTIONS_FILE, INSTANCES_FILE]:
        if not path.is_file():
            raise FileNotFoundError(f"找不到標註檔：{path}")

    captions_data = load_json(CAPTIONS_FILE)
    instances_data = load_json(INSTANCES_FILE)

    image_dir = IMAGE_DIR
    if not image_dir.is_dir():
        raise FileNotFoundError(f"找不到圖片目錄：{image_dir}")

    # 建立 image_id -> image filename 對應
    images = {
        item["id"]: item["file_name"]
        for item in captions_data["images"]
    }

    # 每張圖片選擇一段 caption，使用固定排序以確保可重現
    caption_groups = {}
    for item in captions_data["annotations"]:
        caption_groups.setdefault(item["image_id"], []).append(item["caption"])

    selected_captions = {
        image_id: sorted(texts)[0]
        for image_id, texts in caption_groups.items()
        if texts
    }

    # 建立類別 ID 與類別名稱對應
    categories = sorted(
        instances_data["categories"],
        key=lambda item: item["id"]
    )
    category_map = {
        item["id"]: item["name"]
        for item in categories
    }
    label_map = {
        str(index): item["name"]
        for index, item in enumerate(categories)
    }
    category_to_index = {
        item["id"]: index
        for index, item in enumerate(categories)
    }

    # 一張圖片可以有多個標籤；同一類別只記錄一次
    labels_by_image = {}
    for item in instances_data["annotations"]:
        labels_by_image.setdefault(item["image_id"], set()).add(
            item["category_id"]
        )

    records = []
    missing = []
    skipped_no_labels = 0

    for image_id in sorted(images):
        filename = images[image_id]
        image_path = image_dir / filename

        if not image_path.is_file():
            missing.append(f"image:{image_id}")
            continue
        if image_id not in selected_captions:
            missing.append(f"caption:{image_id}")
            continue

        # 沒有物件標註的圖片不當成全負類樣本，直接排除
        if image_id not in labels_by_image or not labels_by_image[image_id]:
            skipped_no_labels += 1
            continue

        category_ids = sorted(labels_by_image[image_id])
        labels = sorted({
            category_to_index[cid] for cid in category_ids
        })

        records.append({
            "image_id": image_id,
            "image": str(
                Path("data") / "raw" / "val2017" / "val2017" / filename
            ),
            "caption": selected_captions[image_id],
            "labels": labels,
        })

    if missing:
        raise ValueError(
            f"有 {len(missing)} 筆資料缺少圖片或 caption。"
            f"範例：{missing[:10]}"
        )

    print(f"排除無物件標註圖片：{skipped_no_labels} 張")
    print(f"保留有效圖片：{len(records)} 張")

    # 依圖片 ID 隨機打散，避免同一張圖片出現在不同集合
    rng = random.Random(SEED)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }

    # 輸出 JSONL：每行一筆資料
    for split_name, split_records in splits.items():
        path = OUT / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (OUT / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # 檢查三個集合沒有圖片重疊
    split_ids = {
        name: {record["image_id"] for record in items}
        for name, items in splits.items()
    }
    assert split_ids["train"].isdisjoint(split_ids["val"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["val"].isdisjoint(split_ids["test"])
    assert sum(len(items) for items in splits.values()) == n

    label_counts = Counter()
    for record in records:
        label_counts.update(record["labels"])

    summary = {
        "dataset": "COCO 2017 validation images, internally split for coursework",
        "seed": SEED,
        "num_images": n,
        "num_classes": len(categories),
        "split_sizes": {
            name: len(items) for name, items in splits.items()
        },
        "split_proportions": {
            name: round(len(items) / n, 4) for name, items in splits.items()
        },
        "num_unique_image_ids": len(set(images) & {
            record["image_id"] for record in records
        }),
        "num_labels_per_image": {
            "min": min(len(record["labels"]) for record in records),
            "max": max(len(record["labels"]) for record in records),
            "mean": round(
                sum(len(record["labels"]) for record in records) / n, 3
            ),
        },
        "top_10_label_counts": [
            {
                "class_id": index,
                "class_name": label_map[str(index)],
                "image_count": label_counts[index],
            }
            for index, count in label_counts.most_common(10)
        ],
        "note": (
            "Splits are made by image_id. This is a small internal split "
            "of COCO val2017, not the official COCO test set."
        ),
    }

    with (OUT / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("資料集建立完成")
    print("=" * 60)
    print("總圖片數：", n)
    print("類別數：", len(categories))
    print("切分結果：", summary["split_sizes"])
    print("每張圖片標籤數：", summary["num_labels_per_image"])
    print("三個集合的圖片 ID 不重疊：通過")
    print("輸出目錄：", OUT)
    print("前 10 個常見類別：")
    for item in summary["top_10_label_counts"]:
        print(
            f"  {item['class_name']}: "
            f"{item['image_count']} 張圖片"
        )

if __name__ == "__main__":
    main()
