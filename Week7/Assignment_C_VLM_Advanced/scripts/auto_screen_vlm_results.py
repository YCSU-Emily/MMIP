import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/eval_dpo_lora_v1/paired_manual_audit.csv"
OUT_DIR = ROOT / "results/auto_screen_v1"
OUT_CSV = OUT_DIR / "automatic_screening.csv"
OUT_JSON = OUT_DIR / "screening_summary.json"

# 常見動物名稱與 COCO 標註名稱的對照。
# 這只是初篩詞彙表，不是完整的物件辨識器。
ANIMAL_ALIASES = {
    "elephant": ["elephant", "elephants"],
    "giraffe": ["giraffe", "giraffes"],
    "cow": ["cow", "cows", "cattle"],
    "cat": ["cat", "cats", "kitten", "kittens"],
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "bird": ["bird", "birds"],
    "horse": ["horse", "horses"],
    "sheep": ["sheep"],
    "zebra": ["zebra", "zebras"],
    "bear": ["bear", "bears"],
    "deer": ["deer"],
    "elephant": ["elephant", "elephants"],
}

def normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def mentions(text, words):
    text = normalize(text)
    return any(
        re.search(r"\b" + re.escape(word) + r"\b", text)
        for word in words
    )

def find_animals(text):
    return [
        animal for animal, aliases in ANIMAL_ALIASES.items()
        if mentions(text, aliases)
    ]

def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing input CSV: {INPUT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Input CSV is empty")

    output = []
    changed_count = 0
    identical_count = 0
    animal_mismatch_count = 0

    for row in rows:
        baseline = row.get("baseline_prediction", "") or ""
        dpo = row.get("dpo_prediction", "") or ""
        objects = [
            x.strip().lower()
            for x in (row.get("coco_objects", "") or "").split(";")
            if x.strip()
        ]

        same = normalize(baseline) == normalize(dpo)
        if same:
            identical_count += 1
        else:
            changed_count += 1

        base_animals = find_animals(baseline)
        dpo_animals = find_animals(dpo)

        # 只有在兩個描述的動物種類不一致時才標記；
        # 不能單靠 COCO 標註缺少某類，就斷定是幻覺。
        animal_changed = set(base_animals) != set(dpo_animals)
        if animal_changed:
            animal_mismatch_count += 1

        # 依差異與動物種類變化排序，分數只代表審查優先度。
        priority = 0
        reasons = []

        if not same:
            priority += 1
            reasons.append("描述有變化")

        if animal_changed:
            priority += 3
            reasons.append("描述中的動物種類有變化")

        # 動物名稱與 COCO 標註不一致時，列為待核對線索，
        # 而不是判定錯誤。
        for animal in set(base_animals + dpo_animals):
            aliases = ANIMAL_ALIASES[animal]
            if mentions(" ".join(objects), aliases):
                continue
            if mentions(dpo, aliases):
                priority += 1
                reasons.append(f"DPO 提及 {animal}，COCO 物件標註未列出")
            elif mentions(baseline, aliases):
                priority += 1
                reasons.append(f"Baseline 提及 {animal}，COCO 物件標註未列出")

        if same:
            reasons.append("兩個描述相同，可降低審查優先度")

        output.append({
            "review_priority": priority,
            "image_id": row.get("image_id", ""),
            "image": row.get("image", ""),
            "baseline_prediction": baseline,
            "dpo_prediction": dpo,
            "coco_objects": row.get("coco_objects", ""),
            "baseline_animals_detected_by_keywords": "; ".join(base_animals),
            "dpo_animals_detected_by_keywords": "; ".join(dpo_animals),
            "baseline_vs_dpo_identical": same,
            "animal_mentions_changed": animal_changed,
            "screening_reasons": "; ".join(dict.fromkeys(reasons)),
            "human_verdict": "待人工確認",
        })

    # 高優先度排前面，同分時先排描述不同者。
    output.sort(
        key=lambda r: (
            -r["review_priority"],
            r["baseline_vs_dpo_identical"],
            str(r["image_id"]),
        )
    )

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)

    summary = {
        "input_file": str(INPUT.relative_to(ROOT)),
        "output_csv": str(OUT_CSV.relative_to(ROOT)),
        "num_images": len(output),
        "identical_descriptions": identical_count,
        "changed_descriptions": changed_count,
        "animal_mention_changes": animal_mismatch_count,
        "high_priority_cases": sum(
            r["review_priority"] >= 3 for r in output
        ),
        "important_limitations": [
            "這是關鍵字初篩，不是可靠的幻覺判定器。",
            "COCO 物件標註可能漏掉影像中實際存在的物體。",
            "分數只代表人工審查優先度，不代表模型品質。",
            "最後的品質判斷仍需查看原圖。"
        ],
    }

    OUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("===== Automatic screening complete =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nCSV: {OUT_CSV}")
    print(f"Summary: {OUT_JSON}")
    print("\n===== Top 15 cases for review =====")

    for row in output[:15]:
        print(
            f"\nID={row['image_id']} "
            f"priority={row['review_priority']}"
        )
        print("Reason:", row["screening_reasons"])
        print("Baseline:", row["baseline_prediction"])
        print("DPO:", row["dpo_prediction"])

if __name__ == "__main__":
    main()
