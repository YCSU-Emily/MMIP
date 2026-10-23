import csv
import random
from pathlib import Path

ROOT = Path.cwd()
AUDIT = ROOT / "results/eval_dpo_lora_v1/paired_manual_audit.csv"
SCREEN = ROOT / "results/auto_screen_v1/automatic_screening.csv"
OUT = ROOT / "results/auto_screen_v1/manual_review_sample_20.csv"

# 這 10 筆是根據前面初篩結果挑選的高資訊量案例。
# 包含物件種類、動作、位置或細節的改變。
PRIORITY_IDS = [
    "270386", "47010", "229997", "64868", "132408",
    "52017", "8211", "2149", "4795", "7108",
]

SEED = 20261024

def main():
    for p in [AUDIT, SCREEN]:
        if not p.is_file():
            raise FileNotFoundError(f"找不到檔案：{p}")

    with AUDIT.open(encoding="utf-8-sig", newline="") as f:
        audit_rows = list(csv.DictReader(f))

    with SCREEN.open(encoding="utf-8-sig", newline="") as f:
        screen_rows = list(csv.DictReader(f))

    audit_by_id = {str(r["image_id"]): r for r in audit_rows}
    screen_by_id = {str(r["image_id"]): r for r in screen_rows}

    missing = [i for i in PRIORITY_IDS if i not in audit_by_id]
    if missing:
        raise ValueError(f"以下指定 ID 不在審查資料中：{missing}")

    priority = [audit_by_id[i] for i in PRIORITY_IDS]

    already_selected = set(PRIORITY_IDS)
    remaining = [
        r for r in audit_rows
        if str(r["image_id"]) not in already_selected
    ]

    rng = random.Random(SEED)
    random_rows = rng.sample(remaining, min(10, len(remaining)))

    fields = [
        "sample_group",
        "image_id",
        "image",
        "coco_objects",
        "reference_captions",
        "baseline_prediction",
        "dpo_prediction",
        "baseline_overall_grounded",
        "dpo_overall_grounded",
        "baseline_hallucinated_object",
        "dpo_hallucinated_object",
        "baseline_unsupported_action_or_attribute",
        "dpo_unsupported_action_or_attribute",
        "baseline_important_object_omitted",
        "dpo_important_object_omitted",
        "baseline_severity",
        "dpo_severity",
        "which_is_better",
        "review_notes",
        "screening_priority",
        "screening_reasons",
    ]

    result = []

    for group, rows in [
        ("high_priority", priority),
        ("random_sample", random_rows),
    ]:
        for r in rows:
            image_id = str(r["image_id"])
            s = screen_by_id.get(image_id, {})
            item = {
                "sample_group": group,
                "image_id": image_id,
                "image": r.get("image", ""),
                "coco_objects": r.get("coco_objects", ""),
                "reference_captions": r.get("reference_captions", ""),
                "baseline_prediction": r.get("baseline_prediction", ""),
                "dpo_prediction": r.get("dpo_prediction", ""),
                "baseline_overall_grounded": "",
                "dpo_overall_grounded": "",
                "baseline_hallucinated_object": "",
                "dpo_hallucinated_object": "",
                "baseline_unsupported_action_or_attribute": "",
                "dpo_unsupported_action_or_attribute": "",
                "baseline_important_object_omitted": "",
                "dpo_important_object_omitted": "",
                "baseline_severity": "",
                "dpo_severity": "",
                "which_is_better": "",
                "review_notes": "",
                "screening_priority": s.get("review_priority", ""),
                "screening_reasons": s.get("screening_reasons", ""),
            }
            result.append(item)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)

    print(f"Created: {OUT}")
    print(f"Total: {len(result)}")
    print(f"High-priority cases: {len(priority)}")
    print(f"Random sample: {len(random_rows)}")
    print(f"Random seed: {SEED}")
    print("\nReview groups:")
    for r in result:
        print(f"{r['sample_group']:14} ID={r['image_id']}")

if __name__ == "__main__":
    main()
