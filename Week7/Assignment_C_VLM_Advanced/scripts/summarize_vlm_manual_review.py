import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
INPUT = ROOT / "results/auto_screen_v1/manual_review_sample_20_scored.csv"
OUT_JSON = ROOT / "results/auto_screen_v1/manual_review_summary.json"
OUT_TXT = ROOT / "results/auto_screen_v1/manual_review_summary.txt"

if not INPUT.is_file():
    raise FileNotFoundError(f"找不到評分檔案：{INPUT}")

with INPUT.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("評分 CSV 沒有資料")

def counts_for(subset, field):
    return dict(Counter(
        (r.get(field) or "").strip()
        for r in subset
        if (r.get(field) or "").strip()
    ))

def rate_yes(subset, field):
    vals = [(r.get(field) or "").strip().lower() for r in subset]
    valid = [v for v in vals if v in ("yes", "no")]
    return {
        "yes": valid.count("yes"),
        "no": valid.count("no"),
        "rated": len(valid),
    }

def summarize_group(subset):
    winners = counts_for(subset, "which_is_better")
    result = {
        "n": len(subset),
        "baseline_groundedness": counts_for(
            subset, "baseline_overall_grounded"
        ),
        "dpo_groundedness": counts_for(
            subset, "dpo_overall_grounded"
        ),
        "baseline_hallucinated_object": rate_yes(
            subset, "baseline_hallucinated_object"
        ),
        "dpo_hallucinated_object": rate_yes(
            subset, "dpo_hallucinated_object"
        ),
        "baseline_unsupported_action_or_attribute": rate_yes(
            subset, "baseline_unsupported_action_or_attribute"
        ),
        "dpo_unsupported_action_or_attribute": rate_yes(
            subset, "dpo_unsupported_action_or_attribute"
        ),
        "baseline_important_object_omitted": rate_yes(
            subset, "baseline_important_object_omitted"
        ),
        "dpo_important_object_omitted": rate_yes(
            subset, "dpo_important_object_omitted"
        ),
        "baseline_severity": counts_for(subset, "baseline_severity"),
        "dpo_severity": counts_for(subset, "dpo_severity"),
        "pairwise_preference": winners,
    }

    # 只有在兩邊都有有效評分時才計算配對改善/退步。
    grounded_order = {"good": 0, "minor_issue": 1, "major_issue": 2}
    paired = []
    for r in subset:
        b = (r.get("baseline_overall_grounded") or "").strip()
        d = (r.get("dpo_overall_grounded") or "").strip()
        if b in grounded_order and d in grounded_order:
            paired.append((b, d))

    result["paired_groundedness"] = {
        "rated_pairs": len(paired),
        "dpo_better": sum(
            grounded_order[d] < grounded_order[b] for b, d in paired
        ),
        "same": sum(b == d for b, d in paired),
        "dpo_worse": sum(
            grounded_order[d] > grounded_order[b] for b, d in paired
        ),
    }
    return result

groups = {
    "all": rows,
    "high_priority": [
        r for r in rows if r.get("sample_group") == "high_priority"
    ],
    "random_sample": [
        r for r in rows if r.get("sample_group") == "random_sample"
    ],
}

summary = {
    "input": str(INPUT.relative_to(ROOT)),
    "total_rows": len(rows),
    "groups": {name: summarize_group(subset)
               for name, subset in groups.items()},
    "caveats": [
        "高優先組是依初篩疑點選出的，不能代表整體資料分布。",
        "隨機組只有 10 張，結果僅適合探索性分析。",
        "人工評分結果不等於統計顯著性檢定。",
        "若只有單人評分，尚未估計評分者間一致性。",
    ],
}

OUT_JSON.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "VLM Baseline vs DPO — Manual Review Summary",
    f"Total images: {len(rows)}",
    "",
]

for name, result in summary["groups"].items():
    lines.extend([
        f"=== {name} (n={result['n']}) ===",
        "Baseline groundedness: " + json.dumps(
            result["baseline_groundedness"], ensure_ascii=False
        ),
        "DPO groundedness: " + json.dumps(
            result["dpo_groundedness"], ensure_ascii=False
        ),
        "Baseline hallucinated objects: " + json.dumps(
            result["baseline_hallucinated_object"], ensure_ascii=False
        ),
        "DPO hallucinated objects: " + json.dumps(
            result["dpo_hallucinated_object"], ensure_ascii=False
        ),
        "Baseline unsupported action/attribute: " + json.dumps(
            result["baseline_unsupported_action_or_attribute"],
            ensure_ascii=False
        ),
        "DPO unsupported action/attribute: " + json.dumps(
            result["dpo_unsupported_action_or_attribute"],
            ensure_ascii=False
        ),
        "Baseline important object omitted: " + json.dumps(
            result["baseline_important_object_omitted"],
            ensure_ascii=False
        ),
        "DPO important object omitted: " + json.dumps(
            result["dpo_important_object_omitted"],
            ensure_ascii=False
        ),
        "Pairwise preference: " + json.dumps(
            result["pairwise_preference"], ensure_ascii=False
        ),
        "Paired groundedness: " + json.dumps(
            result["paired_groundedness"], ensure_ascii=False
        ),
        "",
    ])

lines.extend(["=== Caveats ==="] + ["- " + x for x in summary["caveats"]])
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n".join(lines))
print(f"\nSaved JSON: {OUT_JSON}")
print(f"Saved TXT:  {OUT_TXT}")
