import json
import re
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/baseline"
OUT_DIR = BASELINE / "scoring"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def normalize(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、；：,.!?;:'\"「」『』（）()\[\]{}]", "", text)
    return text

# 這些詞只用於初步偵測「有表達不確定或指出前提不成立」。
UNCERTAINTY_MARKERS = [
    "沒有證據", "沒有可信證據", "沒有可靠證據",
    "目前沒有", "尚未", "並不存在", "不存在",
    "無法確認", "無法確定", "無法提供",
    "沒有資料", "沒有紀錄", "未有紀錄",
    "前提不成立", "資訊不足", "無法判斷",
    "不清楚", "不能確認", "沒有可信紀錄",
    "no evidence", "does not exist", "do not exist",
    "cannot confirm", "unable to verify", "not enough information",
]

# 一般能力評分：先使用標準答案正規化後的完全相同或包含比對。
# 此方法透明但不等於完整語意評估。
def general_correct(row):
    answer = normalize(row.get("model_answer", ""))
    reference = normalize(row.get("reference", ""))

    if not answer or not reference:
        return False, "empty"

    if answer == reference:
        return True, "exact"

    # 短答案可安全使用包含比對，例如數字或單字。
    if len(reference) <= 24 and reference in answer:
        return True, "reference_in_answer"

    return False, "needs_review"

def score_target(row):
    answer = row.get("model_answer", "")
    lower = answer.lower()

    expressed_uncertainty = any(
        marker in lower for marker in UNCERTAINTY_MARKERS
    )

    # 不把「有表達不確定」當作完全正確。
    # 只做一個需要人工複查的風險標記。
    specific_detail_patterns = [
        r"\b(?:19|20|21|22)\d{2}\b",
        r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,2}\b",
        r"\d+(?:\.\d+)?\s*(?:%|倍|年|月|日|公里|公尺|攝氏度)",
    ]
    has_specific_detail = any(
        re.search(pattern, answer) for pattern in specific_detail_patterns
    )

    return {
        **row,
        "expressed_uncertainty_heuristic": expressed_uncertainty,
        "specific_detail_flag_for_review": has_specific_detail,
        "automatic_target_pass": expressed_uncertainty and not has_specific_detail,
        "scoring_note": (
            "Heuristic only; manually inspect factual claims and the question premise."
        ),
    }

def main():
    target_rows = read_jsonl(BASELINE / "target_predictions.jsonl")
    general_rows = read_jsonl(BASELINE / "general_predictions.jsonl")

    target_scored = [score_target(row) for row in target_rows]

    target_path = OUT_DIR / "target_scored.jsonl"
    with target_path.open("w", encoding="utf-8") as f:
        for row in target_scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    general_scored = []
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0, "needs_review": 0})

    for row in general_rows:
        correct, method = general_correct(row)
        scored = {
            **row,
            "automatic_correct": correct,
            "scoring_method": method,
        }
        general_scored.append(scored)

        stats = category_stats[row["category"]]
        stats["total"] += 1
        stats["correct"] += int(correct)
        stats["needs_review"] += int(method == "needs_review")

    general_path = OUT_DIR / "general_scored.jsonl"
    with general_path.open("w", encoding="utf-8") as f:
        for row in general_scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    target_summary = {
        "total": len(target_scored),
        "expressed_uncertainty_heuristic": sum(
            r["expressed_uncertainty_heuristic"] for r in target_scored
        ),
        "specific_detail_flag_for_review": sum(
            r["specific_detail_flag_for_review"] for r in target_scored
        ),
        "automatic_target_pass": sum(
            r["automatic_target_pass"] for r in target_scored
        ),
        "warning": (
            "These are heuristic indicators, not validated semantic or factual scores. "
            "Manual review is required, especially for answers with specific-detail flags."
        ),
    }

    general_summary = {}
    for category, stats in sorted(category_stats.items()):
        total = stats["total"]
        general_summary[category] = {
            **stats,
            "automatic_accuracy": round(stats["correct"] / total, 4) if total else 0,
        }

    summary = {
        "evaluation_tag": "baseline",
        "target": target_summary,
        "general": general_summary,
        "overall_general": {
            "total": len(general_scored),
            "automatic_correct": sum(r["automatic_correct"] for r in general_scored),
            "automatic_accuracy": round(
                sum(r["automatic_correct"] for r in general_scored) / len(general_scored), 4
            ) if general_scored else 0,
            "needs_manual_review": sum(
                r["scoring_method"] == "needs_review" for r in general_scored
            ),
        },
    }

    summary_path = OUT_DIR / "scoring_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV 摘要：方便放進報告或試算表。
    csv_path = OUT_DIR / "category_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "category", "total", "automatic_correct",
                         "automatic_accuracy", "needs_review"])
        for category, stats in sorted(category_stats.items()):
            writer.writerow([
                "general", category, stats["total"], stats["correct"],
                round(stats["correct"] / stats["total"], 4) if stats["total"] else 0,
                stats["needs_review"],
            ])
        writer.writerow([
            "target", "all_target_categories", target_summary["total"],
            target_summary["automatic_target_pass"],
            round(
                target_summary["automatic_target_pass"] / target_summary["total"], 4
            ) if target_summary["total"] else 0,
            target_summary["specific_detail_flag_for_review"],
        ])

    print("===== BASELINE SCORING SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(target_path)
    print(general_path)
    print(summary_path)
    print(csv_path)

if __name__ == "__main__":
    main()
