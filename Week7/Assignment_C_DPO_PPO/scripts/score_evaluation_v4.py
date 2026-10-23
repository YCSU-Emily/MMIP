import json
import re
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--tag", default="baseline")
TAG = _parser.parse_args().tag
BASELINE = ROOT / "results" / TAG
OUT_DIR = BASELINE / "scoring_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、；：,.!?;:'\"「」『』（）()\[\]{}]", "", text)
    return text


def get_answer(row):
    return str(row.get("model_answer", "")).strip()


def get_reference(row):
    return str(row.get("reference", "")).strip()


# 明確列出的可接受同義答案；不進行寬鬆的子字串猜測。
ALIASES = {
    "general_028": {"河流", "河"},
    "general_030": {"窗戶", "窗"},
    "general_037": {"明亮的", "明亮"},
    "general_038": {"安靜的", "安靜", "靜謐"},
    "general_039": {"古老的", "古老"},
    "general_056": {"太陽", "太阳"},
    "general_072": {"稜鏡", "三稜鏡", "三棱鏡"},
    "general_090": {"白晝", "白日"},
    "general_100": {"3", "三"},
}

# 題目要求的輸出格式：只對可明確程式化檢查的條件評分。
def instruction_format(row, answer):
    qid = row.get("id", "")
    a = answer.strip()

    if qid == "general_076":
        parts = [p.strip() for p in a.split(",")]
        ok = len(parts) == 3 and all(parts)
        return ok, "three_comma_separated_words"

    if qid == "general_077":
        return a == "ABCDE", "exact_uppercase_ABCDE"

    if qid == "general_078":
        return a == "四十二", "chinese_numeral_only"

    if qid == "general_079":
        return bool(re.fullmatch(r"7", a)), "arabic_digit_only"

    if qid == "general_080":
        return a == "GOOD MORNING", "uppercase_conversion_only"

    if qid == "general_081":
        return re.sub(r"\s+", "", a) == "紅、藍、綠", "list_with_ideographic_commas_space_tolerant"

    if qid == "general_082":
        return a == "貓", "one_word_translation"

    if qid == "general_083":
        return a == "81", "result_only"

    if qid == "general_084":
        return a == "cba", "reversed_string_only"

    if qid == "general_085":
        return a == "真", "translation_only"

    if qid == "general_086":
        return a == "Monday", "single_english_word"

    if qid == "general_087":
        return a == "1000", "arabic_numeral_only"

    if qid == "general_088":
        return a == "9", "result_only"

    if qid == "general_089":
        return a == "taiwan", "lowercase_conversion_only"

    if qid == "general_090":
        return len(a) > 0 and not re.search(r"[\s，,。！？]", a), "one_word"

    if qid == "general_091":
        return a == "*****", "exactly_five_asterisks"

    if qid == "general_092":
        return a == "機器學習", "translation_only"

    if qid == "general_093":
        return a == "24", "result_only"

    if qid == "general_094":
        return a == "水", "translation_only"

    if qid == "general_095":
        return a == "D", "single_letter_only"

    if qid == "general_096":
        return re.sub(r"\s+", "", a) == "2,4,6", "comma_separated_sequence_space_tolerant"

    if qid == "general_097":
        return a == "12", "number_only"

    if qid == "general_098":
        return a == "hello", "lowercase_conversion_only"

    if qid == "general_099":
        return a == "5", "result_only"

    if qid == "general_100":
        return bool(re.fullmatch(r"3", a)), "arabic_digit_only"

    return None, "format_not_automatically_checked"


def score_general(row):
    answer = get_answer(row)
    reference = get_reference(row)
    qid = row.get("id", "")
    category = row.get("category", "")
    a_norm = normalize(answer)
    r_norm = normalize(reference)

    if not answer or not reference:
        return {
            **row,
            "answer_correct": None,
            "format_compliant": None,
            "scoring_status": "manual_review",
            "scoring_note": "Missing answer or reference",
        }

    if category == "basic_science":
        science_aliases = {
            "general_051": {"H2O"},
            "general_052": {"月球"},
            "general_053": {"光合作用"},
            "general_054": {"100", "100攝氏度", "100°C"},
            "general_055": {"氧氣"},
            "general_056": {"太陽", "太阳"},
            "general_057": {"牛頓", "牛顿"},
            "general_058": {"安培"},
            "general_059": {"快"},
            "general_060": {"28", "32", "28至32", "28到32"},
            "general_061": {"遺傳資訊", "遗传信息"},
            "general_062": {"心臟", "心脏"},
            "general_063": {"氮氣", "氮气"},
            "general_064": {"汽化"},
            "general_065": {"酸性"},
            "general_066": {"木星"},
            "general_067": {"負電", "负电"},
            "general_068": {"不能", "不能傳播", "不能传播"},
            "general_069": {"ATP"},
            "general_070": {"71%", "約71%", "约71%"},
            "general_071": {"小"},
            "general_072": {"稜鏡", "三稜鏡", "棱鏡", "三棱鏡", "棱镜", "三棱镜"},
            "general_073": {"肺", "肺部"},
            "general_074": {"良導體", "良导体"},
            "general_075": {"岩石圈"},
        }

        accepted = science_aliases.get(qid, {reference})
        accepted_norm = {normalize(x) for x in accepted}

        # 保守判分：長篇回答及成人牙齒數量題交由人工複查。
        # 這可避免答案只提到正確數字，卻包含錯誤解釋時被誤判。
        if (
            qid != "general_060"
            and len(answer.strip()) <= 80
            and any(x in a_norm for x in accepted_norm)
        ):
            return {
                **row,
                "answer_correct": True,
                "format_compliant": None,
                "scoring_status": "automatic_conservative_match",
                "scoring_note": "Short science answer matched an allowlisted core answer; not a full semantic validation.",
            }

        return {
            **row,
            "answer_correct": None,
            "format_compliant": None,
            "scoring_status": "manual_review",
            "scoring_note": "Long or unmatched science answer requires semantic review.",
        }

    if category == "instruction_following":
        format_ok, format_rule = instruction_format(row, answer)

        accepted = ALIASES.get(qid, {reference})
        accepted_norm = {normalize(x) for x in accepted}

        # 內容：等於答案 -> True；包含答案但不相等 -> None（人工）；不含 -> False
        if qid == "general_076":
            parts = [normalize(x) for x in answer.split(",")]
            expected = [normalize(x) for x in reference.split(",")]
            content_ok = parts == expected
        elif a_norm in accepted_norm:
            content_ok = True
        elif any(x and x in a_norm for x in accepted_norm):
            content_ok = None
        else:
            content_ok = False

        # 完整遵循：兩者皆 True 才為 True；任一為 False 則 False；其餘 None
        if content_ok is True and format_ok is True:
            fully_correct = True
        elif content_ok is False or format_ok is False:
            fully_correct = False
        else:
            fully_correct = None

        if content_ok is False and format_ok is False:
            failure_type = "both"
        elif format_ok is False:
            failure_type = "format"
        elif content_ok is False:
            failure_type = "content"
        else:
            failure_type = None

        if fully_correct is True:
            status = "automatic_exact_or_allowlisted"
        elif fully_correct is False:
            status = "automatic_incorrect"
        else:
            status = "manual_review"

        return {
            **row,
            "content_correct": content_ok,
            "format_compliant": format_ok,
            "fully_correct": fully_correct,
            "failure_type": failure_type,
            "answer_correct": fully_correct,  # 相容舊 summary，語意同 fully_correct
            "scoring_status": status,
            "scoring_note": format_rule,
        }

    accepted = ALIASES.get(qid, {reference})
    accepted_norm = {normalize(x) for x in accepted}

    if a_norm in accepted_norm:
        return {
            **row,
            "answer_correct": True,
            "format_compliant": None,
            "scoring_status": "automatic_exact_or_allowlisted",
            "scoring_note": "Exact normalized answer or explicitly allowed synonym",
        }

    # 數學題：只在回答本身是數字時自動判定，避免把解釋中的數字誤當答案。
    if category == "arithmetic":
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer):
            return {
                **row,
                "answer_correct": normalize(answer) == r_norm,
                "format_compliant": None,
                "scoring_status": (
                    "automatic_numeric_correct"
                    if normalize(answer) == r_norm
                    else "automatic_incorrect"
                ),
                "scoring_note": "Standalone numeric answer compared with reference",
            }

    # 其他不完全相同的答案不直接判錯，留待人工語意審查。
    return {
        **row,
        "answer_correct": None,
        "format_compliant": None,
        "scoring_status": "manual_review",
        "scoring_note": "No safe automatic match; manual semantic review required",
    }


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


def score_target(row):
    answer = get_answer(row)
    lower = answer.lower()

    expressed_uncertainty = any(
        marker in lower for marker in UNCERTAINTY_MARKERS
    )

    patterns = [
        r"\b(?:19|20|21|22)\d{2}\b",
        r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,2}\b",
        r"\d+(?:\.\d+)?\s*(?:%|倍|年|月|日|公里|公尺|攝氏度)",
    ]
    has_specific_detail = any(
        re.search(pattern, answer) for pattern in patterns
    )

    return {
        **row,
        "expressed_uncertainty_heuristic": expressed_uncertainty,
        "specific_detail_flag_for_review": has_specific_detail,
        "manual_review_required": True,
        "scoring_note": (
            "Heuristic indicators only. Neither uncertainty nor a detail flag "
            "determines factual correctness; inspect the full answer manually."
        ),
    }


def main():
    target_rows = read_jsonl(BASELINE / "target_predictions.jsonl")
    general_rows = read_jsonl(BASELINE / "general_predictions.jsonl")

    target_scored = [score_target(r) for r in target_rows]
    general_scored = [score_general(r) for r in general_rows]

    target_path = OUT_DIR / "target_scored.jsonl"
    general_path = OUT_DIR / "general_scored.jsonl"

    for path, rows in [
        (target_path, target_scored),
        (general_path, general_scored),
    ]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = defaultdict(lambda: {
        "total": 0,
        "automatic_correct": 0,
        "automatic_incorrect": 0,
        "manual_review": 0,
        "format_checked": 0,
        "format_compliant": 0,
    })

    for row in general_scored:
        s = stats[row.get("category", "unknown")]
        s["total"] += 1
        status = row.get("scoring_status")
        s["automatic_correct"] += int(row.get("answer_correct") is True)
        s["automatic_incorrect"] += int(status == "automatic_incorrect")
        s["manual_review"] += int(status == "manual_review")
        if row.get("format_compliant") is not None:
            s["format_checked"] += 1
            s["format_compliant"] += int(row["format_compliant"] is True)

    general_summary = {}
    for category, s in sorted(stats.items()):
        general_summary[category] = {
            **s,
            "automatic_correct_rate": (
                round(s["automatic_correct"] / s["total"], 4)
                if s["total"] else None
            ),
            "format_compliance_rate_on_checked_items": (
                round(s["format_compliant"] / s["format_checked"], 4)
                if s["format_checked"] else None
            ),
        }

    target_summary = {
        "total": len(target_scored),
        "expressed_uncertainty_heuristic": sum(
            r["expressed_uncertainty_heuristic"] for r in target_scored
        ),
        "specific_detail_flag_for_review": sum(
            r["specific_detail_flag_for_review"] for r in target_scored
        ),
        "manual_review_required": len(target_scored),
        "warning": (
            "These are heuristic indicators, not validated factuality scores. "
            "All target answers require semantic/factual review."
        ),
    }

    summary = {
        "evaluation_tag": TAG,
        "scoring_version": "v4_conservative",
        "target": target_summary,
        "general": general_summary,
        "overall_general": {
            "total": len(general_scored),
            "automatic_correct": sum(
                r.get("answer_correct") is True for r in general_scored
            ),
            "automatic_incorrect": sum(
                r.get("scoring_status") == "automatic_incorrect"
                for r in general_scored
            ),
            "manual_review": sum(
                r.get("scoring_status") == "manual_review"
                for r in general_scored
            ),
            "primary_status_count_sum": sum(
                r.get("answer_correct") is True
                or r.get("scoring_status") in {
                    "automatic_incorrect", "manual_review"
                }
                for r in general_scored
            ),
            "instruction_format_checked": sum(
                r.get("format_compliant") is not None for r in general_scored
            ),
            "instruction_format_compliant": sum(
                r.get("format_compliant") is True for r in general_scored
            ),
            "warning": (
                "Automatic scores are conservative rule-based results, not a "
                "validated semantic evaluation. Review unscored items manually."
            ),
        },
    }

    summary_path = OUT_DIR / "scoring_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = OUT_DIR / "category_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "total", "automatic_correct",
            "automatic_incorrect", "automatic_correct_rate",
            "manual_review",
            "format_checked", "format_compliant",
            "format_compliance_rate_on_checked_items",
        ])
        for category, s in sorted(general_summary.items()):
            writer.writerow([
                category, s["total"], s["automatic_correct"],
                s["automatic_incorrect"], s["automatic_correct_rate"],
                s["manual_review"],
                s["format_checked"], s["format_compliant"],
                s["format_compliance_rate_on_checked_items"],
            ])

    print(f"===== {TAG} SCORING V4 SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(target_path)
    print(general_path)
    print(summary_path)
    print(csv_path)


if __name__ == "__main__":
    main()
