import csv
import json
from pathlib import Path

BASE = Path.home() / "MMIP" / "Week7" / "Assignment_A_LLM"
RESULTS = BASE / "results"

FILES = {
    "FP16": RESULTS / "fp16_instruction_results.json",
    "4-bit NF4": RESULTS / "4bit_instruction_results.json",
}

all_results = {}

for label, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"找不到結果檔案：{path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results", [])

    if len(rows) != 20:
        raise ValueError(f"{label} 應有 20 題，實際為 {len(rows)} 題")

    all_results[label] = {
        "metadata": data,
        "rows": rows,
    }

# 確認兩組題目一致，避免錯誤比較。
fp16_by_id = {r["id"]: r for r in all_results["FP16"]["rows"]}
int4_by_id = {r["id"]: r for r in all_results["4-bit NF4"]["rows"]}

if set(fp16_by_id) != set(int4_by_id):
    raise ValueError("FP16 和 4-bit 的題目 ID 不一致")

def coverage(row):
    keywords = row.get("evaluation_keywords", [])
    answer = row.get("model_answer", "")
    if not keywords:
        return None, [], 0
    matched = [kw for kw in keywords if kw in answer]
    return len(matched) / len(keywords), matched, len(keywords)

def summarize(rows):
    coverages = [coverage(r)[0] for r in rows]
    coverages = [x for x in coverages if x is not None]

    total_tokens = sum(r.get("generated_tokens", 0) for r in rows)
    total_time = sum(r.get("latency_seconds", 0) for r in rows)

    return {
        "questions": len(rows),
        "average_keyword_coverage": (
            sum(coverages) / len(coverages) if coverages else None
        ),
        "average_latency_seconds": (
            sum(r.get("latency_seconds", 0) for r in rows) / len(rows)
        ),
        "total_generated_tokens": total_tokens,
        "overall_tokens_per_second": total_tokens / total_time if total_time else 0,
        "responses_at_token_limit": sum(
            bool(r.get("hit_max_new_tokens", False)) for r in rows
        ),
        "average_peak_allocated_gib": (
            sum(r["peak_allocated_gib"] for r in rows if "peak_allocated_gib" in r)
            / sum("peak_allocated_gib" in r for r in rows)
            if any("peak_allocated_gib" in r for r in rows) else None
        ),
        "max_peak_allocated_gib": (
            max(r["peak_allocated_gib"] for r in rows if "peak_allocated_gib" in r)
            if any("peak_allocated_gib" in r for r in rows) else None
        ),
    }

summaries = {
    label: summarize(data["rows"])
    for label, data in all_results.items()
}

print("=" * 82)
print("FP16 vs 4-bit NF4: 20-question instruction benchmark")
print("=" * 82)

for label, summary in summaries.items():
    print(f"\n[{label}]")
    print(f"  Questions: {summary['questions']}")
    print(f"  Average keyword coverage: {summary['average_keyword_coverage']:.2%}")
    print(f"  Average latency: {summary['average_latency_seconds']:.4f} sec/question")
    print(f"  Total generated tokens: {summary['total_generated_tokens']}")
    print(f"  Overall generation speed: {summary['overall_tokens_per_second']:.2f} tokens/sec")
    print(f"  Responses at token limit: {summary['responses_at_token_limit']}")

    if summary["average_peak_allocated_gib"] is not None:
        print(f"  Average measured peak allocated memory: {summary['average_peak_allocated_gib']:.3f} GiB")
        print(f"  Maximum measured peak allocated memory: {summary['max_peak_allocated_gib']:.3f} GiB")
    else:
        print("  Per-question peak memory: not recorded in this result file")

fp16 = summaries["FP16"]
q4 = summaries["4-bit NF4"]

def pct_change(new, old):
    if old == 0:
        return None
    return (new - old) / old * 100

print("\n" + "-" * 82)
print("4-bit change relative to FP16")
print("-" * 82)

latency_change = pct_change(
    q4["average_latency_seconds"], fp16["average_latency_seconds"]
)
speed_change = pct_change(
    q4["overall_tokens_per_second"], fp16["overall_tokens_per_second"]
)
coverage_diff_pp = (
    q4["average_keyword_coverage"] - fp16["average_keyword_coverage"]
) * 100

print(f"Keyword coverage difference: {coverage_diff_pp:+.2f} percentage points")
print(f"Average latency change: {latency_change:+.2f}%")
print(f"Generation speed change: {speed_change:+.2f}%")
print(
    "Token-limit responses: "
    f"{fp16['responses_at_token_limit']} (FP16) -> "
    f"{q4['responses_at_token_limit']} (4-bit)"
)

per_question = []
for qid in sorted(fp16_by_id):
    a = fp16_by_id[qid]
    b = int4_by_id[qid]
    cov_a, matched_a, count_a = coverage(a)
    cov_b, matched_b, count_b = coverage(b)

    if a["prompt"] != b["prompt"]:
        raise ValueError(f"ID={qid} 的 prompt 不一致")
    if a.get("evaluation_keywords", []) != b.get("evaluation_keywords", []):
        raise ValueError(f"ID={qid} 的評估關鍵字不一致")

    per_question.append({
        "id": qid,
        "category": a["category"],
        "fp16_coverage": cov_a,
        "4bit_coverage": cov_b,
        "fp16_matched_keywords": matched_a,
        "4bit_matched_keywords": matched_b,
        "keyword_count": count_a,
        "fp16_latency_seconds": a["latency_seconds"],
        "4bit_latency_seconds": b["latency_seconds"],
        "fp16_generated_tokens": a["generated_tokens"],
        "4bit_generated_tokens": b["generated_tokens"],
        "fp16_hit_max_tokens": a.get("hit_max_new_tokens", False),
        "4bit_hit_max_tokens": b.get("hit_max_new_tokens", False),
    })

print("\n" + "-" * 82)
print("PER-QUESTION KEYWORD COVERAGE")
print(f"{'ID':>3}  {'Category':<12} {'FP16':>9} {'4-bit':>9} {'Difference':>12}")
print("-" * 82)

for row in per_question:
    a = row["fp16_coverage"]
    b = row["4bit_coverage"]
    a_text = f"{a:.0%}" if a is not None else "N/A"
    b_text = f"{b:.0%}" if b is not None else "N/A"
    diff = f"{(b-a)*100:+.1f} pp" if a is not None and b is not None else "N/A"
    print(f"{row['id']:>3}  {row['category']:<12} {a_text:>9} {b_text:>9} {diff:>12}")

report = {
    "description": (
        "FP16 versus 4-bit NF4 on the same 20 prompts. "
        "Keyword coverage is a simple string-match proxy, not semantic accuracy. "
        "Memory fields are only compared when recorded by the respective run."
    ),
    "summaries": summaries,
    "per_question": [
        {
            **row,
            "fp16_coverage": round(row["fp16_coverage"], 4)
                if row["fp16_coverage"] is not None else None,
            "4bit_coverage": round(row["4bit_coverage"], 4)
                if row["4bit_coverage"] is not None else None,
        }
        for row in per_question
    ],
}

json_path = RESULTS / "fp16_vs_4bit_comparison.json"
json_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

csv_path = RESULTS / "fp16_vs_4bit_per_question.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    columns = list(per_question[0].keys())
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(per_question)

print("\nSaved comparison files:")
print(json_path)
print(csv_path)
print("=" * 82)
