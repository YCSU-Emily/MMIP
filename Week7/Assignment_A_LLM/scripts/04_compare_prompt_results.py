import json
from pathlib import Path

BASE = Path.home() / "MMIP" / "Week7" / "Assignment_A_LLM"
FILES = {
    "Baseline": BASE / "results" / "fp16_instruction_results.json",
    "Optimized": BASE / "results" / "optimized_prompt_results.json",
}

all_results = {}

for name, path in FILES.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]

    evaluated = []
    for row in rows:
        keywords = row.get("evaluation_keywords", [])
        answer = row.get("model_answer", "")
        matched = [kw for kw in keywords if kw in answer]
        coverage = len(matched) / len(keywords) if keywords else None

        evaluated.append({
            "id": row["id"],
            "category": row["category"],
            "coverage": coverage,
            "matched_keywords": matched,
            "keyword_count": len(keywords),
            "latency_seconds": row["latency_seconds"],
            "generated_tokens": row["generated_tokens"],
            "hit_max_new_tokens": row.get("hit_max_new_tokens", False),
        })

    all_results[name] = evaluated

print("=" * 76)
print("PROMPT COMPARISON: SAME KEYWORD RULES")
print("=" * 76)

for name, rows in all_results.items():
    valid = [r for r in rows if r["coverage"] is not None]
    avg_coverage = sum(r["coverage"] for r in valid) / len(valid)
    avg_latency = sum(r["latency_seconds"] for r in rows) / len(rows)
    total_tokens = sum(r["generated_tokens"] for r in rows)
    total_time = sum(r["latency_seconds"] for r in rows)
    speed = total_tokens / total_time if total_time else 0
    truncated = sum(r["hit_max_new_tokens"] for r in rows)

    print(f"\n{name}")
    print(f"  Questions: {len(rows)}")
    print(f"  Average keyword coverage: {avg_coverage:.2%}")
    print(f"  Average latency: {avg_latency:.4f} sec/question")
    print(f"  Overall generation speed: {speed:.2f} tokens/sec")
    print(f"  Responses at token limit: {truncated}")

print("\n" + "-" * 76)
print("PER-QUESTION KEYWORD COVERAGE")
print(f"{'ID':>3}  {'Category':<12} {'Baseline':>10} {'Optimized':>10}")
print("-" * 76)

base = {r["id"]: r for r in all_results["Baseline"]}
opt = {r["id"]: r for r in all_results["Optimized"]}

for qid in sorted(base):
    b, o = base[qid], opt[qid]
    b_cov = f"{b['coverage']:.0%}" if b["coverage"] is not None else "N/A"
    o_cov = f"{o['coverage']:.0%}" if o["coverage"] is not None else "N/A"
    print(f"{qid:>3}  {b['category']:<12} {b_cov:>10} {o_cov:>10}")

output = {
    "description": "Keyword coverage recalculated with identical rules for both prompt settings.",
    "results": {
        name: [
            {
                **row,
                "coverage": round(row["coverage"], 4)
                    if row["coverage"] is not None else None
            }
            for row in rows
        ]
        for name, rows in all_results.items()
    },
}

out_path = BASE / "results" / "prompt_comparison.json"
out_path.write_text(
    json.dumps(output, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"\nSaved: {out_path}")
