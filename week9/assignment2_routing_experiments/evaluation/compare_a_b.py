import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

A_PATH = ROOT / "results/experiment_a_v3_results.json"
B_PATH = (
    ROOT / "experiments/B_sft_rlhf/results/"
    "benchmark_v1/experiment_b_results.json"
)

OUT = ROOT / "results/comparison"

a_results = json.loads(A_PATH.read_text(encoding="utf-8"))
b_results = json.loads(B_PATH.read_text(encoding="utf-8"))

a_by_id = {x["id"]: x for x in a_results}
b_by_id = {x["id"]: x for x in b_results}

assert set(a_by_id) == set(b_by_id), "Benchmark IDs do not match"

metrics = [
    "intent",
    "product",
    "image_type",
    "department",
    "action",
    "clarification",
]

rows = []

for case_id in sorted(a_by_id):
    a = a_by_id[case_id]
    b = b_by_id[case_id]

    row = {
        "id": case_id,
        "A_latency_ms": a["latency_ms"],
        "B_latency_ms": b["latency_ms"],
        "A_output_tokens": a["output_tokens"],
        "B_output_tokens": b["output_tokens"],
    }

    for metric in metrics:
        row[f"A_{metric}"] = a["scores"][metric]
        row[f"B_{metric}"] = b["scores"][metric]

    row["A_combined_routing"] = (
        a["scores"]["department"] and a["scores"]["action"]
    )
    row["B_combined_routing"] = (
        b["scores"]["department"] and b["scores"]["action"]
    )

    rows.append(row)


def average(items, key):
    return sum(x[key] for x in items) / len(items)


summary = {
    "benchmark_cases": len(rows),
    "A": {
        "schema_validity": sum(x["schema_valid"] for x in a_results) / len(rows),
        "average_latency_ms": average(rows, "A_latency_ms"),
        "average_output_tokens": average(rows, "A_output_tokens"),
        "combined_routing_accuracy": sum(
            x["A_combined_routing"] for x in rows
        ) / len(rows),
    },
    "B": {
        "schema_validity": sum(x["schema_valid"] for x in b_results) / len(rows),
        "average_latency_ms": average(rows, "B_latency_ms"),
        "average_output_tokens": average(rows, "B_output_tokens"),
        "combined_routing_accuracy": sum(
            x["B_combined_routing"] for x in rows
        ) / len(rows),
    },
}

summary["latency_reduction_percent"] = (
    1 - summary["B"]["average_latency_ms"]
    / summary["A"]["average_latency_ms"]
) * 100

summary["output_token_reduction_percent"] = (
    1 - summary["B"]["average_output_tokens"]
    / summary["A"]["average_output_tokens"]
) * 100

OUT.mkdir(parents=True, exist_ok=True)

with (OUT / "a_b_per_case.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

(OUT / "a_b_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nSaved comparison files to:", OUT)
