import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "comparison"

FILES = {
    "A": ROOT / "results/experiment_a_v3_results.json",
    "B": ROOT / "experiments/B_sft_rlhf/results/benchmark_v1/experiment_b_results.json",
    "C": ROOT / "results/experiment_c_v1_results.json",
    "D": ROOT / "results/experiment_d_v1_results.json",
}

METRICS = [
    "intent",
    "product",
    "image_type",
    "department",
    "action",
    "clarification",
]


def load_results():
    results = {}

    for method, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        if not data:
            raise ValueError(f"Empty result file: {path}")

        results[method] = data

    reference_ids = {x["id"] for x in results["A"]}

    for method, data in results.items():
        ids = {x["id"] for x in data}
        if ids != reference_ids:
            raise ValueError(
                f"Benchmark IDs mismatch: {method}"
            )

    return results


def average(data, key):
    return sum(float(x[key]) for x in data) / len(data)


def accuracy(data, metric):
    return sum(
        bool(x["scores"].get(metric, False)) for x in data
    ) / len(data)


def combined_routing_accuracy(data):
    return sum(
        bool(x["scores"].get("department", False))
        and bool(x["scores"].get("action", False))
        for x in data
    ) / len(data)


def summarize_method(method, data):
    count = len(data)

    # A, B, C all invoke the model for every benchmark case.
    # D records the actual invocation status per case.
    if method == "D":
        invoked = sum(bool(x["model_invoked"]) for x in data)
        fast = sum(x["selected_path"] == "fast" for x in data)
        deep = sum(x["selected_path"] == "deep" for x in data)
    else:
        invoked = count
        fast = 0
        deep = count

    summary = {
        "benchmark_cases": count,
        "schema_validity": sum(
            bool(x["schema_valid"]) for x in data
        ) / count,
        "average_latency_ms": average(data, "latency_ms"),
        "average_input_tokens": average(data, "input_tokens"),
        "average_output_tokens": average(data, "output_tokens"),
        "model_invocation_count": invoked,
        "model_invocation_ratio": invoked / count,
        "fast_path_cases": fast,
        "deep_path_cases": deep,
        "combined_routing_accuracy": combined_routing_accuracy(data),
        "api_cost": None,
        "cost_note": (
            "Local inference; API billing unavailable. "
            "GPU electricity and hardware costs are excluded."
        ),
    }

    for metric in METRICS:
        summary[f"{metric}_accuracy"] = accuracy(data, metric)

    return summary


def build_per_case_rows(results):
    rows = []

    ids = sorted(x["id"] for x in results["A"])

    indexed = {
        method: {x["id"]: x for x in data}
        for method, data in results.items()
    }

    for case_id in ids:
        row = {"id": case_id}

        for method in ("A", "B", "C", "D"):
            item = indexed[method][case_id]

            row[f"{method}_latency_ms"] = item["latency_ms"]
            row[f"{method}_input_tokens"] = item["input_tokens"]
            row[f"{method}_output_tokens"] = item["output_tokens"]
            row[f"{method}_schema_valid"] = item["schema_valid"]

            row[f"{method}_combined_routing"] = (
                bool(item["scores"].get("department", False))
                and bool(item["scores"].get("action", False))
            )

            if method == "D":
                row["D_selected_path"] = item["selected_path"]
                row["D_routing_reason"] = item["routing_reason"]
                row["D_model_invoked"] = item["model_invoked"]

            for metric in METRICS:
                row[f"{method}_{metric}"] = bool(
                    item["scores"].get(metric, False)
                )

        rows.append(row)

    return rows


def main():
    results = load_results()

    summaries = {
        method: summarize_method(method, data)
        for method, data in results.items()
    }

    baseline_latency = summaries["A"]["average_latency_ms"]
    baseline_output_tokens = summaries["A"]["average_output_tokens"]

    for method, summary in summaries.items():
        summary["latency_change_vs_A_percent"] = (
            (summary["average_latency_ms"] / baseline_latency) - 1
        ) * 100

        summary["output_token_change_vs_A_percent"] = (
            (summary["average_output_tokens"] / baseline_output_tokens) - 1
        ) * 100

    summary_output = {
        "experiment": "A-D Unified Comparison",
        "benchmark_cases": summaries["A"]["benchmark_cases"],
        "benchmark_ids": sorted(x["id"] for x in results["A"]),
        "methods": summaries,
        "comparison_notes": [
            "All four methods use the same 10 benchmark IDs.",
            "A, B, C invoke Qwen for every case.",
            "D uses rule-based Fast Path and Qwen Deep Path.",
            "D average tokens include zero-token Fast Path cases.",
            "Latency excludes model loading time.",
            "API billing is unavailable; local GPU electricity and hardware costs are excluded.",
            "Results are limited to this 10-case benchmark."
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)

    rows = build_per_case_rows(results)

    csv_path = OUT / "a_b_c_d_per_case.csv"
    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT / "a_b_c_d_summary.json"
    json_path.write_text(
        json.dumps(summary_output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("===== A-D UNIFIED COMPARISON =====")
    print(json.dumps(
        summary_output,
        ensure_ascii=False,
        indent=2
    ))

    print("\n===== COMPACT COMPARISON TABLE =====")
    print(
        f"{'Method':<8}"
        f"{'Schema':>10}"
        f"{'Routing':>10}"
        f"{'Latency(ms)':>15}"
        f"{'In Tokens':>12}"
        f"{'Out Tokens':>12}"
        f"{'Model Calls':>13}"
    )

    for method, summary in summaries.items():
        print(
            f"{method:<8}"
            f"{summary['schema_validity'] * 100:>9.1f}%"
            f"{summary['combined_routing_accuracy'] * 100:>9.1f}%"
            f"{summary['average_latency_ms']:>15.3f}"
            f"{summary['average_input_tokens']:>12.1f}"
            f"{summary['average_output_tokens']:>12.1f}"
            f"{summary['model_invocation_count']:>13}"
        )

    print("\nSaved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
