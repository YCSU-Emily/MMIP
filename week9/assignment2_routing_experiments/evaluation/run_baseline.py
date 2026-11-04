import csv
import json
import sys
import time
import uuid
from pathlib import Path

from jsonschema import validate, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT1_ROOT = PROJECT_ROOT.parent / "assignment1_multimodal_chatbot"

sys.path.insert(0, str(ASSIGNMENT1_ROOT / "src"))
import chatbot

BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "benchmark_v1.json"
SCHEMA_PATH = ASSIGNMENT1_ROOT / "config" / "output_schema.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

records = []

for case in cases:
    state = {
        "session_id": str(uuid.uuid4()),
        "turn_count": 1,
        "previous_intent": None,
        "pending_intent": None
    }

    image_path = None
    if case.get("image_path"):
        image_path = (PROJECT_ROOT / case["image_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{case['id']}: image not found: {image_path}"
            )

    start = time.perf_counter()
    output, state = chatbot.build_output(
        case["user_text"],
        state,
        image_path=str(image_path) if image_path else None,
        enable_vision=False
    )
    latency_ms = (time.perf_counter() - start) * 1000

    try:
        validate(instance=output, schema=schema)
        schema_valid = True
    except ValidationError as exc:
        schema_valid = False
        print(f"[SCHEMA ERROR] {case['id']}: {exc.message}")

    intent_correct = (
        output["intent"]["category"] == case["expected_intent"]
    )
    product_correct = (
        output["entities"]["product_name"] == case["expected_product_name"]
    )
    image_type_correct = (
        output["image_analysis"]["image_type"] == case["expected_image_type"]
    )
    department_correct = (
        output["routing"]["department"] == case["expected_department"]
    )
    action_correct = (
        output["routing"]["action"] == case["expected_action"]
    )
    clarification_correct = (
        output["intent"]["needs_clarification"]
        == case["expected_needs_clarification"]
    )

    routing_correct = department_correct and action_correct

    records.append({
        "id": case["id"],
        "description": case["description"],
        "latency_ms": round(latency_ms, 3),
        "schema_valid": schema_valid,
        "intent_correct": intent_correct,
        "product_correct": product_correct,
        "image_type_correct": image_type_correct,
        "department_correct": department_correct,
        "action_correct": action_correct,
        "routing_correct": routing_correct,
        "clarification_correct": clarification_correct,
        "expected_intent": case["expected_intent"],
        "predicted_intent": output["intent"]["category"],
        "expected_department": case["expected_department"],
        "predicted_department": output["routing"]["department"],
        "expected_action": case["expected_action"],
        "predicted_action": output["routing"]["action"],
        "expected_image_type": case["expected_image_type"],
        "predicted_image_type": output["image_analysis"]["image_type"],
        "expected_needs_clarification": case["expected_needs_clarification"],
        "predicted_needs_clarification": output["intent"]["needs_clarification"],
        "output": output
    })

    print(
        f"{case['id']} | "
        f"intent={'PASS' if intent_correct else 'FAIL'} | "
        f"routing={'PASS' if routing_correct else 'FAIL'} | "
        f"schema={'PASS' if schema_valid else 'FAIL'} | "
        f"{latency_ms:.3f} ms"
    )

def accuracy(key):
    return sum(bool(row[key]) for row in records) / len(records)

summary = {
    "experiment": "Rule-based lightweight baseline",
    "benchmark_cases": len(records),
    "vision_model_enabled": False,
    "schema_validity": accuracy("schema_valid"),
    "intent_accuracy": accuracy("intent_correct"),
    "product_accuracy": accuracy("product_correct"),
    "image_type_accuracy": accuracy("image_type_correct"),
    "department_accuracy": accuracy("department_correct"),
    "action_accuracy": accuracy("action_correct"),
    "routing_accuracy": accuracy("routing_correct"),
    "clarification_accuracy": accuracy("clarification_correct"),
    "average_latency_ms": sum(r["latency_ms"] for r in records) / len(records),
    "total_latency_ms": sum(r["latency_ms"] for r in records)
}

json_path = RESULTS_DIR / "baseline_results.json"
csv_path = RESULTS_DIR / "baseline_results.csv"
summary_path = RESULTS_DIR / "baseline_summary.json"

json_path.write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

csv_fields = [key for key in records[0] if key != "output"]
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for row in records:
        writer.writerow({k: row[k] for k in csv_fields})

print("\n===== BASELINE SUMMARY =====")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f} ({value * 100:.2f}%)" if value <= 1
              else f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")

print("\nSaved:")
print(json_path)
print(csv_path)
print(summary_path)
