import argparse
import csv
import json
import time
import sys
from pathlib import Path

# Support direct execution and package-style imports.
PROJECT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent

for import_dir in (PROJECT_DIR, EVALUATION_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import torch
from PIL import Image
from jsonschema import validate, ValidationError
from transformers import AutoProcessor, AutoModelForImageTextToText

from routing_prompt import build_routing_prompt
from experiments.D_system_one.router import select_path
from experiments.D_system_one import router
import chatbot


ROOT = Path(__file__).resolve().parents[1]
WEEK9 = ROOT.parent

BENCHMARK_PATH = ROOT / "benchmark" / "benchmark_v1.json"
SCHEMA_PATH = (
    WEEK9 / "assignment1_multimodal_chatbot"
    / "config" / "output_schema.json"
)

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/"
      "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)

MAX_NEW_TOKENS = 768


def load_model():
    print("\nLoading Qwen3.5-4B for Deep Path...")

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True
    )

    model.eval()

    print("Deep Path model loaded.")
    print("Model device:", model.device)

    return processor, model


def resolve_image_path(image_path):
    if not image_path:
        return None

    path = (ROOT / image_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    return path


def initial_state(case_id):
    return {
        "session_id": f"experiment_d_{case_id}",
        "turn_count": 1,
        "previous_intent": None,
        "pending_intent": None,
    }


def run_deep_path(case, image_path, processor, model):
    session_id = f"experiment_d_{case['id']}"

    prompt = build_routing_prompt(
        user_text=case["user_text"],
        session_id=session_id
    )

    content = []

    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        content.append({
            "type": "image",
            "image": image
        })

    content.append({
        "type": "text",
        "text": prompt
    })

    messages = [{
        "role": "user",
        "content": content
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_tokens = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    new_tokens = generated[
        :, inputs["input_ids"].shape[-1]:
    ]

    output_tokens = int(new_tokens.shape[-1])

    raw_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True
    )[0]

    return raw_text, input_tokens, output_tokens


def score_output(parsed, schema_valid, case):
    if parsed is None or not schema_valid:
        return {
            "intent": False,
            "product": False,
            "image_type": False,
            "department": False,
            "action": False,
            "clarification": False,
        }

    intent = parsed.get("intent", {})
    entities = parsed.get("entities", {})
    image_analysis = parsed.get("image_analysis", {})
    routing = parsed.get("routing", {})

    return {
        "intent": (
            intent.get("category") == case["expected_intent"]
        ),
        "product": (
            entities.get("product_name")
            == case["expected_product_name"]
        ),
        "image_type": (
            image_analysis.get("image_type")
            == case["expected_image_type"]
        ),
        "department": (
            routing.get("department")
            == case["expected_department"]
        ),
        "action": (
            routing.get("action")
            == case["expected_action"]
        ),
        "clarification": (
            intent.get("needs_clarification")
            == case["expected_needs_clarification"]
        ),
    }


def run_one_case(case, schema, processor=None, model=None):
    case_id = case["id"]
    user_text = case["user_text"]

    image_path = resolve_image_path(case.get("image_path"))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    decision = select_path(
        user_text=user_text,
        image_path=image_path
    )

    selected_path = decision["selected_path"]

    input_tokens = 0
    output_tokens = 0
    raw_text = ""
    parsed = None
    json_valid = False
    schema_valid = False
    schema_error = None

    if selected_path == "fast":
        state = initial_state(case_id)

        parsed, updated_state = chatbot.build_output(
            user_text,
            state,
            image_path=None,
            enable_vision=False
        )

        raw_text = json.dumps(
            parsed,
            ensure_ascii=False
        )

        json_valid = True

        try:
            validate(instance=parsed, schema=schema)
            schema_valid = True
        except ValidationError as exc:
            schema_error = exc.message

    else:
        if processor is None or model is None:
            raise RuntimeError("Deep Path model is not loaded.")

        raw_text, input_tokens, output_tokens = run_deep_path(
            case,
            image_path,
            processor,
            model
        )

        try:
            if raw_text.strip().startswith("```"):
                raise ValueError("Markdown code fences are not allowed.")

            parsed = json.loads(raw_text)
            json_valid = True

            try:
                validate(instance=parsed, schema=schema)
                schema_valid = True
            except ValidationError as exc:
                schema_error = exc.message

        except (json.JSONDecodeError, ValueError) as exc:
            schema_error = str(exc)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - start_time) * 1000

    scores = score_output(
        parsed,
        schema_valid,
        case
    )

    return {
        "id": case_id,
        "user_text": user_text,
        "image_path": str(image_path) if image_path else None,
        "selected_path": selected_path,
        "routing_reason": decision["routing_reason"],
        "detected_intent": decision["detected_intent"],
        "matched_intents": decision["matched_intents"],
        "model_invoked": selected_path == "deep",
        "raw_output": raw_text,
        "parsed_output": parsed,
        "json_parse_valid": json_valid,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "latency_ms": round(latency_ms, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "scores": scores,
    }


def summarize(results):
    total = len(results)

    def rate(key):
        return (
            sum(r["scores"][key] for r in results) / total
            if total else 0
        )

    fast_count = sum(
        r["selected_path"] == "fast" for r in results
    )
    deep_count = sum(
        r["selected_path"] == "deep" for r in results
    )
    model_calls = sum(r["model_invoked"] for r in results)

    combined_routing = sum(
        r["scores"]["department"] and r["scores"]["action"]
        for r in results
    ) / total if total else 0

    return {
        "experiment": "D - System One Hybrid Fast/Deep Routing",
        "benchmark_cases": total,
        "model": "Qwen3.5-4B",
        "fast_path_cases": fast_count,
        "deep_path_cases": deep_count,
        "fast_path_ratio": fast_count / total if total else 0,
        "deep_path_ratio": deep_count / total if total else 0,
        "model_invocation_count": model_calls,
        "model_invocation_ratio": model_calls / total if total else 0,
        "schema_validity": (
            sum(r["schema_valid"] for r in results) / total
            if total else 0
        ),
        "intent_accuracy": rate("intent"),
        "product_accuracy": rate("product"),
        "image_type_accuracy": rate("image_type"),
        "department_accuracy": rate("department"),
        "action_accuracy": rate("action"),
        "combined_routing_accuracy": combined_routing,
        "clarification_accuracy": rate("clarification"),
        "average_latency_ms": (
            sum(r["latency_ms"] for r in results) / total
            if total else 0
        ),
        "total_latency_ms": sum(r["latency_ms"] for r in results),
        "average_input_tokens_per_request": (
            sum(r["input_tokens"] for r in results) / total
            if total else 0
        ),
        "average_output_tokens_per_request": (
            sum(r["output_tokens"] for r in results) / total
            if total else 0
        ),
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "api_cost": None,
        "cost_note": (
            "Local rule-based and GPU inference; no API billing data. "
            "GPU electricity and hardware costs are not included."
        ),
        "method_note": (
            "Fast Path uses Assignment 1 rule-based build_output. "
            "Deep Path uses Qwen3.5-4B with the same prompt as A. "
            "Actual images and ambiguous requests are routed to Deep Path."
        ),
    }


def save_results(results, summary, prefix):
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_json = results_dir / f"{prefix}_results.json"
    results_csv = results_dir / f"{prefix}_results.csv"
    summary_json = results_dir / f"{prefix}_summary.json"

    results_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    fields = [
        "id",
        "selected_path",
        "routing_reason",
        "model_invoked",
        "json_parse_valid",
        "schema_valid",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "scores",
        "schema_error",
    ]

    with results_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in results:
            row = dict(item)
            row["scores"] = json.dumps(
                item["scores"],
                ensure_ascii=False
            )
            writer.writerow({
                key: row.get(key) for key in fields
            })

    print("\nSaved:")
    print(results_json)
    print(results_csv)
    print(summary_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of benchmark cases; 0 means all cases."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    cases = json.loads(
        BENCHMARK_PATH.read_text(encoding="utf-8")
    )

    schema = json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8")
    )

    if args.limit > 0:
        cases = cases[:args.limit]

    print("Benchmark cases:", len(cases))
    print("Schema:", schema.get("title"))

    # Determine whether the selected test cases need the model.
    needs_deep = any(
        select_path(
            case["user_text"],
            resolve_image_path(case.get("image_path"))
        )["selected_path"] == "deep"
        for case in cases
    )

    processor = None
    model = None

    # Model startup is reported separately and excluded from per-case latency.
    if needs_deep:
        processor, model = load_model()

    results = []

    for index, case in enumerate(cases, start=1):
        print(
            f"\n===== {index}/{len(cases)}: {case['id']} ====="
        )

        try:
            result = run_one_case(
                case,
                schema,
                processor,
                model
            )
            results.append(result)

            print("Path:", result["selected_path"])
            print("Reason:", result["routing_reason"])
            print("Schema valid:", result["schema_valid"])
            print("Latency:", result["latency_ms"], "ms")
            print(
                "Tokens:",
                result["input_tokens"],
                "/",
                result["output_tokens"]
            )
            print("Scores:", result["scores"])

        except Exception as exc:
            print("ERROR:", repr(exc))
            raise

    summary = summarize(results)

    print("\n===== D EXPERIMENT SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    save_results(
        results,
        summary,
        "experiment_d_v1"
    )


if __name__ == "__main__":
    main()
