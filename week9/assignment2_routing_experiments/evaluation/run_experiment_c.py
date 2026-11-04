import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from jsonschema import validate, ValidationError
from transformers import AutoProcessor, AutoModelForImageTextToText

# Transformers 5.19 compatibility for lm-format-enforcer 0.11.3
import transformers.tokenization_utils as _old_tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_old_tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn
)

from routing_prompt import build_routing_prompt


ROOT = Path(__file__).resolve().parents[1]
WEEK9 = ROOT.parent

BENCHMARK_PATH = ROOT / "benchmark" / "benchmark_v1.json"
SCHEMA_PATH = (
    WEEK9
    / "assignment1_multimodal_chatbot"
    / "config"
    / "output_schema.json"
)

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/"
      "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)

MAX_NEW_TOKENS = 768


def load_model():
    print("Loading Qwen3.5-4B...")
    print("Model path:", MODEL_PATH)

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

    print("Model loaded.")
    print("Model device:", model.device)

    return processor, model


def resolve_image_path(image_path):
    if not image_path:
        return None

    path = (ROOT / image_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    return path


def extract_json(raw_text):
    """
    Strict JSON parsing.
    No missing-field repair or semantic post-processing.
    """
    text = raw_text.strip()

    if text.startswith("```"):
        raise ValueError("Model returned Markdown code fences, not raw JSON.")

    return json.loads(text)


def run_one_case(case, processor, model, schema):
    case_id = case["id"]
    user_text = case["user_text"]
    session_id = f"experiment_a_{case_id}"

    image_path = resolve_image_path(case.get("image_path"))

    prompt = build_routing_prompt(
        user_text=user_text,
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

    # Build JSON Schema constrained decoding.
    # The callback starts from the JSON parser root and constrains
    # newly generated tokens; the existing prompt is not parsed as JSON.
    json_parser = JsonSchemaParser(schema)
    lmfe_callback = (
        build_transformers_prefix_allowed_tokens_fn(
            processor.tokenizer,
            json_parser
        )
    )

    # Once the complete JSON Schema is satisfied, allow EOS only.
    # This prevents the model from generating trailing whitespace
    # after the JSON object has already been completed.
    token_enforcer = lmfe_callback.token_enforcer

    def prefix_allowed_tokens_fn(batch_id, sent):
        token_sequence = sent.tolist()

        # Let LMFE update its parser state and enforce the schema.
        allowed_tokens = lmfe_callback(batch_id, sent)

        state = token_enforcer.prefix_states.get(
            tuple(token_sequence)
        )

        if state is not None and state.parser.can_end():
            eos_id = token_enforcer.eos_token_id

            if isinstance(eos_id, list):
                return eos_id

            return [eos_id]

        return allowed_tokens

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - start_time) * 1000

    new_tokens = generated[
        :, inputs["input_ids"].shape[-1]:
    ]

    output_tokens = int(new_tokens.shape[-1])

    raw_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True
    )[0]

    result = {
        "id": case_id,
        "user_text": user_text,
        "image_path": str(image_path) if image_path else None,
        "raw_output": raw_text,
        "parsed_output": None,
        "json_parse_valid": False,
        "schema_valid": False,
        "schema_error": None,
        "latency_ms": round(latency_ms, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    try:
        parsed = extract_json(raw_text)
        result["json_parse_valid"] = True
        result["parsed_output"] = parsed

        try:
            validate(instance=parsed, schema=schema)
            result["schema_valid"] = True
        except ValidationError as exc:
            result["schema_error"] = exc.message

    except (json.JSONDecodeError, ValueError) as exc:
        result["schema_error"] = str(exc)

    # Score only the model's parsed output.
    # Invalid JSON or schema output receives no correctness credit.
    parsed = result["parsed_output"]

    if parsed is not None and result["schema_valid"]:
        intent = parsed.get("intent", {})
        entities = parsed.get("entities", {})
        image_analysis = parsed.get("image_analysis", {})
        routing = parsed.get("routing", {})

        result["scores"] = {
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
    else:
        result["scores"] = {
            "intent": False,
            "product": False,
            "image_type": False,
            "department": False,
            "action": False,
            "clarification": False,
        }

    return result


def summarize(results):
    total = len(results)

    def rate(key):
        return (
            sum(r["scores"][key] for r in results) / total
            if total else 0
        )

    schema_validity = (
        sum(r["schema_valid"] for r in results) / total
        if total else 0
    )

    return {
        "experiment": "C - Qwen3.5-4B constrained JSON decoding",
        "benchmark_cases": total,
        "model": "Qwen3.5-4B",
        "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else "CPU",
        "schema_validity": schema_validity,
        "intent_accuracy": rate("intent"),
        "product_accuracy": rate("product"),
        "image_type_accuracy": rate("image_type"),
        "department_accuracy": rate("department"),
        "action_accuracy": rate("action"),
        "clarification_accuracy": rate("clarification"),
        "routing_accuracy": (
            sum(
                r["scores"]["department"] and r["scores"]["action"]
                for r in results
            ) / total if total else 0
        ),
        "average_latency_ms": (
            sum(r["latency_ms"] for r in results) / total
            if total else 0
        ),
        "total_latency_ms": sum(r["latency_ms"] for r in results),
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "api_cost": None,
        "cost_note": (
            "Local GPU inference; no API billing data. "
            "GPU electricity and hardware costs are not included."
        ),
    }


def save_results(results, summary, prefix):
    results_json = ROOT / "results" / f"{prefix}_results.json"
    results_csv = ROOT / "results" / f"{prefix}_results.csv"
    summary_json = ROOT / "results" / f"{prefix}_summary.json"

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
        "json_parse_valid",
        "schema_valid",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "scores",
        "schema_error",
    ]

    with results_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in results:
            row = dict(item)
            row["scores"] = json.dumps(
                item["scores"], ensure_ascii=False
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

    processor, model = load_model()

    results = []

    for index, case in enumerate(cases, start=1):
        print(
            f"\n===== {index}/{len(cases)}: {case['id']} ====="
        )
        print("User:", case["user_text"])
        print("Image:", case.get("image_path"))

        try:
            result = run_one_case(
                case, processor, model, schema
            )
            results.append(result)

            print("JSON valid:", result["json_parse_valid"])
            print("Schema valid:", result["schema_valid"])
            print("Latency:", result["latency_ms"], "ms")
            print("Tokens:", result["input_tokens"],
                  "/", result["output_tokens"])
            print("Scores:", result["scores"])

            if result["schema_error"]:
                print("Validation error:", result["schema_error"])

        except Exception as exc:
            print("CASE FAILED:", repr(exc))
            results.append({
                "id": case["id"],
                "user_text": case["user_text"],
                "image_path": case.get("image_path"),
                "raw_output": None,
                "parsed_output": None,
                "json_parse_valid": False,
                "schema_valid": False,
                "schema_error": repr(exc),
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "scores": {
                    "intent": False,
                    "product": False,
                    "image_type": False,
                    "department": False,
                    "action": False,
                    "clarification": False,
                },
            })

    prefix = (
        "experiment_c_v1_smoke"
        if args.limit > 0
        else "experiment_c_v1"
    )

    summary = summarize(results)
    save_results(results, summary, prefix)

    print("\n===== EXPERIMENT SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
