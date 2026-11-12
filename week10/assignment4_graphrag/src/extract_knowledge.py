import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


BASE = Path(__file__).resolve().parents[1]
TEXT_PATH = BASE / "results" / "paper_text.txt"
OUTPUT_DIR = BASE / "results" / "extraction_chunks"
FINAL_PATH = BASE / "results" / "extracted_knowledge.json"

MODEL_PATH = Path(
    "/home/emily/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-4B/snapshots/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)

MAX_CHARS = 6000
MAX_NEW_TOKENS = 2800


def parse_pages(text):
    """Split extracted paper text using [PAGE n] markers."""
    matches = list(re.finditer(r"^\[PAGE\s+(\d+)\]\s*$", text, re.MULTILINE))
    pages = {}

    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pages[page_num] = text[start:end].strip()

    return pages


def split_text(text, max_chars=MAX_CHARS):
    """Split long pages into manageable pieces, preferring paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start:start + max_chars])
            continue

        candidate = current + "\n" + paragraph if current else paragraph

        if len(candidate) > max_chars:
            if current:
                chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def extract_json(raw_text):
    """Extract a JSON object even if the model adds surrounding text."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model output")

    return json.loads(cleaned[start:end + 1])


def build_prompt(page_num, part_num, source_text):
    page_specific_instruction = ""

    if page_num == 26:
        page_specific_instruction = """
SPECIAL RULES FOR PAGE 26 - STATISTICAL ANALYSIS:
1. Focus on the statistical methodology and overall experimental findings.
2. Extract important methods: Shapiro-Wilk test, Wilcoxon signed-rank test,
   and Holm-Bonferroni correction.
3. Extract the four evaluation metrics:
   Comprehensiveness, Diversity, Empowerment, and Directness.
4. Extract the experimental scope: 125 questions, two datasets,
   Podcast Transcripts and News Articles, and six evaluation conditions,
   only when supported by the source.
5. Do NOT create individual relationships for every numerical table row.
6. Do NOT treat C0, C1, C2, C3, TS, and SS as separate entities
   merely because they appear in pairwise comparison rows.
7. Summarize only meaningful methodological relationships supported
   by the explanatory text and table caption.
8. Return no more than 10 entities and 8 relationships.
9. Prioritize meaningful research knowledge over exhaustive table values.
10. Every evidence quote must be copied verbatim from the source.
"""

    return f"""
You are extracting factual knowledge from a research paper.

Analyze ONLY the source text below. Do not use outside knowledge.
Do not invent entities, methods, results, or relationships.
Keep names consistent with the paper.
Include short verbatim evidence quotes from the source text.

Return exactly one valid JSON object. Do not use Markdown fences.

Required schema:
{{
  "entities": [
    {{
      "name": "canonical entity name",
      "type": "Method | Concept | Model | Dataset | Metric | Organization | Other",
      "description": "A concise description supported by the source",
      "evidence": "A short verbatim quote"
    }}
  ],
  "relationships": [
    {{
      "source": "entity name",
      "target": "entity name",
      "type": "uses | proposes | addresses | evaluates | compares | consists_of | improves | other",
      "description": "A concise explanation of the relationship",
      "evidence": "A short verbatim quote"
    }}
  ]
}}

Rules:
1. Extract important research concepts, not ordinary words.
2. Only create a relationship when the source text supports it.
3. Both relationship endpoints should appear in the entities list.
4. Evidence must be copied from the source text, not paraphrased.
5. If no reliable item exists, return an empty array.
6. Avoid duplicate entities within this response.
7. Return approximately 4-10 entities and 4-10 relationships when supported.
8. Do not force the requested number if the source does not support it.

{page_specific_instruction}

Source page: {page_num}
Source part: {part_num}

SOURCE TEXT:
{source_text}
""".strip()


def validate_result(data):
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")

    if not isinstance(data.get("entities"), list):
        raise ValueError("Missing or invalid entities array")

    if not isinstance(data.get("relationships"), list):
        raise ValueError("Missing or invalid relationships array")

    entities = []
    seen_entities = set()

    for item in data["entities"]:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        if not name or name.lower() in seen_entities:
            continue

        seen_entities.add(name.lower())

        entities.append({
            "name": name,
            "type": str(item.get("type", "Other")).strip(),
            "description": str(item.get("description", "")).strip(),
            "evidence": str(item.get("evidence", "")).strip()
        })

    entity_names = {item["name"].lower() for item in entities}

    relationships = []

    for item in data["relationships"]:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()

        if not source or not target:
            continue

        if source.lower() not in entity_names:
            continue

        if target.lower() not in entity_names:
            continue

        relationships.append({
            "source": source,
            "target": target,
            "type": str(item.get("type", "other")).strip(),
            "description": str(item.get("description", "")).strip(),
            "evidence": str(item.get("evidence", "")).strip()
        })

    return {
        "entities": entities,
        "relationships": relationships
    }


def load_model():
    print("\n========== LOAD QWEN ==========")

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True
    )

    model.eval()

    print("Processor:", type(processor).__name__)
    print("Model:", type(model).__name__)
    print("Device:", model.device)

    return processor, model


def generate_extraction(processor, model, page_num, part_num, source_text):
    prompt = build_prompt(page_num, part_num, source_text)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]

    raw_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True
    )[0]

    return raw_text


def aggregate_results():
    all_entities = {}
    all_relationships = {}
    chunk_files = sorted(OUTPUT_DIR.glob("page_*_part_*.json"))

    for path in chunk_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("Skip invalid chunk:", path.name, exc)
            continue

        for item in data.get("entities", []):
            key = item["name"].strip().lower()

            if key not in all_entities:
                all_entities[key] = {
                    **item,
                    "source_pages": [],
                    "evidence_records": []
                }

            page = data["page"]
            if page not in all_entities[key]["source_pages"]:
                all_entities[key]["source_pages"].append(page)

            evidence = item.get("evidence", "").strip()
            if evidence and evidence not in all_entities[key]["evidence_records"]:
                all_entities[key]["evidence_records"].append(evidence)

        for item in data.get("relationships", []):
            key = (
                item["source"].strip().lower(),
                item["target"].strip().lower(),
                item["type"].strip().lower()
            )

            if key not in all_relationships:
                all_relationships[key] = {
                    **item,
                    "source_pages": [],
                    "evidence_records": []
                }

            page = data["page"]
            if page not in all_relationships[key]["source_pages"]:
                all_relationships[key]["source_pages"].append(page)

            evidence = item.get("evidence", "").strip()
            if evidence and evidence not in all_relationships[key]["evidence_records"]:
                all_relationships[key]["evidence_records"].append(evidence)

    result = {
        "paper_title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "source": "arXiv:2404.16130v2",
        "extraction_model": "Qwen3.5-4B",
        "entities": list(all_entities.values()),
        "relationships": list(all_relationships.values()),
        "statistics": {
            "processed_chunk_files": len(chunk_files),
            "unique_entities": len(all_entities),
            "unique_relationships": len(all_relationships)
        }
    }

    FINAL_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not TEXT_PATH.exists():
        raise FileNotFoundError(TEXT_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    text = TEXT_PATH.read_text(encoding="utf-8")
    pages = parse_pages(text)

    if not pages:
        raise RuntimeError("No [PAGE n] markers found")

    selected_pages = [
        page for page in sorted(pages)
        if args.start_page <= page <= args.end_page
    ]

    if not selected_pages:
        raise ValueError("No pages selected. Check start/end page.")

    print("=" * 60)
    print("QWEN PAPER KNOWLEDGE EXTRACTION")
    print("=" * 60)
    print("Available pages:", len(pages))
    print("Selected pages:", selected_pages)
    print("Output directory:", OUTPUT_DIR)

    processor, model = load_model()

    for page_num in selected_pages:
        page_chunks = split_text(pages[page_num])

        for part_num, source_text in enumerate(page_chunks, start=1):
            output_path = OUTPUT_DIR / (
                f"page_{page_num:03d}_part_{part_num:02d}.json"
            )
            raw_path = output_path.with_suffix(".raw.txt")

            if output_path.exists() and not args.force:
                print("\nSkip existing:", output_path.name)
                continue

            print(
                f"\nExtracting page {page_num}, "
                f"part {part_num}/{len(page_chunks)} "
                f"({len(source_text)} characters)"
            )

            raw_text = generate_extraction(
                processor,
                model,
                page_num,
                part_num,
                source_text
            )

            raw_path.write_text(raw_text, encoding="utf-8")

            try:
                parsed = extract_json(raw_text)
                validated = validate_result(parsed)

                result = {
                    "page": page_num,
                    "part": part_num,
                    "source_characters": len(source_text),
                    **validated
                }

                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

                print(
                    "PASS:",
                    len(validated["entities"]), "entities,",
                    len(validated["relationships"]), "relationships"
                )

            except Exception as exc:
                print("JSON EXTRACTION FAILED:", exc)
                print("Raw response preserved at:", raw_path)
                raise

    result = aggregate_results()

    print("\n========== AGGREGATED GRAPH DATA ==========")
    print("Chunk files:", result["statistics"]["processed_chunk_files"])
    print("Unique entities:", result["statistics"]["unique_entities"])
    print("Unique relationships:", result["statistics"]["unique_relationships"])
    print("Saved:", FINAL_PATH)
    print("\nEXTRACTION FINISHED")


if __name__ == "__main__":
    main()
