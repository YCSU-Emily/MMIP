import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"

MODEL_PATH = Path(
    "/home/emily/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-4B/snapshots/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)

GRAPH_FILE = RESULTS / "knowledge_graph.json"
ASSIGNMENT_FILE = RESULTS / "community_assignments.json"

OUTPUT_FILE = RESULTS / "community_summaries.json"
RAW_DIR = RESULTS / "community_summary_raw"

MAX_NEW_TOKENS = 600


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_model():
    print("\n========== LOAD LOCAL QWEN ==========")

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model.eval()

    print("Processor:", type(processor).__name__)
    print("Model:", type(model).__name__)
    print("Device:", model.device)

    return processor, model


def build_community_input(community_id, assignments, nodes, edges):
    members = [
        item for item in assignments
        if item["community_id"] == community_id
    ]

    member_names = {item["entity"] for item in members}

    community_nodes = []

    for item in members:
        node = nodes[item["entity"]]

        community_nodes.append({
            "name": item["entity"],
            "type": item["entity_type"],
            "description": item.get("description", ""),
            "source_pages": item.get("source_pages", []),
            "evidence_records": node.get("evidence_records", []),
        })

    internal_relationships = []
    boundary_relationships = []

    for edge in edges:
        source_inside = edge["source"] in member_names
        target_inside = edge["target"] in member_names

        if source_inside and target_inside:
            internal_relationships.append({
                "source": edge["source"],
                "relation": edge["relation_type"],
                "target": edge["target"],
                "description": edge.get("description", ""),
                "source_pages": edge.get("source_pages", []),
                "evidence_records": edge.get("evidence_records", []),
            })

        elif source_inside or target_inside:
            boundary_relationships.append({
                "source": edge["source"],
                "relation": edge["relation_type"],
                "target": edge["target"],
                "source_pages": edge.get("source_pages", []),
                "evidence_records": edge.get("evidence_records", []),
            })

    return {
        "community_id": community_id,
        "member_count": len(members),
        "entities": community_nodes,
        "internal_relationships": internal_relationships,
        "boundary_relationships": boundary_relationships,
    }


def build_prompt(community):
    community_json = json.dumps(
        community,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are generating a grounded knowledge graph community summary
for a research paper.

Your task is to summarize ONE community using only the supplied data.

STRICT RULES:
1. Use only the provided entity descriptions, relationships, and evidence.
2. Do not introduce facts that are not supported by the evidence.
3. Do not invent causal relationships or connections.
4. Distinguish internal relationships from boundary relationships.
5. Boundary relationships provide context only. Do not describe them
   as internal community relationships.
6. Preserve important technical names and method names.
7. Include source page numbers for the supported information.
8. If the community contains only one entity, summarize that entity
   without inventing connections.
9. If evidence is insufficient, explicitly say so.
10. Write in Traditional Chinese. Keep technical terms in English.

Return ONLY valid JSON in this format:

{{
  "main_topic": "A concise topic title",
  "summary": "A grounded summary of this community",
  "key_entities": ["entity names"],
  "key_relationships": ["supported relationship descriptions"],
  "source_pages": [1, 2],
  "limitations": "Information that cannot be concluded from this community"
}}

COMMUNITY DATA:
{community_json}
""".strip()


def generate_summary(processor, model, community):
    prompt = build_prompt(community)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]

    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0].strip()


def parse_summary(raw_text):
    text = raw_text.strip()

    # Remove optional Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Extract the JSON object if extra text is present.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON object found.")

    return json.loads(text[start:end + 1])


def validate_summary(summary, community, valid_pages):
    required = {
        "main_topic",
        "summary",
        "key_entities",
        "key_relationships",
        "source_pages",
        "limitations",
    }

    missing = required - set(summary)

    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")

    if not isinstance(summary["source_pages"], list):
        raise ValueError("source_pages must be a list.")

    cited_pages = set(summary["source_pages"])

    if not all(isinstance(page, int) for page in cited_pages):
        raise ValueError("All source pages must be integers.")

    if not cited_pages.issubset(valid_pages):
        raise ValueError(
            f"Unsupported source pages: {cited_pages - valid_pages}"
        )

    member_names = {
        entity["name"] for entity in community["entities"]
    }

    if not isinstance(summary["key_entities"], list):
        raise ValueError("key_entities must be a list.")

    unknown_entities = set(summary["key_entities"]) - member_names

    if unknown_entities:
        raise ValueError(
            f"Unknown key entities: {unknown_entities}"
        )

    if not isinstance(summary["key_relationships"], list):
        raise ValueError("key_relationships must be a list.")

    return summary


def main():
    print("=" * 70)
    print("QWEN COMMUNITY SUMMARIZATION")
    print("=" * 70)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    if RAW_DIR.exists():
        raise FileExistsError(
            f"Raw output directory already exists: {RAW_DIR}"
        )

    graph_data = load_json(GRAPH_FILE)
    assignments = load_json(ASSIGNMENT_FILE)

    nodes = {
        item["id"]: item
        for item in graph_data["nodes"]
    }

    edges = graph_data["edges"]

    community_ids = sorted({
        item["community_id"] for item in assignments
    })

    if len(assignments) != len(nodes):
        raise ValueError("Assignment and graph node counts differ.")

    if set(nodes) != {item["entity"] for item in assignments}:
        raise ValueError("Graph and assignment node names differ.")

    RAW_DIR.mkdir(parents=True, exist_ok=False)

    processor, model = load_model()

    summaries = []
    failures = []

    all_valid_pages = {
        page
        for node in graph_data["nodes"]
        for page in node.get("source_pages", [])
    }

    all_valid_pages.update(
        page
        for edge in edges
        for page in edge.get("source_pages", [])
    )

    for position, community_id in enumerate(community_ids, 1):
        community = build_community_input(
            community_id,
            assignments,
            nodes,
            edges,
        )

        print(
            f"\n[{position}/{len(community_ids)}] "
            f"Community {community_id} "
            f"({community['member_count']} entities)"
        )

        raw_text = generate_summary(
            processor,
            model,
            community,
        )

        raw_path = RAW_DIR / f"community_{community_id:03d}.raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8")

        try:
            summary = parse_summary(raw_text)

            summary = validate_summary(
                summary,
                community,
                all_valid_pages,
            )

            summaries.append({
                "community_id": community_id,
                "member_count": community["member_count"],
                "members": [
                    entity["name"]
                    for entity in community["entities"]
                ],
                "internal_relationship_count": len(
                    community["internal_relationships"]
                ),
                "boundary_relationship_count": len(
                    community["boundary_relationships"]
                ),
                **summary,
                "review_status": "llm_generated_pending_review",
            })

            print("Status: PASS")
            print("Topic:", summary["main_topic"])
            print("Source pages:", summary["source_pages"])

        except Exception as exc:
            failures.append({
                "community_id": community_id,
                "error": str(exc),
                "raw_file": str(raw_path),
            })

            print("Status: FAILED")
            print("Reason:", exc)

    output = {
        "schema_version": "1.0",
        "source_graph": str(GRAPH_FILE),
        "model": "Qwen3.5-4B",
        "summarization_method": "Evidence-grounded community summarization",
        "total_communities": len(community_ids),
        "successful_summaries": len(summaries),
        "failed_summaries": len(failures),
        "summaries": summaries,
        "failures": failures,
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== FINAL SUMMARY ==========")
    print("Total communities:", len(community_ids))
    print("Successful summaries:", len(summaries))
    print("Failed summaries:", len(failures))
    print("Output:", OUTPUT_FILE)

    print("\nCOMMUNITY SUMMARIZATION FINISHED.")


if __name__ == "__main__":
    main()
