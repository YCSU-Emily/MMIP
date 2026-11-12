import argparse
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

from query_engine import GraphRAGRetriever


MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/"
    / "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)


def build_context(result):
    """Format GraphRAG retrieval results as grounded context."""

    sections = []

    sections.append("=== RETRIEVED ENTITIES ===")
    for item in result["entities"]:
        sections.append(
            f"- {item['label']} ({item['entity_type']}); "
            f"description: {item['description']}; "
            f"note: entity metadata, not directly citable"
        )

    sections.append("\n=== RETRIEVED RELATIONSHIPS ===")
    for item in result["relationships"]:
        sections.append(
            f"- {item['source']} --[{item['relation_type']}]--> "
            f"{item['target']}\n"
            f"  Description: {item['description']}\n"
            f"  Evidence: {item['evidence']}\n"
            f"  Note: graph relationship metadata, not directly citable"
        )

    sections.append("\n=== COMMUNITY SUMMARIES ===")
    for item in result["communities"]:
        sections.append(
            f"- Community C{item['community_id']:02d}: "
            f"{item['main_topic']}\n"
            f"  Summary: {item['summary']}\n"
            f"  Key entities: {item['key_entities']}\n"
            f"  Key relationships: {item['key_relationships']}\n"
            f"  Note: community summary, not directly citable"
        )

    sections.append(
        "\n=== ORIGINAL PAPER EVIDENCE "
        "(THE ONLY CITABLE SOURCE) ==="
    )
    for item in result["evidence_chunks"]:
        sections.append(
            f"[{item['chunk_id']}, page {item['page']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(sections)


def load_qwen():
    print("\n[Qwen] Loading local model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Qwen model directory not found: {MODEL_PATH}"
        )

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    print("[Qwen] Model ready.")
    return processor, model


def generate_answer(question, retriever, processor, model):
    print("\n[GraphRAG] Retrieving evidence...")
    result = retriever.retrieve(
        question,
        top_entities=8,
        top_edges=10,
        top_communities=5,
        top_chunks=7,
    )

    context = build_context(result)

    allowed_pages = sorted({
        int(item["page"])
        for item in result["evidence_chunks"]
    })
    allowed_pages_text = ", ".join(str(p) for p in allowed_pages)

    system_prompt = f"""
You are a research assistant answering questions about one academic paper.

Rules:
1. Answer using ONLY the supplied retrieved context.
2. Do not invent facts, methods, results, or citations.
3. The ONLY allowed citation pages are: {allowed_pages_text}.
4. Every citation must use exactly [p. N] format.
5. Never cite a page outside the allowed page list.
6. Do not use page ranges.
7. If evidence is insufficient, explicitly say so.
8. Distinguish paper claims from your own interpretation.
9. Do not treat community summaries as original paper evidence.
"""

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Answer using only the retrieved context and allowed citation pages.
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = processor(
        text=[prompt],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    print("[Qwen] Generating grounded answer...")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_length:]

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    cited_pages = sorted({
        int(item["page"])
        for item in result["evidence_chunks"]
    })

    # Parse all page numbers, including grouped citations such as
    # [p. 1, 2, 6] and ranges such as [pp. 1-3].
    answer_citations_set = set()
    has_page_range = False

    citation_pattern = re.compile(
        r"\[(pp?\.)\s*([0-9,\s–-]+)\]",
        re.IGNORECASE,
    )

    for match in citation_pattern.finditer(answer):
        citation_body = match.group(2)

        if re.search(r"\d+\s*[-–]\s*\d+", citation_body):
            has_page_range = True

        numbers = re.findall(r"\d+", citation_body)
        answer_citations_set.update(int(number) for number in numbers)

    answer_citations = sorted(answer_citations_set)

    unsupported_citations = sorted(
        set(answer_citations) - set(cited_pages)
    )

    citation_valid = (
        len(answer_citations) > 0
        and len(unsupported_citations) == 0
        and not has_page_range
    )

    return {
        "question": question,
        "answer": answer,
        "retrieval_method": "GraphRAG",
        "answer_citations": answer_citations,
        "unsupported_citations": unsupported_citations,
        "citation_valid": citation_valid,
        "retrieved_entities": [
            item["label"] for item in result["entities"]
        ],
        "retrieved_communities": [
            item["community_id"] for item in result["communities"]
        ],
        "evidence_chunks": [
            {
                "chunk_id": item["chunk_id"],
                "page": item["page"],
                "score": item["score"],
            }
            for item in result["evidence_chunks"]
        ],
        "candidate_pages": cited_pages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-grounded answers with GraphRAG and Qwen."
    )
    parser.add_argument(
        "--question",
        type=str,
        default=(
            "How does GraphRAG use community summaries "
            "to answer global questions?"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This script expects a GPU.")

    print("===== GRAPH RAG ANSWER GENERATOR =====")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Question:", args.question)

    retriever = GraphRAGRetriever(device="cuda")
    processor, model = load_qwen()

    result = generate_answer(
        args.question,
        retriever,
        processor,
        model,
    )

    print("\n" + "=" * 75)
    print("QUESTION")
    print(result["question"])

    print("\nGENERATED ANSWER")
    print(result["answer"])

    print("\nRETRIEVED EVIDENCE")
    for item in result["evidence_chunks"]:
        print(
            f"- {item['chunk_id']} | "
            f"page {item['page']} | "
            f"score {item['score']}"
        )

    print("\nCANDIDATE PAGES:", result["candidate_pages"])
    print("ANSWER CITATIONS:", result["answer_citations"])
    print("UNSUPPORTED CITATIONS:", result["unsupported_citations"])
    print(
        "CITATION VALIDATION:",
        "PASS" if result["citation_valid"] else "FAIL",
    )
    print("\n===== GENERATION COMPLETE =====")


if __name__ == "__main__":
    main()
