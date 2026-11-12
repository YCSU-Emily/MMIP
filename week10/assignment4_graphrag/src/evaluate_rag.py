import argparse
import json
import re
import time
from pathlib import Path

import chromadb
import torch

from answer_generator import build_context, load_qwen
from query_engine import GraphRAGRetriever


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
BENCHMARK = RESULTS / "benchmark.json"
DB_DIR = RESULTS / "chroma_vector_db"
COLLECTION_NAME = "graphrag_paper_chunks"


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_citations(answer):
    pages = set()
    has_range = False
    pattern = re.compile(r"\[(pp?\.)\s*([0-9,\s–-]+)\]", re.I)

    for match in pattern.finditer(answer):
        body = match.group(2)
        if re.search(r"\d+\s*[-–]\s*\d+", body):
            has_range = True
        pages.update(int(x) for x in re.findall(r"\d+", body))

    return sorted(pages), has_range


def build_vector_context(chunks):
    sections = [
        "=== ORIGINAL PAPER EVIDENCE (THE ONLY CITABLE SOURCE) ==="
    ]
    for item in chunks:
        sections.append(
            f"[{item['chunk_id']}, page {item['page']}]\n{item['text']}"
        )
    return "\n\n".join(sections)


def generate_answer(question, context, allowed_pages, processor, model):
    allowed = ", ".join(str(p) for p in allowed_pages) or "none"

    system_prompt = f"""
You are a research assistant answering questions about one academic paper.

Rules:
1. Answer using ONLY the supplied retrieved context.
2. Do not invent facts, methods, results, or citations.
3. The ONLY allowed citation pages are: {allowed}.
4. Every factual paragraph must end with at least one citation.
5. Every numbered step and every factual bullet point must include a citation.
6. Place citations immediately after the sentence or item they support.
7. Every citation must use exactly [p. N] format.
8. Only cite pages in the allowed page list: {allowed}.
9. Never cite a page that is not represented in the retrieved context.
10. Do not use page ranges.
11. If evidence is insufficient, explicitly say so instead of guessing.
12. Distinguish paper claims from your own interpretation.
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

    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    prompt_tokens = int(inputs["input_ids"].shape[1])

    sync_cuda()
    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
        )

    sync_cuda()
    generation_seconds = time.perf_counter() - start

    answer = processor.batch_decode(
        output_ids[:, prompt_tokens:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return answer, prompt_tokens, generation_seconds


def calculate_metrics(item, answer, retrieved_pages):
    reference_pages = sorted(set(item["reference_pages"]))
    retrieved_pages = sorted(set(retrieved_pages))
    overlap = sorted(set(reference_pages) & set(retrieved_pages))

    recall = len(overlap) / len(reference_pages) if reference_pages else 0
    precision = len(overlap) / len(retrieved_pages) if retrieved_pages else 0

    keywords = item.get("keywords", [])
    matched = [k for k in keywords if k.casefold() in answer.casefold()]
    keyword_coverage = len(matched) / len(keywords) if keywords else 0

    citations, has_range = parse_citations(answer)
    unsupported = sorted(set(citations) - set(retrieved_pages))

    citation_valid = (
        bool(citations) and not unsupported and not has_range
    )

    citation_precision = (
        len(set(citations) & set(retrieved_pages)) / len(set(citations))
        if citations else 0
    )

    return {
        "reference_pages": reference_pages,
        "retrieved_pages": retrieved_pages,
        "reference_page_overlap": overlap,
        "retrieval_page_recall": round(recall, 4),
        "retrieval_page_precision": round(precision, 4),
        "reference_page_hit": bool(overlap),
        "matched_keywords": matched,
        "keyword_coverage": round(keyword_coverage, 4),
        "answer_citations": citations,
        "unsupported_citations": unsupported,
        "citation_valid": citation_valid,
        "citation_precision": round(citation_precision, 4),
        "citation_has_page_range": has_range,
    }


def summarize(records):
    if not records:
        return {"completed_questions": 0}

    n = len(records)

    def mean(key):
        return round(
            sum(r["metrics"][key] for r in records) / n, 4
        )

    return {
        "completed_questions": n,
        "reference_page_hit_rate": round(
            sum(r["metrics"]["reference_page_hit"] for r in records) / n, 4
        ),
        "mean_retrieval_page_recall": mean("retrieval_page_recall"),
        "mean_retrieval_page_precision": mean("retrieval_page_precision"),
        "mean_keyword_coverage": mean("keyword_coverage"),
        "citation_valid_rate": round(
            sum(r["metrics"]["citation_valid"] for r in records) / n, 4
        ),
        "mean_citation_precision": mean("citation_precision"),
        "mean_retrieval_seconds": round(
            sum(r["retrieval_seconds"] for r in records) / n, 4
        ),
        "mean_generation_seconds": round(
            sum(r["generation_seconds"] for r in records) / n, 4
        ),
        "mean_prompt_tokens": round(
            sum(r["prompt_tokens"] for r in records) / n, 2
        ),
    }


def save_results(path, payload):
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        choices=["both", "graphrag", "vector"],
        default="both",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of questions for a pilot run.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    if args.limit > 0:
        benchmark = benchmark[:args.limit]

    methods = (
        ["graphrag", "vector"]
        if args.methods == "both"
        else [args.methods]
    )

    print("===== RAG EVALUATION =====")
    print("Questions:", len(benchmark))
    print("Methods:", methods)
    print("GPU:", torch.cuda.get_device_name(0))

    print("\n[1] Loading GraphRAG retriever and BGE-M3...")
    retriever = GraphRAGRetriever(device="cuda")
    bge_model = retriever.model

    print("\n[2] Loading existing ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    if collection.count() != 95:
        raise RuntimeError(
            f"Expected 95 vectors, found {collection.count()}"
        )

    print("Collection:", collection.name)
    print("Vectors:", collection.count())

    print("\n[3] Loading Qwen...")
    processor, qwen = load_qwen()

    for method in methods:
        output_path = RESULTS / f"evaluation_{method}.json"

        if output_path.exists() and not (
            args.resume or args.overwrite
        ):
            raise FileExistsError(
                f"{output_path} exists. Use --resume or --overwrite."
            )

        if args.overwrite:
            records = []
        elif args.resume and output_path.exists():
            old = json.loads(output_path.read_text(encoding="utf-8"))
            records = old.get("records", [])
        else:
            records = []

        completed = {r["id"] for r in records}

        print(f"\n===== {method.upper()} =====")
        print("Already completed:", len(completed))

        for index, item in enumerate(benchmark, start=1):
            if item["id"] in completed:
                continue

            question = item["question"]
            print(
                f"\n[{index}/{len(benchmark)}] "
                f"{item['id']}: {question}",
                flush=True,
            )

            sync_cuda()
            retrieval_start = time.perf_counter()

            if method == "graphrag":
                retrieved = retriever.retrieve(
                    question,
                    top_entities=8,
                    top_edges=10,
                    top_communities=5,
                    top_chunks=7,
                )

                chunks = retrieved["evidence_chunks"]
                context = build_context(retrieved)
                extra = {
                    "retrieved_entities": [
                        x["label"] for x in retrieved["entities"]
                    ],
                    "retrieved_communities": [
                        x["community_id"] for x in retrieved["communities"]
                    ],
                    "graph_candidate_pages": retrieved["candidate_pages"],
                }

            else:
                query_vector = bge_model.encode(
                    [question],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )

                result = collection.query(
                    query_embeddings=query_vector.tolist(),
                    n_results=7,
                    include=[
                        "documents",
                        "metadatas",
                        "distances",
                    ],
                )

                chunks = []
                for j, chunk_id in enumerate(result["ids"][0]):
                    metadata = result["metadatas"][0][j]
                    chunks.append({
                        "chunk_id": chunk_id,
                        "page": int(metadata["page"]),
                        "text": result["documents"][0][j],
                        "score": round(
                            1.0 - float(result["distances"][0][j]), 4
                        ),
                    })

                context = build_vector_context(chunks)
                extra = {
                    "retrieved_entities": [],
                    "retrieved_communities": [],
                    "graph_candidate_pages": [],
                }

            sync_cuda()
            retrieval_seconds = time.perf_counter() - retrieval_start

            retrieved_pages = sorted({
                int(c["page"]) for c in chunks
            })

            answer, prompt_tokens, generation_seconds = generate_answer(
                question,
                context,
                retrieved_pages,
                processor,
                qwen,
            )

            metrics = calculate_metrics(
                item, answer, retrieved_pages
            )

            record = {
                "id": item["id"],
                "topic": item["topic"],
                "difficulty": item["difficulty"],
                "question": question,
                "reference_answer": item["reference_answer"],
                "answer": answer,
                "method": method,
                "retrieved_chunks": chunks,
                "retrieved_chunk_count": len(chunks),
                "context_characters": len(context),
                "prompt_tokens": prompt_tokens,
                "retrieval_seconds": round(retrieval_seconds, 4),
                "generation_seconds": round(generation_seconds, 4),
                "total_seconds": round(
                    retrieval_seconds + generation_seconds, 4
                ),
                "metrics": metrics,
                **extra,
            }

            records.append(record)
            completed.add(item["id"])

            payload = {
                "experiment": "GraphRAG-inspired vs. Vector RAG",
                "method": method,
                "generation_model": "Qwen3.5-4B",
                "embedding_model": "BAAI/bge-m3",
                "top_k_chunks": 7,
                "generation_config": {
                    "max_new_tokens": 600,
                    "do_sample": False,
                },
                "benchmark_total": len(benchmark),
                "records": records,
                "summary": summarize(records),
            }

            save_results(output_path, payload)

            print(
                f"  retrieval={retrieval_seconds:.2f}s"
                f" | generation={generation_seconds:.2f}s"
                f" | page_recall={metrics['retrieval_page_recall']:.2f}"
                f" | keyword_coverage={metrics['keyword_coverage']:.2f}"
                f" | citation={'PASS' if metrics['citation_valid'] else 'FAIL'}",
                flush=True,
            )

        print("\n===== SUMMARY =====")
        print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
        print("Saved:", output_path)

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()
