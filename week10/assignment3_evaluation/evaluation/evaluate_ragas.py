import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import pandas as pd
from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from pydantic import ConfigDict

from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)


ROOT = Path(__file__).resolve().parents[3]
RAG_PROJECT = ROOT / "week10" / "assignment1_rag"
TEST_FILE = (
    RAG_PROJECT
    / "data"
    / "splits_canonical_v4"
    / "test.jsonl"
)
REPORT_DIR = ROOT / "week10" / "assignment3_evaluation" / "reports"


class LocalQwenLLM(LLM):
    """Expose the already-loaded RAGEngine Qwen model to LangChain/RAGAS."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: Any
    max_new_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "local-qwen3.5-4b"

    @property
    def _identifying_params(self):
        return {
            "model": "Qwen/Qwen3.5-4B",
            "max_new_tokens": self.max_new_tokens,
        }

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        prompt_text = str(prompt)

        # RAGAS calls this prompt when the first JSON output is invalid.
        is_format_repair = (
            "The output string did not satisfy the constraints" in prompt_text
        )

        if is_format_repair:
            repair_rules = """
IMPORTANT: THIS IS A JSON FORMAT REPAIR TASK.

The required output model for this repair operation is StringIO.
Return ONLY one valid outer JSON object:
{"text": "..."}

The value of "text" must be a string containing the corrected
JSON object for the ORIGINAL evaluation task.

Read the original task prompt and preserve its exact output structure.
Every required field must be present in every record.
Do not omit fields, replace arrays with strings, or invent evidence.
Do not return the inner JSON object directly.
Do not use Markdown fences or explanations outside the JSON.
"""

            # Context Recall
            if "attributed" in prompt_text and "classifications" in prompt_text:
                repair_rules += """
The original task is Context Recall.
The inner JSON must have:
{
  "classifications": [
    {
      "statement": "string",
      "reason": "string",
      "attributed": 0
    }
  ]
}
Every classification MUST contain all three fields:
- statement: a non-empty string copied from or faithfully representing
  one sentence in the answer being evaluated.
- reason: a non-empty string explaining the decision.
- attributed: integer 0 or 1.

Every answer sentence must be evaluated exactly once.
Do not create empty classifications.
Do not omit the final sentence.
Do not leave any field blank or missing.
Use attributed=1 only when the statement is supported by the context.
Use attributed=0 when the statement cannot be supported by the context.
Return valid JSON only.
"""

            # Faithfulness
            elif "statements" in prompt_text and "verdict" in prompt_text:
                repair_rules += """
The original task is Faithfulness.
The inner JSON must have:
{
  "statements": [
    {
      "statement": "string",
      "reason": "string",
      "verdict": 0
    }
  ]
}
Every statement MUST contain statement, reason, and verdict.
The verdict value must be integer 0 or 1.
Judge support using the provided context.
"""

            # Answer Relevancy
            elif "noncommittal" in prompt_text:
                repair_rules += """
The original task is Answer Relevancy.
The inner JSON must contain:
{
  "question": "string",
  "noncommittal": 0
}
Both fields are required. noncommittal must be integer 0 or 1.
"""

            # Context Precision
            elif "reason" in prompt_text and "verdict" in prompt_text:
                repair_rules += """
The original task is Context Precision.
The inner JSON must contain:
{
  "reason": "string",
  "verdict": 0
}
Both fields are required. verdict must be integer 0 or 1.
"""

            prompt_text += "\\n\\n" + repair_rules


        messages = [
            {
                "role": "user",
                "content": prompt_text,
            }
        ]

        formatted = self.engine.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.engine.tokenizer(
            formatted,
            return_tensors="pt",
        ).to(self.engine.device)

        with torch.inference_mode():
            output = self.engine.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]

        response = self.engine.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        if stop:
            for stop_text in stop:
                if stop_text in response:
                    response = response.split(stop_text)[0]

        return response


class ExistingBGEEmbeddings(Embeddings):
    """Reuse the BGE-M3 model already loaded by FAQRetriever."""

    def __init__(self, embedder):
        self.embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        vector = self.embedder.embed_query(text)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedder.embed_documents(texts)
        return vectors.tolist()


def load_test_data():
    records = []

    with TEST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def format_context(documents):
    """Match the context format actually sent to the original RAG model."""
    parts = []

    for index, doc in enumerate(documents, start=1):
        parts.append(
            f"[資料 {index}]\n"
            f"問題：{doc.get('question', '')}\n"
            f"答案：{doc.get('answer', '')}\n"
            f"分類：{doc.get('category', '')}"
        )

    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of test records. Use 1 for smoke test.",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse existing raw RAG results.",
    )
    args = parser.parse_args()

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_test_data()

    if args.limit is not None:
        records = records[:args.limit]

    if not records:
        raise RuntimeError("No test records found.")

    print("========== RAGAS EVALUATION ==========")
    print("Test file:", TEST_FILE)
    print("Number of records:", len(records))
    print("GPU:", torch.cuda.get_device_name(0))

    # Import the original project without modifying its source files.
    sys.path.insert(0, str(RAG_PROJECT))

    from src.rag_engine import RAGEngine

    engine = None

    try:
        print("\n[1/4] Loading original RAGEngine...")

        engine = RAGEngine(
            persist_directory=str(
                RAG_PROJECT / "vector_db_canonical_v4"
            ),
            collection_name="smart3c_faq_canonical_v4",
            top_k=3,
            max_history_turns=4,
        )

        print("[OK] RAGEngine loaded.")

        raw_path = REPORT_DIR / "ragas_raw_results.json"

        evaluation_rows = []
        raw_records = []

        if args.reuse_raw:
            print("\n[2/4] Reusing existing raw RAG results...")

            if not raw_path.exists():
                raise FileNotFoundError(raw_path)

            with raw_path.open("r", encoding="utf-8") as f:
                raw_records = json.load(f)

            if not isinstance(raw_records, list):
                raise TypeError("Raw results must be a JSON list.")

            raw_records = raw_records[:len(records)]

            if len(raw_records) != len(records):
                raise ValueError(
                    f"Expected {len(records)} raw records, "
                    f"found {len(raw_records)}."
                )

            for item, raw in zip(records, raw_records):
                if raw.get("id") != item.get("id"):
                    raise ValueError(
                        f"ID mismatch: {raw.get('id')} vs {item.get('id')}"
                    )

                if raw.get("question") != item.get("question"):
                    raise ValueError(
                        f"Question mismatch for {item.get('id')}"
                    )

                if raw.get("reference_answer") != item.get("answer"):
                    raise ValueError(
                        f"Reference mismatch for {item.get('id')}"
                    )

                documents = raw.get("retrieved_documents", [])

                evaluation_rows.append({
                    "user_input": raw["question"],
                    "response": raw["generated_answer"],
                    "retrieved_contexts": [
                        format_context([doc]) for doc in documents
                    ],
                    "reference": raw["reference_answer"],
                })

            print(f"[OK] Reused {len(evaluation_rows)} raw records.")

        else:
            print("\n[2/4] Generating answers and collecting contexts...")

            for index, item in enumerate(records, start=1):
                question = item["question"]
                session_id = f"ragas_eval_{item['id']}"

                print(
                    f"\n[{index}/{len(records)}] "
                    f"{item['id']}: {question}"
                )

                result = engine.ask(
                    question,
                    session_id=session_id,
                    top_k=3,
                )

                documents = result["retrieved_documents"]
                context = format_context(documents)

                evaluation_rows.append({
                    "user_input": question,
                    "response": result["answer"],
                    "retrieved_contexts": [
                        format_context([doc]) for doc in documents
                    ],
                    "reference": item["answer"],
                })

                raw_records.append({
                    "id": item["id"],
                    "category": item.get("category"),
                    "question": question,
                    "reference_answer": item["answer"],
                    "generated_answer": result["answer"],
                    "retrieval_query": result["retrieval_query"],
                    "retrieved_context": context,
                    "retrieved_documents": documents,
                    "references": result["references"],
                })

                print("Generated answer:", result["answer"])
                print("Retrieved documents:", len(documents))

            with raw_path.open("w", encoding="utf-8") as f:
                json.dump(
                    raw_records,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print("\n[OK] Raw results saved:", raw_path)

        print("\n[3/4] Preparing local RAGAS evaluator...")
        print("\n[3/4] Preparing local RAGAS evaluator...")

        local_llm = LocalQwenLLM(engine=engine)
        ragas_llm = LangchainLLMWrapper(local_llm)

        local_embeddings = ExistingBGEEmbeddings(
            embedder=engine.retriever.embedder
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            local_embeddings
        )

        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]

        for metric in metrics:
            metric.llm = ragas_llm

            if hasattr(metric, "embeddings"):
                metric.embeddings = ragas_embeddings

        dataset = EvaluationDataset.from_list(evaluation_rows)

        print("[OK] Dataset and metrics prepared.")

        print("\n[4/4] Running RAGAS evaluation...")

        run_config = RunConfig(
            timeout=600,
            max_retries=3,
            max_wait=30,
            max_workers=1,
        )

        print(
            "[INFO] RAGAS RunConfig:",
            f"timeout={run_config.timeout}",
            f"max_retries={run_config.max_retries}",
            f"max_workers={run_config.max_workers}",
        )

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=True,
        )

        scores_df = result.to_pandas()
        score_records = scores_df.to_dict(orient="records")

        scores_path = REPORT_DIR / "ragas_scores.json"

        with scores_path.open("w", encoding="utf-8") as f:
            json.dump(
                score_records,
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        summary = {}

        for column in scores_df.columns:
            # Only calculate averages for numeric evaluation scores.
            values = pd.to_numeric(
                scores_df[column],
                errors="coerce",
            ).dropna()

            if len(values) > 0:
                summary[column] = float(values.mean())

        summary_data = {
            "experiment": "Smart3C Canonical V4 RAGAS Evaluation",
            "ragas_version": "0.4.3",
            "model": "Qwen/Qwen3.5-4B",
            "embedding_model": "BAAI/bge-m3",
            "retrieval_top_k": 3,
            "test_samples": len(records),
            "reference_note": (
                "Reference answers are taken from the existing "
                "Canonical V4 test dataset; they are not claimed "
                "to be independently human-validated."
            ),
            "metrics": summary,
        }

        summary_path = REPORT_DIR / "ragas_summary.json"

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                summary_data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\n========== RAGAS RESULTS ==========")
        print(scores_df.to_string(index=False))

        print("\n========== SUMMARY ==========")
        print(json.dumps(
            summary_data,
            ensure_ascii=False,
            indent=2,
        ))

        print("\nSaved:")
        print(raw_path)
        print(scores_path)
        print(summary_path)

    finally:
        if engine is not None:
            print("\nClosing RAGEngine...")
            engine.close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
