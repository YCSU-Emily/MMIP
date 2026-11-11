import json
from pathlib import Path

from src.rag_engine import RAGEngine


PROMPT_A = (
    "你是 Smart3C 客服助理，請使用繁體中文回答。"
    "請根據提供的參考資料回答使用者問題。"
    "回答應簡潔、清楚，並提供可執行的建議。"
)

PROMPT_B = (
    "你是 Smart3C 客服助理，請使用繁體中文回答。"
    "回答必須以本次檢索到的參考資料為依據。"
    "只有參考資料明確支持的資訊，才能表述為已確認事實。"
    "不得把一般常識、推測或資料未提及的原因、步驟、數字，"
    "寫成參考資料已確認的內容。"
    "若資料不足，請直接說明不足，不要自行推測補齊。"
    "若提供延伸建議，必須明確標示為一般建議，"
    "且不可與參考資料矛盾。"
    "請區分不同裝置、故障症狀與使用情境。"
    "回答應簡潔、清楚，必要時按步驟列出。"
)

TEST_CASES = [
    {
        "id": "PROMPT001",
        "question": "我的手機充電很慢，應該怎麼處理？",
    },
    {
        "id": "PROMPT002",
        "question": "家中多台裝置都無法連接同一個 Wi-Fi，應該如何排查？",
    },
]

REPORT_PATH = Path("reports/prompt_ab/prompt_ab_v1.json")


def build_context(documents):
    parts = []

    for index, doc in enumerate(documents, start=1):
        parts.append(
            f"[資料 {index}]\n"
            f"問題：{doc.get('question', '')}\n"
            f"答案：{doc.get('answer', '')}\n"
            f"分類：{doc.get('category', '')}"
        )

    return "\n\n".join(parts)


def build_references(documents):
    references = []
    seen_urls = set()

    for doc in documents:
        url = doc.get("source_url")
        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        references.append({
            "title": doc.get("source_title"),
            "publisher": doc.get("source_publisher"),
            "url": url,
        })

    return references


def main():
    if REPORT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {REPORT_PATH}"
        )

    engine = None
    results = []

    try:
        engine = RAGEngine(top_k=3)

        for case in TEST_CASES:
            question = case["question"]

            # Retrieve once: both prompts share exactly the same evidence.
            documents = engine.retriever.retrieve(question, top_k=3)
            context = build_context(documents)

            user_content = (
                "先前對話：\n無\n\n"
                f"本次檢索問題：{question}\n\n"
                f"參考資料：\n{context or '沒有找到相關資料。'}\n\n"
                f"使用者最新問題：{question}"
            )

            messages_a = [
                {"role": "system", "content": PROMPT_A},
                {"role": "user", "content": user_content},
            ]

            messages_b = [
                {"role": "system", "content": PROMPT_B},
                {"role": "user", "content": user_content},
            ]

            answer_a = engine._generate(messages_a)
            answer_b = engine._generate(messages_b)

            results.append({
                "id": case["id"],
                "question": question,
                "retrieved_documents": [
                    {
                        "id": doc.get("id"),
                        "question": doc.get("question"),
                        "answer": doc.get("answer"),
                        "category": doc.get("category"),
                    }
                    for doc in documents
                ],
                "references": build_references(documents),
                "prompt_a_answer": answer_a,
                "prompt_b_answer": answer_b,
            })

            print(f"\n===== {case['id']} =====")
            print("Question:", question)
            print("\nPrompt A:", answer_a[:500])
            print("\nPrompt B:", answer_b[:500])
            print("Retrieved documents:", len(documents))

        report = {
            "experiment": "Prompt A/B Comparison",
            "model": "Qwen/Qwen3.5-4B",
            "top_k": 3,
            "prompt_a": PROMPT_A,
            "prompt_b": PROMPT_B,
            "generation": {
                "do_sample": False,
                "enable_thinking": False,
            },
            "results": results,
        }

        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n===== EXPERIMENT COMPLETE =====")
        print("Report:", REPORT_PATH)

    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
