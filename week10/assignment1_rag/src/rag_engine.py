import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from src.retriever import FAQRetriever


class RAGEngine:
    def __init__(
        self,
        model_id="Qwen/Qwen3.5-4B",
        persist_directory="vector_db_canonical_v4",
        collection_name="smart3c_faq_canonical_v4",
        top_k=3,
        max_history_turns=4,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this configuration.")

        self.device = "cuda:0"
        self.top_k = top_k
        self.max_history_turns = max_history_turns

        self.retriever = FAQRetriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=self.device,
            local_files_only=True,
        )
        self.model.eval()

        self.sessions = {}

    def _generate(self, messages, max_new_tokens=180):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

    def _rewrite_query(self, question, history):
        if not history:
            return question

        recent = history[-self.max_history_turns:]
        history_text = "\n".join(
            f"使用者：{item['user']}\n助理：{item['assistant']}"
            for item in recent
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是搜尋查詢改寫助手。"
                    "根據對話歷史，把使用者最新問題改寫成"
                    "一個可獨立搜尋的繁體中文問題。"
                    "保留前文中的產品、型號與必要條件。"
                    "只輸出改寫後的問題，不要回答問題。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"對話歷史：\n{history_text}\n\n"
                    f"最新問題：{question}\n\n"
                    "改寫後的搜尋問題："
                ),
            },
        ]

        rewritten = self._generate(messages, max_new_tokens=64)
        rewritten = rewritten.strip().strip('"「」')

        return rewritten if rewritten else question

    def ask(self, question, session_id="default", top_k=None):
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        history = self.sessions.setdefault(session_id, [])

        retrieval_query = self._rewrite_query(question, history)

        documents = self.retriever.retrieve(
            retrieval_query,
            top_k=top_k or self.top_k,
        )

        context_parts = []
        for index, doc in enumerate(documents, start=1):
            context_parts.append(
                f"[資料 {index}]\n"
                f"問題：{doc.get('question', '')}\n"
                f"答案：{doc.get('answer', '')}\n"
                f"分類：{doc.get('category', '')}"
            )

        context = "\n\n".join(context_parts)
        recent_history = history[-self.max_history_turns:]

        history_text = "\n".join(
            f"使用者：{item['user']}\n助理：{item['assistant']}"
            for item in recent_history
        )

        messages = [
            {
                "role": "system",
                "content": (
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
                ),
            },
            {
                "role": "user",
                "content": (
                    f"先前對話：\n{history_text or '無'}\n\n"
                    f"本次檢索問題：{retrieval_query}\n\n"
                    f"參考資料：\n{context or '沒有找到相關資料。'}\n\n"
                    f"使用者最新問題：{question}"
                ),
            },
        ]

        answer = self._generate(messages)

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
                "verified_at": doc.get("verified_at"),
            })

        history.append({
            "user": question,
            "assistant": answer,
        })

        return {
            "question": question,
            "retrieval_query": retrieval_query,
            "answer": answer,
            "references": references,
            "retrieved_documents": documents,
        }

    def clear_history(self, session_id="default"):
        self.sessions.pop(session_id, None)

    def close(self):
        del self.model
        self.model = None
        self.tokenizer = None
        self.sessions.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
