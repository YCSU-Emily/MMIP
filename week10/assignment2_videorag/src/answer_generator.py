from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/"
    / "models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
    / "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)


class VideoAnswerGenerator:
    """
    Generate answers strictly grounded in retrieved video evidence.
    """

    def __init__(self, max_new_tokens=240):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Local Qwen model not found: {MODEL_PATH}"
            )

        torch.set_num_threads(4)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        self.max_new_tokens = max_new_tokens

        print(
            f"[VideoAnswerGenerator] Loading Qwen on {self.device}..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            dtype=dtype,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print("[VideoAnswerGenerator] Qwen loaded successfully.")

    @staticmethod
    def format_evidence(evidence):
        if not evidence:
            return "沒有檢索到相關影片證據。"

        formatted = []

        for item in evidence:
            metadata = item.get("metadata", {})

            record_id = item.get("id", "unknown")
            video_id = metadata.get("video_id", "unknown")
            modality = metadata.get("modality", "unknown")
            start = float(metadata.get("start_seconds", 0))
            end = float(metadata.get("end_seconds", start))
            text = item.get("text", "")

            formatted.append(
                f"[證據 ID: {record_id}]\n"
                f"影片: {video_id}\n"
                f"資料類型: {modality}\n"
                f"時間: {start:.2f}–{end:.2f} 秒\n"
                f"內容: {text}"
            )

        return "\n\n".join(formatted)

    @staticmethod
    def format_history(history, max_turns=4):
        """
        Use previous user questions for context.
        Do not use previous assistant answers as factual evidence.
        """
        if not history:
            return "目前沒有先前對話。"

        user_messages = [
            item.get("content", "")
            for item in history
            if item.get("role") == "user" and item.get("content")
        ]

        user_messages = user_messages[-max_turns:]

        if not user_messages:
            return "目前沒有先前使用者問題。"

        return "\n".join(
            f"先前使用者問題 {index}：{message}"
            for index, message in enumerate(user_messages, start=1)
        )

    def generate_answer(
        self,
        user_query,
        rewritten_query,
        evidence,
        conversation_history=None,
    ):
        if not evidence:
            return (
                "目前找不到足夠的影片證據來回答這個問題。"
                "請提供更多資訊，或重新描述遇到的 Wi-Fi 問題。"
            )

        evidence_text = self.format_evidence(evidence)
        history_text = self.format_history(conversation_history)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是智慧型手機售後服務助理，使用繁體中文回答。\n\n"
                    "【最高優先原則：忠於影片證據】\n"
                    "1. 只能將檢索到的影片證據當作事實依據。\n"
                    "2. 影片沒有提到的操作、時間、原因或建議，禁止自行補充。\n"
                    "3. 不可把合理推測寫成影片明確指示。\n"
                    "4. 如果影片沒有回答使用者的問題，必須明確說明影片未提及。\n"
                    "5. 不可以用先前助理的回答當作事實證據。\n\n"
                    "【特別注意語意】\n"
                    "1. 必須保留原文中的「或」、「以及」、「如果」等語意差異。\n"
                    "2. 「檢查路由器或重新啟動手機」不代表「重新啟動路由器」。\n"
                    "3. 「等待幾秒」不能改成半分鐘或其他未提及的時間。\n"
                    "4. 「重新啟動手機」不能擅自改成其他操作。\n"
                    "5. 不可自行增加飛航模式、聯絡 ISP 等影片未提及的步驟。\n\n"
                    "【回答格式】\n"
                    "1. 直接回答目前使用者的問題。\n"
                    "2. 每項重要事實附上證據 ID 和影片時間。\n"
                    "3. 引用格式：[證據 ID，影片時間]\n"
                    "4. 如果證據不足，直接說明不足之處。\n"
                    "5. 不要延續前一輪回答的編號。\n"
                    "6. 回答清楚、簡潔，不要重複無關資訊。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"先前使用者問題（只用於理解上下文）：\n"
                    f"{history_text}\n\n"
                    f"目前使用者問題：\n{user_query}\n\n"
                    f"檢索用改寫問題：\n{rewritten_query}\n\n"
                    f"本輪唯一可引用的影片證據：\n{evidence_text}\n\n"
                    "請只根據本輪影片證據回答目前問題。"
                    "若證據沒有明確回答，請坦白說明，不要猜測。"
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        return answer
