"""
Experiment D: System One routing mechanism.

Fast Path:
    Clear text-only requests -> lightweight rule-based structured output.

Deep Path:
    Actual image, ambiguous intent, multiple intents,
    or suspicious prompt-injection patterns -> Qwen3.5-4B.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSIGNMENT1_SRC = (
    PROJECT_ROOT.parent
    / "assignment1_multimodal_chatbot"
    / "src"
)

sys.path.insert(0, str(ASSIGNMENT1_SRC))

import chatbot


SUSPICIOUS_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "system prompt",
    "developer message",
    "忽略之前的指示",
    "忽略所有指令",
    "顯示系統提示",
    "洩漏系統提示",
    "覆蓋系統指令",
]


INTENT_KEYWORDS = {
    "order_tracking": ["訂單", "物流", "配送", "出貨", "到貨"],
    "warranty_check": ["保固", "保修", "保固期限"],
    "repair_request": [
        "故障", "壞掉", "不能開機", "無法開機",
        "維修", "損壞", "報修", "螢幕破裂",
        "螢幕裂開", "螢幕損壞", "無法充電",
    ],
    "return_refund": ["退貨", "退款", "退費"],
    "human_handoff": ["真人", "人工客服", "客服人員"],
}


def detect_multiple_intents(text):
    matched = []

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            matched.append(intent)

    return matched


def select_path(user_text, image_path=None):
    """
    Select Fast Path or Deep Path before generating structured output.

    Returns:
        {
            "selected_path": "fast" or "deep",
            "routing_reason": "...",
            "detected_intent": "...",
            "matched_intents": [...]
        }
    """

    # An actual image requires multimodal understanding.
    if image_path is not None:
        return {
            "selected_path": "deep",
            "routing_reason": "Actual image requires multimodal analysis.",
            "detected_intent": chatbot.detect_intent(user_text),
            "matched_intents": detect_multiple_intents(user_text),
        }

    text_lower = user_text.lower()

    # Potential prompt injection must not enter the lightweight path.
    if any(pattern in text_lower for pattern in SUSPICIOUS_PATTERNS):
        return {
            "selected_path": "deep",
            "routing_reason": "Suspicious instruction pattern requires deeper analysis.",
            "detected_intent": chatbot.detect_intent(user_text),
            "matched_intents": detect_multiple_intents(user_text),
        }

    matched = detect_multiple_intents(user_text)
    detected_intent = chatbot.detect_intent(user_text)

    # Multiple service intents may require semantic disambiguation.
    if len(matched) > 1:
        return {
            "selected_path": "deep",
            "routing_reason": "Multiple service intents detected.",
            "detected_intent": detected_intent,
            "matched_intents": matched,
        }

    # Unknown requests are routed to the VLM/LLM for deeper interpretation.
    if detected_intent == "unknown":
        return {
            "selected_path": "deep",
            "routing_reason": "Ambiguous or unknown intent requires deeper analysis.",
            "detected_intent": detected_intent,
            "matched_intents": matched,
        }

    # Clear, text-only requests can use the lightweight path.
    return {
        "selected_path": "fast",
        "routing_reason": "Clear single-intent text request.",
        "detected_intent": detected_intent,
        "matched_intents": matched,
    }
