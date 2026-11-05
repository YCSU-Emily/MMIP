"""Session memory and topic-shift detection."""

import uuid
import re
from datetime import datetime, timezone


TOPIC_KEYWORDS = {
    "order_tracking": ["訂單", "物流", "配送", "包裹", "出貨", "到貨"],
    "warranty_check": ["保固", "保修", "保固期限"],
    "repair_request": ["故障", "壞掉", "維修", "損壞", "螢幕破裂", "無法充電"],
    "return_refund": ["退貨", "退款", "退費"],
    "human_handoff": ["真人客服", "人工客服", "客服人員"],
    "faq": ["如何", "怎麼", "請問", "設定", "使用方法"],
}

# Explicit cues for a self-contained question outside the 3C service workflow.
UNRELATED_MARKERS = [
    "什麼是",
    "what is ",
    "explain ",
    "天氣",
    "幫我寫",
    "寫一封",
    "自我介紹",
    "翻譯",
    "transformer",
    "python",
    "數學",
    "股票",
    "旅遊",
    "食譜",
]

TOPIC_SWITCH_MARKERS = [
    "換個問題",
    "另外想問",
    "我想問另一件事",
    "先不管剛剛",
    "先不討論剛剛",
]

FOLLOW_UP_MARKERS = [
    "那",
    "這個",
    "那個",
    "它",
    "上述",
    "剛剛",
    "還有",
    "所以",
    "這樣",
    "為什麼",
    "how about",
    "what about",
]


def create_session():
    return {
        "session_id": str(uuid.uuid4()),
        "turn_count": 0,
        "previous_intent": None,
        "pending_intent": None,
        "entities": {
            "product_name": None,
            "product_model": None,
            "order_id": None,
            "purchase_date": None,
            "problem_description": None,
        },
        # Complete record for audit and conversation display.
        "history": [],
        # Only turns relevant to the current topic are sent to the model.
        "context_history": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def detect_topic(text):
    """Return a keyword-based service topic, or None if not recognized."""
    text_lower = text.lower()

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword.lower() in text_lower for keyword in keywords):
            return topic

    return None


def _is_explicit_unrelated(text):
    text_lower = text.lower()
    return any(marker in text_lower for marker in UNRELATED_MARKERS)


def _is_explicit_topic_switch(text):
    text_lower = text.lower()
    return any(marker in text_lower for marker in TOPIC_SWITCH_MARKERS)


def _looks_like_follow_up(text):
    text_lower = text.lower().strip()
    return any(marker in text_lower for marker in FOLLOW_UP_MARKERS)



def _is_order_id_supplement(text):
    """Recognize an order ID supplied as extra information for a repair case."""
    tracking_cues = (
        "查詢物流",
        "查物流",
        "物流狀態",
        "配送狀態",
        "包裹到哪",
        "出貨進度",
        "到貨進度",
        "追蹤訂單",
        "shipping status",
        "track my order",
    )

    text_lower = text.lower()
    if any(cue in text_lower for cue in tracking_cues):
        return False

    pattern = (
        r"(?:訂單編號|訂單號碼|訂單號)"
        r"\s*(?:是|為|：|:)?\s*"
        r"[A-Z0-9][A-Z0-9-]{5,}"
    )
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def detect_topic_shift(text, previous_intent):
    """Conservatively detect service-topic changes and unrelated questions."""

    if not previous_intent:
        return {
            "topic_shift": False,
            "previous_topic": previous_intent,
            "new_topic": detect_topic(text),
            "reason": "No previous topic to compare",
        }

    # Explicit unrelated questions take precedence over broad FAQ keywords.
    if _is_explicit_unrelated(text):
        return {
            "topic_shift": True,
            "previous_topic": previous_intent,
            "new_topic": "unrelated",
            "reason": "A self-contained question outside the current service topic was detected",
        }

    # An explicit request to change subject is also a topic transition.
    if _is_explicit_topic_switch(text):
        return {
            "topic_shift": True,
            "previous_topic": previous_intent,
            "new_topic": "unrelated",
            "reason": "The user explicitly requested a different topic",
        }

    # An order ID may be supplementary information for an active repair case.
    # It should not be mistaken for a request to track delivery.
    if (
        previous_intent == "repair_request"
        and _is_order_id_supplement(text)
    ):
        return {
            "topic_shift": False,
            "previous_topic": previous_intent,
            "new_topic": previous_intent,
            "reason": "Order ID supplied as supplementary repair information",
            "entity_supplement": True,
        }

    # Short references usually depend on the previous turn.
    if _looks_like_follow_up(text):
        return {
            "topic_shift": False,
            "previous_topic": previous_intent,
            "new_topic": detect_topic(text),
            "reason": "Likely contextual follow-up",
        }

    new_topic = detect_topic(text)

    if not new_topic:
        return {
            "topic_shift": False,
            "previous_topic": previous_intent,
            "new_topic": None,
            "reason": "Insufficient topic evidence",
        }

    changed = new_topic != previous_intent

    return {
        "topic_shift": changed,
        "previous_topic": previous_intent,
        "new_topic": new_topic,
        "reason": (
            "A different service topic was detected"
            if changed else "The current topic appears consistent"
        ),
    }


def add_turn(session, role, content, image_attached=False, max_history=10):
    """Append a turn to both the complete history and active context."""
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")

    turn = {
        "role": role,
        "content": str(content),
        "image_attached": bool(image_attached),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    session["history"].append(turn)
    session["history"] = session["history"][-max_history:]

    if "context_history" not in session:
        session["context_history"] = []

    session["context_history"].append(turn.copy())
    session["context_history"] = session["context_history"][-max_history:]

    return session


def update_session(session, user_text, intent=None):
    session["turn_count"] += 1

    shift = detect_topic_shift(
        user_text,
        session.get("previous_intent")
    )

    if shift["topic_shift"]:
        session["pending_intent"] = None

    if shift["new_topic"] == "unrelated":
        # Preserve complete history, but isolate the new topic from old context.
        session["context_history"] = []

        # Clear stale customer-service entities from the active session.
        for key in (
            "product_name",
            "product_model",
            "order_id",
            "purchase_date",
            "problem_description",
        ):
            session[key] = None
            session.setdefault("entities", {})[key] = None

        session["previous_intent"] = None
        session["pending_intent"] = None

    if intent and intent != "unknown":
        session["previous_intent"] = intent
        session["pending_intent"] = intent

    return shift
