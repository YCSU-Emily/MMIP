import json
import re
import uuid
from pathlib import Path
from jsonschema import validate
try:
    from .image_handler import analyze_image
    from .vision_backend import analyze_image_with_qwen
except ImportError:
    from image_handler import analyze_image
    from vision_backend import analyze_image_with_qwen

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "config" / "output_schema.json"

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    SCHEMA = json.load(f)


def detect_intent(text):
    rules = {
        "order_tracking": ["訂單", "物流", "配送", "出貨", "到貨"],
        "warranty_check": ["保固", "保修", "保固期限"],
        "repair_request": [
            "故障", "壞掉", "不能開機", "無法開機",
            "維修", "損壞", "報修",
            "產品有問題", "商品有問題", "筆電有問題",
            "手機有問題", "電腦有問題",
            "無法使用", "不能使用", "無法正常使用",
            "螢幕破裂", "螢幕裂開", "螢幕損壞",
            "當機", "閃退", "沒有反應", "沒反應",
            "無法充電", "不能充電", "充不了電",
            "過熱", "異常"
        ],
        "return_refund": ["退貨", "退款", "退費"],
        "human_handoff": ["真人", "人工客服", "客服人員"],
    }

    for intent, keywords in rules.items():
        if any(word in text for word in keywords):
            return intent

    if any(word in text for word in ["你好", "您好", "怎麼", "如何", "請問"]):
        return "faq"

    return "unknown"



def classify_image_type(text):
    """
    Infer the intended image category from the user's text.
    This is text-based classification, not visual recognition.
    """

    rules = {
        "damage": [
            "裂痕", "裂開", "裂了", "破裂", "摔壞", "損壞",
            "刮傷", "破掉", "螢幕壞", "螢幕裂", "外觀受損"
        ],
        "receipt": [
            "收據", "發票", "購買證明", "付款證明"
        ],
        "order_information": [
            "訂單截圖", "訂單畫面", "物流截圖",
            "出貨截圖", "訂單資訊"
        ],
        "product": [
            "產品照片", "商品照片", "產品圖片",
            "商品圖片", "產品外觀"
        ]
    }

    for image_type, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            return image_type

    return "unknown"


def extract_entities(text, state):
    order = re.search(r"\b[A-Z0-9]{6,}\b", text)
    state["order_id"] = order.group(0) if order else state.get("order_id")

    if "ASUS" in text.upper():
        state["product_name"] = "ASUS"
    elif "MacBook" in text or "macbook" in text.lower():
        state["product_name"] = "Apple MacBook"
    elif "Apple" in text:
        state["product_name"] = "Apple"
    elif "iPhone" in text:
        state["product_name"] = "iPhone"
    elif "Samsung" in text:
        state["product_name"] = "Samsung"

    # Preserve the first problem description across follow-up turns.
    # Later replies such as a brand name or order ID should not overwrite it.
    if not state.get("problem_description"):
        state["problem_description"] = text
    return state


def build_output(text, state, image_path=None, enable_vision=False):
    intent = detect_intent(text)
    state = extract_entities(text, state)

    # Analyze an optional image supplied by the user.
    image_analysis = {
        "image_provided": False,
        "image_type": "none",
        "observations": [],
        "visible_damage": None,
        "limitations": []
    }

    if image_path is not None:
        # Always validate the image before any AI inference.
        image_result = analyze_image(image_path)
        image_analysis["image_provided"] = True

        if not image_result["success"]:
            image_analysis["image_type"] = "unknown"
            image_analysis["limitations"] = [
                f"圖片驗證失敗：{image_result['error']}"
            ]

        elif enable_vision:
            # Real chatbot mode: use Qwen for visual understanding.
            image_analysis = analyze_image_with_qwen(
                image_path,
                user_text=text
            )

        else:
            # Lightweight test mode: do not load the vision model.
            image_analysis["image_type"] = classify_image_type(text)
            image_analysis["observations"] = [
                f"Image format: {image_result['image_format']}",
                f"Image dimensions: {image_result['width']}x{image_result['height']}",
                f"Color mode: {image_result['color_mode']}"
            ]
            image_analysis["limitations"] = [
                "目前使用輕量測試模式，圖片類型依使用者文字推測，並非 AI 視覺辨識結果。",
                "未啟用 Qwen 視覺模型，因此無法根據圖片判斷產品是否損壞。"
            ]

    # Continue the previous task when the user supplies missing information.
    if intent == "unknown" and state.get("pending_intent"):
        intent = state["pending_intent"]

    departments = {
        "faq": ("general_support", "answer_faq"),
        "order_tracking": ("order_support", "track_order"),
        "warranty_check": ("warranty_service", "check_warranty"),
        "repair_request": ("repair_center", "create_repair_request"),
        "return_refund": ("returns_refunds", "start_return_refund"),
        "human_handoff": ("human_agent", "handoff_to_human"),
        "unknown": ("general_support", "ask_follow_up"),
    }

    department, action = departments[intent]

    missing = []
    if intent == "order_tracking" and not state.get("order_id"):
        missing.append("order_id")

    if intent == "repair_request" and not state.get("product_name"):
        missing.append("product_name")

    needs_clarification = bool(missing) or intent == "unknown"

    message = (
        "請提供訂單編號，以便查詢物流。"
        if "order_id" in missing else
        "請問您的產品品牌或型號是什麼？"
        if "product_name" in missing else
        "請問您想詢問哪一項客服服務？"
        if intent == "unknown" else
        f"已辨識您的需求，將由 {department} 部門處理。"
    )

    if needs_clarification:
        action = "ask_follow_up"
        state["pending_intent"] = intent
    else:
        state["pending_intent"] = None

    result = {
        "intent": {
            "category": intent,
            "confidence": 0.9 if intent != "unknown" else 0.3,
            "needs_clarification": needs_clarification
        },
        "entities": {
            "product_name": state.get("product_name"),
            "product_model": None,
            "order_id": state.get("order_id"),
            "purchase_date": None,
            "problem_description": state.get("problem_description")
        },
        "image_analysis": image_analysis,
        "conversation_state": {
            "session_id": state["session_id"],
            "turn_count": state["turn_count"],
            "collected_information": [
                key for key in ["product_name", "order_id"]
                if state.get(key)
            ],
            "missing_information": missing,
            "previous_intent": state.get("previous_intent"),
            "awaiting_user_input": needs_clarification
        },
        "routing": {
            "department": department,
            "action": action,
            "priority": "normal",
            "requires_human": intent == "human_handoff",
            "reason": "Intent-based routing"
        },
        "response": {
            "message": message,
            "language": "zh-TW",
            "suggested_questions": missing
        }
    }

    state["previous_intent"] = intent
    return result, state


def main():
    state = {
        "session_id": str(uuid.uuid4()),
        "turn_count": 0,
        "previous_intent": None,
        "pending_intent": None
    }

    print("=== 3C Multimodal Customer Service Bot ===")
    print("輸入 /quit 結束，輸入 /reset 重新開始。")
    print("每一輪都可以輸入圖片路徑，直接按 Enter 可略過。")

    while True:
        text = input("\nUser: ").strip()

        if text == "/quit":
            print("Bot: 感謝使用，再見！")
            break

        if text == "/reset":
            state = {
                "session_id": str(uuid.uuid4()),
                "turn_count": 0,
                "previous_intent": None,
                "pending_intent": None
            }
            print("Bot: 已重新開始對話。")
            continue

        if not text:
            continue

        image_input = input("圖片路徑（沒有圖片請直接按 Enter）: ").strip()
        image_path = image_input if image_input else None

        state["turn_count"] += 1
        result, state = build_output(
            text,
            state,
            image_path=image_path,
            enable_vision=True
        )

        validate(instance=result, schema=SCHEMA)

        print("\nBot:", result["response"]["message"])
        print("Department:", result["routing"]["department"])
        print("Intent:", result["intent"]["category"])

        print("\nStructured JSON:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
