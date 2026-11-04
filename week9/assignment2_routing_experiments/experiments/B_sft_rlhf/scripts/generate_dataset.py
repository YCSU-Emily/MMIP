import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B_DIR = ROOT / "experiments" / "B_sft_rlhf"
DATA_DIR = B_DIR / "data"

sys.path.insert(0, str(ROOT / "evaluation"))
from routing_prompt import build_routing_prompt


# ---------------------------------------------------------
# 1. Synthetic customer messages
#    These are independent training/validation examples.
#    Benchmark B01-B10 are never loaded here.
# ---------------------------------------------------------

TRAIN = {
    "faq": [
        ("我的 ASUS 筆電第一次開機，應該怎麼設定？", "ASUS", None, None),
        ("請問 Lenovo 筆電要如何連接家裡的 Wi-Fi？", "Lenovo", None, None),
        ("Dell 筆電的藍牙要去哪裡開啟？", "Dell", None, None),
        ("Acer 筆電觸控板突然不能使用，有沒有設定方法？", "Acer", None, None),
        ("HP 筆電平常應該怎麼保養電池？", "HP", None, None),
        ("MacBook 要怎麼查看目前的 macOS 版本？", "Apple MacBook", None, None),
        ("請教我如何在 ASUS 筆電上安裝印表機。", "ASUS", None, None),
        ("Lenovo 電腦如何調整螢幕亮度？", "Lenovo", None, None),
        ("Dell 筆電可以怎麼切換輸入法？", "Dell", None, None),
        ("Acer 筆電要如何清理暫存檔案？", "Acer", None, None),
        ("HP 筆電如何設定開機密碼？", "HP", None, None),
        ("MacBook 的截圖快捷鍵是什麼？", "Apple MacBook", None, None),
        ("ASUS 筆電如何連接外接螢幕？", "ASUS", None, None),
        ("Lenovo 筆電可以在哪裡查看硬碟容量？", "Lenovo", None, None),
    ],
    "order_tracking": [
        ("請幫我查詢訂單 AB731205 的配送狀態。", None, None, "AB731205"),
        ("訂單 CD820416 到哪裡了？", None, None, "CD820416"),
        ("我想知道訂單 EF194728 什麼時候送達。", None, None, "EF194728"),
        ("麻煩追蹤一下訂單 GH520183。", None, None, "GH520183"),
        ("請問訂單 JK630914 是否已經出貨？", None, None, "JK630914"),
        ("可以查一下訂單 LM482760 的物流嗎？", None, None, "LM482760"),
        ("我的訂單 NP715204 還沒收到，想確認進度。", None, None, "NP715204"),
        ("請協助查詢訂單 QR390621 的配送資訊。", None, None, "QR390621"),
        ("訂單 ST841206 目前在哪個物流站？", None, None, "ST841206"),
        ("我想追蹤一下 UV275193 的包裹。", None, None, "UV275193"),
        ("請問我之前買的商品寄出了嗎？", None, None, None),
        ("我的包裹還沒到，可以幫忙查物流嗎？", None, None, None),
        ("我想查一下購物訂單目前的進度。", None, None, None),
        ("請問商品配送到哪個階段了？", None, None, None),
    ],
    "warranty_check": [
        ("我的 ASUS 筆電還在保固期內嗎？", "ASUS", None, None),
        ("想確認 Lenovo 筆電目前是否有保固。", "Lenovo", None, None),
        ("Dell 螢幕的保固期限還有效嗎？", "Dell", None, None),
        ("請問 Acer 筆電可以查詢剩餘保固時間嗎？", "Acer", None, None),
        ("HP 印表機目前還有原廠保固嗎？", "HP", None, None),
        ("我的 MacBook 是否還在保固範圍？", "Apple MacBook", None, None),
        ("想查一下 ASUS 主機板的保固狀態。", "ASUS", None, None),
        ("Lenovo 平板的保固到什麼時候？", "Lenovo", None, None),
        ("請幫我確認 Dell 筆電的原廠保固。", "Dell", None, None),
        ("Acer 螢幕是否還能申請保固服務？", "Acer", None, None),
        ("HP 筆電的保固期限可以幫我確認嗎？", "HP", None, None),
        ("我的 MacBook Air 還有 Apple 保固嗎？", "Apple MacBook", "MacBook Air", None),
        ("請問 ASUS 顯示卡是否仍在保固內？", "ASUS", None, None),
        ("我想查詢 Lenovo 電腦的保固期限。", "Lenovo", None, None),
    ],
    "repair_request": [
        ("我的 ASUS 筆電無法開機，希望申請維修。", "ASUS", None, "無法開機"),
        ("Lenovo 筆電鍵盤有幾個按鍵失靈，想送修。", "Lenovo", None, "鍵盤按鍵失靈"),
        ("Dell 螢幕出現閃爍問題，需要維修。", "Dell", None, "螢幕閃爍"),
        ("Acer 筆電的充電孔接觸不良，請協助安排維修。", "Acer", None, "充電孔接觸不良"),
        ("HP 印表機一直卡紙，我想申請檢修。", "HP", None, "印表機卡紙"),
        ("MacBook 螢幕出現異常，希望送修。", "Apple MacBook", None, "螢幕異常"),
        ("ASUS 筆電風扇有異音，想請維修人員檢查。", "ASUS", None, "風扇異音"),
        ("Lenovo 筆電電池無法正常充電，需要維修。", "Lenovo", None, "電池無法充電"),
        ("Dell 筆電的 USB 插槽故障，想申請維修服務。", "Dell", None, "USB 插槽故障"),
        ("Acer 筆電開機後一直當機，請幫我安排檢修。", "Acer", None, "開機後當機"),
        ("HP 筆電的喇叭沒有聲音，需要送修。", "HP", None, "喇叭無聲"),
        ("MacBook Air 的觸控板失靈，想要維修。", "Apple MacBook", "MacBook Air", "觸控板失靈"),
        ("我的 ASUS 筆電散熱異常，想請你們檢查。", "ASUS", None, "散熱異常"),
        ("Lenovo 平板螢幕觸控沒有反應，需要維修。", "Lenovo", None, "觸控失效"),
    ],
    "return_refund": [
        ("我想退回剛買的 ASUS 筆電，請問如何辦理？", "ASUS", None, None),
        ("收到的 Lenovo 商品不符合需求，希望申請退貨。", "Lenovo", None, None),
        ("Dell 螢幕買錯尺寸了，我想辦理退款。", "Dell", None, None),
        ("請問 Acer 商品可以申請退貨退款嗎？", "Acer", None, None),
        ("HP 印表機收到後發現不適合，想退貨。", "HP", None, None),
        ("我想取消購買的 MacBook 並申請退款。", "Apple MacBook", None, None),
        ("ASUS 商品還沒使用，可以協助我辦理退貨嗎？", "ASUS", None, None),
        ("買錯 Lenovo 型號了，想詢問退換貨流程。", "Lenovo", None, None),
        ("Dell 商品已經收到，但我希望辦理退款。", "Dell", None, None),
        ("請幫我處理 Acer 筆電的退貨申請。", "Acer", None, None),
        ("HP 商品有購買後不需要的情況，想申請退貨。", "HP", None, None),
        ("我想將 MacBook 退回並拿回款項。", "Apple MacBook", None, None),
        ("請問 ASUS 筆電的退貨程序要怎麼開始？", "ASUS", None, None),
        ("Lenovo 商品想辦理退貨，請協助處理。", "Lenovo", None, None),
    ],
    "human_handoff": [
        ("我想直接和真人客服人員聯絡。", None, None, None),
        ("請幫我轉接人工客服，不想再使用自動回覆。", None, None, None),
        ("這個問題需要專人協助，麻煩安排客服聯繫。", None, None, None),
        ("我希望由真人客服處理我的案件。", None, None, None),
        ("可以幫我轉給客服專員嗎？", None, None, None),
        ("我需要和客服人員通話，請協助轉接。", None, None, None),
        ("請安排人工服務，我想向專人說明問題。", None, None, None),
        ("自動客服無法解決，請轉接真人。", None, None, None),
    ],
    "unknown": [
        ("我有一個問題想詢問。", None, None, None),
        ("可以幫幫我嗎？", None, None, None),
        ("我想了解一些事情，但不知道怎麼說。", None, None, None),
        ("您好，我需要協助。", None, None, None),
        ("我遇到狀況了，請問可以處理嗎？", None, None, None),
        ("有件事情想請教你們。", None, None, None),
        ("我不太確定該詢問哪一項服務。", None, None, None),
        ("請問有人可以協助我嗎？", None, None, None),
    ],
    "other": [
        ("我想詢問你們門市的營業時間。", None, None, None),
        ("請問最近的服務據點在哪裡？", None, None, None),
        ("我想知道客服服務時間。", None, None, None),
        ("請問有沒有產品型錄可以參考？", None, None, None),
    ],
}

VALIDATION = {
    "faq": [
        ("請問 HP 筆電如何設定外接滑鼠？", "HP", None, None),
        ("ASUS 筆電如何調整鍵盤背光？", "ASUS", None, None),
    ],
    "order_tracking": [
        ("我的物流單還沒有更新，可以查詢嗎？", None, None, None),
        ("請查詢訂單 WX618205 的出貨進度。", None, None, "WX618205"),
    ],
    "warranty_check": [
        ("請協助確認 Acer 筆電是否仍有保固。", "Acer", None, None),
    ],
    "repair_request": [
        ("Dell 筆電的電源鍵壞掉了，我想送修。", "Dell", None, "電源鍵故障"),
    ],
    "return_refund": [
        ("我收到的 HP 商品想辦理退貨退款。", "HP", None, None),
    ],
    "human_handoff": [
        ("請讓我和真人服務人員談話。", None, None, None),
    ],
    "unknown": [
        ("我不知道要怎麼描述遇到的問題。", None, None, None),
    ],
    "other": [
        ("請問你們客服幾點開始服務？", None, None, None),
    ],
}


ROUTES = {
    "faq": ("general_support", "answer_faq"),
    "order_tracking": ("order_support", "track_order"),
    "warranty_check": ("warranty_service", "check_warranty"),
    "repair_request": ("repair_center", "create_repair_request"),
    "return_refund": ("returns_refunds", "start_return_refund"),
    "human_handoff": ("human_agent", "handoff_to_human"),
    "unknown": ("general_support", "ask_follow_up"),
    "other": ("general_support", "ask_follow_up"),
}


def make_record(split, index, category, item):
    text, product, model, problem_or_order = item

    order_id = None
    problem = None

    if category == "order_tracking":
        order_id = problem_or_order
    elif category == "repair_request":
        problem = problem_or_order

    needs_clarification = (
        category == "unknown"
        or (category == "order_tracking" and order_id is None)
    )

    department, action = ROUTES[category]

    # Order tracking without an order ID requires clarification.
    if category == "order_tracking" and order_id is None:
        action = "ask_follow_up"

    session_id = f"B_SFT_{split}_{index:03d}"

    collected = []
    missing = []

    for key, value in [
        ("product_name", product),
        ("product_model", model),
        ("order_id", order_id),
        ("problem_description", problem),
    ]:
        if value is not None:
            collected.append(f"{key}: {value}")

    if category == "order_tracking" and order_id is None:
        missing.append("order_id")

    if category in ("warranty_check", "return_refund", "repair_request"):
        missing.append("purchase_date_or_order_id")

    if category == "repair_request" and product is None:
        missing.append("product_name")

    if category == "unknown":
        missing.append("specific_request")

    if category == "order_tracking" and order_id is None:
        response_message = "請提供訂單編號，才能協助查詢物流進度。"
    elif category == "unknown":
        response_message = "請描述您遇到的問題或想辦理的服務，我會協助您確認。"
    elif category == "human_handoff":
        response_message = "已收到您希望聯繫人工客服的需求，將協助您轉接專人。"
    elif category == "repair_request":
        response_message = "已收到您的維修需求，可以先協助建立維修案件；後續再確認產品及購買資訊。"
    elif category == "return_refund":
        response_message = "已收到您的退貨退款需求，將協助啟動申請流程，後續再確認相關購買資訊。"
    elif category == "warranty_check":
        response_message = "已收到您的保固查詢需求，接下來可確認產品及購買資訊。"
    elif category == "order_tracking":
        response_message = f"已收到訂單 {order_id} 的查詢需求，將協助確認物流進度。"
    elif category == "faq":
        response_message = "已收到您的產品使用問題，我可以協助您查詢相關操作方式。"
    else:
        response_message = "已收到您的詢問，請提供更多需求細節，以便協助您。"

    output = {
        "intent": {
            "category": category,
            "confidence": 0.95,
            "needs_clarification": needs_clarification
        },
        "entities": {
            "product_name": product,
            "product_model": model,
            "order_id": order_id,
            "purchase_date": None,
            "problem_description": problem
        },
        "image_analysis": {
            "image_provided": False,
            "image_type": "none",
            "observations": [],
            "visible_damage": None,
            "limitations": []
        },
        "conversation_state": {
            "session_id": session_id,
            "turn_count": 1,
            "collected_information": collected,
            "missing_information": missing,
            "previous_intent": None,
            "awaiting_user_input": needs_clarification
        },
        "routing": {
            "department": department,
            "action": action,
            "priority": (
                "high" if category in ("repair_request", "human_handoff")
                else "normal"
            ),
            "requires_human": category == "human_handoff",
            "reason": f"根據客戶需求判定為 {category}，分派至 {department}。"
        },
        "response": {
            "message": response_message,
            "language": "zh-TW",
            "suggested_questions": (
                ["請提供訂單編號。"]
                if category == "order_tracking" and order_id is None
                else []
            )
        }
    }

    prompt = build_routing_prompt(text, session_id)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": json.dumps(output, ensure_ascii=False)
            }
        ],
        "metadata": {
            "sample_id": session_id,
            "intent": category,
            "has_image": False
        }
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, source in [
        ("train", TRAIN),
        ("validation", VALIDATION)
    ]:
        records = []
        index = 0

        for category, examples in source.items():
            for item in examples:
                index += 1
                records.append(
                    make_record(split, index, category, item)
                )

        output_path = DATA_DIR / f"{split}.jsonl"

        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"{split}: {len(records)} samples -> {output_path}")

    summary = {
        "experiment": "B - Qwen3.5-4B LoRA SFT",
        "training_samples": sum(map(len, TRAIN.values())),
        "validation_samples": sum(map(len, VALIDATION.values())),
        "image_training_samples": 0,
        "test_benchmark": "benchmark/benchmark_v1.json",
        "test_samples_reserved": 10,
        "data_generation": "deterministic synthetic examples",
        "note": (
            "All training and validation samples are text-only. "
            "The benchmark B01-B10 is reserved for final evaluation."
        )
    }

    (DATA_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nDataset generation completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
