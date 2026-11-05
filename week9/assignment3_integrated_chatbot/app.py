import sys
import tempfile
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------
# Import Assignment 3 backend
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "src"))

from integrated_chatbot import IntegratedChatbot


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="centered",
)


# ---------------------------------------------------------
# Persistent session state
# ---------------------------------------------------------
if "bot" not in st.session_state:
    st.session_state.bot = IntegratedChatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "您好！我是 AI 智慧客服助手。"
                "您可以詢問產品資訊、訂單、保固或維修問題，"
                "也可以上傳產品圖片讓我協助辨識。"
            ),
        }
    ]


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("🤖 AI 客服系統")
    st.caption("Assignment 3 - Integrated Multimodal Chatbot")

    st.divider()
    st.subheader("系統功能")
    st.markdown(
        """
        - 💬 多輪對話
        - 🖼️ 圖片辨識
        - ⚡ Fast / Deep Routing
        - 🔐 Prompt Injection 防禦
        - 🧠 Session Memory
        - 🔄 主題切換偵測
        """
    )

    st.divider()

    if st.button("🗑️ 重新開始對話", use_container_width=True):
        st.session_state.bot.reset_session()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "您好！請問今天有什麼問題需要協助？",
            }
        ]
        st.rerun()

    st.caption("Powered by Qwen + Assignment 1 & 2")


# ---------------------------------------------------------
# Main interface
# ---------------------------------------------------------
st.title("💬 AI 智慧客服")
st.caption("支援文字、圖片、多輪對話及智慧分流。")
st.divider()


# ---------------------------------------------------------
# Display conversation history
# ---------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("image"):
            st.image(message["image"], caption="使用者上傳圖片")

        if message.get("metadata") or message.get("security"):
            with st.expander("查看系統處理資訊"):
                st.json({
                    "routing_metadata": message.get("metadata", {}),
                    "security": message.get("security", {}),
                })


# ---------------------------------------------------------
# Input form: text + image
# ---------------------------------------------------------
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input("請輸入您的問題")
    uploaded_image = st.file_uploader(
        "上傳產品圖片（JPG / PNG / WEBP，最多 10 MB）",
        type=["jpg", "jpeg", "png", "webp"],
    )
    submitted = st.form_submit_button("送出訊息", use_container_width=True)


# ---------------------------------------------------------
# Process user turn
# ---------------------------------------------------------
if submitted:
    if not user_text.strip() and uploaded_image is None:
        st.warning("請輸入文字或上傳圖片。")
        st.stop()

    image_bytes = uploaded_image.getvalue() if uploaded_image else None

    st.session_state.messages.append({
        "role": "user",
        "content": user_text.strip() or "請協助辨識這張圖片。",
        "image": image_bytes,
    })

    temp_image_path = None

    try:
        with st.spinner("AI 正在分析您的問題，請稍候..."):
            if uploaded_image is not None:
                suffix = Path(uploaded_image.name).suffix.lower()

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as temp_file:
                    temp_file.write(image_bytes)
                    temp_image_path = temp_file.name

            result = st.session_state.bot.process_turn(
                user_text=user_text,
                image_path=temp_image_path,
                enable_deep=True,
            )

        assistant_text = result["response"]["message"]

        metadata = dict(st.session_state.bot.last_metadata)
        security = dict(st.session_state.bot.last_security)

    except Exception as exc:
        assistant_text = (
            "處理訊息時發生錯誤，請稍後再試。\n\n"
            f"錯誤資訊：`{type(exc).__name__}: {exc}`"
        )
        metadata = {}
        security = {}

    finally:
        if temp_image_path and Path(temp_image_path).exists():
            Path(temp_image_path).unlink()

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_text,
        "metadata": metadata,
        "security": security,
    })

    st.rerun()
