import uuid

import streamlit as st

from src.rag_engine import RAGEngine


st.set_page_config(
    page_title="Smart3C RAG 客服助理",
    page_icon="💻",
    layout="wide",
)


@st.cache_resource
def load_rag_engine():
    return RAGEngine(
        persist_directory="vector_db_canonical_v4",
        collection_name="smart3c_faq_canonical_v4",
        top_k=3,
        max_history_turns=4,
    )


st.title("💻 Smart3C 智慧客服助理")
st.caption("結合 BGE-M3 向量檢索與 Qwen3.5-4B，支援多輪對話及參考資料查詢。")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ 設定")
    top_k = st.slider("檢索文件數量（Top-K）", 1, 5, 3)

    st.caption("目前使用 Smart3C Canonical V4 知識庫。")

    if st.button("🗑️ 清除對話", use_container_width=True):
        if "rag_engine" in st.session_state:
            st.session_state.rag_engine.clear_history(
                st.session_state.session_id
            )
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

try:
    engine = load_rag_engine()
    st.session_state.rag_engine = engine
except Exception as error:
    st.error(f"RAG Engine 載入失敗：{error}")
    st.stop()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "result" in message:
            result = message["result"]

            st.caption(f"檢索問題：{result['retrieval_query']}")

            with st.expander("📚 References 參考來源"):
                references = result["references"]

                if references:
                    for index, ref in enumerate(references, start=1):
                        st.markdown(
                            f"{index}. **{ref.get('title') or '未提供標題'}**"
                        )
                        st.write(f"發布者：{ref.get('publisher') or '未提供'}")
                        st.write(f"驗證日期：{ref.get('verified_at') or '未提供'}")
                        st.markdown(f"[開啟來源]({ref['url']})")
                else:
                    st.info("本次檢索沒有可顯示的參考來源。")

            with st.expander("🔎 檢索文件與相似度"):
                for doc in result["retrieved_documents"]:
                    st.markdown(
                        f"**{doc.get('rank', '')}. "
                        f"{doc.get('question', '未提供問題')}**"
                    )
                    st.write(f"分類：{doc.get('category', '未分類')}")
                    st.write(
                        f"相似度：{doc.get('similarity', '未提供')}"
                    )
                    st.write(doc.get("answer", ""))
                    st.divider()


question = st.chat_input("請輸入你的手機、電腦或網路問題……")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    try:
        with st.spinner("正在檢索資料並產生回答……"):
            result = engine.ask(
                question,
                session_id=st.session_state.session_id,
                top_k=top_k,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "result": result,
        })

    except Exception as error:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"處理問題時發生錯誤：{error}",
        })

    st.rerun()
