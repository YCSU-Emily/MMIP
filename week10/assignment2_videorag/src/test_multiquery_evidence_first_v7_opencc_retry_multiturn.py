import json
import os
import re
import unicodedata

from opencc import OpenCC

from retriever import VideoRetriever
from conversation_manager import ConversationManager
from test_multiquery_rrf_v2 import multi_query_rrf
from evidence_query_matcher import match_evidence_to_queries
from answer_generator_evidence_first_v4 import (
    EvidenceFirstVideoAnswerGenerator,
)


FAILURE_TEXT = (
    "目前沒有足夠且通過格式驗證的影片證據回答這個問題。"
)

NO_FACT_TEXTS = {
    "沒有可用的明確事實",
    "没有可用的明确事实",
}

# 將最終回答統一轉換為台灣繁體中文。
TRADITIONAL_CONVERTER = OpenCC("s2twp")


def to_traditional_chinese(text):
    return TRADITIONAL_CONVERTER.convert(str(text or ""))


def normalize_answer(text):
    """移除引用標記與標點，供重複答案比對。"""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()

    text = re.sub(
        r"\s*\[證據\s*id\s*[:：].*?秒[］\]]",
        "",
        text,
        flags=re.S,
    )

    return re.sub(
        r"[\s，。！？、；：,.!?;:「」『』（）()\[\]—–\-]+",
        "",
        text,
    )


def mentions_router_restart(text):
    """辨識明確的『重新啟動路由器』表述。"""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()

    patterns = [
        r"(?:重新啟動|重新启动|重啟|重启)\s*(?:一下|一次)?\s*(?:無線|无线)?路由器",
        r"(?:無線|无线)?路由器\s*(?:重新啟動|重新启动|重啟|重启)",
        r"\b(?:reboot|restart)\s+(?:the\s+)?(?:wireless\s+)?router\b",
        r"\b(?:wireless\s+)?router\s+(?:reboot|restart)\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def validate_answer_against_source(answer, source_text):
    """
    定向來源一致性檢查：
    若答案主張重新啟動路由器，但原始影片文字沒有這項操作，
    就拒絕該生成答案。
    """
    if not mentions_router_restart(answer):
        return True, None

    if mentions_router_restart(source_text):
        return True, None

    answer_text = unicodedata.normalize(
        "NFKC", str(answer or "")
    )

    # 允許明確否定或指出影片沒有提及該操作的回答。
    denial_phrases = (
        "沒有提到",
        "未提及",
        "沒有明確提到",
        "沒有明確說明",
        "未說明",
        "並未提到",
        "不需要",
        "不用",
        "不建議",
        "並非",
    )

    for match in re.finditer(
        r"(?:重新啟動|重新启动|重啟|重启)\s*(?:一下|一次)?\s*(?:無線|无线)?路由器",
        answer_text,
    ):
        prefix = answer_text[max(0, match.start() - 20):match.start()]
        if any(phrase in prefix for phrase in denial_phrases):
            return True, None

    return (
        False,
        "答案提到重新啟動路由器，但原始影片證據沒有支持這項操作。",
    )


def is_duplicate_answer(answer, previous_history, current_answers):
    """只過濾與先前回答完全相同的答案，不排除重複使用證據。"""
    candidate = normalize_answer(answer)

    if not candidate:
        return False

    previous_answers = []

    for item in previous_history or []:
        if item.get("role") == "assistant":
            previous_answers.extend(
                item.get("content", "").splitlines()
            )

    previous_answers.extend(current_answers or [])

    return any(
        candidate == normalize_answer(old_answer)
        for old_answer in previous_answers
        if normalize_answer(old_answer)
    )


def is_router_restart_question(query):
    query = unicodedata.normalize(
        "NFKC", str(query or "")
    ).lower()

    has_router = "路由器" in query or "router" in query
    has_restart = any(
        term in query
        for term in (
            "重新啟動",
            "重新启动",
            "重啟",
            "重启",
            "restart",
            "reboot",
        )
    )

    return has_router and has_restart


def build_safe_router_fallback(
    user_query,
    source_text,
    evidence_id,
    start,
    end,
):
    """
    僅在原始證據同時提到：
    1. 檢查路由器；
    2. 重新啟動手機；
    3. 沒有提到重新啟動路由器；
    且使用者確實詢問路由器是否需要重啟時，才提供保守回答。
    """
    source = unicodedata.normalize(
        "NFKC", str(source_text or "")
    )

    if not is_router_restart_question(user_query):
        return None

    has_router = "路由器" in source or "router" in source.lower()

    has_router_check = any(
        term in source
        for term in (
            "檢查",
            "检查",
            "確認",
            "确认",
            "是否正常運作",
            "是否正常运行",
        )
    )

    has_phone_restart = any(
        term in source
        for term in (
            "重新啟動手機",
            "重新启动手机",
            "重啟手機",
            "重启手机",
        )
    )

    if (
        not has_router
        or not has_router_check
        or not has_phone_restart
        or mentions_router_restart(source)
    ):
        return None

    answer = (
        "這段影片內容沒有明確提到需要重新啟動路由器；"
        "只提到檢查無線路由器是否正常運作，"
        "或者嘗試重新啟動手機。"
    )

    return (
        f"{answer} "
        f"[證據 ID: {evidence_id}，"
        f"影片時間：{start:.2f}–{end:.2f} 秒]"
    )



# 三輪測試對話
CONVERSATION = [
    "我的手機 Wi-Fi 無法連線，怎麼辦？",
    "那如果還是不行呢？",
    "路由器也需要重新啟動嗎？",
]


def build_turn_queries(user_query, rewritten_query):
    """
    建立本輪動態檢索查詢。

    每輪一定有目前問題。
    如果是追問，再加入包含前文的上下文查詢。
    """

    queries = {
        "Q0_CURRENT": user_query,
    }

    if rewritten_query.strip() != user_query.strip():
        queries["Q1_CONTEXT"] = rewritten_query

    return queries


def main():

    print("\n========== 1. INITIALIZE RETRIEVER ==========")

    retriever = VideoRetriever(per_modality_k=8)

    print("\n========== 2. INITIALIZE V4 GENERATOR ==========")

    generator = EvidenceFirstVideoAnswerGenerator(
        max_new_tokens=160
    )

    conversation = ConversationManager()

    all_turn_results = []

    for turn_index, user_query in enumerate(
        CONVERSATION,
        start=1,
    ):

        print("\n" + "#" * 85)
        print(f"CONVERSATION TURN {turn_index}")
        print("#" * 85)

        print("\nUser:", user_query)

        # -------------------------------------------------
        # A. Conversation-aware query rewriting
        # -------------------------------------------------

        rewritten_query = conversation.prepare_query(
            user_query
        )

        # 目前使用者問題已加入 history，
        # 因此排除最後一筆，取得先前對話。
        previous_history = conversation.history[:-1]

        print("\n========== A. CONVERSATION CONTEXT ==========")

        print("Is follow-up:",
              conversation.history[-1]["is_follow_up"])

        print("Rewritten query:", rewritten_query)

        print("Previous history:")
        for item in previous_history:
            print(
                f"- {item['role']}: {item['content']}"
            )

        # -------------------------------------------------
        # B. Build dynamic multi-query set
        # -------------------------------------------------

        turn_queries = build_turn_queries(
            user_query,
            rewritten_query,
        )

        print("\n========== B. DYNAMIC QUERIES ==========")

        for query_name, query_text in turn_queries.items():
            print(f"{query_name}: {query_text}")

        # -------------------------------------------------
        # C. Multi-query RRF retrieval
        # -------------------------------------------------

        print("\n========== C. MULTI-QUERY RRF ==========")

        evidence_list, unique_count = multi_query_rrf(
            retriever,
            turn_queries,
            final_k=8,
        )

        print("Unique candidates:", unique_count)
        print("Final evidence count:", len(evidence_list))

        # -------------------------------------------------
        # D. Evidence matching + evidence-first generation
        # -------------------------------------------------

        turn_results = []
        successful_answers = []

        print("\n========== D. EVIDENCE-FIRST GENERATION ==========")

        for evidence_index, target in enumerate(
            evidence_list,
            start=1,
        ):

            evidence_id = target["id"]
            metadata = target.get("metadata", {})

            start = float(
                metadata.get("start_seconds", 0)
            )

            end = float(
                metadata.get("end_seconds", start)
            )

            print("\n" + "=" * 75)
            print(
                f"EVIDENCE {evidence_index}/"
                f"{len(evidence_list)}: {evidence_id}"
            )

            # -------------------------------------------------
            # D1. Match evidence to retrieval queries
            # -------------------------------------------------

            matches = match_evidence_to_queries(
                model=retriever.model,
                evidence_text=target["text"],
                query_sources=target.get(
                    "retrieved_from_queries", []
                ),
                query_texts=turn_queries,
                top_k=2,
            )

            print("\nMatched queries:")

            for rank, match in enumerate(
                matches,
                start=1,
            ):
                print(
                    f"Top-{rank}: {match['query_name']} "
                    f"| similarity={match['similarity']:.6f}"
                )

            if not matches:
                turn_results.append({
                    "evidence_id": evidence_id,
                    "status": "NO_MATCH",
                    "attempts": [],
                    "final_answer": None,
                })
                continue

            # -------------------------------------------------
            # D2. Stage 1: Extract facts from one evidence
            # -------------------------------------------------

            print("\n========== STAGE 1: FACT EXTRACTION ==========")

            fact = generator.extract_single_evidence(
                target
            )

            print("Extracted fact:", fact)

            if (
                not fact
                or not fact.strip()
                or fact.strip() in NO_FACT_TEXTS
            ):

                print("Skip: no usable fact (NO_FACT).")

                turn_results.append({
                    "evidence_id": evidence_id,
                    "status": "NO_FACT",
                    "extracted_fact": fact,
                    "attempts": [],
                    "final_answer": None,
                })

                continue

            fact_record = {
                "id": evidence_id,
                "start": start,
                "end": end,
                "text": target.get("text", ""),
                "fact": fact,
                "retrieved_from_queries": matches,
                "rrf_score": target.get("rrf_score"),
            }

            # -------------------------------------------------
            # D3. Stage 2: Adaptive Top-1 -> Top-2
            # -------------------------------------------------

            attempts = []
            final_answer = None
            selected_query = None
            answer_origin = None

            for rank, match in enumerate(
                matches,
                start=1,
            ):

                query_name = match["query_name"]
                query_text = match["query_text"]

                print("\n" + "-" * 65)
                print(f"STAGE 2 ATTEMPT {rank}")
                print("Query:", query_text)

                answer = generator.generate_answer_from_facts(
                    user_query=query_text,
                    facts=[fact_record],
                    conversation_history=previous_history,
                )

                success = (
                    answer.strip() != FAILURE_TEXT
                    and bool(answer.strip())
                )

                # 保留 answer 作為原始模型輸出；
                # answer_for_output 僅供最終回答使用。
                answer_for_output = (
                    to_traditional_chinese(answer)
                    if success
                    else answer
                )

                rejection_reason = None

                if success:
                    valid, rejection_reason = (
                        validate_answer_against_source(
                            answer,
                            target.get("text", ""),
                        )
                    )

                    if not valid:
                        success = False
                        print(
                            "[V7 VALIDATION REJECTED]",
                            rejection_reason,
                        )

                if success and is_duplicate_answer(
                    answer_for_output,
                    previous_history,
                    successful_answers,
                ):
                    success = False
                    rejection_reason = (
                        "與先前對話或本輪已接受的答案完全重複。"
                    )
                    print(
                        "[V7 DUPLICATE FILTERED]",
                        rejection_reason,
                    )

                attempts.append({
                    "rank": rank,
                    "query_name": query_name,
                    "query_text": query_text,
                    "similarity": match["similarity"],
                    "answer": answer,
                    "answer_for_output": answer_for_output,
                    "success": success,
                    "rejection_reason": rejection_reason,
                })

                if success:
                    selected_query = query_name
                    final_answer = answer_for_output
                    answer_origin = "QWEN_V4_VALIDATED"

                    print("\nFallback result: SUCCESS")
                    break

                print("\nFallback result: REJECTED_OR_NO_ANSWER")

            # -------------------------------------------------
            # D4. Save evidence-level result
            # -------------------------------------------------

            # -------------------------------------------------
            # D3.5. Conservative source-grounded fallback
            # -------------------------------------------------

            if final_answer is None:
                safe_fallback = build_safe_router_fallback(
                    user_query=query_text,
                    source_text=target.get("text", ""),
                    evidence_id=evidence_id,
                    start=start,
                    end=end,
                )

                if safe_fallback and not is_duplicate_answer(
                    safe_fallback,
                    previous_history,
                    successful_answers,
                ):
                    final_answer = safe_fallback
                    selected_query = "RULE_BASED_SAFE_FALLBACK"
                    answer_origin = "RULE_BASED_SOURCE_GROUNDED_FALLBACK"

                    print(
                        "[V7 SAFE FALLBACK] "
                        "使用原始影片文字組成保守回答。"
                    )

            status = (
                "SUCCESS"
                if final_answer is not None
                else "NO_ANSWER"
            )

            if final_answer is not None:
                successful_answers.append(final_answer)

            turn_results.append({
                "evidence_id": evidence_id,
                "status": status,
                "evidence_text": target.get("text"),
                "extracted_fact": fact,
                "matched_queries": matches,
                "attempts": attempts,
                "selected_query": selected_query,
                "answer_origin": answer_origin,
                "final_answer": final_answer,
            })

            print("\nEvidence status:", status)
            print("Selected query:", selected_query)
            print("Answer:", final_answer)

        # -----------------------------------------------------
        # E. Combine this turn's grounded answers
        # -----------------------------------------------------

        final_text = "\n".join(successful_answers)

        if not final_text:
            final_text = FAILURE_TEXT

        # 將本輪回答加入對話歷史，
        # 供下一輪 ConversationManager 使用。
        conversation.add_assistant_message(
            final_text
        )

        success_count = sum(
            item["status"] == "SUCCESS"
            for item in turn_results
        )

        print("\n" + "=" * 75)
        print(f"TURN {turn_index} FINAL ANSWER")
        print("=" * 75)
        print(final_text)

        print("\nTurn evidence count:", len(evidence_list))
        print("Successful evidence count:", success_count)

        all_turn_results.append({
            "turn": turn_index,
            "user_query": user_query,
            "is_follow_up": rewritten_query.strip() != user_query.strip(),
            "rewritten_query": rewritten_query,
            "turn_queries": turn_queries,
            "previous_history": previous_history,
            "retrieval_method": "Dynamic Multi-query RRF Top-8",
            "matching_method": "BGE-M3 Top-2",
            "answer_method": "V4 Evidence-first + V7 source validation and duplicate filtering",
            "evidence_count": len(evidence_list),
            "unique_candidate_count": unique_count,
            "success_count": success_count,
            "results": turn_results,
            "final_answer": final_text,
        })

    # ---------------------------------------------------------
    # F. Save final report
    # ---------------------------------------------------------

    os.makedirs("outputs", exist_ok=True)

    report = {
        "project": "VideoRAG Multi-turn Conversational RAG",
        "conversation_turns": len(all_turn_results),
        "retrieval_method": (
            "Conversation-aware Dynamic Multi-query RRF Top-8"
        ),
        "matching_method": "BGE-M3 Top-2",
        "generation_method": (
            "V4 Evidence-first Stage 1 + Stage 2, with V7 validation"
        ),
        "conversation_history_policy": (
            "Previous user questions are used as context; "
            "previous assistant answers are not treated as video evidence."
        ),
        "turns": all_turn_results,
    }

    report_path = (
        "outputs/v7_opencc_retry_multiturn_videorag_report.json"
    )

    answer_path = (
        "outputs/v7_opencc_retry_multiturn_videorag_answers.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(answer_path, "w", encoding="utf-8") as f:
        for item in all_turn_results:
            f.write(
                f"Turn {item['turn']}\n"
                f"User: {item['user_query']}\n"
                f"Rewritten: {item['rewritten_query']}\n"
                f"Answer:\n{item['final_answer']}\n\n"
                + "=" * 75
                + "\n\n"
            )

    # ---------------------------------------------------------
    # G. Final summary
    # ---------------------------------------------------------

    print("\n" + "#" * 85)
    print("V7 MULTI-TURN VIDEORAG SUMMARY")
    print("#" * 85)

    for item in all_turn_results:
        print(f"\nTurn {item['turn']}: {item['user_query']}")
        print("Rewritten:", item["rewritten_query"])
        print("Evidence:", item["evidence_count"])
        print("Success:", item["success_count"])
        print("Answer:", item["final_answer"])

    print("\nReport saved:", report_path)
    print("Answers saved:", answer_path)


if __name__ == "__main__":
    main()
