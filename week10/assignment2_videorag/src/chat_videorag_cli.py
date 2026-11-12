import json
from datetime import datetime
from pathlib import Path

from retriever import VideoRetriever
from conversation_manager import ConversationManager
from test_multiquery_rrf_v2 import multi_query_rrf
from evidence_query_matcher import match_evidence_to_queries
from answer_generator_evidence_first_v4 import (
    EvidenceFirstVideoAnswerGenerator,
)

from test_multiquery_evidence_first_v7_opencc_retry_multiturn import (
    to_traditional_chinese,
    validate_answer_against_source,
    is_duplicate_answer,
    build_safe_router_fallback,
    build_turn_queries,
    FAILURE_TEXT,
    NO_FACT_TEXTS,
)


def save_session(report, report_path):
    """每輪結束後儲存對話紀錄。"""
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )


def answer_one_turn(
    user_query,
    retriever,
    generator,
    conversation,
):
    """執行一輪完整的 VideoRAG 檢索與回答流程。"""

    rewritten_query = conversation.prepare_query(user_query)
    previous_history = conversation.history[:-1]

    is_follow_up = conversation.history[-1]["is_follow_up"]

    print("\n[對話管理]")
    print("是否為追問：", is_follow_up)
    print("改寫後查詢：", rewritten_query)

    turn_queries = build_turn_queries(
        user_query,
        rewritten_query,
    )

    print("\n[多查詢檢索]")
    evidence_list, unique_count = multi_query_rrf(
        retriever,
        turn_queries,
        final_k=8,
    )

    print("\n候選證據數：", unique_count)
    print("最終證據數：", len(evidence_list))

    successful_answers = []
    evidence_results = []

    for index, target in enumerate(evidence_list, start=1):
        evidence_id = target["id"]
        metadata = target.get("metadata", {})

        start = float(metadata.get("start_seconds", 0))
        end = float(metadata.get("end_seconds", start))
        source_text = target.get("text", "")

        print(
            f"\n[證據 {index}/{len(evidence_list)}] "
            f"{evidence_id} ({start:.2f}–{end:.2f}s)"
        )

        # D1. 證據與查詢匹配
        matches = match_evidence_to_queries(
            model=retriever.model,
            evidence_text=source_text,
            query_sources=target.get(
                "retrieved_from_queries", []
            ),
            query_texts=turn_queries,
            top_k=2,
        )

        if not matches:
            evidence_results.append({
                "evidence_id": evidence_id,
                "status": "NO_MATCH",
                "final_answer": None,
            })
            continue

        # D2. Stage 1：獨立擷取單筆證據事實
        fact = generator.extract_single_evidence(target)

        print("擷取事實：", fact)

        if (
            not fact
            or not fact.strip()
            or fact.strip() in NO_FACT_TEXTS
        ):
            evidence_results.append({
                "evidence_id": evidence_id,
                "status": "NO_FACT",
                "extracted_fact": fact,
                "final_answer": None,
            })
            continue

        fact_record = {
            "id": evidence_id,
            "start": start,
            "end": end,
            "text": source_text,
            "fact": fact,
            "retrieved_from_queries": matches,
            "rrf_score": target.get("rrf_score"),
        }

        # D3. Stage 2：依序嘗試 Top-1、Top-2 查詢
        attempts = []
        final_answer = None
        selected_query = None

        for rank, match in enumerate(matches, start=1):
            query_text = match["query_text"]

            raw_answer = generator.generate_answer_from_facts(
                user_query=query_text,
                facts=[fact_record],
                conversation_history=previous_history,
            )

            answer_for_output = (
                to_traditional_chinese(raw_answer)
                if raw_answer.strip() != FAILURE_TEXT
                else raw_answer
            )

            success = (
                bool(raw_answer.strip())
                and raw_answer.strip() != FAILURE_TEXT
            )

            rejection_reason = None

            # V7：來源一致性檢查
            if success:
                valid, rejection_reason = (
                    validate_answer_against_source(
                        raw_answer,
                        source_text,
                    )
                )
                if not valid:
                    success = False

            # V7：避免與先前回答完全重複
            if success and is_duplicate_answer(
                answer_for_output,
                previous_history,
                successful_answers,
            ):
                success = False
                rejection_reason = "與先前回答完全重複。"

            attempts.append({
                "rank": rank,
                "query_name": match["query_name"],
                "query_text": query_text,
                "similarity": match["similarity"],
                "raw_answer": raw_answer,
                "success": success,
                "rejection_reason": rejection_reason,
            })

            if success:
                final_answer = answer_for_output
                selected_query = match["query_name"]
                break

        answer_origin = (
            "QWEN_V4_VALIDATED"
            if final_answer is not None
            else None
        )

        # D4. V7：特定路由器問題的保守 fallback
        if final_answer is None:
            safe_fallback = build_safe_router_fallback(
                user_query=user_query,
                source_text=source_text,
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

        status = (
            "SUCCESS"
            if final_answer is not None
            else "NO_ANSWER"
        )

        if final_answer is not None:
            successful_answers.append(final_answer)

        evidence_results.append({
            "evidence_id": evidence_id,
            "status": status,
            "evidence_text": source_text,
            "extracted_fact": fact,
            "matched_queries": matches,
            "attempts": attempts,
            "selected_query": selected_query,
            "answer_origin": answer_origin,
            "final_answer": final_answer,
        })

    final_text = "\n".join(successful_answers)

    if not final_text:
        final_text = FAILURE_TEXT

    # 保存本輪回答，供下一輪理解上下文
    conversation.add_assistant_message(final_text)

    return {
        "user_query": user_query,
        "is_follow_up": is_follow_up,
        "rewritten_query": rewritten_query,
        "turn_queries": turn_queries,
        "retrieval_method": "Dynamic Multi-query RRF Top-8",
        "matching_method": "BGE-M3 Top-2",
        "answer_method": "V4 Evidence-first + V7 validation",
        "unique_candidate_count": unique_count,
        "evidence_count": len(evidence_list),
        "success_count": sum(
            item["status"] == "SUCCESS"
            for item in evidence_results
        ),
        "results": evidence_results,
        "final_answer": final_text,
    }


def main():
    print("=" * 65)
    print("       VideoRAG Multi-turn Conversational Chatbot")
    print("=" * 65)

    # 建立獨立的互動紀錄資料夾
    output_dir = Path("outputs/interactive_sessions")
    output_dir.mkdir(parents=True, exist_ok=True)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = (
        output_dir
        / f"videorag_session_{session_id}.json"
    )

    report = {
        "project": "VideoRAG Interactive Multi-turn Chatbot",
        "session_id": session_id,
        "turns": [],
    }

    print("\n[1/2] 載入 BGE-M3 與 Video Corpus...")
    retriever = VideoRetriever(per_modality_k=8)

    print("\n[2/2] 載入 Qwen2.5-3B-Instruct...")
    generator = EvidenceFirstVideoAnswerGenerator(
        max_new_tokens=160
    )

    conversation = ConversationManager()

    print("\n" + "=" * 65)
    print("VideoRAG 已準備完成！")
    print("直接輸入問題即可開始聊天。")
    print("輸入 reset：重設對話")
    print("輸入 exit：結束聊天")
    print("=" * 65)

    turn_index = 0

    try:
        while True:
            try:
                user_query = input("\n你：").strip()
            except EOFError:
                break

            if not user_query:
                continue

            command = user_query.lower()

            if command in ("exit", "quit", "q"):
                print("\n已結束 VideoRAG 對話。")
                break

            if command == "reset":
                conversation.reset()
                print("\n已清空對話歷史，可以開始新的主題。")
                continue

            turn_index += 1

            print("\n" + "-" * 65)
            print(f"開始第 {turn_index} 輪對話")
            print("-" * 65)

            try:
                result = answer_one_turn(
                    user_query=user_query,
                    retriever=retriever,
                    generator=generator,
                    conversation=conversation,
                )
            except Exception as error:
                # 本輪發生錯誤時，不保留不完整的使用者訊息。
                if (
                    conversation.history
                    and conversation.history[-1].get("role") == "user"
                ):
                    conversation.history.pop()

                print("\n本輪處理發生錯誤：", repr(error))
                print("可以重新輸入問題，或輸入 exit 結束。")
                turn_index -= 1
                continue

            report["turns"].append({
                "turn": turn_index,
                **result,
            })

            save_session(report, report_path)

            print("\n" + "=" * 65)
            print("VideoRAG：")
            print(result["final_answer"])
            print("=" * 65)

            print(
                f"本輪證據數：{result['evidence_count']} | "
                f"成功回答數：{result['success_count']}"
            )

            print("對話紀錄已保存：", report_path)

    except KeyboardInterrupt:
        print("\n收到中斷訊號，正在結束對話...")

    finally:
        save_session(report, report_path)
        print("\n本次互動紀錄：", report_path)
        print("總對話輪數：", len(report["turns"]))


if __name__ == "__main__":
    main()
