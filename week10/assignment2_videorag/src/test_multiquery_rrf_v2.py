from retriever import VideoRetriever


QUERIES = {
    "Q0_ORIGINAL": "我的手機 Wi-Fi 無法連線，怎麼辦？",
    "Q1_WIFI_ENABLED": "手機 Wi-Fi 已經開啟但仍然無法連線，如何排除故障？",
    "Q2_RECONNECTION_FAILED": "重新連線後問題仍然存在，應該怎麼辦？",
    "Q3_ROUTER_CHECK": "檢查無線路由器是否正常運作，或嘗試重新啟動手機。",
}


def multi_query_rrf(retriever, queries, final_k=8):
    fused = {}
    rrf_constant = 60

    for query_name, query in queries.items():
        print("\n" + "=" * 75)
        print(f"{query_name}: {query}")

        results = retriever.retrieve(
            query,
            final_k=8,
        )

        print(f"Retrieved candidates: {len(results)}")

        for query_rank, item in enumerate(results, start=1):
            doc_id = item["id"]

            if doc_id not in fused:
                fused[doc_id] = {
                    **item,
                    "rrf_score": 0.0,
                    "best_distance": item["distance"],
                    "retrieved_from_queries": [],
                }

            fused[doc_id]["rrf_score"] += item["rrf_score"]

            fused[doc_id]["best_distance"] = min(
                fused[doc_id]["best_distance"],
                item["distance"],
            )

            fused[doc_id]["retrieved_from_queries"].append({
                "query_name": query_name,
                "query_rank": query_rank,
                "modality_ranks": item["retrieved_from"],
            })

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -item["rrf_score"],
            item["best_distance"],
        ),
    )

    return ranked[:final_k], len(fused)


def main():
    print("========== 1. INITIALIZE RETRIEVER ==========")

    retriever = VideoRetriever(per_modality_k=8)

    print("\n========== 2. MULTI-QUERY RRF ==========")

    results, unique_count = multi_query_rrf(
        retriever,
        QUERIES,
        final_k=8,
    )

    print("\n========== 3. FUSED TOP 8 ==========")

    speech_count = 0
    visual_count = 0

    for rank, item in enumerate(results, start=1):
        metadata = item["metadata"]
        modality = metadata["modality"]

        if modality == "speech":
            speech_count += 1
        elif modality == "visual_ocr":
            visual_count += 1

        print(f"\n[{rank}] ID: {item['id']}")
        print("Modality:", modality)
        print("Video:", metadata["video_id"])
        print(
            "Timestamp:",
            f"{metadata['start_seconds']:.2f}s - "
            f"{metadata['end_seconds']:.2f}s",
        )
        print("Multi-query RRF score:", f"{item['rrf_score']:.6f}")
        print("Text:", item["text"])

        print("Retrieved from:")
        for source in item["retrieved_from_queries"]:
            print(
                f"  - {source['query_name']} "
                f"(query rank {source['query_rank']})"
            )

    print("\n========== 4. SUMMARY ==========")
    print("Number of queries:", len(QUERIES))
    print("Unique candidates:", unique_count)
    print("Final evidence count:", len(results))
    print("Speech:", speech_count)
    print("Visual OCR:", visual_count)


if __name__ == "__main__":
    main()
