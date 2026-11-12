"""
Evidence-Query Matching
Use BGE-M3 to match video evidence with retrieved query provenance.

Query text is relevance context only.
It must never be treated as video evidence.
"""


def match_evidence_to_queries(
    model,
    evidence_text,
    query_sources,
    query_texts,
    top_k=2,
):
    """
    Match one evidence item against the queries that retrieved it.

    Args:
        model: Existing SentenceTransformer BGE-M3 model.
        evidence_text: Extracted fact from one video evidence item.
        query_sources: Its retrieved_from_queries list.
        query_texts: Dict mapping query_name to original query text.
        top_k: Number of top matches to return.

    Returns:
        Ranked list of query matches with similarity and retrieval rank.
    """
    candidates = []

    for source in query_sources:
        query_name = source.get("query_name")
        query_text = query_texts.get(query_name)

        if not query_name or not query_text:
            continue

        candidates.append({
            "query_name": query_name,
            "query_text": query_text,
            "query_rank": source.get("query_rank"),
            "modality_ranks": source.get("modality_ranks", []),
        })

    if not evidence_text.strip() or not candidates:
        return []

    texts = [evidence_text] + [
        candidate["query_text"] for candidate in candidates
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    evidence_vector = embeddings[0]
    query_vectors = embeddings[1:]

    for candidate, query_vector in zip(candidates, query_vectors):
        candidate["similarity"] = float(
            evidence_vector @ query_vector
        )

    ranked = sorted(
        candidates,
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return ranked[:max(1, top_k)]
