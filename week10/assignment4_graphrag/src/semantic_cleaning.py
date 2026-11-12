import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("week10/assignment4_graphrag")
INPUT_FILE = PROJECT / "results/cleaned_knowledge.json"

# --------------------------------------------------
# 1. Canonical entity names
# --------------------------------------------------

ENTITY_ALIASES = {
    "Graph RAG": "GraphRAG",
    "entity knowledge graph": "Knowledge Graph",
    "knowledge graph": "Knowledge Graph",
    "community-level summaries": "Community Summaries",
    "community summaries": "Community Summaries",
    "Podcast dataset": "Podcast Transcripts",
    "Podcast transcripts": "Podcast Transcripts",
    "News dataset": "News Articles",
    "News articles": "News Articles",
    "retrieval-augmented generation": "Retrieval-Augmented Generation (RAG)",
    "RAG": "Retrieval-Augmented Generation (RAG)",
    "conventional vector RAG (SS)": "SS (Semantic Search / Vector RAG)",
    "SS": "SS (Semantic Search / Vector RAG)",
    "na¨ıve RAG": "Naive RAG",
}

# --------------------------------------------------
# 2. Entity type corrections
# --------------------------------------------------

ENTITY_TYPE_OVERRIDES = {
    "LangChain": "Framework",
    "explicit fact retrieval": "Task",
    "Token limit": "Parameter",
    "distance thresholds": "Parameter",
    "fixed context window size of 8k tokens": "Parameter",
    "Experiment 1": "Experiment",
    "Experiment 2": "Experiment",
    "C0": "Experimental Condition",
    "C1": "Experimental Condition",
    "C2": "Experimental Condition",
    "C3": "Experimental Condition",
    "TS": "Experimental Condition",
    "SS (Semantic Search / Vector RAG)": "Experimental Condition",
}

ENTITY_DESCRIPTION_OVERRIDES = {
    "C0": "Experimental condition using root-level community summaries.",
    "C1": "Experimental condition using a higher level of the GraphRAG community hierarchy.",
    "C2": "Experimental condition using an intermediate level of the GraphRAG community hierarchy.",
    "C3": "Experimental condition using low-level community summaries.",
    "TS": "Text Summarization baseline applying map-reduce directly to source texts without a graph index.",
    "SS (Semantic Search / Vector RAG)": "Semantic Search baseline implementing vector RAG by retrieving text chunks into the context window.",
    "LangChain": "Open-source framework that provides an extension integrating the GraphRAG approach.",
    "explicit fact retrieval": "A task focused on retrieving specific facts, contrasted with global sensemaking.",
    "distance thresholds": "Experimental parameter used when analyzing clustering results.",
    "fixed context window size of 8k tokens": "Experimental parameter specifying an 8,000-token context window.",
}

# These are illustrative entities from the paper's example corpus,
# not central entities in the GraphRAG method.
EXCLUDED_ENTITIES = {
    "NeoChip",
    "Quantum Systems",
    "NewTech Exchange",
}


def canonical_name(name):
    return ENTITY_ALIASES.get(name, name)


def merge_entities(entities):
    merged = {}

    for entity in entities:
        original_name = entity["name"]

        if original_name in EXCLUDED_ENTITIES:
            continue

        name = canonical_name(original_name)

        if name not in merged:
            merged[name] = {
                **entity,
                "name": name,
                "source_pages": [],
                "evidence_records": [],
            }

        target = merged[name]

        # Preserve every page/evidence pair.
        pages = entity.get("source_pages", [])
        records = entity.get("evidence_records", [])

        if len(pages) != len(records):
            raise ValueError(
                f"Evidence/page mismatch for entity: {original_name}"
            )

        for page, evidence in zip(pages, records):
            pair = (page, evidence)
            existing = list(zip(
                target["source_pages"],
                target["evidence_records"]
            ))

            if pair not in existing:
                target["source_pages"].append(page)
                target["evidence_records"].append(evidence)

        # Apply corrected entity types.
        if name in ENTITY_TYPE_OVERRIDES:
            target["type"] = ENTITY_TYPE_OVERRIDES[name]

        # Prefer curated descriptions where available.
        if name in ENTITY_DESCRIPTION_OVERRIDES:
            target["description"] = ENTITY_DESCRIPTION_OVERRIDES[name]

        target["review_status"] = "semantically_reviewed"

    # Rebuild compact evidence fields and sort page/evidence pairs.
    result = []

    for name, entity in merged.items():
        pairs = sorted(
            zip(entity["source_pages"], entity["evidence_records"]),
            key=lambda item: (item[0], item[1])
        )

        entity["source_pages"] = [page for page, _ in pairs]
        entity["evidence_records"] = [evidence for _, evidence in pairs]
        entity["evidence"] = " | ".join(dict.fromkeys(
            entity["evidence_records"]
        ))

        result.append(entity)

    return sorted(result, key=lambda item: item["name"])




# --------------------------------------------------
# 3. Relationship semantic corrections
# --------------------------------------------------

# Original relationship IDs are 1-based and refer to
# the original ordering in cleaned_knowledge.json.

RELATIONSHIP_REMOVALS = {
    29: "Insufficient evidence for a direct GraphRAG-Claims relation.",
    30: "Illustrative NeoChip example corpus.",
    31: "Illustrative NeoChip example corpus.",
    32: "Illustrative NeoChip example corpus.",
    36: "Incorrectly represents duplicate-entity handling.",
    45: "Semantically invalid Global Answer-Token limit relation.",
    75: "Evidence does not support the consists_of relation.",
    81: "Comprehensiveness and Diversity are independent metrics.",
    88: "Evidence is from the bibliography rather than the main text.",
    95: "Dataset is not a component of the Comprehensiveness metric.",
    96: "Dataset is not a component of the Comprehensiveness metric.",
}

# Each rule overrides only fields that require correction.
# All other fields and original evidence are preserved.

RELATIONSHIP_RULES = {
    2: {
        "target": "RAG baseline",
    },
    3: {
        "type": "constructs",
    },
    4: {
        "type": "generates",
        "target": "Community Summaries",
    },
    6: {
        "type": "overcomes_limitations_of",
    },
    7: {
        "type": "has_limitations_for",
    },
    9: {
        "type": "constructs",
    },
    10: {
        "type": "generates",
        "target": "Community Summaries",
    },
    14: {
        "type": "available_as_extension_to",
    },
    15: {
        "type": "collectively_referred_to_as",
    },
    35: {
        "type": "uses_for_entity_matching",
    },
    17: {
        "type": "is_similar_to",
    },
    20: {
        "type": "generates_queries_for",
    },
    21: {
        "source": "adaptive benchmarking",
        "type": "uses",
        "target": "LLM-based persona generation",
    },
    25: {
        "type": "yields",
    },
    26: {
        "type": "yields",
    },
    28: {
        "type": "enables",
    },
    37: {
        "type": "generates",
        "target": "Community Summaries",
    },
    38: {
        "type": "generates",
        "target": "Community Summaries",
    },
    39: {
        "type": "generates",
    },
    40: {
        "type": "contributes_to",
    },
    42: {
        "type": "scores_helpfulness_of",
    },
    44: {
        "source": "Higher-level communities",
        "type": "derived_from",
        "target": "Leaf-level communities",
    },
    47: {
        "type": "generates_questions_for",
    },
    52: {
        "type": "used_to_generate_questions_for",
    },
    53: {
        "type": "used_to_generate_questions_for",
    },
    56: {
        "type": "judges_answers_from",
    },
    57: {
        "type": "judges_answers_from",
    },
    58: {
        "type": "judges_answers_from",
    },
    59: {
        "type": "evaluated_on",
    },
    60: {
        "type": "evaluated_on",
    },
    63: {
        "type": "uses_domain_adapted_prompts_for",
    },
    64: {
        "source": "Graph indexing",
        "type": "uses",
        "target": "gpt-4-turbo",
    },
    69: {
        "type": "evaluated_on",
        "target": "Podcast Transcripts",
    },
    70: {
        "type": "evaluated_on",
        "target": "News Articles",
    },
    71: {
        "type": "is_condition_of",
        "target": "GraphRAG",
    },
    72: {
        "type": "is_condition_of",
        "target": "GraphRAG",
    },
    73: {
        "type": "compared_with",
        "target": "GraphRAG",
    },
    76: {
        "type": "outperforms_on_comprehensiveness",
        "target": "SS (Semantic Search / Vector RAG)",
    },
    77: {
        "type": "outperforms_on_comprehensiveness",
        "target": "SS (Semantic Search / Vector RAG)",
    },
    78: {
        "type": "outperforms_on_comprehensiveness",
        "target": "SS (Semantic Search / Vector RAG)",
    },
    79: {
        "type": "outperforms_on_comprehensiveness",
        "target": "SS (Semantic Search / Vector RAG)",
    },
    80: {
        "type": "outperforms_on_comprehensiveness",
        "target": "SS (Semantic Search / Vector RAG)",
    },
    83: {
        "type": "compared_with",
    },
    84: {
        "type": "uses",
    },
    85: {
        "type": "may_combine_with",
    },
    86: {
        "type": "mitigates_risks_to",
    },
    87: {
        "type": "supports_exploration_guided_by",
    },
    89: {
        "type": "motivates_use_of",
    },
    97: {
        "type": "compared_with",
    },
    98: {
        "type": "compared_with",
    },
}

RELATIONSHIP_DESCRIPTION_OVERRIDES = {
    2: "GraphRAG is compared with the conventional RAG baseline.",
    6: "RAG retrieves a subset of records to handle data exceeding an LLM context window.",
    7: "Vector RAG has limitations for sensemaking queries requiring global understanding.",
    20: "Adaptive benchmarking generates corpus-specific questions for global sensemaking evaluation.",
    21: "Adaptive benchmarking uses LLM-based persona generation to create representative user queries.",
    25: "Text chunks yield extracted entity instances.",
    26: "Text chunks yield extracted relationships between entities.",
    28: "Community detection enables parallel summarization that contributes to global answer generation.",
    37: "Graph communities are summarized to produce community summaries.",
    39: "Community summaries are used to generate intermediate community answers.",
    40: "Intermediate community answers contribute to the final global answer.",
    42: "The LLM assigns helpfulness scores to intermediate community answers.",
    44: "Higher-level communities are derived from lower-level communities.",
    47: "GENERATEQUESTIONS generates questions requiring global understanding of the corpus.",
    63: "GraphRAG uses domain-adapted extraction prompts for the Podcast Transcripts dataset.",
    64: "Graph indexing uses the gpt-4-turbo endpoint.",
    76: "C0 has higher answer comprehensiveness than SS across the evaluated datasets.",
    77: "C1 has higher answer comprehensiveness than SS across the evaluated datasets.",
    78: "C2 has higher answer comprehensiveness than SS across the evaluated datasets.",
    79: "C3 has higher answer comprehensiveness than SS across the evaluated datasets.",
    80: "TS has higher answer comprehensiveness than SS across the evaluated datasets.",
    89: "The Shapiro-Wilk test indicated non-normality, motivating the use of the Wilcoxon signed-rank test.",
}


def merge_relationships(relationships, valid_entity_names):
    merged = {}
    removed = []

    for original_id, relationship in enumerate(relationships, 1):
        if original_id in RELATIONSHIP_REMOVALS:
            removed.append({
                "id": original_id,
                "reason": RELATIONSHIP_REMOVALS[original_id],
            })
            continue

        source = canonical_name(relationship["source"])
        target = canonical_name(relationship["target"])
        relation_type = relationship["type"]

        # Apply manually reviewed corrections.
        rule = RELATIONSHIP_RULES.get(original_id, {})

        source = rule.get("source", source)
        target = rule.get("target", target)
        relation_type = rule.get("type", relation_type)

        # Remove relations involving excluded example entities.
        if source in EXCLUDED_ENTITIES or target in EXCLUDED_ENTITIES:
            removed.append({
                "id": original_id,
                "reason": "Relationship references an excluded entity.",
            })
            continue

        if source not in valid_entity_names:
            raise ValueError(
                f"Relationship #{original_id}: unknown source entity: {source}"
            )

        if target not in valid_entity_names:
            raise ValueError(
                f"Relationship #{original_id}: unknown target entity: {target}"
            )

        pages = relationship.get("source_pages", [])
        records = relationship.get("evidence_records", [])

        if len(pages) != len(records):
            raise ValueError(
                f"Relationship #{original_id}: page/evidence mismatch."
            )

        key = (source, relation_type, target)

        if key not in merged:
            merged[key] = {
                **relationship,
                "source": source,
                "type": relation_type,
                "target": target,
                "source_pages": [],
                "evidence_records": [],
                "description": RELATIONSHIP_DESCRIPTION_OVERRIDES.get(
                    original_id,
                    relationship.get("description", "")
                ),
                "original_ids": [],
            }

        item = merged[key]
        item["original_ids"].append(original_id)

        if original_id in RELATIONSHIP_DESCRIPTION_OVERRIDES:
            item["description"] = RELATIONSHIP_DESCRIPTION_OVERRIDES[
                original_id
            ]

        existing_pairs = set(zip(
            item["source_pages"],
            item["evidence_records"]
        ))

        for page, evidence in zip(pages, records):
            pair = (page, evidence)
            if pair not in existing_pairs:
                item["source_pages"].append(page)
                item["evidence_records"].append(evidence)
                existing_pairs.add(pair)

        item["review_status"] = "semantically_reviewed"

    result = []

    for item in merged.values():
        pairs = sorted(
            zip(item["source_pages"], item["evidence_records"]),
            key=lambda x: (x[0], x[1])
        )

        item["source_pages"] = [page for page, _ in pairs]
        item["evidence_records"] = [evidence for _, evidence in pairs]
        item["evidence"] = " | ".join(dict.fromkeys(
            item["evidence_records"]
        ))

        result.append(item)

    result.sort(key=lambda x: (
        x["source"], x["type"], x["target"]
    ))

    return result, removed


def main():
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    entities = merge_entities(data["entities"])
    valid_names = {entity["name"] for entity in entities}

    relationships, removed = merge_relationships(
        data["relationships"],
        valid_names
    )

    print("=" * 75)
    print("SEMANTIC CLEANING - ENTITY AND RELATIONSHIP DRY RUN")
    print("=" * 75)

    print("\n========== ENTITIES ==========")
    print("Original entities:", len(data["entities"]))
    print("Canonical entities:", len(entities))

    for entity_type, count in sorted(
        Counter(e["type"] for e in entities).items()
    ):
        print(f"  {entity_type}: {count}")

    print("\n========== RELATIONSHIPS ==========")
    print("Original relationships:", len(data["relationships"]))
    print("After semantic cleaning and deduplication:", len(relationships))
    print("Removed relationships:", len(removed))

    print("\nRemoved relationship records:")
    for item in removed:
        print(f"  #{item['id']:03d}: {item['reason']}")

    print("\nCanonical relationships:")
    for index, relation in enumerate(relationships, 1):
        original_ids = relation.get("original_ids", [])
        print(
            f"  [{index:03d}] "
            f"{relation['source']} "
            f"--{relation['type']}--> "
            f"{relation['target']} "
            f"(original IDs={original_ids})"
        )

    # --------------------------------------------------
    # Evidence and graph integrity validation
    # --------------------------------------------------

    entity_errors = [
        e["name"] for e in entities
        if len(e["source_pages"]) != len(e["evidence_records"])
    ]

    relationship_errors = [
        (
            r["source"],
            r["type"],
            r["target"]
        )
        for r in relationships
        if len(r["source_pages"]) != len(r["evidence_records"])
    ]

    invalid_endpoints = [
        (r["source"], r["target"])
        for r in relationships
        if (
            r["source"] not in valid_names
            or r["target"] not in valid_names
        )
    ]

    duplicate_keys = []
    keys = [
        (r["source"], r["type"], r["target"])
        for r in relationships
    ]

    seen = set()
    for key in keys:
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)

    print("\n========== INTEGRITY CHECK ==========")
    print("Entity evidence alignment errors:", len(entity_errors))
    print("Relationship evidence alignment errors:", len(relationship_errors))
    print("Invalid relationship endpoints:", len(invalid_endpoints))
    print("Duplicate relationship keys:", len(duplicate_keys))

    if entity_errors or relationship_errors or invalid_endpoints or duplicate_keys:
        raise SystemExit("INTEGRITY CHECK FAILED.")


    # --------------------------------------------------
    # Export semantic knowledge after all checks pass
    # --------------------------------------------------

    knowledge_path = PROJECT / "results/semantic_knowledge.json"
    report_path = PROJECT / "results/semantic_review_report.json"

    # Never overwrite existing deliverables.
    if knowledge_path.exists() or report_path.exists():
        raise SystemExit(
            "OUTPUT FILE EXISTS. No files were written. "
            "Please inspect existing outputs first."
        )

    entity_type_distribution = dict(sorted(
        Counter(e["type"] for e in entities).items()
    ))

    merged_relationships = [
        {
            "source": r["source"],
            "type": r["type"],
            "target": r["target"],
            "original_ids": r["original_ids"],
            "merged_record_count": len(r["original_ids"]),
        }
        for r in relationships
        if len(r["original_ids"]) > 1
    ]

    knowledge = {
        "schema_version": "1.0",
        "source_file": str(INPUT_FILE),
        "cleaning_method": "manual semantic normalization and evidence-preserving deduplication",
        "entities": entities,
        "relationships": relationships,
    }

    report = {
        "schema_version": "1.0",
        "source_file": str(INPUT_FILE),
        "statistics": {
            "original_entity_count": len(data["entities"]),
            "semantic_entity_count": len(entities),
            "original_relationship_count": len(data["relationships"]),
            "semantic_relationship_count": len(relationships),
            "removed_relationship_count": len(removed),
            "entity_type_distribution": entity_type_distribution,
        },
        "removed_relationships": removed,
        "merged_relationships": merged_relationships,
        "validation": {
            "entity_evidence_alignment_errors": len(entity_errors),
            "relationship_evidence_alignment_errors": len(relationship_errors),
            "invalid_relationship_endpoints": len(invalid_endpoints),
            "duplicate_relationship_keys": len(duplicate_keys),
            "passed": True,
        },
    }

    knowledge_tmp = knowledge_path.with_suffix(".json.tmp")
    report_tmp = report_path.with_suffix(".json.tmp")

    try:
        knowledge_tmp.write_text(
            json.dumps(knowledge, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_tmp.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        knowledge_tmp.replace(knowledge_path)
        report_tmp.replace(report_path)

    except Exception:
        knowledge_tmp.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)
        raise

    print("\\n========== FORMAL EXPORT ==========")
    print("Semantic knowledge:", knowledge_path)
    print("Review report:", report_path)
    print("Entities exported:", len(entities))
    print("Relationships exported:", len(relationships))
    print("Merged relationship groups:", len(merged_relationships))
    print("Validation: PASS")
    print("\\nSEMANTIC CLEANING COMPLETED SUCCESSFULLY.")



if __name__ == "__main__":
    main()
