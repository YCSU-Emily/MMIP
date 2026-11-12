import json
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path("week10/assignment4_graphrag")
RESULT_DIR = BASE_DIR / "results"
CHUNK_DIR = RESULT_DIR / "extraction_chunks"

INPUT_FILE = RESULT_DIR / "extracted_knowledge.json"
OUTPUT_FILE = RESULT_DIR / "cleaned_knowledge.json"
REPORT_FILE = RESULT_DIR / "cleaning_report.json"

# Main paper + statistical analysis
KEEP_PAGES = set(range(1, 13)) | {26}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize(value):
    return str(value).strip().casefold()


def load_original_chunks():
    entity_evidence = defaultdict(list)
    relation_evidence = defaultdict(list)

    chunk_files = sorted(CHUNK_DIR.glob("*.json"))

    if not chunk_files:
        raise FileNotFoundError(
            f"No extraction chunks found in {CHUNK_DIR}"
        )

    for path in chunk_files:
        chunk = load_json(path)
        page = chunk.get("page")

        if not isinstance(page, int):
            raise ValueError(f"Invalid page number in {path}")

        for entity in chunk.get("entities", []):
            key = normalize(entity.get("name", ""))

            evidence = entity.get("evidence", "").strip()

            if key and evidence:
                entity_evidence[key].append((page, evidence))

        for relation in chunk.get("relationships", []):
            key = (
                normalize(relation.get("source", "")),
                normalize(relation.get("type", "")),
                normalize(relation.get("target", ""))
            )

            evidence = relation.get("evidence", "").strip()

            if all(key) and evidence:
                relation_evidence[key].append((page, evidence))

    return entity_evidence, relation_evidence, len(chunk_files)


def unique_pairs(pairs):
    # Deduplicate only identical page + evidence pairs.
    result = []
    seen = set()

    for page, evidence in pairs:
        key = (page, evidence)

        if key not in seen:
            seen.add(key)
            result.append((page, evidence))

    return result


def main():
    original = load_json(INPUT_FILE)

    entity_evidence, relation_evidence, chunk_count = (
        load_original_chunks()
    )

    cleaned_entities = []
    cleaned_relationships = []

    excluded_entities = []
    excluded_relationships = []

    missing_entity_evidence = []
    missing_relation_evidence = []
    removed_dangling_relationships = []

    # --------------------------------------------------
    # 1. Rebuild entity evidence from original chunks
    # --------------------------------------------------
    for entity in original.get("entities", []):
        name = entity.get("name", "")
        key = normalize(name)

        pairs = unique_pairs(entity_evidence.get(key, []))

        kept_pairs = [
            (page, evidence)
            for page, evidence in pairs
            if page in KEEP_PAGES
        ]

        if not pairs:
            missing_entity_evidence.append(name)

        if not kept_pairs:
            excluded_entities.append({
                "name": name,
                "reason": "No evidence in selected pages",
                "original_pages": sorted(
                    {page for page, _ in pairs}
                )
            })
            continue

        item = dict(entity)

        # Page number and evidence now have strict 1:1 alignment.
        item["source_pages"] = [
            page for page, _ in kept_pairs
        ]
        item["evidence_records"] = [
            evidence for _, evidence in kept_pairs
        ]
        item["evidence"] = kept_pairs[0][1]
        item["review_status"] = "pending_semantic_review"

        cleaned_entities.append(item)

    # --------------------------------------------------
    # 2. Entity lookup
    # --------------------------------------------------
    entity_lookup = {
        normalize(entity["name"]): entity["name"]
        for entity in cleaned_entities
    }

    # --------------------------------------------------
    # 3. Rebuild relationship evidence from chunks
    # --------------------------------------------------
    for relation in original.get("relationships", []):
        source = relation.get("source", "")
        target = relation.get("target", "")
        rel_type = relation.get("type", "")

        key = (
            normalize(source),
            normalize(rel_type),
            normalize(target)
        )

        pairs = unique_pairs(relation_evidence.get(key, []))

        kept_pairs = [
            (page, evidence)
            for page, evidence in pairs
            if page in KEEP_PAGES
        ]

        if not pairs:
            missing_relation_evidence.append({
                "source": source,
                "target": target,
                "type": rel_type
            })

        if not kept_pairs:
            excluded_relationships.append({
                "source": source,
                "target": target,
                "type": rel_type,
                "reason": "No evidence in selected pages",
                "original_pages": sorted(
                    {page for page, _ in pairs}
                )
            })
            continue

        source_key = normalize(source)
        target_key = normalize(target)

        if (
            source_key not in entity_lookup
            or target_key not in entity_lookup
        ):
            removed_dangling_relationships.append({
                "source": source,
                "target": target,
                "type": rel_type,
                "reason": "Endpoint missing after entity filtering"
            })
            continue

        item = dict(relation)
        item["source"] = entity_lookup[source_key]
        item["target"] = entity_lookup[target_key]

        item["source_pages"] = [
            page for page, _ in kept_pairs
        ]
        item["evidence_records"] = [
            evidence for _, evidence in kept_pairs
        ]
        item["evidence"] = kept_pairs[0][1]
        item["review_status"] = "pending_semantic_review"

        cleaned_relationships.append(item)

    # --------------------------------------------------
    # 4. Validate strict alignment
    # --------------------------------------------------
    alignment_errors = []

    for entity in cleaned_entities:
        if len(entity["source_pages"]) != len(
            entity["evidence_records"]
        ):
            alignment_errors.append(
                f'Entity: {entity["name"]}'
            )

    for relation in cleaned_relationships:
        if len(relation["source_pages"]) != len(
            relation["evidence_records"]
        ):
            alignment_errors.append(
                f'Relationship: {relation["source"]} -> '
                f'{relation["target"]}'
            )

    if alignment_errors:
        raise ValueError(
            "Evidence alignment validation failed: "
            + "; ".join(alignment_errors)
        )

    # --------------------------------------------------
    # 5. Create cleaned knowledge
    # --------------------------------------------------
    statistics = {
        "original_entities": len(original.get("entities", [])),
        "original_relationships": len(
            original.get("relationships", [])
        ),
        "cleaned_entities": len(cleaned_entities),
        "cleaned_relationships": len(cleaned_relationships),
        "excluded_entities": len(excluded_entities),
        "excluded_relationships": len(excluded_relationships),
        "removed_dangling_relationships": len(
            removed_dangling_relationships
        ),
        "missing_entity_chunk_evidence": len(
            missing_entity_evidence
        ),
        "missing_relationship_chunk_evidence": len(
            missing_relation_evidence
        ),
        "processed_chunks": chunk_count,
        "alignment_errors": len(alignment_errors)
    }

    cleaned = {
        "paper_title": original.get("paper_title"),
        "source": original.get("source"),
        "extraction_model": original.get("extraction_model"),
        "cleaning_scope": {
            "included_pages": sorted(KEEP_PAGES),
            "excluded_pages": sorted(
                set(range(1, 27)) - KEEP_PAGES
            ),
            "description": (
                "Main paper pages 1-12 and statistical analysis "
                "page 26. Evidence rebuilt from original chunks."
            )
        },
        "entities": cleaned_entities,
        "relationships": cleaned_relationships,
        "statistics": statistics
    }

    report = {
        "included_pages": sorted(KEEP_PAGES),
        "excluded_pages": sorted(set(range(1, 27)) - KEEP_PAGES),
        "statistics": statistics,
        "entity_type_distribution": dict(
            Counter(e.get("type", "Unknown") for e in cleaned_entities)
        ),
        "relationship_type_distribution": dict(
            Counter(
                r.get("type", "Unknown")
                for r in cleaned_relationships
            )
        ),
        "excluded_entities": excluded_entities,
        "excluded_relationships": excluded_relationships,
        "missing_entity_evidence": missing_entity_evidence,
        "missing_relationship_evidence": missing_relation_evidence,
        "removed_dangling_relationships": (
            removed_dangling_relationships
        ),
        "evidence_rebuilt_from_original_chunks": True,
        "alignment_validation": "PASSED",
        "semantic_review_required": True
    }

    save_json(OUTPUT_FILE, cleaned)
    save_json(REPORT_FILE, report)

    print("=" * 70)
    print("KNOWLEDGE GRAPH CLEANING COMPLETE")
    print("=" * 70)

    print("\nStatistics:")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))

    print("\nEntity type distribution:")
    for key, value in report["entity_type_distribution"].items():
        print(f"  {key:20s}: {value}")

    print("\nRelationship type distribution:")
    for key, value in report["relationship_type_distribution"].items():
        print(f"  {key:25s}: {value}")

    print("\nAlignment validation: PASSED")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("\nOriginal extraction file was NOT modified.")


if __name__ == "__main__":
    main()
