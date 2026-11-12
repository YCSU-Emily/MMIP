import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
RESULTS = PROJECT / "results"

sys.path.insert(0, str(SRC))
import summarize_communities as sc

ORIGINAL_FILE = RESULTS / "community_summaries.json"
OUTPUT_FILE = RESULTS / "community_summaries_repaired.json"
RAW_DIR = RESULTS / "community_summary_repair_raw"

FAILED_IDS = [0, 2, 3, 5, 16, 19]
MAX_NEW_TOKENS = 1800


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    print("=" * 70)
    print("REPAIR FAILED COMMUNITY SUMMARIES")
    print("=" * 70)

    # Preserve all original files.
    if OUTPUT_FILE.exists():
        raise FileExistsError(f"Repair output already exists: {OUTPUT_FILE}")

    if RAW_DIR.exists():
        raise FileExistsError(f"Repair raw directory already exists: {RAW_DIR}")

    original = load_json(ORIGINAL_FILE)
    graph_data = load_json(RESULTS / "knowledge_graph.json")
    assignments = load_json(RESULTS / "community_assignments.json")

    if original["successful_summaries"] != 14:
        raise ValueError("Unexpected original successful summary count.")

    if original["failed_summaries"] != 6:
        raise ValueError("Unexpected original failed summary count.")

    original_summaries = {
        item["community_id"]: item
        for item in original["summaries"]
    }

    original_failures = {
        item["community_id"]: item
        for item in original["failures"]
    }

    if set(original_failures) != set(FAILED_IDS):
        raise ValueError(
            f"Failed community IDs changed: {sorted(original_failures)}"
        )

    nodes = {
        item["id"]: item
        for item in graph_data["nodes"]
    }
    edges = graph_data["edges"]

    all_valid_pages = {
        page
        for node in graph_data["nodes"]
        for page in node.get("source_pages", [])
    }
    all_valid_pages.update(
        page
        for edge in edges
        for page in edge.get("source_pages", [])
    )

    # Increase output length for the two truncated JSON responses.
    sc.MAX_NEW_TOKENS = MAX_NEW_TOKENS

    RAW_DIR.mkdir(parents=True)

    print("Original successful summaries:", len(original_summaries))
    print("Communities to repair:", FAILED_IDS)
    print("MAX_NEW_TOKENS:", sc.MAX_NEW_TOKENS)

    processor, model = sc.load_model()

    repaired = []
    failures = []

    for position, community_id in enumerate(FAILED_IDS, 1):
        print(
            f"\n[{position}/{len(FAILED_IDS)}] "
            f"Repair Community {community_id}"
        )

        community = sc.build_community_input(
            community_id,
            assignments,
            nodes,
            edges,
        )

        raw_text = sc.generate_summary(
            processor,
            model,
            community,
        )

        raw_path = RAW_DIR / f"community_{community_id:03d}.raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8")

        try:
            summary = sc.parse_summary(raw_text)

            # Only community members may appear in key_entities.
            member_names = {
                entity["name"]
                for entity in community["entities"]
            }

            proposed_entities = summary.get("key_entities", [])

            if not isinstance(proposed_entities, list):
                raise ValueError("key_entities must be a list.")

            excluded_entities = [
                name for name in proposed_entities
                if name not in member_names
            ]

            summary["key_entities"] = [
                name for name in proposed_entities
                if name in member_names
            ]

            # Keep the original strict validation for pages and fields.
            summary = sc.validate_summary(
                summary,
                community,
                all_valid_pages,
            )

            result = {
                "community_id": community_id,
                "member_count": community["member_count"],
                "members": [
                    entity["name"]
                    for entity in community["entities"]
                ],
                "internal_relationship_count": len(
                    community["internal_relationships"]
                ),
                "boundary_relationship_count": len(
                    community["boundary_relationships"]
                ),
                **summary,
                "excluded_external_key_entities": excluded_entities,
                "review_status": "llm_generated_pending_manual_review",
            }

            repaired.append(result)

            print("Status: PASS")
            print("Topic:", summary["main_topic"])
            print("Key entities:", summary["key_entities"])
            print("Excluded external entities:", excluded_entities)
            print("Source pages:", summary["source_pages"])

        except Exception as exc:
            failures.append({
                "community_id": community_id,
                "error": str(exc),
                "raw_file": str(raw_path),
            })
            print("Status: FAILED")
            print("Reason:", exc)

    # Merge repaired summaries with the 14 original successful summaries.
    merged = dict(original_summaries)

    for item in repaired:
        merged[item["community_id"]] = item

    all_summaries = [
        merged[cid]
        for cid in sorted(merged)
    ]

    output = {
        "schema_version": "1.0",
        "source_graph": original["source_graph"],
        "model": "Qwen3.5-4B",
        "summarization_method":
            "Evidence-grounded community summarization with failed-item repair",
        "total_communities": original["total_communities"],
        "successful_summaries": len(all_summaries),
        "failed_summaries": len(failures),
        "summaries": all_summaries,
        "failures": failures,
        "repair": {
            "repaired_community_ids": [x["community_id"] for x in repaired],
            "failed_community_ids": [x["community_id"] for x in failures],
            "max_new_tokens": MAX_NEW_TOKENS,
            "preserved_original_successful_summaries": len(original_summaries),
        },
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("REPAIR RESULT")
    print("=" * 70)
    print("Original successful summaries preserved:", len(original_summaries))
    print("Newly repaired summaries:", len(repaired))
    print("Total successful summaries:", len(all_summaries))
    print("Remaining failures:", len(failures))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
