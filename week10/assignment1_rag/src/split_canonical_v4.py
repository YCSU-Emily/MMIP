import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data/processed/smart3c_qa_v3.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/splits_canonical_v4"

SEED = 42
SPLITS = ("train", "validation", "test")
RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}


def canonicalize_url(url):
    """Normalize URL variants that refer to the same official document."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("A record has a missing source_url.")

    parts = urlsplit(url.strip())

    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Invalid source_url: {url}")

    scheme = parts.scheme.lower()
    host = parts.netloc.lower()

    if host.startswith("www."):
        host = host[4:]

    path = re.sub(r"/+", "/", parts.path.lower())

    # Remove a leading locale segment, e.g. en-us, en-gb, zh-tw.
    segments = [segment for segment in path.split("/") if segment]

    if segments and re.fullmatch(r"[a-z]{2}-[a-z]{2}", segments[0]):
        segments = segments[1:]

    normalized_path = "/" + "/".join(segments)

    # Query strings and fragments are excluded from document identity.
    return urlunsplit((
        scheme,
        host,
        normalized_path.rstrip("/"),
        "",
        "",
    ))


def load_records():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError("The source dataset is empty.")

    ids = [record.get("id") for record in records]

    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("QA IDs are missing or duplicated.")

    for record in records:
        record["_canonical_url"] = canonicalize_url(
            record.get("source_url", "")
        )

        if not record.get("category"):
            raise ValueError(f"Missing category in QA {record['id']}")

    return records


def build_groups(records):
    groups = defaultdict(list)

    for record in records:
        groups[record["_canonical_url"]].append(record)

    return dict(groups)


def summarize_assignment(assignment, group_items, categories):
    result = {}

    for split in SPLITS:
        selected = [
            record
            for group_index, group in enumerate(group_items)
            if assignment[group_index] == split
            for record in group
        ]

        result[split] = {
            "qa_count": len(selected),
            "group_count": sum(
                1 for value in assignment if value == split
            ),
            "category_distribution": dict(
                sorted(Counter(r["category"] for r in selected).items())
            ),
            "records": selected,
        }

    return result


def objective(assignment, group_items, categories, total_qa):
    """Balance QA counts and category distributions without splitting groups."""
    targets = {
        split: total_qa * RATIOS[split]
        for split in SPLITS
    }

    total_by_category = Counter(
        record["category"]
        for group in group_items
        for record in group
    )

    stats = summarize_assignment(assignment, group_items, categories)

    cost = 0.0

    for split in SPLITS:
        actual = stats[split]["qa_count"]
        target = targets[split]

        cost += 5.0 * ((actual - target) / max(target, 1)) ** 2

        group_target = len(group_items) * RATIOS[split]
        group_actual = stats[split]["group_count"]

        cost += 0.25 * (
            (group_actual - group_target) / max(group_target, 1)
        ) ** 2

        distribution = Counter(stats[split]["category_distribution"])

        for category in categories:
            category_target = total_by_category[category] * RATIOS[split]
            category_actual = distribution[category]

            cost += 0.8 * (
                (category_actual - category_target)
                / max(category_target, 1)
            ) ** 2

            # Encourage category coverage in validation and test.
            if split in ("validation", "test") and category_actual == 0:
                cost += 0.8

    return cost


def optimize_groups(group_items, categories, total_qa):
    """Deterministic multi-start simulated annealing over whole document groups."""
    rng = random.Random(SEED)
    group_count = len(group_items)

    if group_count < 3:
        raise ValueError("At least three document groups are required.")

    best_assignment = None
    best_cost = float("inf")

    restarts = 120
    steps_per_restart = 5000

    for _ in range(restarts):
        current = [
            rng.choice(SPLITS)
            for _ in range(group_count)
        ]

        # Ensure all three splits are represented.
        for split_index, split in enumerate(SPLITS):
            current[split_index] = split

        current_cost = objective(
            current, group_items, categories, total_qa
        )

        local_best = current[:]
        local_best_cost = current_cost

        for step in range(steps_per_restart):
            candidate = current[:]
            index = rng.randrange(group_count)
            candidate[index] = rng.choice(SPLITS)

            candidate_cost = objective(
                candidate, group_items, categories, total_qa
            )

            progress = step / steps_per_restart
            temperature = 0.08 * (1.0 - progress) + 0.001

            delta = candidate_cost - current_cost

            if delta <= 0 or rng.random() < (
                __import__("math").exp(-delta / temperature)
            ):
                current = candidate
                current_cost = candidate_cost

            if current_cost < local_best_cost:
                local_best = current[:]
                local_best_cost = current_cost

        if local_best_cost < best_cost:
            best_assignment = local_best
            best_cost = local_best_cost

    if best_assignment is None:
        raise RuntimeError("Failed to generate a split assignment.")

    return best_assignment, best_cost


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}"
        )

    records = load_records()
    groups = build_groups(records)

    group_items = [
        groups[key]
        for key in sorted(groups)
    ]

    categories = sorted({
        record["category"]
        for record in records
    })

    assignment, final_cost = optimize_groups(
        group_items,
        categories,
        len(records),
    )

    stats = summarize_assignment(
        assignment,
        group_items,
        categories,
    )

    # Verify all QA records are assigned exactly once.
    assigned_records = [
        record
        for split in SPLITS
        for record in stats[split]["records"]
    ]

    original_ids = {record["id"] for record in records}
    assigned_ids = [record["id"] for record in assigned_records]

    if len(assigned_ids) != len(set(assigned_ids)):
        raise RuntimeError("Duplicate QA IDs found after splitting.")

    if set(assigned_ids) != original_ids:
        raise RuntimeError("QA IDs were lost or added during splitting.")

    # Verify canonical document groups never cross splits.
    group_to_split = {}

    for group_index, group in enumerate(group_items):
        split = assignment[group_index]
        canonical_url = group[0]["_canonical_url"]

        if canonical_url in group_to_split:
            raise RuntimeError(
                f"Canonical URL assigned more than once: {canonical_url}"
            )

        group_to_split[canonical_url] = split

    # Prepare manifest before creating the output directory.
    split_manifest = {}

    for split in SPLITS:
        selected = stats[split]["records"]

        split_manifest[split] = {
            "qa_count": len(selected),
            "canonical_document_group_count": stats[split]["group_count"],
            "source_ids": sorted({
                record.get("source_id", "")
                for record in selected
            }),
            "category_distribution": stats[split]["category_distribution"],
            "canonical_urls": sorted({
                record["_canonical_url"]
                for record in selected
            }),
        }

    manifest = {
        "version": "Canonical Document Split V4",
        "source_dataset": "data/processed/smart3c_qa_v3.jsonl",
        "total_qa": len(records),
        "total_canonical_document_groups": len(groups),
        "split_method": "canonical_url_grouped_simulated_annealing",
        "seed": SEED,
        "target_ratios": RATIOS,
        "optimization_cost": final_cost,
        "canonicalization": {
            "lowercase_scheme_host_path": True,
            "remove_leading_locale_segment": True,
            "remove_query_and_fragment": True,
            "remove_trailing_slash": True,
            "remove_www_prefix": True,
        },
        "splits": split_manifest,
        "integrity_checks": {
            "all_qa_ids_assigned_once": True,
            "canonical_document_overlap": 0,
            "missing_source_urls": 0,
        },
    }

    # Create a new directory only after all in-memory checks pass.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    try:
        for split in SPLITS:
            output_file = OUTPUT_DIR / f"{split}.jsonl"

            with output_file.open("x", encoding="utf-8") as f:
                for record in stats[split]["records"]:
                    clean_record = {
                        key: value
                        for key, value in record.items()
                        if key != "_canonical_url"
                    }
                    f.write(
                        json.dumps(clean_record, ensure_ascii=False) + "\n"
                    )

        manifest_path = OUTPUT_DIR / "split_manifest.json"

        with manifest_path.open("x", encoding="utf-8") as f:
            json.dump(
                manifest,
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")

    except Exception:
        print(
            "WARNING: Output creation was interrupted. "
            "Inspect the new directory before retrying."
        )
        raise

    print("===== CANONICAL DOCUMENT SPLIT V4 =====")
    print("Total QA:", len(records))
    print("Canonical document groups:", len(groups))
    print("Optimization cost:", round(final_cost, 6))

    for split in SPLITS:
        print(
            f"{split}: "
            f"{stats[split]['qa_count']} QA, "
            f"{stats[split]['group_count']} document groups"
        )
        print(
            "  Categories:",
            stats[split]["category_distribution"],
        )

    print("\nIntegrity checks:")
    print("  All QA IDs assigned exactly once: PASS")
    print("  Canonical document overlap: 0")
    print("  Missing source URLs: 0")
    print("  Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
