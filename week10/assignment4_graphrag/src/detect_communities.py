import json
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx


PROJECT = Path("week10/assignment4_graphrag")
GRAPH_FILE = PROJECT / "results/knowledge_graph.graphml"
ASSIGNMENT_FILE = PROJECT / "results/community_assignments.json"
REPORT_FILE = PROJECT / "results/community_report.json"

RANDOM_SEED = 42
RESOLUTION = 1.0


def main():
    print("=" * 70)
    print("LEIDEN COMMUNITY DETECTION")
    print("=" * 70)

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(GRAPH_FILE)

    for path in (ASSIGNMENT_FILE, REPORT_FILE):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Inspect the existing file before rerunning."
            )

    # --------------------------------------------------
    # 1. Load original directed multigraph
    # --------------------------------------------------
    directed_graph = nx.read_graphml(GRAPH_FILE)

    if not directed_graph.is_directed():
        raise ValueError("Original graph must be directed.")

    if not directed_graph.is_multigraph():
        raise ValueError("Original graph must be a multigraph.")

    nodes = sorted(directed_graph.nodes())
    node_to_index = {
        name: index for index, name in enumerate(nodes)
    }

    # --------------------------------------------------
    # 2. Build undirected weighted projection
    # --------------------------------------------------
    pair_weights = Counter()

    for source, target, key in directed_graph.edges(keys=True):
        if source == target:
            continue

        pair = tuple(sorted((source, target)))
        pair_weights[pair] += 1

    weighted_edges = [
        (
            node_to_index[source],
            node_to_index[target],
        )
        for source, target in pair_weights
    ]

    weights = list(pair_weights.values())

    projection = ig.Graph(
        n=len(nodes),
        edges=weighted_edges,
        directed=False,
    )

    projection.vs["name"] = nodes
    projection.es["weight"] = weights

    # --------------------------------------------------
    # 3. Run Leiden
    # --------------------------------------------------
    partition = leidenalg.find_partition(
        projection,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=RANDOM_SEED,
    )

    membership = partition.membership

    communities = defaultdict(list)

    for node_index, community_id in enumerate(membership):
        communities[community_id].append(nodes[node_index])

    # --------------------------------------------------
    # 4. Community statistics
    # --------------------------------------------------
    node_attributes = dict(directed_graph.nodes(data=True))

    community_records = []

    for community_id, members in sorted(communities.items()):
        type_counts = Counter(
            node_attributes[name].get("entity_type", "Unknown")
            for name in members
        )

        internal_edges = 0
        external_edges = 0

        member_set = set(members)

        for source, target, key in directed_graph.edges(keys=True):
            if source in member_set and target in member_set:
                internal_edges += 1
            elif source in member_set or target in member_set:
                external_edges += 1

        community_records.append({
            "community_id": int(community_id),
            "size": len(members),
            "members": sorted(members),
            "entity_type_distribution": dict(type_counts),
            "internal_directed_edges": internal_edges,
            "external_directed_edges": external_edges,
        })

    # --------------------------------------------------
    # 5. Build node assignments
    # --------------------------------------------------
    assignments = []

    for node_index, name in enumerate(nodes):
        attrs = node_attributes[name]

        assignments.append({
            "entity": name,
            "entity_type": attrs.get("entity_type", "Unknown"),
            "community_id": int(membership[node_index]),
            "description": attrs.get("description", ""),
            "source_pages": json.loads(
                attrs.get("source_pages", "[]")
            ),
        })

    # --------------------------------------------------
    # 6. Calculate graph-level statistics
    # --------------------------------------------------
    isolated_nodes = [
        nodes[index]
        for index, degree in enumerate(projection.degree())
        if degree == 0
    ]

    report = {
        "algorithm": "Leiden",
        "partition_type": "ModularityVertexPartition",
        "random_seed": RANDOM_SEED,
        "resolution": RESOLUTION,
        "projection": {
            "original_graph_type": "MultiDiGraph",
            "community_graph_type": "Undirected Weighted Graph",
            "node_count": projection.vcount(),
            "unique_undirected_edge_count": projection.ecount(),
            "total_relationship_weight": sum(weights),
            "isolated_node_count": len(isolated_nodes),
            "isolated_nodes": isolated_nodes,
        },
        "results": {
            "community_count": len(communities),
            "modularity": float(partition.modularity),
            "community_sizes": sorted(
                [len(members) for members in communities.values()],
                reverse=True,
            ),
            "communities": community_records,
        },
        "node_assignments": assignments,
    }

    # --------------------------------------------------
    # 7. Integrity checks
    # --------------------------------------------------
    assigned_nodes = [
        item["entity"] for item in assignments
    ]

    if len(assigned_nodes) != len(nodes):
        raise ValueError("Node assignment count mismatch.")

    if len(set(assigned_nodes)) != len(nodes):
        raise ValueError("Duplicate node assignments.")

    if sum(
        item["size"] for item in community_records
    ) != len(nodes):
        raise ValueError("Community size total mismatch.")

    # --------------------------------------------------
    # 8. Export results
    # --------------------------------------------------
    ASSIGNMENT_FILE.write_text(
        json.dumps(assignments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 9. Display results
    # --------------------------------------------------
    print("\n========== PROJECTION GRAPH ==========")
    print("Original directed nodes:", directed_graph.number_of_nodes())
    print("Original directed edges:", directed_graph.number_of_edges())
    print("Projection nodes:", projection.vcount())
    print("Projection edges:", projection.ecount())
    print("Total edge weight:", sum(weights))
    print("Isolated nodes:", len(isolated_nodes))

    print("\n========== LEIDEN RESULTS ==========")
    print("Algorithm: Leiden")
    print("Partition: ModularityVertexPartition")
    print("Random seed:", RANDOM_SEED)
    print("Resolution:", RESOLUTION)
    print("Communities:", len(communities))
    print("Modularity:", round(partition.modularity, 6))
    print("Community sizes:", report["results"]["community_sizes"])

    print("\n========== COMMUNITY DETAILS ==========")

    for community in community_records:
        print(
            f"\nCommunity {community['community_id']}: "
            f"{community['size']} nodes"
        )
        print("Entity types:", community["entity_type_distribution"])
        print("Members:", ", ".join(community["members"]))

    print("\n========== OUTPUT FILES ==========")
    for path in (ASSIGNMENT_FILE, REPORT_FILE):
        print(f"{path}: {path.stat().st_size:,} bytes")

    print("\nLEIDEN COMMUNITY DETECTION SUCCESSFUL.")


if __name__ == "__main__":
    main()
