import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network


PROJECT = Path("week10/assignment4_graphrag")
INPUT_FILE = PROJECT / "results/semantic_knowledge.json"
RESULTS = PROJECT / "results"

GRAPHML_FILE = RESULTS / "knowledge_graph.graphml"
JSON_FILE = RESULTS / "knowledge_graph.json"
HTML_FILE = RESULTS / "knowledge_graph.html"


def main():
    print("=" * 70)
    print("BUILD NETWORKX KNOWLEDGE GRAPH")
    print("=" * 70)

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    entities = data["entities"]
    relationships = data["relationships"]

    # Never overwrite existing graph outputs.
    for path in (GRAPHML_FILE, JSON_FILE, HTML_FILE):
        if path.exists():
            raise SystemExit(
                f"Output already exists: {path}\n"
                "Please inspect it before rerunning."
            )

    # Use MultiDiGraph to preserve parallel relationships.
    graph = nx.MultiDiGraph(
        name="GraphRAG Paper Knowledge Graph",
        schema_version=data["schema_version"],
    )

    # --------------------------------------------------
    # 1. Add entity nodes
    # --------------------------------------------------
    for entity in entities:
        name = entity["name"]

        graph.add_node(
            name,
            label=name,
            entity_type=entity["type"],
            description=entity.get("description", ""),
            evidence=entity.get("evidence", ""),
            source_pages=json.dumps(
                entity.get("source_pages", []),
                ensure_ascii=False,
            ),
            evidence_records=json.dumps(
                entity.get("evidence_records", []),
                ensure_ascii=False,
            ),
            review_status=entity.get("review_status", ""),
        )

    # --------------------------------------------------
    # 2. Add relationship edges
    # --------------------------------------------------
    for index, relation in enumerate(relationships, 1):
        source = relation["source"]
        target = relation["target"]

        if source not in graph or target not in graph:
            raise ValueError(
                f"Invalid relationship endpoint: {source} -> {target}"
            )

        graph.add_edge(
            source,
            target,
            key=f"relation_{index:03d}",
            relation_type=relation["type"],
            description=relation.get("description", ""),
            evidence=relation.get("evidence", ""),
            source_pages=json.dumps(
                relation.get("source_pages", []),
                ensure_ascii=False,
            ),
            evidence_records=json.dumps(
                relation.get("evidence_records", []),
                ensure_ascii=False,
            ),
            original_ids=json.dumps(
                relation.get("original_ids", []),
                ensure_ascii=False,
            ),
            review_status=relation.get("review_status", ""),
        )

    # --------------------------------------------------
    # 3. Integrity checks
    # --------------------------------------------------
    if graph.number_of_nodes() != len(entities):
        raise ValueError("Entity count mismatch.")

    if graph.number_of_edges() != len(relationships):
        raise ValueError("Relationship count mismatch.")

    if not nx.is_directed(graph):
        raise ValueError("Graph must be directed.")

    if not graph.is_multigraph():
        raise ValueError("Graph must preserve parallel edges.")

    missing_evidence = [
        (u, v, key)
        for u, v, key, attrs in graph.edges(keys=True, data=True)
        if not attrs.get("evidence_records")
    ]

    print("\n========== GRAPH STATISTICS ==========")
    print("Graph type:", type(graph).__name__)
    print("Directed:", nx.is_directed(graph))
    print("Multigraph:", graph.is_multigraph())
    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())
    print("Parallel edges:", sum(
        count - 1
        for count in [
            len(list(graph.get_edge_data(u, v).values()))
            for u, v in {
                (u, v) for u, v in graph.edges()
            }
        ]
        if count > 1
    ))
    print("Edges without evidence:", len(missing_evidence))

    if missing_evidence:
        raise ValueError("Some graph edges have no evidence.")

    # --------------------------------------------------
    # 4. Export GraphML
    # --------------------------------------------------
    nx.write_graphml(graph, GRAPHML_FILE)

    # --------------------------------------------------
    # 5. Export JSON
    # --------------------------------------------------
    graph_json = {
        "metadata": {
            "name": "GraphRAG Paper Knowledge Graph",
            "schema_version": data["schema_version"],
            "graph_type": "MultiDiGraph",
            "directed": True,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        },
        "nodes": [
            {
                "id": name,
                **dict(attrs),
                "source_pages": json.loads(attrs["source_pages"]),
                "evidence_records": json.loads(attrs["evidence_records"]),
            }
            for name, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "key": key,
                **dict(attrs),
                "source_pages": json.loads(attrs["source_pages"]),
                "evidence_records": json.loads(attrs["evidence_records"]),
                "original_ids": json.loads(attrs["original_ids"]),
            }
            for source, target, key, attrs
            in graph.edges(keys=True, data=True)
        ],
    }

    JSON_FILE.write_text(
        json.dumps(graph_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 6. Create interactive PyVis visualization
    # --------------------------------------------------
    net = Network(
        height="850px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )

    net.barnes_hut(
        gravity=-30000,
        central_gravity=0.25,
        spring_length=180,
        spring_strength=0.04,
        damping=0.09,
    )

    for name, attrs in graph.nodes(data=True):
        title = (
            f"<b>{name}</b><br>"
            f"Type: {attrs['entity_type']}<br>"
            f"Description: {attrs['description']}<br>"
            f"Pages: {attrs['source_pages']}<br>"
            f"Evidence: {attrs['evidence']}"
        )

        net.add_node(
            name,
            label=name,
            title=title,
            group=attrs["entity_type"],
            shape="dot",
            size=18,
        )

    for source, target, key, attrs in graph.edges(
        keys=True,
        data=True,
    ):
        title = (
            f"<b>{attrs['relation_type']}</b><br>"
            f"{source} → {target}<br>"
            f"{attrs['description']}<br>"
            f"Evidence: {attrs['evidence']}<br>"
            f"Pages: {attrs['source_pages']}<br>"
            f"Original IDs: {attrs['original_ids']}"
        )

        net.add_edge(
            source,
            target,
            label=attrs["relation_type"],
            title=title,
            arrows="to",
        )

    net.write_html(str(HTML_FILE), notebook=False)

    # --------------------------------------------------
    # 7. Final verification
    # --------------------------------------------------
    reloaded = nx.read_graphml(GRAPHML_FILE)

    print("\n========== EXPORT VERIFICATION ==========")
    print("GraphML nodes:", reloaded.number_of_nodes())
    print("GraphML edges:", reloaded.number_of_edges())
    print("JSON nodes:", len(graph_json["nodes"]))
    print("JSON edges:", len(graph_json["edges"]))
    print("HTML exists:", HTML_FILE.exists())

    if (
        reloaded.number_of_nodes() != 75
        or reloaded.number_of_edges() != 82
        or len(graph_json["nodes"]) != 75
        or len(graph_json["edges"]) != 82
        or not HTML_FILE.exists()
    ):
        raise ValueError("Export verification failed.")

    print("\n========== OUTPUT FILES ==========")
    for path in (GRAPHML_FILE, JSON_FILE, HTML_FILE):
        print(f"{path}: {path.stat().st_size:,} bytes")

    print("\nKNOWLEDGE GRAPH BUILD SUCCESSFUL.")


if __name__ == "__main__":
    main()
