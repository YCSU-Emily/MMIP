import json
from pathlib import Path

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    / "5617a9f61b028005a4858fdac845db406aefb181"
)


def load_json(filename):
    path = RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value):
    return str(value or "").strip()


class GraphRAGRetriever:
    def __init__(self, device="cuda"):
        print("[GraphRAG] Loading graph and retrieval data...")

        graph_data = load_json("knowledge_graph.json")
        assignment_data = load_json("community_assignments.json")
        summary_data = load_json("community_summaries_repaired.json")
        chunk_data = load_json("paper_chunks.json")

        self.nodes = graph_data["nodes"]
        self.edges = graph_data["edges"]
        self.assignments = assignment_data
        self.summaries = summary_data["summaries"]
        self.chunks = chunk_data["chunks"]

        self.node_by_id = {
            str(node["id"]): node for node in self.nodes
        }

        self.assignment_by_entity = {
            str(item["entity"]): item
            for item in self.assignments
        }

        self.summary_by_id = {
            int(item["community_id"]): item
            for item in self.summaries
        }

        # Use an undirected view for one-hop neighborhood discovery.
        # Original edge direction and relation type are preserved below.
        self.graph = nx.MultiDiGraph()

        for node in self.nodes:
            self.graph.add_node(str(node["id"]), **node)

        for edge in self.edges:
            edge_attributes = {
                k: v for k, v in edge.items()
                if k not in {"source", "target", "key"}
            }
            self.graph.add_edge(
                str(edge["source"]),
                str(edge["target"]),
                key=str(edge["key"]),
                **edge_attributes,
            )

        print("[GraphRAG] Loading BGE-M3...")
        self.model = SentenceTransformer(
            str(MODEL_PATH),
            local_files_only=True,
            device=device,
        )

        print("[GraphRAG] Preparing retrieval embeddings...")

        self.node_texts = [
            " | ".join([
                normalize_text(n.get("label")),
                normalize_text(n.get("entity_type")),
                normalize_text(n.get("description")),
                normalize_text(n.get("evidence")),
            ])
            for n in self.nodes
        ]

        self.edge_texts = [
            " | ".join([
                normalize_text(e.get("source")),
                normalize_text(e.get("relation_type")),
                normalize_text(e.get("target")),
                normalize_text(e.get("description")),
                normalize_text(e.get("evidence")),
            ])
            for e in self.edges
        ]

        self.summary_texts = [
            " | ".join([
                normalize_text(s.get("main_topic")),
                normalize_text(s.get("summary")),
                " ".join(s.get("key_entities", [])),
                " ".join(s.get("key_relationships", [])),
            ])
            for s in self.summaries
        ]

        self.chunk_texts = [
            normalize_text(c.get("text"))
            for c in self.chunks
        ]

        self.node_embeddings = self._encode(self.node_texts)
        self.edge_embeddings = self._encode(self.edge_texts)
        self.summary_embeddings = self._encode(self.summary_texts)
        self.chunk_embeddings = self._encode(self.chunk_texts)

        print("[GraphRAG] Ready.")
        print("  Nodes:", len(self.nodes))
        print("  Edges:", len(self.edges))
        print("  Communities:", len(self.summaries))
        print("  Chunks:", len(self.chunks))

    def _encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    @staticmethod
    def _rank(query_embedding, document_embeddings, top_k):
        scores = np.dot(document_embeddings, query_embedding)
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            (int(i), float(scores[i]))
            for i in indices
        ]

    def retrieve(self, query, top_entities=5, top_edges=8,
                 top_communities=4, top_chunks=5):

        query = normalize_text(query)
        if not query:
            raise ValueError("Query cannot be empty.")

        query_embedding = self._encode([query])[0]

        # 1. Entity Retrieval
        entity_ranking = self._rank(
            query_embedding,
            self.node_embeddings,
            top_entities,
        )

        selected_nodes = []
        selected_node_ids = set()

        for index, score in entity_ranking:
            node = self.nodes[index]
            selected_node_ids.add(str(node["id"]))

            selected_nodes.append({
                "id": node["id"],
                "label": node["label"],
                "entity_type": node.get("entity_type"),
                "score": round(score, 4),
                "description": node.get("description", ""),
                "source_pages": node.get("source_pages", []),
            })

        # 2. Relationship Traversal
        # Retrieve all original directed edges incident to matched nodes.
        candidate_edge_indices = []

        for index, edge in enumerate(self.edges):
            source = str(edge["source"])
            target = str(edge["target"])

            if (
                source in selected_node_ids
                or target in selected_node_ids
            ):
                candidate_edge_indices.append(index)

        # Score candidate edges semantically against the query.
        edge_scores = []
        for index in candidate_edge_indices:
            score = float(
                np.dot(self.edge_embeddings[index], query_embedding)
            )
            edge_scores.append((index, score))

        edge_scores.sort(key=lambda x: x[1], reverse=True)
        edge_scores = edge_scores[:top_edges]

        selected_edges = []
        for index, score in edge_scores:
            edge = self.edges[index]
            selected_edges.append({
                "source": edge["source"],
                "target": edge["target"],
                "relation_type": edge.get("relation_type"),
                "description": edge.get("description", ""),
                "evidence": edge.get("evidence", ""),
                "source_pages": edge.get("source_pages", []),
                "score": round(score, 4),
            })

        # 3. Community Retrieval
        # Combine communities linked to retrieved entities with
        # semantically similar community summaries.
        linked_community_ids = set()

        for node_id in selected_node_ids:
            assignment = self.assignment_by_entity.get(node_id)
            if assignment:
                linked_community_ids.add(
                    int(assignment["community_id"])
                )

        summary_ranking = self._rank(
            query_embedding,
            self.summary_embeddings,
            len(self.summaries),
        )

        summary_scores = {
            int(self.summaries[index]["community_id"]): score
            for index, score in summary_ranking
        }

        candidate_community_ids = (
            linked_community_ids
            | {
                int(self.summaries[index]["community_id"])
                for index, _ in summary_ranking[:top_communities]
            }
        )

        ranked_communities = sorted(
            candidate_community_ids,
            key=lambda cid: summary_scores.get(cid, -1.0),
            reverse=True,
        )[:top_communities]

        selected_communities = []

        for cid in ranked_communities:
            summary = self.summary_by_id[cid]
            selected_communities.append({
                "community_id": cid,
                "main_topic": summary["main_topic"],
                "summary": summary["summary"],
                "key_entities": summary.get("key_entities", []),
                "key_relationships": summary.get(
                    "key_relationships", []
                ),
                "source_pages": summary.get("source_pages", []),
                "score": round(summary_scores.get(cid, 0.0), 4),
                "linked_to_retrieved_entity": cid in linked_community_ids,
            })

        # 4. Evidence Retrieval
        # Candidate pages come from matched nodes, edges, and summaries.
        candidate_pages = set()

        for node in selected_nodes:
            candidate_pages.update(node["source_pages"])

        for edge in selected_edges:
            candidate_pages.update(edge["source_pages"])

        for community in selected_communities:
            candidate_pages.update(community["source_pages"])

        candidate_chunk_indices = [
            i for i, chunk in enumerate(self.chunks)
            if int(chunk["page"]) in candidate_pages
        ]

        # If graph-derived pages are unavailable, use the full corpus.
        if not candidate_chunk_indices:
            candidate_chunk_indices = list(range(len(self.chunks)))

        chunk_scores = [
            (
                i,
                float(np.dot(self.chunk_embeddings[i], query_embedding))
            )
            for i in candidate_chunk_indices
        ]

        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        selected_chunks = []
        for index, score in chunk_scores[:top_chunks]:
            chunk = self.chunks[index]
            selected_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": round(score, 4),
            })

        return {
            "query": query,
            "retrieval_method": "GraphRAG",
            "entities": selected_nodes,
            "relationships": selected_edges,
            "communities": selected_communities,
            "evidence_chunks": selected_chunks,
            "candidate_pages": sorted(candidate_pages),
        }


def print_result(result):
    print("\n" + "=" * 75)
    print("QUERY:", result["query"])
    print("=" * 75)

    print("\n[1] ENTITY RETRIEVAL")
    for item in result["entities"]:
        print(
            f"- {item['label']} "
            f"({item['entity_type']}) "
            f"score={item['score']} "
            f"pages={item['source_pages']}"
        )

    print("\n[2] RELATIONSHIP TRAVERSAL")
    for item in result["relationships"]:
        print(
            f"- {item['source']} "
            f"--[{item['relation_type']}]--> "
            f"{item['target']} "
            f"score={item['score']}"
        )
        print("  Evidence:", item["evidence"][:220])

    print("\n[3] COMMUNITY RETRIEVAL")
    for item in result["communities"]:
        print(
            f"- C{item['community_id']:02d}: "
            f"{item['main_topic']} "
            f"score={item['score']} "
            f"entity_link={item['linked_to_retrieved_entity']}"
        )

    print("\n[4] ORIGINAL PAPER EVIDENCE")
    for item in result["evidence_chunks"]:
        print(
            f"- {item['chunk_id']} "
            f"(page {item['page']}, score={item['score']})"
        )
        print(" ", item["text"][:260].replace("\n", " "))


if __name__ == "__main__":
    retriever = GraphRAGRetriever()

    test_queries = [
        "How does GraphRAG construct a knowledge graph from source documents?",
        "How does GraphRAG use community summaries to answer global questions?",
        "Which evaluation metrics are used to compare different RAG approaches?",
    ]

    for query in test_queries:
        result = retriever.retrieve(query)
        print_result(result)
