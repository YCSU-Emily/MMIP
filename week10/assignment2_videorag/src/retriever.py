from pathlib import Path

import torch
import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    / "5617a9f61b028005a4858fdac845db406aefb181"
)

DB_PATH = ROOT / "outputs" / "videorag_chroma_db"

COLLECTION_NAME = "videorag_corpus"


class VideoRetriever:
    def __init__(self, per_modality_k=4):
        torch.set_num_threads(4)

        self.per_modality_k = per_modality_k

        self.model = SentenceTransformer(
            str(MODEL_PATH),
            device="cpu",
            local_files_only=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print("[VideoRetriever] BGE-M3 loaded on CPU.")
        print(
            "[VideoRetriever] Corpus records:",
            self.collection.count(),
        )

    def retrieve(self, query, final_k=6):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

        modality_results = {}

        # 1. Retrieve each modality independently.
        for modality in ["speech", "visual_ocr"]:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=self.per_modality_k,
                where={"modality": modality},
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            modality_results[modality] = results

        # 2. Reciprocal Rank Fusion.
        rrf_constant = 60
        fused = {}

        for modality, results in modality_results.items():
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for rank, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1,
            ):
                if doc_id not in fused:
                    fused[doc_id] = {
                        "id": doc_id,
                        "text": document,
                        "metadata": metadata,
                        "distance": distance,
                        "rrf_score": 0.0,
                        "retrieved_from": [],
                    }

                fused[doc_id]["rrf_score"] += (
                    1.0 / (rrf_constant + rank)
                )

                fused[doc_id]["retrieved_from"].append(
                    {
                        "modality": modality,
                        "rank": rank,
                        "distance": distance,
                    }
                )

        # 3. Sort by fused score.
        ranked_results = sorted(
            fused.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["distance"],
            ),
        )

        return ranked_results[:final_k]
