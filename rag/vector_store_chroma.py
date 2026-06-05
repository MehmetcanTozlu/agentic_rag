from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from config import CONFIG


@dataclass
class ChromaConfig:
    persist_dir: Path = CONFIG.data_paths.chroma_dir
    collection_name: str = "rag_chunks"


class ChromaVectorStore:
    def __init__(self, config: ChromaConfig | None = None):
        self.config = config or ChromaConfig()
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Chroma] Using persist dir: {self.config.persist_dir}")

        self.client = chromadb.PersistentClient(
            path = str(self.config.persist_dir),
            settings = Settings(
                anonymized_telemetry = False, # Entirely Local
            ),
        )

        self.collection = self.client.get_or_create_collection(
            name = self.config.collection_name,
            embedding_function=None,
        )

    def reset_collection(self):
        print(f"[Chroma] Resetting collection: {self.config.collection_name}")
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name = self.config.collection_name,
            embedding_function = None,
        )
    
    def add_embeddings(self, ids: List[str], embeddings, metadatas: List[Dict[str, Any]], documents: List[str]):
        print(f"[Chroma] Adding {len(ids)} embeddings to collection..")
        self.collection.add(
            ids = ids,
            embeddings = embeddings,
            metadatas = metadatas,
            documents = documents,
        )
    
    def search(self, query_embedding, top_k: int = 5) -> List[Dict[str, Any]]:
        if hasattr(query_embedding, "toList"):
            query_embedding = query_embedding.toList()
        
        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results = top_k,
        )

        out: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for _id, meta, doc, dist in zip(ids, metadatas, documents, distances):
            out.append(
                {
                    "id": _id,
                    "distance": distances,
                    "metadata": meta,
                    "document": doc,
                }
            )
        
        return out
