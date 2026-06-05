from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import json
import numpy as np

from config import CONFIG


@dataclass
class VectorStoreConfig:
    index_path: Path = CONFIG.data_paths.index_dir / "faiss.index"
    metadata_path: Path = CONFIG.data_paths.index_dir / "metadata.json"


class FaissVectorStore:
    def __init__(self, config: VectorStoreConfig | None = None):
        self.config = config or VectorStoreConfig()
        self.index = None # Faiss index object
        self.metadata: List[Dict[str, Any]] = []
    
    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Build the vector store from embeddings and metadata"""
        assert embeddings.ndim == 2, "Embeddings shape must be (N, D)"
        n, d = embeddings.shape
        print(f"[FAISS] Building index with {n} vectors, dim={d}")

        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings.astype("float32"))

        self.metadata = metadata
    
    def save(self):
        """Save the vector store to disk"""
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise ValueError("Index is not built yet")
        
        print(f"[FAISS] Saving index to {self.config.index_path}")
        faiss.write_index(self.index, str(self.config.index_path))

        print(f"[FAISS] Saving metadata to {self.config.metadata_path}")
        with open(self.config.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """Load the vector store from disk"""
        print(f"[FAISS] Loading index from {self.config.index_path}")
        self.index = faiss.read_index(str(self.config.index_path))

        print(f"[FAISS] Loading metadata from {self.config.metadata_path}")
        with open(self.config.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the vector store for the most similar vectors"""
        if self.index is None:
            raise ValueError("Index is not loaded or built")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, :]
        
        query_embedding = query_embedding.astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[int(idx)]
            results.append(
                {
                    "distance": float(dist),
                    "metadata": meta,
                }
            )

        return results
