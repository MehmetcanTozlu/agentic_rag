# rag/ingest_chroma.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from pypdf import PdfReader
import numpy as np

from config import CONFIG
from rag.embeddings import EmbeddingModel, EmbeddingConfig
from rag.vector_store_chroma import ChromaVectorStore, ChromaConfig

@dataclass
class IngestChromaConfig:
    docs_dir: Path = CONFIG.data_paths.docs_dir
    chunk_size: int = CONFIG.rag.chunk_size
    chunk_overlap: int = CONFIG.rag.chunk_overlap
    batch_size_chunks: int = 64   # After how many chunks should we compute embeddings and add to Chroma?

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0

    return chunks

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        texts.append(page_text)
    return "\n".join(texts)

def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_csv(path: Path, max_rows: int = 50) -> str:
    df = pd.read_csv(path)
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_string()

def ingest_documents_to_chroma(reset: bool = True):
    cfg = IngestChromaConfig()
    docs_dir = cfg.docs_dir

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    print(f"[Ingest-Chroma] Reading documents from: {docs_dir}")

    # Embedding model
    embed_model = EmbeddingModel(EmbeddingConfig())

    # Chroma store
    store = ChromaVectorStore(ChromaConfig())
    if reset:
        store.reset_collection()

    buffer_texts: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []
    buffer_ids: List[str] = []
    global_chunk_counter = 0

    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()

        if ext == ".pdf":
            print(f"[Ingest-Chroma] Loading PDF: {path.name}")
            raw_text = load_pdf(path)
        elif ext in [".txt", ".md"]:
            print(f"[Ingest-Chroma] Loading text: {path.name}")
            raw_text = load_txt(path)
        elif ext == ".csv":
            print(f"[Ingest-Chroma] Loading CSV: {path.name}")
            raw_text = load_csv(path)
        else:
            print(f"[Ingest-Chroma] Skipping unsupported file: {path.name}")
            continue

        if not raw_text.strip():
            print(f"[Ingest-Chroma] Empty content, skipping: {path.name}")
            continue

        chunks = chunk_text(
            raw_text,
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
        )

        print(f"[Ingest-Chroma] {path.name} -> {len(chunks)} chunks")

        for i, ch in enumerate(chunks):
            chunk_id = f"{path.name}-{i}"
            buffer_ids.append(chunk_id)
            buffer_texts.append(ch)
            buffer_metas.append(
                {
                    "source": str(path),
                    "chunk_id": i,
                }
            )
            global_chunk_counter += 1

            # When the batch is full, compute embeddings and add to Chroma
            if len(buffer_texts) >= cfg.batch_size_chunks:
                _flush_buffer_to_chroma(
                    store,
                    embed_model,
                    buffer_ids,
                    buffer_texts,
                    buffer_metas,
                )
                buffer_ids, buffer_texts, buffer_metas = [], [], []

    # Flush any remaining chunks
    if buffer_texts:
        _flush_buffer_to_chroma(
            store,
            embed_model,
            buffer_ids,
            buffer_texts,
            buffer_metas,
        )

    print(f"[Ingest-Chroma] Done. Total chunks indexed: {global_chunk_counter}")

def _flush_buffer_to_chroma(
    store: ChromaVectorStore,
    embed_model: EmbeddingModel,
    ids: List[str],
    texts: List[str],
    metadatas: List[Dict[str, Any]],
):
    print(f"[Ingest-Chroma] Flushing {len(texts)} chunks to Chroma...")
    embeddings_tensor = embed_model.encode(texts, batch_size=1)  # You can lower batch_size depending on VRAM
    embeddings = embeddings_tensor.numpy()  # (N, D)
    store.add_embeddings(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts,
    )
