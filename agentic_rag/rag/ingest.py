from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from config import CONFIG
from rag.embeddings import EmbeddingModel, EmbeddingConfig
from rag.vector_store import FaissVectorStore, VectorStoreConfig

import pandas as pd
import numpy as np
from pypdf import PdfReader


@dataclass
class IngestConfig:
    docs_dir: Path = CONFIG.data_paths.docs_dir
    chunk_size: int = CONFIG.rag.chunk_size
    chunk_overlap: int = CONFIG.rag.chunk_overlap


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text into smaller chunks"""
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
    """Load a PDF file and return the text"""
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
    """Load a text file and return the text"""
    return path.read_text(encoding="utf-8", errors="ignore")

def load_csv(path: Path, max_rows: int = 50) -> str:
    """Load a CSV file and return the text"""
    df = pd.read_csv(path)
    
    if len(df) > max_rows:
        df = df.head(max_rows)
    
    return df.to_string()

def ingest_documents():
    cfg = IngestConfig()
    docs_dir = cfg.docs_dir

    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    print(f"[Ingest] Reading documents from: {docs_dir}")

    all_text_chunks: List[str] = []
    all_metadata: List[Dict[str, Any]] = []

    # Scan the documents directory
    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
        
        ext = path.suffix.lower()

        if ext == ".pdf":
            print(f"[Ingest] Loading PDF: {path.name}")
            raw_text = load_pdf(path)
        elif ext in [".txt", ".md"]:
            print(f"[Ingest] Loading text file: {path.name}")
            raw_text = load_txt(path)
        elif ext in [".csv"]:
            print(f"[Ingest] Loading CSV file: {path.name}")
        else:
            print(f"[Ingest] Skipping unsupported file type: {path.name}")
            continue
            
        if not raw_text.strip():
            print(f"[Ingest] Empty content, skipping: {path.name}")
            continue
        
        chunks = chunk_text(
            raw_text,
            chunk_size = cfg.chunk_size,
            overlap = cfg.chunk_overlap,
        )

        print(f"[Ingest] {path.name} -> {len(chunks)} chunks")

        for i, ch in enumerate(chunks):
            all_text_chunks.append(ch)
            all_metadata.append(
                {
                    "source": str(path),
                    "chunk_id": i,
                    "text": ch,
                }
            )
    
    if not all_text_chunks:
        print("[Ingest] No chunks generated, nothing to index.")
        return
    
    print(f"[Ingest] Compuing embeddings for {len(all_text_chunks)} chunks...")
    embed_model = EmbeddingModel(EmbeddingConfig())
    embeddings_tensor = embed_model.encode(all_text_chunks, batch_size=16)
    embeddings = embeddings_tensor.numpy()

    store = FaissVectorStore(VectorStoreConfig())
    store.build(embeddings, all_metadata)
    store.save()

    print(f"[Ingest] Done. Index built and saved.")
