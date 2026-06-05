# rag/ingest_chroma_streaming.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable

import pandas as pd
from pypdf import PdfReader

from config import CONFIG
from rag.embeddings import EmbeddingModel, EmbeddingConfig
from rag.vector_store_chroma import ChromaVectorStore, ChromaConfig


@dataclass
class IngestStreamingConfig:
    docs_dir: Path = CONFIG.data_paths.docs_dir
    chunk_size: int = CONFIG.rag.chunk_size
    chunk_overlap: int = CONFIG.rag.chunk_overlap
    # This time we keep it small on purpose.
    # You can even set it to 1 if you want.
    # (batch=64 (each batch is added to the buffer)/buffer_size=64 => flush -> embedding -> Chroma -> clear buffer)
    batch_size_chunks: int = 64 # Determines after how many chunks embeddings are computed (embedding runs once there are 64 chunks)


def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    """
    We return chunks as a generator.
    This way we can iterate chunk by chunk without building a list.
    """
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        start = end - overlap
        if start < 0:
            start = 0


def iter_pdf_chunks(path: Path, cfg: IngestStreamingConfig) -> Iterable[Dict[str, Any]]:
    reader = PdfReader(str(path))
    for page_idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        if not page_text.strip():
            continue

        for i, ch in enumerate(chunk_text(page_text, cfg.chunk_size, cfg.chunk_overlap)):
            yield {
                "id": f"{path.name}-p{page_idx}-c{i}",
                "text": ch,
                "metadata": {
                    "source": str(path),
                    "page": page_idx,
                    "chunk_id": i,
                },
            }


def iter_txt_chunks(path: Path, cfg: IngestStreamingConfig) -> Iterable[Dict[str, Any]]:
    # For large files, read line by line so RAM usage doesn't blow up
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        buffer = []
        current_len = 0
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            buffer.append(line)
            current_len += len(line) + 1

            if current_len >= cfg.chunk_size:
                text = "\n".join(buffer)
                for i, ch in enumerate(chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)):
                    yield {
                        "id": f"{path.name}-c{hash((i, current_len))}",  # generate a unique id
                        "text": ch,
                        "metadata": {
                            "source": str(path),
                        },
                    }
                buffer = []
                current_len = 0

        # Remaining
        if buffer:
            text = "\n".join(buffer)
            for i, ch in enumerate(chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)):
                yield {
                    "id": f"{path.name}-c{hash((i, current_len, 'last'))}",
                    "text": ch,
                    "metadata": {
                        "source": str(path),
                    },
                }


def iter_csv_chunks(path: Path, cfg: IngestStreamingConfig, max_rows: int = 50) -> Iterable[Dict[str, Any]]:
    # Don't load very large CSVs entirely into RAM
    # We read them with chunksize
    row_count = 0
    for df_chunk in pd.read_csv(path, chunksize=10):
        df_chunk_str = df_chunk.to_string()
        for i, ch in enumerate(chunk_text(df_chunk_str, cfg.chunk_size, cfg.chunk_overlap)):
            yield {
                "id": f"{path.name}-r{row_count}-c{i}",
                "text": ch,
                "metadata": {
                    "source": str(path),
                },
            }
        row_count += len(df_chunk)
        if row_count >= max_rows:
            break


def ingest_documents_to_chroma_streaming(reset: bool = True):
    cfg = IngestStreamingConfig()
    docs_dir = cfg.docs_dir

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    print(f"[Ingest-Streaming] Reading documents from: {docs_dir}")

    # Embedding model (stays on the GPU but RAM usage remains constant)
    embed_model = EmbeddingModel(EmbeddingConfig())

    # Chroma store
    store = ChromaVectorStore(ChromaConfig())
    if reset:
        store.reset_collection()

    buffer_ids: List[str] = []
    buffer_texts: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []
    total_chunks = 0

    def flush():
        nonlocal buffer_ids, buffer_texts, buffer_metas, total_chunks
        if not buffer_texts:
            return
        print(f"[Ingest-Streaming] Flushing {len(buffer_texts)} chunks...")
        embeddings_tensor = embed_model.encode(buffer_texts, batch_size=16)  # Set batch_size to 2 or 1 if needed / determines how many chunks are processed at once
        embeddings = embeddings_tensor.numpy()
        store.add_embeddings(
            ids=buffer_ids,
            embeddings=embeddings,
            metadatas=buffer_metas,
            documents=buffer_texts,
        )
        total_chunks += len(buffer_texts)
        buffer_ids, buffer_texts, buffer_metas = [], [], []

    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
            
        doc_count = 0
        if doc_count >= 5: # testing for 5 pages
            break

        ext = path.suffix.lower()

        if ext == ".pdf":
            print(f"[Ingest-Streaming] PDF: {path.name}")
            chunks_iter = iter_pdf_chunks(path, cfg)
        elif ext in [".txt", ".md"]:
            print(f"[Ingest-Streaming] Text: {path.name}")
            chunks_iter = iter_txt_chunks(path, cfg)
        elif ext == ".csv":
            print(f"[Ingest-Streaming] CSV: {path.name}")
            chunks_iter = iter_csv_chunks(path, cfg)
        else:
            print(f"[Ingest-Streaming] Skipping unsupported file: {path.name}")
            continue

        for item in chunks_iter:
            buffer_ids.append(item["id"])
            buffer_texts.append(item["text"])
            buffer_metas.append(item["metadata"])

            if len(buffer_texts) >= cfg.batch_size_chunks:
                flush()

    flush()

    print(f"[Ingest-Streaming] Done. Total chunks indexed: {total_chunks}")
