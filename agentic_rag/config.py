from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelPahts:
    llama_path = ""
    qwen_path = ""
    wiroai_path = ""
    embedding_model_path = ""


@dataclass
class DataPaths:
    docs_dir: Path = Path("data/docs") # PDF, TXT, CSV, etc.
    index_dir: Path = Path("data/index") # FAISS + metadata files
    chroma_dir: Path = Path("data/chroma") # ChromaDB files


@dataclass
class RAGParams:
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass
class AppConfig:
    model_paths: ModelPahts = ModelPahts()
    data_paths: DataPaths = DataPaths()
    rag: RAGParams = RAGParams()
    device: str = "cuda:0"
    dtype: str = "bfloat16"


CONFIG = AppConfig()
