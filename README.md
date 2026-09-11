# Agentic RAG

A lightweight, fully **local** Retrieval-Augmented Generation (RAG) playground built on top of Hugging Face Transformers. It loads local LLMs (Llama / Qwen / WiroAI) and a local embedding model, ingests your documents (PDF / TXT / MD / CSV), indexes them with either **FAISS** or **ChromaDB**, and answers questions grounded only in the retrieved context.

---

## Features

- **Two vector store backends**
  - **FAISS** – flat L2 index persisted to disk with JSON metadata.
  - **ChromaDB** – persistent local collection (telemetry disabled, 100% local).
- **Streaming ingestion** – page-by-page / line-by-line chunking so RAM usage stays flat even on large files.
- **Multi-format loaders** – PDF (`pypdf`), TXT/MD, and CSV (`pandas`).
- **Local embeddings** – mean-pooled + L2-normalized sentence embeddings from any HF encoder model.
- **Multiple LLMs** – switch between `llama`, `qwen`, and `wiroai` via config.
- **Grounded prompting** – system prompts force the model to answer only from retrieved context (with a dedicated CV/résumé Q&A mode).

---

## Project Structure

```
agentic_rag/
├── main.py                      # Entry point: build indexes & run RAG queries
├── config.py                    # Central config (model paths, data paths, RAG params, device)
├── agentic_rag_example.py       # Standalone LangGraph + llama.cpp agentic demo
├── models/
│   └── llm.py                   # LLM wrapper (load HF causal LM + generate)
├── rag/
│   ├── embeddings.py            # Embedding model wrapper (mean pooling + L2 norm)
│   ├── ingest.py                # FAISS ingestion pipeline
│   ├── ingest_chroma.py         # ChromaDB ingestion (batched, in-memory)
│   ├── ingest_chroma_streaming.py  # ChromaDB ingestion (streaming, low RAM)
│   ├── vector_store.py          # FAISS vector store (build/save/load/search)
│   └── vector_store_chroma.py   # ChromaDB vector store wrapper
└── data/
    ├── docs/                    # Put your source documents here
    ├── index/                   # FAISS index + metadata.json
    └── chroma/                  # ChromaDB persistent store
```

---

## How It Works

```
                ┌─────────────┐      ┌──────────────┐      ┌───────────────┐
  documents ──▶ │  Ingestion  │ ──▶  │  Embeddings  │ ──▶  │ Vector Store  │
 (pdf/txt/csv)  │  + chunking │      │ (mean pool)  │      │ FAISS / Chroma│
                └─────────────┘      └──────────────┘      └───────┬───────┘
                                                                    │ top-k search
   question ──────────────────────────────────────────────────────┤
                                                                    ▼
                                                            ┌───────────────┐
                                                            │  LLM.generate │ ──▶ grounded answer
                                                            │ (context+ Q)  │
                                                            └───────────────┘
```

1. **Ingest** – Documents are read, split into overlapping chunks (`chunk_size=800`, `chunk_overlap=100`), embedded, and stored in the chosen vector store.
2. **Retrieve** – The question is embedded and the top-k most similar chunks are fetched.
3. **Generate** – Retrieved chunks are injected into a system prompt and the LLM produces an answer constrained to that context.

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU is recommended (default device is `cuda:0`, dtype `bfloat16`).

Core dependencies:

```bash
pip install torch transformers faiss-cpu chromadb pypdf pandas numpy
```

> Use `faiss-gpu` instead of `faiss-cpu` if you want GPU-accelerated FAISS.

For the standalone `agentic_rag_example.py` demo only:

```bash
pip install langchain-community langgraph llama-cpp-python sentence-transformers
```

---

## Configuration

All paths and parameters live in `config.py`. Before running, set the local paths to your downloaded models:

```python
@dataclass
class ModelPahts:
    llama_path = "/path/to/llama"            # local HF model dir
    qwen_path = "/path/to/qwen"
    wiroai_path = "/path/to/wiroai"
    embedding_model_path = "/path/to/embedding-model"
```

Other tunables:

| Setting | Default | Description |
|---|---|---|
| `chunk_size` | `800` | Characters per chunk |
| `chunk_overlap` | `100` | Overlap between consecutive chunks |
| `device` | `"cuda:0"` | Inference device |
| `dtype` | `"bfloat16"` | Model dtype (`float16` / `bfloat16` / `float32`) |

LLM generation params (in `models/llm.py` → `LLMConfig`): `max_new_tokens=200`, `temperature=0.3`, `top_p=0.9`, `repetition_penalty=1.2`.

---

## Usage

Place your documents in `data/docs/`, configure your model paths, then drive everything from `main.py`.

### 1. Build an index

**ChromaDB (streaming, recommended):**

```python
from main import build_chroma_index
build_chroma_index()   # resets and rebuilds the Chroma collection
```

**FAISS:**

```python
from main import build_index
build_index()          # writes data/index/faiss.index + metadata.json
```

### 2. Ask questions

**RAG over ChromaDB:**

```python
from main import test_rag_with_chroma
test_rag_with_chroma("What is discussed in these documents?", top_k=3, model_name="llama")
```

**RAG over FAISS:**

```python
from main import test_rag_simple
test_rag_simple("What is discussed in these documents?", top_k=5)
```

**CV / résumé Q&A mode** (stricter prompt, answers only from the CV):

```python
from main import ask_cv
ask_cv("What does this person do?", top_k=3, model_name="llama")
```

### Run directly

`main.py` has a `__main__` block you can edit to choose which routine to run:

```bash
python main.py
```

---

## Standalone Agentic Example

`agentic_rag_example.py` is independent from the main pipeline. It wires a small **LangGraph** state machine (router → retrieve → generate) around a **llama.cpp** GGUF model and an in-memory Chroma store. Update `model_path` to your local `.gguf` file before running:

```bash
python agentic_rag_example.py
```

---

# Notes

- The system is designed to run **entirely offline** — embedding and LLM models are loaded from local paths (`local_files_only=True`), and Chroma telemetry is disabled.
- Answers are intentionally constrained to retrieved context; if the information isn't present, the model is instructed to say so rather than hallucinate.
- The streaming ChromaDB ingester is the most memory-friendly option for large document sets.
