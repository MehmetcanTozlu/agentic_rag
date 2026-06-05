from models.llm import LLM, LLMConfig
from rag.ingest import ingest_documents
from rag.ingest_chroma import ChromaVectorStore, ChromaConfig
from rag.embeddings import EmbeddingModel, EmbeddingConfig
from rag.vector_store import FaissVectorStore, VectorStoreConfig
from rag.ingest_chroma_streaming import ingest_documents_to_chroma_streaming
import numpy as np


# ===================== CHROMADB =====================
def build_chroma_index():
    ingest_documents_to_chroma_streaming(reset=True)

def test_rag_with_chroma(question: str, top_k: int = 5, model_name: str ="llama"):
    embed_model = EmbeddingModel(EmbeddingConfig)
    q_emb = embed_model.encode([question], batch_size=1)
    q_emb_np = q_emb.numpy()[0]

    store = ChromaVectorStore(ChromaConfig())
    results = store.search(q_emb_np, top_k=top_k)

    context_parts = []
    for r in results:
        meta = r["metadata"]
        doc = r["document"]
        context_parts.append(
            f"[SOURCE: {meta['source']} / chunk {meta['chunk_id']}]\n{doc}"
        )
    
    print("\n--- DEBUG: Context preview (first 200 characters) ---")
    for i, r in enumerate(results):
        doc_preview = r["document"][:200].replace("\n", " ")
        print(f"[{i+1}] {doc_preview}...")
    print("-----------------------------------------------")

    context = "\n\n---\n\n".join(context_parts)

    llm = LLM(LLMConfig(model_name=model_name))
    system_prompt = (
        "Below are excerpts taken from the documents. "
        "Answer the user's question based ONLY on the information in these excerpts. "
        "Do not repeat yourself; do not write the same sentence or phrase more than once. "
        "Your answer should be at most 4-6 sentences. "
        "If there is not enough information to answer, say so clearly.\n\n"
        f"{context}"
    )

    answer = llm.generate(question, system_prompt=system_prompt)

    print("\n=== RAG + Chroma TEST ===")
    print("Question:")
    print(question)
    print("\nChunks used from context:")
    for i, r in enumerate(results):
        meta = r["metadata"]
        dist = r.get("distance")

        if isinstance(dist, (list, tuple)):
            dist = dist[0]

        try:
            dist_str = f"{float(dist):.4f}"
        except Exception:
            dist_str = str(dist)

        print(f"[{i+1}] source={meta.get('source')} | chunk={meta.get('chunk_id')} | dist={dist_str}")


    print("\nAnswer:\n")
    print(answer)
    print("="*60)

# ===================== FAISS =====================
def test_model(model_name: str, question: str):
    llm = LLM(LLMConfig(model_name=model_name))

    system_prompt = (
        "You are an analytical assistant that gives short but clear answers. "
        "If you need to make an assumption while answering the question, state it explicitly."
    )

    answer = llm.generate(question, system_prompt=system_prompt)
    print(f"\n=== MODEL: {model_name} ===")
    print(f"Question : {question}")
    print(f"Answer:\n{answer}")
    print("="*50)

def build_index():
    ingest_documents()

def test_rag_simple(question: str, top_k: int = 5):
    embed_model = EmbeddingModel(EmbeddingConfig())
    q_emb = embed_model.encode([question])
    q_emb_np = q_emb.numpy()[0]

    store = FaissVectorStore(VectorStoreConfig())
    store.load()

    results = store.search(q_emb_np, top_k=top_k)
    
    context_parts = []
    for r in results:
        meta = r['metadata']
        context_parts.append(
            f"[Source: {meta['source']} / chunk {meta['chunk_id']}]\n{meta['text']}"
        )
    
    context = "\n\n--\n\n".join(context_parts)

    llm = LLM(LLMConfig(model_name="llama"))
    system_prompt = (
        "Below are excerpts taken from the documents. "
        "Answer the user's question based only on this information. "
        "If you don't know, say 'I cannot find this information in the documents I have'.\n\n"
        f"{context}"
    )

    answer = llm.generate(question, system_prompt=system_prompt)

    print("\n=== RAG TEST ===")
    print("Question:")
    print(f"{question}")
    print("\nChunks used from context (summary):")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] source={r['metadata']['source']} | dist={r['distance']:.4f}")
    print("\nAnswer:")
    print(answer)
    print("="*60)


# ===================== FOR ONLY CV EVALUATION =====================
def ask_cv(question: str, top_k: int = 3, model_name: str = "llama"):
    """
    Embedding the question
    Get top_k results from chroma index
    Give these chunks to LLM as context parts
    Create answer
    """

    # Embedding the question
    embed_model = EmbeddingModel(EmbeddingConfig())
    q_emb = embed_model.encode([question], batch_size=1)
    q_emb_np = q_emb.numpy()[0]

    # Get top_k results from chroma index
    store = ChromaVectorStore(ChromaConfig())
    results = store.search(q_emb_np, top_k=top_k)

    if not results:
        print("No results found")
        return
    
    # Give these chunks to LLM as context parts
    context_parts = []
    for r in results:
        meta = r["metadata"]
        doc = r["document"]

        context_parts.append(
            f"[Source: {meta.get('source')} / chunk {meta.get('chunk_id', '?')}]\n{doc}"
        )
    
    context = "\n\n--\n\n".join(context_parts)
    
    # Special system prompt for CV
    llm = LLM(LLMConfig(model_name=model_name))

    system_prompt = (
        "Below are excerpts taken from a resume (CV) document. "
        "Answer the user's question based ONLY on the information in these excerpts. "
        "Do not guess; do not make up information that is not written in the CV. "
        "Write your answer short, clear, and as bullet points when possible. "
        "Do not add editor-like commentary notes; do not write sections like 'editor's note'. "
        "Just answer the question directly. "
        "If the requested information is not in the CV, simply say 'This information is not in the CV.'\n\n"
        f"{context}"
    )

    answer = llm.generate(question, system_prompt=system_prompt)

    # Print the answer
    print("\n=== CV QA ===")
    print("Question:")
    print(question)

    print("\nChunks used:")
    for i, r in enumerate(results):
        meta = r["metadata"]
        print(f"[{i+1}] source={meta.get('source')} | chunk={meta.get('chunk_id', '?')} | dist={r['distance']}")

    print("\nAnswer:\n")
    print(answer)
    print("="*60)


if __name__ == "__main__":
    # ===================== FAISS =====================
    #build_index()

    #q = "What is discussed about the stock market in these documents?"
    #test_rag_simple(q)

    # ===================== CHROMADB =====================
    # build_chroma_index()

    ask_cv("What does this person do?", top_k=3, model_name="llama")

    # q = "Mehmetcan Tozlu is Which he using ai techniques and technologies like python, mlflow, etc.?"
    # test_rag_with_chroma(q, top_k=3, model_name="llama")

    # from pathlib import Path
    # from pypdf import PdfReader

    # path = Path("data/docs/MehmetcanTozlu-CV-eng.pdf")
    # reader = PdfReader(str(path))

    # for page_idx, page in enumerate(reader.pages):
    #     print("==== PAGE", page_idx, "====")
    #     text = page.extract_text() or ""
    #     print(text[:1500])  # first 1500 characters
    #     print()