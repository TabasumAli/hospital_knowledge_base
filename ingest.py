"""
ingest.py
---------
Reads every PDF under KNOWLEDGE_BASE_DIR (including all subfolders like
admissions/, departments/, emergency/, hospital/, patient_safety/),
splits the text into overlapping chunks, embeds the chunks locally
(free, no API key needed), and saves everything into a FAISS index on
disk so app.py can load it instantly at query time.

Run this once initially, and again any time you add/update PDFs.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = "ayub_medical_knowledge_base"
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_all_pdfs(root_dir: str):
    """Walk every subfolder and load each PDF, tagging each page with
    the category (subfolder name) and the original filename so we can
    cite exact sources later with no hallucination."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"'{root_dir}' not found. Download your Drive folder here first "
            f"(see SETUP.md step 1)."
        )

    all_docs = []
    pdf_paths = sorted(root.rglob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found under '{root_dir}'. Check the folder was "
            f"downloaded correctly."
        )

    for pdf_path in pdf_paths:
        category = pdf_path.parent.name  # e.g. "admissions", "emergency"
        filename = pdf_path.name
        print(f"Loading: {category}/{filename}")

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # one Document per PDF page

        for page in pages:
            # Keep clean, exact source metadata for citation at answer time
            page.metadata["source_file"] = filename
            page.metadata["category"] = category
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1

        all_docs.extend(pages)

    print(f"\nLoaded {len(pdf_paths)} PDFs, {len(all_docs)} pages total.")
    return all_docs


def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_and_save_index(chunks):
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}' (runs locally, no API cost)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Creating embeddings and building FAISS index (this can take a few minutes)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"\n✅ FAISS index saved to '{FAISS_INDEX_DIR}/'")


def main():
    print("=" * 60)
    print("Ayub Medical Complex — Knowledge Base Ingestion")
    print("=" * 60)

    documents = load_all_pdfs(KNOWLEDGE_BASE_DIR)
    chunks = split_into_chunks(documents)
    build_and_save_index(chunks)

    print("\nDone. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()