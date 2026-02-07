"""
Document Ingestion & Chunking Pipeline (LangChain)
====================================================
Uses LangChain community document loaders to ingest ~10 documents,
then chunks with RecursiveCharacterTextSplitter (size=600, overlap=100),
logs statistics, and prints sample chunks.

Usage:
    python document_chunking_pipeline.py

Dependencies:
    pip install langchain langchain-community langchain-text-splitters
    pip install pypdf unstructured markdown python-docx

Directory layout expected:
    ./documents/
        ├── report.pdf
        ├── data.csv
        ├── notes.md
        ├── page.html
        ├── script.py
        ├── readme.txt
        └── ...
"""

import os
import json
import glob
import random
import logging
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    PythonLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ────────────────────────────────────────────────────────────
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
DOCS_DIR = "./documents"  # UPDATE to your documents folder
NUM_SAMPLES = 5

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("chunking_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Loader mapping by extension ──────────────────────────────────────────────
# Each loader follows the pattern from the reference slides:
#   loader = SomeLoader(file_path)
#   documents = loader.load()
LOADER_MAP = {
    ".csv":  lambda fp: CSVLoader(file_path=fp),
    ".pdf":  lambda fp: PyPDFLoader(fp),
    ".html": lambda fp: UnstructuredHTMLLoader(file_path=fp),
    ".htm":  lambda fp: UnstructuredHTMLLoader(file_path=fp),
    ".md":   lambda fp: UnstructuredMarkdownLoader(fp),
    ".py":   lambda fp: PythonLoader(fp),
    ".txt":  lambda fp: TextLoader(fp, encoding="utf-8"),
}


# ── 1. Document Ingestion ────────────────────────────────────────────────────

def ingest_documents(docs_dir: str):
    """Load all supported documents from a directory using LangChain loaders."""
    all_docs = []
    file_doc_counts = {}

    for filepath in sorted(glob.glob(os.path.join(docs_dir, "*"))):
        ext = Path(filepath).suffix.lower()
        name = os.path.basename(filepath)

        if ext not in LOADER_MAP:
            logger.warning(f"Unsupported file type '{ext}' — skipping {name}")
            continue

        try:
            loader = LOADER_MAP[ext](filepath)
            documents = loader.load()
            all_docs.extend(documents)
            file_doc_counts[name] = len(documents)
            logger.info(f"Loaded: {name} → {len(documents)} document(s)")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    logger.info(
        f"Total files loaded: {len(file_doc_counts)}, "
        f"total document objects: {len(all_docs)}"
    )
    return all_docs, file_doc_counts


# ── 2. Text Cleaning ─────────────────────────────────────────────────────────

def clean_document(doc):
    """Clean page_content in-place on a LangChain Document."""
    import re

    text = doc.page_content
    # Remove null bytes & non-printable chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize unicode whitespace
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip each line
    text = "\n".join(line.strip() for line in text.splitlines())
    doc.page_content = text.strip()


# ── 3. Chunking ──────────────────────────────────────────────────────────────

def chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split documents using LangChain RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("Pipeline started")

    # --- Validate directory ---
    if not os.path.isdir(DOCS_DIR):
        logger.error(f"Documents directory not found: {DOCS_DIR}")
        print(f"\n❌ Directory '{DOCS_DIR}' does not exist. Update DOCS_DIR and retry.")
        return

    # --- 1. Ingest ---
    all_docs, file_doc_counts = ingest_documents(DOCS_DIR)
    if not all_docs:
        print("\n❌ No documents found. Check DOCS_DIR and file types.")
        return

    print(f"\n✅ Ingested {len(file_doc_counts)} files → {len(all_docs)} document objects")
    for name, count in file_doc_counts.items():
        print(f"   • {name}: {count} doc(s)")

    # --- 2. Clean ---
    for doc in all_docs:
        clean_document(doc)
    print(f"\n✅ Cleaned {len(all_docs)} documents")

    # Show sample metadata (source, row, page, etc.)
    print("\n📄 Sample document metadata:")
    for doc in all_docs[:3]:
        print(
            f"   source={doc.metadata.get('source', 'N/A')}, "
            f"keys={list(doc.metadata.keys())}, "
            f"content_len={len(doc.page_content)}"
        )

    # --- 3. Chunk ---
    chunks = chunk_documents(all_docs)
    print(f"\n✅ Chunking complete (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"   Total chunks: {len(chunks)}")

    # --- 4. Statistics ---
    chunk_lengths = [len(c.page_content) for c in chunks]

    # Chunks per source file
    chunks_per_doc = {}
    for c in chunks:
        src = os.path.basename(c.metadata.get("source", "unknown"))
        chunks_per_doc[src] = chunks_per_doc.get(src, 0) + 1

    stats = {
        "timestamp": datetime.now().isoformat(),
        "config": {"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "total_files": len(file_doc_counts),
        "total_chunks": len(chunks),
        "avg_chunk_length": (
            round(sum(chunk_lengths) / len(chunk_lengths), 1)
            if chunk_lengths
            else 0
        ),
        "min_chunk_length": min(chunk_lengths, default=0),
        "max_chunk_length": max(chunk_lengths, default=0),
        "chunks_per_document": chunks_per_doc,
    }

    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Avg chunk length: {stats['avg_chunk_length']} chars")
    logger.info(f"Chunks per doc: {stats['chunks_per_document']}")

    with open("chunk_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n📊 Pipeline Statistics")
    print(f"   Total files:     {stats['total_files']}")
    print(f"   Total chunks:    {stats['total_chunks']}")
    print(f"   Avg chunk len:   {stats['avg_chunk_length']} chars")
    print(f"   Min chunk len:   {stats['min_chunk_length']} chars")
    print(f"   Max chunk len:   {stats['max_chunk_length']} chars")
    print(f"\n   Chunks per document:")
    for name, count in stats["chunks_per_document"].items():
        print(f"     {name}: {count}")

    # --- 5. Sample Chunks ---
    sample_indices = random.sample(
        range(len(chunks)), min(NUM_SAMPLES, len(chunks))
    )
    print(f"\n📝 {len(sample_indices)} Sample Chunks")
    print("=" * 60)
    for i, idx in enumerate(sample_indices, 1):
        chunk = chunks[idx]
        src = os.path.basename(chunk.metadata.get("source", "unknown"))
        content = chunk.page_content
        preview = content[:500] + ("..." if len(content) > 500 else "")
        print(f"\n--- Sample {i} | source: {src} | length: {len(content)} chars ---")
        print(f"Metadata: {chunk.metadata}")
        print(preview)
        print()

    # --- Done ---
    print("=" * 60)
    print("✅ Pipeline complete. Outputs: chunk_stats.json, chunking_pipeline.log")
    print("\nNext steps:")
    print("  git add document_chunking_pipeline.py chunk_stats.json chunking_pipeline.log")
    print('  git commit -m "feat: add LangChain document chunking pipeline"')
    print("  git push origin main")


if __name__ == "__main__":
    main()
