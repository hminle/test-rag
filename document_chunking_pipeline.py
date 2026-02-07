"""
Document Ingestion & Chunking Pipeline
=======================================
Ingests ~10 documents, cleans text, chunks with size=600 / overlap=100,
logs statistics, and prints sample chunks.

Usage:
    python document_chunking_pipeline.py

Dependencies:
    pip install pymupdf python-docx beautifulsoup4
"""

import os
import re
import json
import glob
import random
import logging
from pathlib import Path
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
CHUNK_SIZE = 600        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
DOCS_DIR = "./documents"  # UPDATE to your documents folder
SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx", ".html", ".csv", ".json"]
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


# ── 1. Document Ingestion ────────────────────────────────────────────────────

def load_document(filepath: str) -> str:
    """Load a single document and return its raw text."""
    ext = Path(filepath).suffix.lower()

    if ext in (".txt", ".md", ".csv"):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    if ext == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)

    if ext == ".html":
        try:
            from bs4 import BeautifulSoup
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator=" ")
        except ImportError:
            logger.warning("bs4 not installed; reading HTML as raw text")
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install pymupdf")
            return ""

    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            logger.error("python-docx not installed. Run: pip install python-docx")
            return ""

    logger.warning(f"Unsupported file type: {ext} — skipping {filepath}")
    return ""


def ingest_documents(docs_dir: str) -> dict:
    """Scan directory and load all supported documents."""
    documents = {}
    for ext in SUPPORTED_EXTENSIONS:
        for fpath in glob.glob(os.path.join(docs_dir, f"*{ext}")):
            name = os.path.basename(fpath)
            text = load_document(fpath)
            if text.strip():
                documents[name] = text
                logger.info(f"Loaded: {name} ({len(text):,} chars)")
            else:
                logger.warning(f"Empty or unreadable: {name}")
    logger.info(f"Total documents ingested: {len(documents)}")
    return documents


# ── 2. Text Cleaning ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean a document's text."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# ── 3. Chunking ──────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end < text_len:
            boundary = -1
            for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos > boundary:
                    boundary = pos + 1
            if boundary > start:
                end = boundary
        else:
            end = text_len

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < text_len else text_len

    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("Pipeline started")

    # --- Ingest ---
    if not os.path.isdir(DOCS_DIR):
        logger.error(f"Documents directory not found: {DOCS_DIR}")
        print(f"\n❌ Directory '{DOCS_DIR}' does not exist. Update DOCS_DIR and retry.")
        return

    raw_docs = ingest_documents(DOCS_DIR)
    if not raw_docs:
        print("\n❌ No documents found. Check DOCS_DIR and SUPPORTED_EXTENSIONS.")
        return

    print(f"\n✅ Ingested {len(raw_docs)} documents")
    for name, text in raw_docs.items():
        print(f"   • {name}: {len(text):,} chars")

    # --- Clean ---
    cleaned_docs = {name: clean_text(text) for name, text in raw_docs.items()}
    print("\n✅ Cleaning complete")
    for name in cleaned_docs:
        before, after = len(raw_docs[name]), len(cleaned_docs[name])
        print(f"   • {name}: {before:,} → {after:,} chars (removed {before - after:,})")

    # --- Chunk ---
    all_chunks = {}
    for name, text in cleaned_docs.items():
        chunks = chunk_text(text)
        all_chunks[name] = chunks
        logger.info(f"Chunked {name}: {len(chunks)} chunks")

    print(f"\n✅ Chunking complete (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    for name, chunks in all_chunks.items():
        print(f"   • {name}: {len(chunks)} chunks")

    # --- Statistics ---
    flat_chunks = [c for chunks in all_chunks.values() for c in chunks]
    chunk_lengths = [len(c) for c in flat_chunks]

    stats = {
        "timestamp": datetime.now().isoformat(),
        "config": {"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "total_documents": len(all_chunks),
        "total_chunks": len(flat_chunks),
        "avg_chunk_length": round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0,
        "min_chunk_length": min(chunk_lengths, default=0),
        "max_chunk_length": max(chunk_lengths, default=0),
        "chunks_per_document": {name: len(chunks) for name, chunks in all_chunks.items()},
    }

    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Avg chunk length: {stats['avg_chunk_length']} chars")
    logger.info(f"Chunks per doc: {stats['chunks_per_document']}")

    with open("chunk_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n📊 Pipeline Statistics")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Total chunks:    {stats['total_chunks']}")
    print(f"   Avg chunk len:   {stats['avg_chunk_length']} chars")
    print(f"   Min chunk len:   {stats['min_chunk_length']} chars")
    print(f"   Max chunk len:   {stats['max_chunk_length']} chars")
    print(f"\n   Chunks per document:")
    for name, count in stats["chunks_per_document"].items():
        print(f"     {name}: {count}")

    # --- Sample Chunks ---
    sample_indices = random.sample(range(len(flat_chunks)), min(NUM_SAMPLES, len(flat_chunks)))
    print(f"\n📝 {len(sample_indices)} Sample Chunks")
    print("=" * 60)
    for i, idx in enumerate(sample_indices, 1):
        chunk = flat_chunks[idx]
        preview = chunk[:500] + ("..." if len(chunk) > 500 else "")
        print(f"\n--- Sample {i} (index={idx}, length={len(chunk)} chars) ---")
        print(preview)
        print()

    # --- Done ---
    print("=" * 60)
    print("✅ Pipeline complete. Outputs: chunk_stats.json, chunking_pipeline.log")
    print("\nNext steps:")
    print("  git add document_chunking_pipeline.py chunk_stats.json chunking_pipeline.log")
    print('  git commit -m "feat: add document chunking pipeline"')
    print("  git push origin main")


if __name__ == "__main__":
    main()
