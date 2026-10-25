#!/usr/bin/env python3
"""
scripts/build_baseline.py
=========================
Offline script: reads all CBSE reference documents from the `documents/` folder,
builds a FAISS vector index, and saves it to `baseline/` along with a manifest.

Run once (or whenever source documents are updated):
    python scripts/build_baseline.py

DO NOT run this automatically inside the Streamlit app. This is a trusted,
controlled indexing process. The resulting baseline is treated as authoritative.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOCUMENTS_DIR: Path = _PROJECT_ROOT / "documents"
BASELINE_DIR: Path = Config.BASELINE_DIR
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Source registry — maps filename → metadata about that CBSE reference doc.
# Extend this whenever you add a new document to `documents/`.
SOURCE_REGISTRY: Dict[str, Dict[str, str]] = {
    "CBSE_Academic_Subjects_Structure.md": {
        "source_id": "cbse-acad-subjects-01",
        "source_title": "CBSE Academic Subjects Structure",
        "issuing_authority": "Central Board of Secondary Education",
        "source_url": "https://cbse.gov.in",
        "source_category": "Academic",
        "applicability": "All CBSE-affiliated schools",
        "publication_date": "2024-01-01",
        "supersession_status": "current",
    },
    "CBSE_Assessment_Attendance_Rules.md": {
        "source_id": "cbse-assessment-attendance-01",
        "source_title": "CBSE Assessment and Attendance Rules",
        "issuing_authority": "Central Board of Secondary Education",
        "source_url": "https://cbse.gov.in",
        "source_category": "Assessment & Attendance",
        "applicability": "All CBSE-affiliated schools",
        "publication_date": "2024-01-01",
        "supersession_status": "current",
    },
    "CBSE_Fees_Admissions_Operational_Compliance.md": {
        "source_id": "cbse-fees-admissions-01",
        "source_title": "CBSE Fees, Admissions & Operational Compliance",
        "issuing_authority": "Central Board of Secondary Education",
        "source_url": "https://cbse.gov.in",
        "source_category": "Fees & Admissions",
        "applicability": "All CBSE-affiliated schools",
        "publication_date": "2024-01-01",
        "supersession_status": "current",
    },
    "CBSE_Lab_Infrastructure_Mandates.md": {
        "source_id": "cbse-lab-infra-01",
        "source_title": "CBSE Lab and Infrastructure Mandates",
        "issuing_authority": "Central Board of Secondary Education",
        "source_url": "https://cbse.gov.in",
        "source_category": "Infrastructure",
        "applicability": "All CBSE-affiliated schools",
        "publication_date": "2024-01-01",
        "supersession_status": "current",
    },
    "User_Upload_Verification_Matrix.md": {
        "source_id": "cbse-verification-matrix-01",
        "source_title": "User Upload Verification Matrix",
        "issuing_authority": "Central Board of Secondary Education",
        "source_url": "https://cbse.gov.in",
        "source_category": "Verification",
        "applicability": "All CBSE-affiliated schools",
        "publication_date": "2024-01-01",
        "supersession_status": "current",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_sections(text: str, source_meta: Dict[str, str]) -> List[Document]:
    """
    Split a Markdown document into logical sections using heading lines
    (## or ###) as section boundaries.  Each section is then chunked by
    the text splitter so individual chunks know which section they came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    # Split on level-2 headings
    raw_sections: List[tuple[str, str]] = []  # (heading, body)
    lines = text.split("\n")
    current_heading = source_meta.get("source_title", "Introduction")
    current_body: List[str] = []

    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if current_body:
                raw_sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.lstrip("#").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        raw_sections.append((current_heading, "\n".join(current_body).strip()))

    docs: List[Document] = []
    for section_heading, body in raw_sections:
        if not body.strip():
            continue
        chunks = splitter.split_text(body)
        for i, chunk in enumerate(chunks):
            meta = {
                **source_meta,
                "section_or_clause": section_heading,
                "chunk_id": f"{source_meta['source_id']}-sec-{i}",
                "index_version": "1.0",
            }
            docs.append(Document(page_content=chunk, metadata=meta))

    return docs


def load_documents() -> List[Document]:
    """Read every .md file in the documents directory and return chunked Documents."""
    all_docs: List[Document] = []
    md_files = sorted(DOCUMENTS_DIR.glob("*.md"))

    if not md_files:
        print(f"[ERROR] No markdown files found in {DOCUMENTS_DIR}")
        sys.exit(1)

    for md_path in md_files:
        fname = md_path.name
        source_meta = SOURCE_REGISTRY.get(fname)
        if source_meta is None:
            print(f"[WARN] {fname} not in SOURCE_REGISTRY — skipping.")
            continue

        text = md_path.read_text(encoding="utf-8")
        docs = _parse_sections(text, {**source_meta, "local_file": str(md_path)})
        print(f"  ✓ {fname} → {len(docs)} chunks")
        all_docs.extend(docs)

    return all_docs


def build_index(docs: List[Document]) -> FAISS:
    """Embed all documents and build a FAISS index."""
    print(f"\n[INFO] Embedding {len(docs)} chunks with {Config.EMBEDDING_MODEL} …")
    embeddings = SentenceTransformerEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def save_manifest(md_files: List[Path]) -> Dict[str, Any]:
    """Write baseline/manifest.json and return its content."""
    source_inventory = []
    for md_path in sorted(DOCUMENTS_DIR.glob("*.md")):
        if md_path.name not in SOURCE_REGISTRY:
            continue
        meta = SOURCE_REGISTRY[md_path.name]
        source_inventory.append(
            {
                **meta,
                "local_file": md_path.name,
                "sha256": _sha256(md_path),
                "retrieval_date": datetime.now(timezone.utc).isoformat(),
            }
        )

    # Compute checksums of the saved FAISS artifacts
    index_faiss = BASELINE_DIR / f"{Config.BASELINE_INDEX_NAME}.faiss"
    index_pkl = BASELINE_DIR / f"{Config.BASELINE_INDEX_NAME}.pkl"

    manifest: Dict[str, Any] = {
        "index_version": "1.0",
        "embedding_model": Config.EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "index_name": Config.BASELINE_INDEX_NAME,
        "artifact_checksums": {
            f"{Config.BASELINE_INDEX_NAME}.faiss": _sha256(index_faiss) if index_faiss.exists() else "",
            f"{Config.BASELINE_INDEX_NAME}.pkl": _sha256(index_pkl) if index_pkl.exists() else "",
        },
        "source_inventory": source_inventory,
    }

    manifest_path = BASELINE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    print("=" * 60)
    print("  CBSE Compliance Auditor — Baseline Builder")
    print("=" * 60)
    print(f"\n[INFO] Source documents : {DOCUMENTS_DIR}")
    print(f"[INFO] Baseline output  : {BASELINE_DIR}")
    print(f"[INFO] Embedding model  : {Config.EMBEDDING_MODEL}\n")

    print("[STEP 1] Loading and chunking documents …")
    docs = load_documents()
    print(f"\n[INFO] Total chunks ready for indexing: {len(docs)}")

    print("\n[STEP 2] Building FAISS index …")
    vectorstore = build_index(docs)

    print("\n[STEP 3] Saving FAISS artifacts …")
    vectorstore.save_local(
        folder_path=str(BASELINE_DIR),
        index_name=Config.BASELINE_INDEX_NAME,
    )
    print(f"  ✓ Saved → {BASELINE_DIR}/{Config.BASELINE_INDEX_NAME}.faiss")
    print(f"  ✓ Saved → {BASELINE_DIR}/{Config.BASELINE_INDEX_NAME}.pkl")

    print("\n[STEP 4] Writing manifest …")
    manifest = save_manifest(list(DOCUMENTS_DIR.glob("*.md")))
    print(f"  ✓ manifest.json written (index_version={manifest['index_version']})")

    print("\n✅  Baseline build complete.\n")
    print(f"   Sources indexed : {len(manifest['source_inventory'])}")
    print(f"   Total chunks    : {len(docs)}")
    print(f"   Built at        : {manifest['built_at']}")
    print("\nYou can now start the Streamlit app. It will load this index on startup.")


if __name__ == "__main__":
    main()
