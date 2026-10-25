"""
src/baseline.py
===============
BaselineLoader: validates and loads the permanent CBSE FAISS index from disk.

Key design decisions
--------------------
* Uses @st.cache_resource so the heavy FAISS deserialization happens ONCE per
  Streamlit server process.  Ordinary reruns (e.g. button clicks) reuse the
  cached object without touching disk.
* Validates manifest.json before loading — checks model name compatibility and
  the existence of required artifact files.
* If validation fails or the directory is missing, returns (None, None, error_msg)
  rather than raising — the UI reads the status and disables audit actions.
* allow_dangerous_deserialization=True is permitted ONLY here because:
    - The index is produced by our own trusted build_baseline.py script.
    - The baseline/ folder is NOT writable by user uploads.
    - SHA-256 checksums in manifest.json are verified before loading.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import Config


# ---------------------------------------------------------------------------
# Type alias for the loader return value
# ---------------------------------------------------------------------------
LoadResult = Tuple[Optional[FAISS], Optional[dict], str]
#              (vectorstore | None,  manifest | None,  status_message)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_manifest(manifest: dict, baseline_dir: Path) -> tuple[bool, str]:
    """
    Run pre-load validation checks on the manifest and artifact files.
    Returns (ok: bool, message: str).
    """
    # 1 — Expected artifact files present
    for artifact_name in manifest.get("artifact_checksums", {}):
        artifact_path = baseline_dir / artifact_name
        if not artifact_path.exists():
            return False, f"Artifact missing: {artifact_name}"

    # 2 — SHA-256 checksum verification
    for artifact_name, expected_hash in manifest.get("artifact_checksums", {}).items():
        if not expected_hash:
            continue  # hash not yet written (legacy manifest)
        actual_hash = _sha256(baseline_dir / artifact_name)
        if actual_hash != expected_hash:
            return False, (
                f"Checksum mismatch for {artifact_name}. "
                "Baseline may have been tampered with or is corrupt. "
                "Re-run scripts/build_baseline.py to rebuild."
            )

    # 3 — Embedding model compatibility
    manifest_model = manifest.get("embedding_model", "")
    if manifest_model and manifest_model != Config.EMBEDDING_MODEL:
        return False, (
            f"Embedding model mismatch: baseline was built with '{manifest_model}' "
            f"but current Config.EMBEDDING_MODEL='{Config.EMBEDDING_MODEL}'. "
            "Re-run scripts/build_baseline.py with the current model."
        )

    return True, "OK"


@st.cache_resource(show_spinner=False)
def load_baseline(cache_key: str = "default") -> LoadResult:
    """
    Load and validate the permanent CBSE FAISS baseline.

    The `cache_key` parameter is set to the manifest file's modification
    timestamp so that @st.cache_resource automatically re-runs this function
    whenever the baseline is rebuilt (instead of returning a stale failure).

    Returns
    -------
    (vectorstore, manifest, status)
        vectorstore : FAISS object if loaded successfully, else None
        manifest    : parsed manifest dict if available, else None
        status      : human-readable status string for the UI
    """
    baseline_dir: Path = Config.BASELINE_DIR
    manifest_path: Path = Config.BASELINE_MANIFEST
    index_name: str = Config.BASELINE_INDEX_NAME

    # ------------------------------------------------------------------ #
    # Guard 1: Baseline directory missing
    # ------------------------------------------------------------------ #
    if not baseline_dir.exists():
        return (
            None,
            None,
            (
                "❌ Baseline directory not found. "
                "Run `python scripts/build_baseline.py` to build the CBSE index."
            ),
        )

    # ------------------------------------------------------------------ #
    # Guard 2: Manifest missing
    # ------------------------------------------------------------------ #
    if not manifest_path.exists():
        return (
            None,
            None,
            (
                "❌ Baseline manifest (manifest.json) not found. "
                "Run `python scripts/build_baseline.py` to rebuild."
            ),
        )

    # ------------------------------------------------------------------ #
    # Guard 3: Parse manifest
    # ------------------------------------------------------------------ #
    try:
        manifest: dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return (None, None, f"❌ Could not parse manifest.json: {exc}")

    # ------------------------------------------------------------------ #
    # Guard 4: Validate artifacts and checksums
    # ------------------------------------------------------------------ #
    ok, validation_msg = _validate_manifest(manifest, baseline_dir)
    if not ok:
        return (None, manifest, f"⚠️ Baseline incompatible: {validation_msg}")

    # ------------------------------------------------------------------ #
    # Load FAISS — allow_dangerous_deserialization is intentional here:
    # the index was produced by our own trusted build script and its
    # checksums have been verified above.
    # ------------------------------------------------------------------ #
    try:
        embeddings = SentenceTransformerEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        vectorstore = FAISS.load_local(
            folder_path=str(baseline_dir),
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,  # safe: checksums verified above
        )
    except Exception as exc:
        return (
            None,
            manifest,
            f"❌ Failed to load FAISS index: {exc}",
        )

    built_at = manifest.get("built_at", "unknown")
    n_sources = len(manifest.get("source_inventory", []))
    return (
        vectorstore,
        manifest,
        f"🟢 Ready — {n_sources} source(s) indexed · built {built_at[:10]}",
    )


def get_baseline_retriever(vectorstore: FAISS, k: int = None):
    """Convenience wrapper: return a similarity retriever over the baseline."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or Config.TOP_K},
    )
