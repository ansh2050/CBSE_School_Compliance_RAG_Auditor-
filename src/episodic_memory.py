"""
src/episodic_memory.py
======================
SQLite-backed Episodic Memory with Precondition-Verified Invalidation.

Stores past query→response trajectories (episodes) alongside verifiable
preconditions. On lookup, every stored precondition is re-verified against
the current runtime context before returning a cached episode. If any
precondition fails — e.g. the baseline manifest changed, an uploaded
document was replaced, or the TTL expired — the episode is logged as
``STALE_MEMORY_INVALIDATED`` and skipped.

Precondition Types
------------------
- ``baseline_version``  — SHA-256 hash of the baseline ``manifest.json``
- ``document_hash``     — SHA-256 hash of the uploaded document content
- ``ttl``               — Wall-clock expiry (epoch seconds)

Matching Strategy
-----------------
1. Normalised query hash (SHA-256 of lowercased, stripped query text).
2. Optional cosine similarity over ``all-MiniLM-L6-v2`` embeddings when
   sentence-transformers is available.

The SQLite database path is configurable via ``Config.EPISODIC_DB_PATH``
and the file is git-ignored (``*.db`` in ``.gitignore``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────────────────────

class EpisodePrecondition(BaseModel):
    """
    A single verifiable precondition attached to a cached episode.

    Call ``.verify(current_context)`` to check whether the precondition
    still holds.  The ``current_context`` dict should contain:

      - ``"baseline_manifest_hash"`` — SHA-256 of the current manifest.json
      - ``"document_hashes"``        — dict[filename, SHA-256]
      - ``"now"``                    — current epoch time (float)
    """

    condition_type: Literal["baseline_version", "document_hash", "ttl"] = Field(
        description="Type of precondition to verify."
    )
    condition_key: str = Field(
        description=(
            "Key for the check — e.g. manifest path, document filename, "
            "or 'expiry_epoch'."
        )
    )
    condition_value: str = Field(
        description=(
            "Expected value — SHA-256 hash for version/document checks, "
            "or epoch timestamp string for TTL."
        )
    )

    def verify(self, current_context: Dict[str, Any]) -> bool:
        """
        Return True if this precondition is still satisfied.

        Parameters
        ----------
        current_context : dict
            Runtime context containing current hashes and timestamps.
        """
        if self.condition_type == "baseline_version":
            current_hash = current_context.get("baseline_manifest_hash", "")
            ok = current_hash == self.condition_value
            if not ok:
                logger.info(
                    "Precondition FAILED: baseline_version mismatch "
                    "(expected=%s, current=%s)",
                    self.condition_value[:12],
                    current_hash[:12],
                )
            return ok

        elif self.condition_type == "document_hash":
            doc_hashes = current_context.get("document_hashes", {})
            current_hash = doc_hashes.get(self.condition_key, "")
            ok = current_hash == self.condition_value
            if not ok:
                logger.info(
                    "Precondition FAILED: document_hash mismatch for '%s'",
                    self.condition_key,
                )
            return ok

        elif self.condition_type == "ttl":
            now = float(current_context.get("now", time.time()))
            expiry = float(self.condition_value)
            ok = now < expiry
            if not ok:
                logger.info(
                    "Precondition FAILED: TTL expired (expiry=%s, now=%s)",
                    self.condition_value,
                    now,
                )
            return ok

        # Unknown type — fail safe
        logger.warning("Unknown precondition type: %s", self.condition_type)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_query_hash(query: str) -> str:
    """SHA-256 hash of lowercased, whitespace-normalised query text."""
    normalised = " ".join(query.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Optional: Embedding-based similarity for fuzzy matching
# ──────────────────────────────────────────────────────────────────────────────

_sentence_model = None


def _get_sentence_model():
    """Lazily load sentence-transformers model for cosine similarity lookup."""
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _sentence_model
    except Exception:
        return None


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ──────────────────────────────────────────────────────────────────────────────
# EpisodicMemoryStore
# ──────────────────────────────────────────────────────────────────────────────

class EpisodicMemoryStore:
    """
    SQLite-backed cache of past RAG trajectories with precondition verification.

    Each stored episode contains:
      - The original query (text + normalised hash)
      - Retrieved chunk IDs and the gate result
      - The generated response
      - A set of preconditions that must hold for the episode to be reusable

    On ``lookup()``, all preconditions are re-verified against the live
    runtime context.  If any fail, the episode is logged as stale and
    ``None`` is returned.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    default_ttl_seconds : int
        Default time-to-live for new episodes (default: 1 hour).
    similarity_threshold : float
        Cosine similarity threshold for fuzzy query matching (default: 0.92).
    """

    def __init__(
        self,
        db_path: str | Path = "episodic_memory.db",
        default_ttl_seconds: int = 3600,
        similarity_threshold: float = 0.92,
    ):
        self.db_path = str(db_path)
        self.default_ttl_seconds = default_ttl_seconds
        self.similarity_threshold = similarity_threshold

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    # ── Schema initialisation ──────────────────────────────────────────────

    def _create_tables(self) -> None:
        """Create the episodes and preconditions tables if they don't exist."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id    TEXT PRIMARY KEY,
                    query_text    TEXT NOT NULL,
                    query_hash    TEXT NOT NULL,
                    query_embedding TEXT,
                    chunk_ids     TEXT NOT NULL,
                    gate_result   TEXT NOT NULL,
                    response      TEXT NOT NULL,
                    created_at    REAL NOT NULL,
                    hit_count     INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_query_hash
                    ON episodes(query_hash);

                CREATE TABLE IF NOT EXISTS preconditions (
                    precondition_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id       TEXT NOT NULL,
                    condition_type   TEXT NOT NULL,
                    condition_key    TEXT NOT NULL,
                    condition_value  TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_preconditions_episode
                    ON preconditions(episode_id);
                """
            )

    # ── Lookup ─────────────────────────────────────────────────────────────

    def lookup(
        self,
        query: str,
        current_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Find a cached episode matching ``query`` and verify its preconditions.

        Parameters
        ----------
        query : str
            The user's current query.
        current_context : dict
            Runtime context for precondition verification.  Expected keys:
              - ``baseline_manifest_hash`` (str)
              - ``document_hashes`` (dict[str, str])
              - ``now`` (float, epoch seconds) — defaults to time.time()

        Returns
        -------
        dict or None
            The cached episode dict if found and all preconditions pass,
            otherwise ``None``.
        """
        if "now" not in current_context:
            current_context = {**current_context, "now": time.time()}

        query_hash = _normalise_query_hash(query)

        # ── Exact hash match ──────────────────────────────────────────────
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE query_hash = ?", (query_hash,)
        ).fetchall()

        # ── Fuzzy embedding match (if no exact match and model available) ─
        if not rows:
            model = _get_sentence_model()
            if model is not None:
                try:
                    query_emb = model.encode(query).tolist()
                    all_rows = self._conn.execute(
                        "SELECT * FROM episodes WHERE query_embedding IS NOT NULL"
                    ).fetchall()
                    for row in all_rows:
                        stored_emb = json.loads(row["query_embedding"])
                        sim = _cosine_similarity(query_emb, stored_emb)
                        if sim >= self.similarity_threshold:
                            rows = [row]
                            break
                except Exception as exc:
                    logger.debug("Embedding-based lookup failed: %s", exc)

        if not rows:
            return None

        # ── Verify preconditions for the best candidate ───────────────────
        for row in rows:
            episode_id = row["episode_id"]
            precondition_rows = self._conn.execute(
                "SELECT * FROM preconditions WHERE episode_id = ?",
                (episode_id,),
            ).fetchall()

            preconditions = [
                EpisodePrecondition(
                    condition_type=pr["condition_type"],
                    condition_key=pr["condition_key"],
                    condition_value=pr["condition_value"],
                )
                for pr in precondition_rows
            ]

            all_valid = all(pc.verify(current_context) for pc in preconditions)

            if not all_valid:
                logger.warning(
                    "STALE_MEMORY_INVALIDATED: episode %s failed "
                    "precondition verification — skipping.",
                    episode_id,
                )
                continue

            # Preconditions passed — return the episode
            self._conn.execute(
                "UPDATE episodes SET hit_count = hit_count + 1 WHERE episode_id = ?",
                (episode_id,),
            )
            self._conn.commit()

            return {
                "episode_id": episode_id,
                "query_text": row["query_text"],
                "chunk_ids": json.loads(row["chunk_ids"]),
                "gate_result": json.loads(row["gate_result"]),
                "response": row["response"],
                "created_at": row["created_at"],
                "hit_count": row["hit_count"] + 1,
                "source": "episodic_memory",
            }

        return None

    # ── Store ──────────────────────────────────────────────────────────────

    def store_episode(
        self,
        query: str,
        chunk_ids: List[str],
        gate_result: Dict[str, Any],
        response: str,
        preconditions: List[EpisodePrecondition],
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Persist a verified query→response trajectory with its preconditions.

        This should only be called AFTER the retrieval gate has passed.

        Parameters
        ----------
        query : str
            The original query text.
        chunk_ids : list[str]
            IDs of the retrieved chunks that were used.
        gate_result : dict
            Serialised ``RetrievalGateResult``.
        response : str
            The generated LLM response.
        preconditions : list[EpisodePrecondition]
            Preconditions that must hold for this episode to remain valid.
        ttl_seconds : int, optional
            Override for the default TTL.

        Returns
        -------
        str
            The generated episode ID.
        """
        query_hash = _normalise_query_hash(query)
        now = time.time()
        episode_id = hashlib.sha256(
            f"{query_hash}:{now}".encode()
        ).hexdigest()[:16]

        # Compute embedding for fuzzy matching
        query_embedding_json: Optional[str] = None
        model = _get_sentence_model()
        if model is not None:
            try:
                emb = model.encode(query).tolist()
                query_embedding_json = json.dumps(emb)
            except Exception:
                pass

        # Add a TTL precondition
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        ttl_precondition = EpisodePrecondition(
            condition_type="ttl",
            condition_key="expiry_epoch",
            condition_value=str(now + ttl),
        )
        all_preconditions = list(preconditions) + [ttl_precondition]

        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO episodes
                    (episode_id, query_text, query_hash, query_embedding,
                     chunk_ids, gate_result, response, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    query,
                    query_hash,
                    query_embedding_json,
                    json.dumps(chunk_ids),
                    json.dumps(gate_result),
                    response,
                    now,
                ),
            )

            for pc in all_preconditions:
                self._conn.execute(
                    """
                    INSERT INTO preconditions
                        (episode_id, condition_type, condition_key, condition_value)
                    VALUES (?, ?, ?, ?)
                    """,
                    (episode_id, pc.condition_type, pc.condition_key, pc.condition_value),
                )

        logger.info(
            "Stored episode %s (%d preconditions, TTL=%ds)",
            episode_id,
            len(all_preconditions),
            ttl,
        )
        return episode_id

    # ── Maintenance ────────────────────────────────────────────────────────

    def purge_expired(self) -> int:
        """
        Remove episodes whose TTL precondition has expired.
        Returns the number of purged episodes.
        """
        now = time.time()
        expired_ids = []

        rows = self._conn.execute(
            """
            SELECT DISTINCT p.episode_id
            FROM preconditions p
            WHERE p.condition_type = 'ttl'
              AND CAST(p.condition_value AS REAL) < ?
            """,
            (now,),
        ).fetchall()

        for row in rows:
            expired_ids.append(row[0])

        if expired_ids:
            placeholders = ",".join("?" * len(expired_ids))
            with self._conn:
                self._conn.execute(
                    f"DELETE FROM preconditions WHERE episode_id IN ({placeholders})",
                    expired_ids,
                )
                self._conn.execute(
                    f"DELETE FROM episodes WHERE episode_id IN ({placeholders})",
                    expired_ids,
                )

        if expired_ids:
            logger.info("Purged %d expired episodes.", len(expired_ids))
        return len(expired_ids)

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
