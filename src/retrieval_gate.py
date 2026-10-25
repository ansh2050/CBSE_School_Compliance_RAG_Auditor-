"""
src/retrieval_gate.py
=====================
Corrective RAG (CRAG) Precondition Retrieval Gate.

Implements a two-phase verification pipeline for retrieved document chunks:

  Phase 1 — Deterministic precondition checks:
    • Supersession status from baseline manifest metadata
    • Keyword / lexical overlap scoring between query and chunk
    • Minimum content-length thresholds

  Phase 2 — LLM-assisted structured evaluation (optional):
    • Uses the configured LLM to score relevance and query-intent coverage
    • Falls back to the deterministic heuristic if the LLM is unavailable,
      rate-limited, or disabled via config

All inputs and outputs are validated through Pydantic v2 schemas so that
downstream consumers (episodic memory, telemetry, the Streamlit UI) receive
structured, type-safe data.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────────────────────

class ChunkEvaluation(BaseModel):
    """Per-chunk relevance evaluation produced by the retrieval gate."""

    chunk_id: str = Field(
        description="Unique identifier for the chunk (source:page:idx or hash)."
    )
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="Normalised relevance score: 0 = irrelevant, 1 = perfect match."
    )
    is_stale: bool = Field(
        default=False,
        description="True if the chunk's source document has been superseded."
    )
    covers_query_intent: bool = Field(
        default=True,
        description="True if the chunk substantively addresses the query intent."
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of the evaluation decision."
    )


class RetrievalGateResult(BaseModel):
    """Aggregate gate decision returned after evaluating all retrieved chunks."""

    action: Literal["proceed", "filter", "rewrite_query", "insufficient_evidence"] = Field(
        description=(
            "Gate action: 'proceed' = all chunks passed; 'filter' = some rejected; "
            "'rewrite_query' = query needs reformulation; "
            "'insufficient_evidence' = too few relevant chunks remain."
        )
    )
    passed_chunks: List[str] = Field(
        default_factory=list,
        description="Chunk IDs that passed the gate."
    )
    rejected_chunks: List[str] = Field(
        default_factory=list,
        description="Chunk IDs that were filtered out."
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Suggested rewritten query (populated when action='rewrite_query')."
    )
    evaluations: List[ChunkEvaluation] = Field(
        default_factory=list,
        description="Per-chunk evaluation details."
    )


class AuditFinding(BaseModel):
    """Structured compliance finding produced during an audit query."""

    finding_id: str = Field(
        description="Unique identifier for this finding."
    )
    category: str = Field(
        default="general",
        description="Finding category (e.g. 'fees', 'infrastructure', 'academic')."
    )
    severity: Literal["info", "warning", "violation", "critical"] = Field(
        default="info",
        description="Severity level of the finding."
    )
    title: str = Field(
        description="Short summary title of the finding."
    )
    description: str = Field(
        default="",
        description="Detailed description of the compliance finding."
    )
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk IDs from which this finding was derived."
    )
    regulation_reference: Optional[str] = Field(
        default=None,
        description="Reference to the specific CBSE regulation or circular."
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Recommended corrective action."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval Gate Implementation
# ──────────────────────────────────────────────────────────────────────────────

# Deterministic scoring thresholds
_MIN_RELEVANCE_SCORE = 0.25
_MIN_PASS_RATIO = 0.20  # at least 20% of chunks must pass to proceed


def _chunk_id(doc: Document) -> str:
    """Derive a stable chunk identifier from a document's metadata."""
    md = doc.metadata or {}
    source = md.get("source", "unknown")
    page = md.get("page", 0)
    chunk_idx = md.get("chunk_id", 0)
    return f"{source}:p{page}:c{chunk_idx}"


def _lexical_overlap(query: str, text: str) -> float:
    """
    Compute a fast lexical overlap score between the query and the chunk text.

    Returns a float between 0.0 and 1.0 representing the fraction of
    query tokens found in the chunk text (case-insensitive).
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return 0.0
    text_tokens = set(re.findall(r"\w+", text.lower()))
    overlap = query_tokens & text_tokens
    return len(overlap) / len(query_tokens)


def _is_superseded(doc: Document) -> bool:
    """
    Check whether the document's source is marked as superseded in metadata.

    The baseline manifest tags each source with `supersession_status`.
    Values other than 'current' indicate the source has been replaced.
    """
    md = doc.metadata or {}
    status = md.get("supersession_status", "current")
    return status.lower() not in ("current", "active", "")


class RetrievalGate:
    """
    Two-phase retrieval gate implementing the Corrective RAG pattern.

    Usage::

        gate = RetrievalGate()
        filtered_docs, gate_result = gate.verify_and_filter(query, docs)
    """

    def __init__(
        self,
        min_relevance: float = _MIN_RELEVANCE_SCORE,
        min_pass_ratio: float = _MIN_PASS_RATIO,
    ):
        self.min_relevance = min_relevance
        self.min_pass_ratio = min_pass_ratio

    # ── Phase 1: Deterministic precondition checks ─────────────────────────

    def _deterministic_evaluate(
        self, query: str, doc: Document
    ) -> ChunkEvaluation:
        """
        Run fast, deterministic checks on a single chunk.

        Checks performed:
          1. Supersession — is the source document still current?
          2. Content length — does the chunk have meaningful content?
          3. Lexical overlap — does the chunk share vocabulary with the query?
        """
        cid = _chunk_id(doc)
        text = doc.page_content or ""

        # Check 1: Supersession
        stale = _is_superseded(doc)

        # Check 2: Content length — extremely short chunks are noise
        if len(text.strip()) < 20:
            return ChunkEvaluation(
                chunk_id=cid,
                relevance_score=0.0,
                is_stale=stale,
                covers_query_intent=False,
                reasoning="Chunk content too short to be meaningful.",
            )

        # Check 3: Lexical overlap
        overlap = _lexical_overlap(query, text)

        # Combine into a single relevance score
        # Stale documents are penalised but not immediately discarded —
        # they may still contain useful (if outdated) context.
        score = overlap
        if stale:
            score *= 0.5

        covers_intent = score >= self.min_relevance

        reasoning_parts = []
        if stale:
            reasoning_parts.append("Source marked as superseded (score halved).")
        reasoning_parts.append(f"Lexical overlap = {overlap:.2f}.")
        if not covers_intent:
            reasoning_parts.append(
                f"Below min relevance threshold ({self.min_relevance})."
            )

        return ChunkEvaluation(
            chunk_id=cid,
            relevance_score=round(score, 4),
            is_stale=stale,
            covers_query_intent=covers_intent,
            reasoning=" ".join(reasoning_parts),
        )

    # ── Phase 2: LLM-assisted evaluation (optional) ───────────────────────

    def _llm_evaluate(
        self, query: str, doc: Document, llm: Any
    ) -> Optional[ChunkEvaluation]:
        """
        Use an LLM to perform structured relevance evaluation.

        Falls back to None (triggering the deterministic path) on any error.
        """
        cid = _chunk_id(doc)
        text = (doc.page_content or "")[:1500]  # truncate for prompt budget

        prompt = (
            "You are a retrieval quality evaluator. Given a user query and a "
            "retrieved text chunk, score the chunk's relevance to the query.\n\n"
            f"QUERY: {query}\n\n"
            f"CHUNK (id={cid}):\n{text}\n\n"
            "Respond with ONLY a JSON object (no markdown fencing) with these fields:\n"
            '  "relevance_score": float between 0.0 and 1.0,\n'
            '  "covers_query_intent": boolean,\n'
            '  "reasoning": brief explanation string\n'
        )

        try:
            from langchain_core.messages import HumanMessage

            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Strip markdown code fencing if present
            if response_text.startswith("```"):
                response_text = re.sub(
                    r"^```(?:json)?\s*", "", response_text
                )
                response_text = re.sub(r"\s*```$", "", response_text)

            import json
            data = json.loads(response_text)

            return ChunkEvaluation(
                chunk_id=cid,
                relevance_score=max(0.0, min(1.0, float(data.get("relevance_score", 0.5)))),
                is_stale=_is_superseded(doc),
                covers_query_intent=bool(data.get("covers_query_intent", True)),
                reasoning=str(data.get("reasoning", "")),
            )

        except Exception as exc:
            logger.warning(
                "LLM evaluation failed for chunk %s, falling back to "
                "deterministic scoring: %s",
                cid,
                exc,
            )
            return None

    # ── Public API ─────────────────────────────────────────────────────────

    def verify_and_filter(
        self,
        query: str,
        docs: List[Document],
        llm: Any = None,
    ) -> tuple[List[Document], RetrievalGateResult]:
        """
        Run the two-phase retrieval gate on the given documents.

        Parameters
        ----------
        query : str
            The user's search query.
        docs : list[Document]
            Retrieved LangChain documents to evaluate.
        llm : optional
            A LangChain-compatible LLM for Phase 2 evaluation.
            If None, only deterministic checks are run.

        Returns
        -------
        (filtered_docs, gate_result)
            filtered_docs  : Documents that passed the gate.
            gate_result    : Structured RetrievalGateResult with per-chunk
                             evaluations and the aggregate action.
        """
        if not docs:
            return [], RetrievalGateResult(
                action="insufficient_evidence",
                passed_chunks=[],
                rejected_chunks=[],
                evaluations=[],
            )

        evaluations: List[ChunkEvaluation] = []

        for doc in docs:
            evaluation = None

            # Phase 2: Try LLM evaluation first (if available)
            if llm is not None:
                evaluation = self._llm_evaluate(query, doc, llm)

            # Phase 1 fallback (or sole path if no LLM)
            if evaluation is None:
                evaluation = self._deterministic_evaluate(query, doc)

            evaluations.append(evaluation)

        # Partition into passed / rejected
        passed_ids: List[str] = []
        rejected_ids: List[str] = []
        filtered_docs: List[Document] = []

        for doc, ev in zip(docs, evaluations):
            if ev.covers_query_intent and ev.relevance_score >= self.min_relevance:
                passed_ids.append(ev.chunk_id)
                filtered_docs.append(doc)
            else:
                rejected_ids.append(ev.chunk_id)

        # Determine aggregate action
        total = len(docs)
        n_passed = len(passed_ids)

        if n_passed == total:
            action = "proceed"
        elif n_passed == 0:
            action = "insufficient_evidence"
        elif n_passed / total < self.min_pass_ratio:
            action = "rewrite_query"
        else:
            action = "filter"

        gate_result = RetrievalGateResult(
            action=action,
            passed_chunks=passed_ids,
            rejected_chunks=rejected_ids,
            rewritten_query=None,
            evaluations=evaluations,
        )

        logger.info(
            "RetrievalGate: %d/%d chunks passed (action=%s)",
            n_passed,
            total,
            action,
        )

        return filtered_docs, gate_result
