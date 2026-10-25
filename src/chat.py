"""
src/chat.py
===========
ChatManager — orchestrates LLM response generation with:

  1. Episodic memory lookup  (if enabled → returns cached response)
  2. CRAG retrieval gate     (if enabled → filters retrieved chunks)
  3. LLM generation          (ConversationalRetrievalChain or fallback)
  4. Episodic memory storage  (persists verified trajectories)

All new layers are wrapped in try/except to preserve the existing
graceful fallback chain so ``app.py`` works seamlessly whether or not
the gate and memory subsystems are available.
"""

from typing import List, Dict, Any, Optional
import hashlib
import json
import logging
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.config import Config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports for optional subsystems
# ──────────────────────────────────────────────────────────────────────────────

_retrieval_gate = None
_episodic_memory = None


def _get_retrieval_gate():
    """Lazily initialise the CRAG retrieval gate."""
    global _retrieval_gate
    if _retrieval_gate is not None:
        return _retrieval_gate
    if not Config.ENABLE_CRAG_GATE:
        return None
    try:
        from src.retrieval_gate import RetrievalGate
        _retrieval_gate = RetrievalGate()
        logger.info("CRAG RetrievalGate initialised.")
        return _retrieval_gate
    except Exception as exc:
        logger.warning("RetrievalGate unavailable: %s", exc)
        return None


def _get_episodic_memory():
    """Lazily initialise the SQLite episodic memory store."""
    global _episodic_memory
    if _episodic_memory is not None:
        return _episodic_memory
    if not Config.ENABLE_EPISODIC_MEMORY:
        return None
    try:
        from src.episodic_memory import EpisodicMemoryStore
        _episodic_memory = EpisodicMemoryStore(
            db_path=Config.EPISODIC_DB_PATH,
            default_ttl_seconds=Config.EPISODIC_TTL_SECONDS,
        )
        logger.info("EpisodicMemoryStore initialised at %s", Config.EPISODIC_DB_PATH)
        return _episodic_memory
    except Exception as exc:
        logger.warning("EpisodicMemoryStore unavailable: %s", exc)
        return None


def _build_runtime_context() -> Dict[str, Any]:
    """
    Build the current runtime context dict for episodic memory
    precondition verification.
    """
    ctx: Dict[str, Any] = {"now": time.time(), "document_hashes": {}}
    try:
        manifest_path = Config.BASELINE_MANIFEST
        if manifest_path.exists():
            h = hashlib.sha256()
            with open(manifest_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            ctx["baseline_manifest_hash"] = h.hexdigest()
    except Exception:
        pass
    return ctx


class ChatManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.memory = None
        self.chain = None
        self.llm = None
        self._last_gate_result = None  # telemetry: last gate result
        self._last_memory_hit = None   # telemetry: last memory lookup result
        self._initialize_components()
        
    def _initialize_components(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=Config.MODEL_NAME, 
                google_api_key=self.api_key,
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.95,
                top_k=40
            )
        except Exception as e:
            print(f"Error initializing LLM, retrying: {str(e)}")
            time.sleep(1)
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash", 
                google_api_key=self.api_key
            )
    
    def _create_chain(self, retriever):
        system_template = """You are a helpful assistant that answers questions based on the provided context.
        If you cannot find the answer in the context, acknowledge that and provide general information if possible.
        Always cite your sources when the information comes from the provided context.
        
        Context:
        {context}
        """
        
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_template + "\nQuestion: {question}"
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False
        )
        return self.chain
    
    def _fallback_generation(self, query: str, context_text: str):
        """Helper to generate response without the chain if chain fails."""
        try:
            messages = [
                SystemMessage(content=f"You are a helpful assistant... Context:\n{context_text}"),
                HumanMessage(content=query),
            ]
            # Retry logic for the fallback itself
            for attempt in range(2):
                try:
                    response = self.llm.invoke(messages)
                    return {"answer": response.content, "sources": []}
                except Exception:
                    time.sleep(2)
            return {"answer": "I am currently overloaded. Please try again in a few seconds.", "sources": []}
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": []}

    def generate_response(self, query: str, context_docs: List[Document]):
        # Rate limit protection (Gemma 3 allows 30 RPM, so 2s is safe)
        time.sleep(2) 

        # Reset telemetry for this request
        self._last_gate_result = None
        self._last_memory_hit = None

        # ── Step 0: Episodic Memory Lookup ────────────────────────────────
        memory_store = _get_episodic_memory()
        if memory_store is not None:
            try:
                ctx = _build_runtime_context()
                cached = memory_store.lookup(query, ctx)
                if cached is not None:
                    self._last_memory_hit = cached
                    logger.info(
                        "Episodic memory HIT for query (episode=%s, hits=%d)",
                        cached.get("episode_id", "?"),
                        cached.get("hit_count", 0),
                    )
                    return {
                        "answer": cached["response"],
                        "sources": [],
                        "_meta": {
                            "source": "episodic_memory",
                            "episode_id": cached.get("episode_id"),
                        },
                    }
            except Exception as exc:
                logger.warning("Episodic memory lookup failed: %s", exc)

        # ── Step 1: CRAG Retrieval Gate ───────────────────────────────────
        gate = _get_retrieval_gate()
        gate_result_dict = None
        passed_chunk_ids = []

        if gate is not None and context_docs:
            try:
                filtered_docs, gate_result = gate.verify_and_filter(
                    query, context_docs, llm=None  # deterministic only for speed
                )
                self._last_gate_result = gate_result
                gate_result_dict = gate_result.model_dump()
                passed_chunk_ids = gate_result.passed_chunks

                if gate_result.action == "insufficient_evidence":
                    logger.warning(
                        "CRAG gate: insufficient evidence — proceeding "
                        "with original docs as fallback."
                    )
                    # Fall through with original docs rather than returning empty
                elif filtered_docs:
                    context_docs = filtered_docs

            except Exception as exc:
                logger.warning("CRAG gate error, skipping: %s", exc)

        # ── Step 2: Generate Response ─────────────────────────────────────
        context_texts = [doc.page_content for doc in context_docs]
        context_text = "\n".join(context_texts)

        response = None

        # 2a. Try to use the RAG Chain first
        if self.chain:
            try:
                result = self.chain.invoke({"question": query})
                answer = result.get("answer") or result.get("output_text") or ""
                sources = []
                for d in result.get("source_documents", []):
                    md = getattr(d, 'metadata', {})
                    sources.append({
                        "text": d.page_content,
                        "metadata": md,
                    })
                response = {"answer": answer, "sources": sources}
            except Exception as e:
                print(f"Chain error: {str(e)}")
                # DO NOT RECURSE. Fallback explicitly.
                pass 
        
        # 2b. Fallback to direct generation if chain missing or failed
        if response is None:
            print("Falling back to direct generation...")
            response = self._fallback_generation(query, context_text)

        # ── Step 3: Episodic Memory Storage ───────────────────────────────
        if memory_store is not None and response:
            try:
                answer_text = response.get("answer", "") if isinstance(response, dict) else str(response)
                if answer_text and gate_result_dict:
                    from src.episodic_memory import EpisodePrecondition
                    preconditions = []

                    # Baseline version precondition
                    ctx = _build_runtime_context()
                    manifest_hash = ctx.get("baseline_manifest_hash")
                    if manifest_hash:
                        preconditions.append(EpisodePrecondition(
                            condition_type="baseline_version",
                            condition_key="manifest.json",
                            condition_value=manifest_hash,
                        ))

                    memory_store.store_episode(
                        query=query,
                        chunk_ids=passed_chunk_ids,
                        gate_result=gate_result_dict,
                        response=answer_text,
                        preconditions=preconditions,
                    )
            except Exception as exc:
                logger.warning("Episodic memory storage failed: %s", exc)

        # Attach gate/memory telemetry to response metadata
        if isinstance(response, dict):
            meta = response.get("_meta", {})
            if gate_result_dict:
                meta["crag_gate"] = {
                    "action": gate_result_dict.get("action"),
                    "passed": len(gate_result_dict.get("passed_chunks", [])),
                    "rejected": len(gate_result_dict.get("rejected_chunks", [])),
                }
            response["_meta"] = meta

        return response
                
    def set_retriever(self, retriever):
        self._create_chain(retriever)
        
    def reset_conversation(self):
        if self.memory:
            self.memory.clear()
            
    def get_conversation_history(self):
        if self.memory:
            return self.memory.chat_memory.messages
        return []