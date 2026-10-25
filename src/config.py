import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Force loading .env from the project root directory
_ROOT = Path(__file__).resolve().parent.parent
env_path = _ROOT / ".env"
load_dotenv(dotenv_path=env_path)

class Config:
    """
    Configuration class for the application.
    Centralizes all configuration parameters and supports environment variable overrides.
    """
    # API Keys
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Text chunking parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Model settings
    MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-27b-it")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Retrieval settings
    TOP_K = int(os.getenv("TOP_K", "4"))

    # Permanent CBSE baseline artifacts (built by scripts/build_baseline.py)
    BASELINE_DIR: Path = _ROOT / "baseline"
    BASELINE_MANIFEST: Path = _ROOT / "baseline" / "manifest.json"
    BASELINE_INDEX_NAME: str = "cbse_index"   # prefix used by FAISS .save_local()
    
    # LLM parameters
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Memory settings
    MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))

    # ── Corrective RAG Gate ───────────────────────────────────────────────
    ENABLE_CRAG_GATE: bool = os.getenv("ENABLE_CRAG_GATE", "true").lower() in ("1", "true", "yes")

    # ── Episodic Memory ──────────────────────────────────────────────────
    ENABLE_EPISODIC_MEMORY: bool = os.getenv("ENABLE_EPISODIC_MEMORY", "true").lower() in ("1", "true", "yes")
    EPISODIC_DB_PATH: Path = _ROOT / os.getenv("EPISODIC_DB_PATH", "episodic_memory.db")
    EPISODIC_TTL_SECONDS: int = int(os.getenv("EPISODIC_TTL_SECONDS", "3600"))
    
    @classmethod
    def get_llm_params(cls) -> Dict[str, Any]:
        """
        Get LLM parameters as a dictionary for easy configuration.
        
        Returns:
            Dict[str, Any]: Dictionary of LLM parameters
        """
        return {
            "temperature": cls.LLM_TEMPERATURE,
            "max_output_tokens": cls.MAX_TOKENS,
        }
    
    @classmethod
    def is_valid(cls) -> bool:
        """
        Check if the configuration is valid (has required API keys).
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return bool(cls.GOOGLE_API_KEY)