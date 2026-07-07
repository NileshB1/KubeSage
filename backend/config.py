"""
KubeSage Project Config
=======================
Centralized configuration manager built on Pydantic Settings.
Handles system directories, embedding models, LLM parameters, 
and Postgres DB connections. Can be customized via a local .env file.

Author: Nilesh (NCI MSc Project)
"""

import os
from pathlib import Path
from typing import Optional, Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    # --- Project Paths ---
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATASETS_DIR: Path = PROJECT_ROOT / "data"
    EMBEDDINGS_DIR: Path = PROJECT_ROOT / "embeddings"
    VECTOR_DB_DIR: Path = PROJECT_ROOT / "vector_db" / "chroma_store"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    FIGURES_DIR: Path = PROJECT_ROOT / "figures"

    # --- Embedding Model ---
    # Preferred: all-MiniLM-L6-v2 (fast, 384-dim)
    # Alternatives: BAAI/bge-base-en-v1.5, intfloat/e5-base
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # 384 for MiniLM, 768 for bge/e5
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"  # "cuda" if GPU available

    # --- Vector Database ---
    # ChromaDB with cosine similarity
    CHROMA_COLLECTION_NAME: str = "k8s_incidents"
    CHROMA_DISTANCE_METRIC: Literal["cosine", "l2", "ip"] = "cosine"
    TOP_K_RETRIEVAL: int = 5  # Default number of similar incidents to retrieve

    # --- LLM Configuration ---
    # CPU-optimized model for structured JSON generation
    # SmolLM2-1.7B-Instruct: 1.7B params, ChatML format, ~3.4GB in float16
    # Alternatives: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct
    LLM_PROVIDER: Literal["llama3", "mistral", "gemma", "qwen", "phi", "smollm"] = "smollm"
    LLM_MODEL_NAME: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.1  # Low temperature for factual consistency
    LLM_DEVICE: str = "cpu"  # "cuda" for GPU
    LLM_LOAD_IN_8BIT: bool = False  # Enable for larger models on CPU (slower)
    LLM_USE_LLAMA_CPP: bool = False  # llama-cpp-python unavailable on this system

    # --- RAG Configuration ---
    RAG_ENABLED: bool = True
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.5  # Minimum cosine similarity

    # --- Database (PostgreSQL) ---
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "kubesage"
    POSTGRES_USER: str = "kubesage"
    POSTGRES_PASSWORD: str = "kubesage"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_CORS_ORIGINS: list[str] = ["http://localhost:8501"]  # Streamlit default

    # --- Logging ---
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )

    # --- Evaluation ---
    EVAL_RANDOM_SEED: int = 42
    EVAL_TEST_SIZE: float = 0.15
    EVAL_VAL_SIZE: float = 0.15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton instance
settings = Settings()

# Ensure directories exist
for dir_attr in [
    "DATASETS_DIR", "EMBEDDINGS_DIR", "VECTOR_DB_DIR",
    "MODELS_DIR", "RESULTS_DIR", "FIGURES_DIR"
]:
    path = getattr(settings, dir_attr)
    path.mkdir(parents=True, exist_ok=True)
