"""
Consolidated backend module containing configuration, logging, database ORM models,
and the FastAPI application routes.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Literal, Any
from datetime import datetime
from uuid import uuid4
from contextlib import asynccontextmanager

from pydantic_settings import BaseSettings
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (Column, String, Text, Float, Integer, 
    DateTime, JSON, ForeignKey, Boolean, create_engine, Index)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ==========================================
# 1. Configuration (formerly config.py)
# ==========================================

class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables or .env file."""

    # --- Project Paths ---
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATASETS_DIR: Path = PROJECT_ROOT / "data"
    EMBEDDING_STORE_DIR: Path = PROJECT_ROOT / "embeddings"
    VECTOR_DB_DIR: Path = PROJECT_ROOT / "vector_db" / "chroma_store"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    FIGURES_DIR: Path = PROJECT_ROOT / "figures"

    # --- Embedding Model ---
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"

    # --- Vector Database ---
    CHROMA_COLLECTION_NAME: str = "k8s_incidents"
    CHROMA_DISTANCE_METRIC: Literal["cosine", "l2", "ip"] = "cosine"
    TOP_K_RETRIEVAL: int = 5

    # --- LLM Configuration ---
    LLM_PROVIDER: Literal["llama3", "mistral", "gemma", "qwen", "phi", "smollm"] = "smollm"
    LLM_MODEL_NAME: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.1
    LLM_DEVICE: str = "cpu" 
    LLM_LOAD_IN_8BIT: bool = False 
    LLM_USE_LLAMA_CPP: bool = False 

    # --- RAG Config ---
    RAG_ENABLED: bool = True
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.5

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
    API_CORS_ORIGINS: list[str] = ["http://localhost:8501"]

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

# Create singleton settings instance
settings = Settings()

# Ensure critical directories exist
for dir_attr in ["DATASETS_DIR", "EMBEDDING_STORE_DIR", "VECTOR_DB_DIR", "MODELS_DIR", "RESULTS_DIR", "FIGURES_DIR"]:
    path = getattr(settings, dir_attr)
    path.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Logging Setup (formerly logging_config.py)
# ==========================================

def setup_logging() -> None:
    """Configure loguru logger with console and file sinks."""
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler
    log_dir = settings.PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "kubesage_{time:YYYY-MM-DD}.log",
        format=settings.LOG_FORMAT,
        level="DEBUG",
        rotation="5 MB",
        retention="40 days",
        compression="gz",
        backtrace=True,
        diagnose=False,
    )

    # Error-only file
    logger.add(
        log_dir / "kubesage_errors_{time:YYYY-MM-DD}.log",
        format=settings.LOG_FORMAT,
        level="ERROR",
        rotation="5 MB",
        retention="90 days",
        backtrace=True,
        diagnose=True,
    )

    logger.info("=" * 40)
    logger.info("KubeSage logging initialized")
    logger.info(f"Log level: {settings.LOG_LEVEL}")
    logger.info(f"Project root: {settings.PROJECT_ROOT}")
    logger.info("=" * 50)

_logger_initialized = False

def get_logger(name: str):
    """Get a logger instance bound to the given module name."""
    global _logger_initialized
    if not _logger_initialized:
        setup_logging()
        _logger_initialized = True
    return logger.bind(module=name)

# Initialize logger for main module
logger_instance = get_logger(__name__)


# ==========================================
# 3. Database ORM Models (formerly db_models.py)
# ==========================================

Base = declarative_base()

class Incident(Base):
    """Stores raw Kubernetes incident data."""
    __tablename__ = "incidents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    root_cause = Column(String(500), nullable=True)
    resolution = Column(Text, nullable=True)
    evidence = Column(JSON, nullable=True)
    affected_services = Column(ARRAY(String), nullable=True)
    affected_pods = Column(ARRAY(String), nullable=True)
    incident_type = Column(String(100), nullable=False, index=True)
    source = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata_ = Column("metadata", JSON, nullable=True)

    reports = relationship("Report", back_populates="incident", cascade="all, delete-orphan")
    embedding = relationship("EmbeddingMetadata", back_populates="incident", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_incident_type_severity", "incident_type", "severity"),
        Index("idx_incident_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Incident {self.incident_id} [{self.severity}] {self.incident_type}>"


class EmbeddingMetadata(Base):
    """Stores embedding vector metadata."""
    __tablename__ = "embedding_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(UUID(as_uuid=True), ForeignKey("incidents.id"), unique=True, nullable=False)
    chroma_id = Column(String(100), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    dimension = Column(Integer, nullable=False)
    text_chunk = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    incident = relationship("Incident", back_populates="embedding")

    def __repr__(self) -> str:
        return f"<Embedding {self.model_name} → {self.incident_id}>"


class Report(Base):
    """Stores generated incident investigation reports."""
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    report_text = Column(Text, nullable=False)
    root_cause = Column(String(500), nullable=True)
    confidence_score = Column(Float, nullable=True)
    rag_enabled = Column(Boolean, default=True)
    llm_model = Column(String(100), nullable=False)
    prompt_template = Column(String(50), nullable=True)
    retrieved_incident_ids = Column(ARRAY(String), nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column("metadata", JSON, nullable=True)

    incident = relationship("Incident", back_populates="reports")

    __table_args__ = (
        Index("idx_report_incident_id", "incident_id"),
    )

    def __repr__(self) -> str:
        return f"<Report for {self.incident_id} [{self.llm_model}]>"


class ExperimentResult(Base):
    """Stores results from evaluation experiments."""
    __tablename__ = "experiment_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    experiment_name = Column(String(200), nullable=False, index=True)
    experiment_config = Column(JSON, nullable=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=True)
    top_k = Column(Integer, nullable=True)
    rag_enabled = Column(Boolean, nullable=True)
    run_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Experiment {self.experiment_name}: {self.metric_name}={self.metric_value}>"


def get_engine(database_url: str | None = None):
    """Create SQLAlchemy engine."""
    url = database_url or settings.postgres_url
    return create_engine(url, echo=False, pool_size=5, max_overflow=10)


def get_session(engine=None):
    """Create a new database session."""
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(engine=None):
    """Create all tables if they do not exist."""
    if engine is None:
        engine = get_engine()       
    Base.metadata.create_all(engine)


# ==========================================
# 4. FastAPI Setup and Routes (formerly main.py)
# ==========================================

class InvestigateRequest(BaseModel):
    """Request body for incident investigation."""
    incident_description: str = Field(
        ...,
        min_length=10,
        description="Description of the current Kubernetes incident",
        example="Pod payment-service-7d4f in namespace production is in CrashLoopBackOff. "
                "Logs show: 'Error: connection refused' to PostgreSQL at 10.0.2.15:5432.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of similar incidents to retrieve")
    llm_provider: str | None = Field(default=None, description="Can override default LLM provider")
    rag_enabled: bool = Field(default=True, description="Enable or disable RAG")


class InvestigateResponse(BaseModel):
    """Response body for incident investigation."""
    report: dict[str, Any]
    retrieved_incidents: list[dict[str, Any]]
    confidence_score: float
    rag_enabled: bool
    processing_time_ms: float


class SearchRequest(BaseModel):
    """Request body for semantic search."""
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger_instance.info("=" * 40)
    logger_instance.info("KubeSage API Starting...")
    logger_instance.info(f"Host: {settings.API_HOST}:{settings.API_PORT}, Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    logger_instance.info(f"   LLM Provider: {settings.LLM_PROVIDER}, RAG Enabled: {settings.RAG_ENABLED}")
    logger_instance.info("=" * 45)
    yield
    logger_instance.info("Application is shutting down")


app = FastAPI(
    title="KubeSage API",
    description="Kubernetes incident investigation system backed by RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "KubeSage API",
        "version": "1.0.0"
    }


@app.get("/api/v1/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Health check with configuration info."""
    return {
        "status": "healthy",
        "rag_enabled": settings.RAG_ENABLED,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_provider": settings.LLM_PROVIDER,
        "vector_db": settings.CHROMA_COLLECTION_NAME
    }


@app.get("/api/v1/incidents", tags=["Incidents"])
async def list_incidents(
    severity: str | None = None,
    incident_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List down all incidents."""
    return {
        "incidents": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
        "message": "DB not connected",
    }


@app.get("/api/v1/incidents/{incident_id}", tags=["Incidents"])
async def get_incident(incident_id: str) -> dict[str, Any]:
    """Get a single incident by ID."""
    return {"incident_id": incident_id, "message": "Not yet implemented"}


@app.post("/api/v1/investigate", response_model=InvestigateResponse, tags=["Investigation"])
async def investigate_incident(request: InvestigateRequest) -> InvestigateResponse:
    """Run the full RAG pipeline."""
    logger_instance.info(f"Investigate request received (rag={request.rag_enabled}, k={request.top_k})")
    return InvestigateResponse(
        report={
            "incident_id": "INC-PENDING",
            "severity": "Unknown",
            "root_cause": "Pending analysis",
            "summary": "RAG pipeline not initialized"
        },
        retrieved_incidents=[],
        confidence_score=0.0,
        rag_enabled=request.rag_enabled,
        processing_time_ms=0.0,
    )


@app.post("/api/v1/search", tags=["Search"])
async def semantic_search(request: SearchRequest) -> dict[str, Any]:
    """Perform semantic search over the incident vector database."""
    return {
        "query": request.query,
        "results": [],
        "message": "Vector database not initialized"
    }


@app.get("/api/v1/eval/metrics", tags=["Evaluation"])
async def get_evaluation_metrics(
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Get evaluation metrics for a specific experiment or all experiments."""
    return {"experiments": [], "message": "No experiments run yet"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
