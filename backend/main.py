"""
FastAPI application exposing endpoints for RAG pipeline queries,
semantic database search, Postgres incident logs, and evaluation metrics
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import settings
from backend.logging_config import get_logger


logger = get_logger(__name__)



# Pydantic Models

class InvestigateRequest(BaseModel):
    """Request body for incident investigation."""
    incident_description: str = Field(
        ...,
        min_length=10,
        description="Description of the current Kubernetes incident",
        example="Pod payment-service-7d4f in namespace production is in CrashLoopBackOff. "
                "Logs show: 'Error: connection refused' to PostgreSQL at 10.0.2.15:5432. "
                "The database pod is running but max_connections has been reached.",
    )

    top_k: int = Field(default=5, ge=1, le=20, description="number of similar incidents to retrieve")
    llm_provider: str | None = Field(default=None, description="can override default llm provider")
    rag_enabled: bool = Field(default=True, description="enable or disable RAG")


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


# Application Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("=" * 40)
    logger.info("KubeSage API Starting...")
    logger.info(f"Host: {settings.API_HOST}:{settings.API_PORT}, Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    logger.info(f"   LLM Provider: {settings.LLM_PROVIDER}, RAG Enabled: {settings.RAG_ENABLED}")
    logger.info("=" * 45)
    yield
    logger.info("Application is shutting down....Ha ha")



# FastAPI App


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



# Health checking


@app.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    """health check endpoint."""
    return {
        "status": "healthy","service": "KubeSage API",
        "version": "1.0.0",
    }


@app.get("/api/v1/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """health check with configuration info."""
    return {
        "status": "healthy",     "rag_enabled": settings.RAG_ENABLED,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_provider": settings.LLM_PROVIDER,     "vector_db": settings.CHROMA_COLLECTION_NAME,
    }



# Incident Endpoints


@app.get("/api/v1/incidents", tags=["Incidents"])
async def list_incidents(
    severity: str | None = None,
    incident_type: str | None = None,
    limit: int = 50,  offset: int = 0,
) -> dict[str, Any]:
    """list down all incidents """
    return {
        "incidents": [],   "total": 0,
        "limit": limit, "offset": offset,
        "message": "DB not connected.....",
    }


@app.get("/api/v1/incidents/{incident_id}", tags=["Incidents"])
async def get_incident(incident_id: str) -> dict[str, Any]:
    """get a single incident by ID."""
    return {"incident_id": incident_id, "message": "Not yet implemented"}



# Investigation (RAG Pipeline)


@app.post("/api/v1/investigate", response_model=InvestigateResponse, tags=["Investigation"])
async def investigate_incident(request: InvestigateRequest) -> InvestigateResponse:
    """Run the full RAG pipeline here"""
    logger.info(f"investigate request received (rag={request.rag_enabled}, k={request.top_k})")

    return InvestigateResponse(
        report={
            "incident_id": "INC-PENDING", "severity": "Unknown",
            "root_cause": "Pending analysis",    "summary": "RAG pipeline not initialized",
        },
        retrieved_incidents=[],        confidence_score=0.0,
        rag_enabled=request.rag_enabled, processing_time_ms=0.0,
    )



# Semantic Search

@app.post("/api/v1/search", tags=["Search"])
async def semantic_search(request: SearchRequest) -> dict[str, Any]:
    """Perform semantic search over the incident vector database."""
    return {
        "query": request.query,  "results": [],
        "message": "Vector database not initialized",
    }



# Evaluation Endpoints

@app.get("/api/v1/eval/metrics", tags=["Evaluation"])
async def get_evaluation_metrics(
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Get evaluation metrics for a specific experiment or all experiments."""
    return {
        "experiments": [],    "message": "No experiments run yet",
    }


# main method

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",   host=settings.API_HOST,
        port=settings.API_PORT,reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )

