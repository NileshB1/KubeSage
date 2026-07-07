# KubeSage - RAG Powered Kubernetes Incident Investigation

This repository contains my MSc research project demonstrating an end-to-end Retrieval-Augmented Generation (RAG) system to automate Kubernetes incident post-mortem analysis.

## Project Scope
KubeSage uses historical incident context from a persistent vector store (ChromaDB) to ground a lightweight local Large Language Model (SmolLM2-1.7B-Instruct) in generating structured post-mortem reports.

## Staged Development Checklist
- **1:** Core Project Scaffolding & FastAPI Backend Rest Skeleton (Current)
- **2:** Synthetic Kubernetes Incidents Generator
- **3:** Text Preprocessing & Cleaning pipeline
- **4:** Sentence-Transformers Embedding Engine
- **5:** Persistent Vector Store (ChromaDB Integration)
- **6:** Local LLM Inference Wrapper
- **7:** RAG Pipeline Orchestrator
- **8:** Interactive Streamlit Dashboard Frontend
-**9:** Evaluation Metrics & Performance Benchmarks

---

## 1: Core Scaffolding & FastAPI Backend REST API
This first module sets up the project base, Docker infrastructure, and exposes REST endpoints via FastAPI:
- `backend/main.py`: Core routing and API skeleton.
- `backend/config.py`: Environment-controlled Pydantic settings.
- `backend/logging_config.py`: Structured logger configured with Loguru.
- `backend/db_models.py`: Database models mapping incidents and reports to PostgreSQL storage.

### Running Backend Locally (Docker)
Ensure Docker is installed, then run:
```bash
docker compose up --build
```
The backend service will be exposed at: `http://localhost:8000/docs`.
