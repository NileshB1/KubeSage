# KubeSage — Work Distribution Document

**Module:** H9DLGA — Deep Learning & Generative AI  
**National College of Ireland — MSc Programme**  
**Project:** KubeSage — A Retrieval-Augmented Generative AI Framework for Automated Kubernetes Incident Investigation  
**Date:** July 2026  
**Submission Deadline:** 27/07/2026

---

## Team Member: Nilesh (Sole Author)

This project was developed as a single-member submission. All research, design, implementation, evaluation, and documentation was carried out by Nilesh. The work is distributed across the CRISP-DM methodology as follows:

---

## 1. Work Distribution by CRISP-DM Phase

| Phase | Activities | Approx. Effort |
|-------|-----------|:---:|
| **Business Understanding** | Defined research objectives, success criteria, stakeholder identification, research questions formulation | 10% |
| **Data Understanding** | Analyzed Kubernetes incident patterns, designed synthetic dataset generator (500 incidents, 7 types), EDA | 15% |
| **Data Preparation** | Built preprocessing pipeline (log parsing, normalization, deduplication, feature engineering), embeddings generation | 15% |
| **Modeling** | Implemented RAG pipeline (Sentence Transformers, ChromaDB, SmolLM2-1.7B LLM), prompt engineering, vector database | 25% |
| **Evaluation** | Designed 6 experiments, implemented evaluation metrics (BLEU, ROUGE-L, Faithfulness, Precision@K, MRR, NDCG), ran real LLM evaluation | 20% |
| **Deployment** | Built FastAPI backend, Streamlit dashboard, Docker containerization, PostgreSQL integration | 10% |
| **Documentation** | Authored IEEE-format paper (10-12 pages), research proposal, presentation slides, README, code documentation | 5% |

---

## 2. Work Distribution by Component

### 2.1 Research & Methodology
- Formulated central research question and 4 sub-questions (RQ1–RQ4)
- Conducted literature review across RAG, Sentence Transformers, LLMs, vector databases
- Selected CRISP-DM methodology and mapped all phases to the project lifecycle
- Justified all key design decisions with literature (embedding model, vector DB, LLM, K=5, prompt template)

### 2.2 Dataset Generation (`data/`)
- Designed synthetic Kubernetes incident generator with 7 incident types
- Implemented parameterized templates with realistic log patterns, root causes, resolutions
- Generated 500-incident dataset with configurable severity distributions
- Ensured reproducibility via fixed random seeds (42)

### 2.3 Preprocessing Pipeline (`data/preprocess.py`)
- Implemented log parsing with regex-based structured extraction
- Built text normalization (lowercasing, IP/timestamp replacement, whitespace standardization)
- Implemented semantic deduplication and feature encoding pipeline
- Created stratified train/validation/test splitting (70/15/15)

### 2.4 Embeddings (`embeddings/`)
- Integrated Sentence Transformer model (`all-MiniLM-L6-v2`, 384-dim)
- Generated embeddings for all 500 incidents
- Implemented L2 normalization for cosine similarity computation

### 2.5 Vector Database (`vector_db/`)
- Configured ChromaDB with persistent SQLite storage and HNSW indexing
- Built vector index with metadata-rich entries (14 fields per incident)
- Implemented semantic search with cosine similarity and metadata filtering
- Created inspection and management utilities

### 2.6 RAG Pipeline (`models/rag_pipeline.py`)
- Implemented full RAG pipeline: Embed → Retrieve → Augment → Generate
- Built PromptBuilder with system prompt, evidence formatting, domain knowledge base
- Developed ReportParser with progressive JSON repair for small-model output
- Implemented dual-mode operation: mock LLM (deterministic testing) and local LLM (SmolLM2-1.7B)
- Created comprehensive unit test suite (108 tests)

### 2.7 LLM Integration (`models/llm_inference.py`)
- Integrated HuggingFace transformers pipeline for SmolLM2-1.7B-Instruct
- Implemented ChatML format prompt construction with `apply_chat_template`
- Added CPU optimizations (float16, dynamic quantization support)
- Built JSON-constrained generation with prefix pre-filling

### 2.8 Backend API (`backend/`)
- Built FastAPI application with 7 REST endpoints
- Implemented Pydantic request/response validation
- Configured CORS middleware for Streamlit cross-origin requests
- Created configuration management with Pydantic Settings (`.env` support)
- Set up structured logging with Loguru

### 2.9 Frontend Dashboard (`frontend/`)
- Developed Streamlit dashboard with 5 interactive sections
- Implemented dark mode with custom CSS styling
- Built KPI cards, Plotly charts (bar, pie, heatmap), interactive filters
- Wired live data from evaluation result JSONs and ChromaDB
- Added JSON/PDF report export functionality

### 2.10 Evaluation Framework (`evaluation/`)
- Implemented RetrievalMetrics (Precision@K, Recall@K, MRR, NDCG@K)
- Implemented GenerationMetrics (BLEU, ROUGE-L)
- Implemented HallucinationMetrics (Faithfulness, Groundedness via SBERT)
- Built ExperimentRunner for systematic ablation studies
- Ran real LLM evaluation (n=5, SmolLM2-1.7B, CPU, float16)
- Ran retrieval evaluation (n=500 queries, leave-one-out)

### 2.11 Testing (`tests/`)
- 108 unit tests covering RAG pipeline, preprocessing, vector DB, evaluation
- Tests for prompt building, report parsing, JSON repair, validation
- Tests run on both CPU and CUDA configurations

### 2.12 Documentation (`paper/`, `README.md`)
- Authored IEEE-format research paper (~8,900 words, 10-12 pages)
- Wrote research proposal aligning with CRISP-DM methodology
- Created presentation slides (7 slides, 7-minute format)
- Authored comprehensive README with setup guide, architecture, troubleshooting
- Created publication-quality figures (system architecture, CRISP-DM workflow, RAG pipeline)

### 2.13 DevOps & Deployment
- Docker containerization (Dockerfile, docker-compose.yml, 3-service orchestration)
- CI/CD screenshots workflow (`.github/workflows/screenshots.yml`)
- Cross-platform setup scripts (`setup.sh`, `setup.ps1`)
- Screenshot capture automation with Playwright

---

## 3. Summary

| Area | Deliverables |
|------|-------------|
| **Research** | 4 research questions, CRISP-DM methodology, literature review (25 references) |
| **Data** | 500 synthetic incidents, 7 types, preprocessing pipeline, 384-dim embeddings |
| **Deep Learning** | Sentence Transformer (SBERT), ChromaDB HNSW vector search |
| **Generative AI** | RAG pipeline, SmolLM2-1.7B LLM, prompt engineering, JSON-constrained generation |
| **Evaluation** | 6 experiments, 7 metrics, real LLM inference evaluation |
| **Software** | FastAPI backend, Streamlit dashboard, Docker deployment, 108 unit tests |
| **Documentation** | IEEE paper (~8,900 words), research proposal, presentation, README |

---

*This document certifies that all work described above was performed by Nilesh for the MSc Deep Learning & Generative AI module (H9DLGA), National College of Ireland, submitted July 2026.*
