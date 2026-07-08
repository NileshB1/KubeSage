# KubeSage - RAG Powered Kubernetes Incident Investigation

This repository contains my MSc research project demonstrating an end-to-end Retrieval-Augmented Generation (RAG) system to automate Kubernetes incident post-mortem analysis.

## Project Scope
KubeSage uses historical incident context from a persistent vector store (ChromaDB) to ground a lightweight local Large Language Model (SmolLM2-1.7B-Instruct) in generating structured post-mortem reports.

## Staged Development Checklist
- **1:** Core Scaffolding & FastAPI API Skeleton: Done
- **2:** Synthetic Kubernetes Incidents Generator: Done
- **3:** Text Preprocessing & Cleaning pipeline: Done
- **4:** Sentence-Transformers Embedding Engine: Done
- **5:** Persistent Vector Store (ChromaDB Integration): Done
- **6:** Local LLM Inference Wrapper: Done
- **7:** RAG Pipeline Orchestrator: Done
- **8:** Interactive Streamlit Dashboard Frontend: Done
- **9:** Evaluation Metrics & Performance Benchmarks: Done

---

## 1: Core Scaffolding & FastAPI Backend REST API
This first module sets up the project base, Docker infrastructure, and exposes REST endpoints via FastAPI.

---

## 2: Synthetic Kubernetes Incidents Generator
This stage includes the synthetic generator engine that seeds the RAG pipeline with 500 realistic incident tickets containing telemetry data:
- `data/generate_synthetic.py`: Incident generator programmatically constructing descriptions, log events, metrics, and resolutions across 7 K8s failure profiles.
- `data/synthetic_incidents.json`: Raw generated dataset file.
- `data/preprocessed_incidents.json`: Structured preprocessed records.

### Execution
To generate the dataset:
```bash
python data/generate_synthetic.py --num-incidents 500 --output data/synthetic_incidents.json
```

---

## 3: Text Preprocessing & Cleaning Pipeline
This stage focuses on formatting and cleaning the unstructured Kubernetes log data and raw descriptions:
- `data/preprocess.py`: Contains the `IncidentPreprocessor` class which cleans logs (removes timestamps, anonymizes IPs and hex container IDs), splits long entries into chunks, and runs categorical label encoding for severity/type.
- `tests/test_preprocessor.py`: Unit tests verifying the log cleaning, normalization, and label encoding logic.

### Running Tests
To verify the preprocessor behavior:
```bash
pytest tests/test_preprocessor.py -v
```

---

## 4: Sentence-Transformers Embedding Engine
This stage implements the vector encoding engine using SBERT:
- `embeddings/generate_embeddings.py`: Wraps `sentence-transformers` models to generate dense numeric vectors from cleaned incident texts. Defaults to `all-MiniLM-L6-v2` (384 dimensions) but supports larger models.
- `embeddings/incident_embeddings.json` & `embeddings/incident_embeddings.npy`: Serialized embeddings files storing the vectors computed over the 500 incidents.

### Execution
To compute embeddings over the generated incidents:
```bash
python embeddings/generate_embeddings.py \
    --input data/synthetic_incidents.json \
    --output embeddings/incident_embeddings.npy
```

---

## 5: Persistent Vector Store (ChromaDB Integration)
This stage sets up the storage layer for our vector database:
- `vector_db/build_index.py`: Instantiates a persistent local client using ChromaDB. Handles custom schema indexing, collection management, similarity searches using cosine distance, and basic CRUD actions.
- `vector_db/chroma_store/`: Persistent SQLite and HNSW index storage directory.

### Execution
To build the vector database from your embeddings and synthetic incidents:
```bash
python -m vector_db.build_index \
    --embeddings embeddings/incident_embeddings.npy \
    --incidents data/synthetic_incidents.json \
    --model all-MiniLM-L6-v2
```

---

## 6: Local LLM Inference Wrapper
This stage implements the local LLM inference manager to execute lightweight, open-weight instruction models on CPU:
- `models/llm_inference.py`: Wraps HuggingFace transformers (`AutoModelForCausalLM` and `AutoTokenizer`) to load and query target models (e.g. `SmolLM2-1.7B-Instruct`). Features memory-optimized CPU float16 configuration fallback and custom JSON parser for structured post-mortem formatting.

---

## 7: RAG Pipeline Orchestrator
This stage bridges retrieval and generation components to form the end-to-end RAG orchestrator:
- `models/rag_pipeline.py`: Contains the `RAGPipeline` class coordinating vector search similarity retrieval, dynamic system/user prompt formatting with context interpolation, and JSON response post-processing.
- `tests/test_rag_pipeline.py` & `tests/test_vector_db.py`: Unit tests verifying prompt rendering, retrieval logic, and orchestrator execution.

### Running Tests
To run pipeline tests:
```bash
pytest tests/test_rag_pipeline.py tests/test_vector_db.py -v
```

---

## 8: Interactive Streamlit Dashboard Frontend
This stage introduces the graphical user interface for interacting with the backend and RAG systems:
- `frontend/app.py`: Streamlit-based web dashboard implementing page routing (Overview metrics, Incident Triage investigation portal, Semantic DB search logs, Saved Reports, and Evaluation statistics). Displays telemetry statistics using Plotly charts and custom dark theme settings.

### Execution
To run the Streamlit frontend locally:
```bash
streamlit run frontend/app.py
```

---

## 9: Evaluation Metrics & Performance Benchmarks
This final stage incorporates metrics calculations, validation suites, and experiments:
- `evaluation/metrics.py`: Computes Precision@K, Recall@K, Mean Reciprocal Rank (MRR), NDCG, and text similarity metrics (BLEU, ROUGE) to evaluate retrieval and generation quality.
- `evaluation/experiments.py`: Defines the experiment execution suite for running performance ablation runs (comparing embedding models, Top-K settings, and hallucination rates).
- `tests/test_evaluation.py`: Unit tests validating metrics calculations.
- `tests/test_capture_substring_regression.py`: Automated regression tests preventing layout label collisions.

### Running Evaluation Tests
To run all verification suites:
```bash
pytest tests/test_evaluation.py tests/test_preprocessor.py tests/test_rag_pipeline.py tests/test_vector_db.py -v
```
