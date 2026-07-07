# KubeSage - RAG Powered Kubernetes Incident Investigation

This repository contains my MSc research project demonstrating an end-to-end Retrieval-Augmented Generation (RAG) system to automate Kubernetes incident post-mortem analysis.

## Project Scope
KubeSage uses historical incident context from a persistent vector store (ChromaDB) to ground a lightweight local Large Language Model (SmolLM2-1.7B-Instruct) in generating structured post-mortem reports.

## Staged Development Checklist
- **1:** Core Scaffolding & FastAPI API Skeleton (Completed)
- **2:** Synthetic Kubernetes Incidents Generator (Completed)
- **3:** Text Preprocessing & Cleaning pipeline (Next)
- **4:** Sentence-Transformers Embedding Engine
- **5:** Persistent Vector Store (ChromaDB Integration)
- **6:** Local LLM Inference Wrapper
- **7:** RAG Pipeline Orchestrator
- **8:** Interactive Streamlit Dashboard Frontend
- **9:** Evaluation Metrics & Performance Benchmarks

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
