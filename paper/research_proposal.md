# Research Proposal: KubeSage

## A Retrieval-Augmented Generative AI Framework for Automated Kubernetes Incident Investigation and Explainable Incident Report Generation

**MSc Deep Learning & Generative AI — National College of Ireland**  
**Author:** Nilesh  
**Date:** July 2026

---

## 1. Introduction & Motivation

Modern cloud-native infrastructure, powered by Kubernetes, generates massive volumes of telemetry data — logs, metrics, alerts, and events — making incident investigation increasingly complex and time-consuming. DevOps and Site Reliability Engineering (SRE) teams spend significant time correlating signals across disparate sources to diagnose root causes and produce incident reports.

**KubeSage** addresses this challenge by combining **Deep Learning embeddings**, **Retrieval-Augmented Generation (RAG)**, and **Large Language Models (LLMs)** to automatically investigate Kubernetes incidents and generate explainable, structured incident reports. Unlike traditional keyword-based search or rule-based systems, KubeSage leverages semantic similarity through Sentence Transformers to retrieve historically similar incidents, augmenting the LLM's context to produce grounded, factual reports with reduced hallucination.

### Problem Statement
Can RAG combined with deep learning embeddings automatically generate accurate, explainable, and reliable Kubernetes incident investigation reports while reducing hallucinations and improving incident response?

---

## 2. Research Questions

| ID   | Question |
|------|----------|
| **MRQ** | Can Retrieval-Augmented Generation combined with Deep Learning embeddings automatically generate accurate, explainable, and reliable Kubernetes incident investigation reports while reducing hallucinations and improving incident response? |
| **RQ1** | Can Sentence Transformer embeddings retrieve semantically similar Kubernetes incidents better than keyword search (measured by Precision@K, Recall@K, MRR, NDCG)? |
| **RQ2** | Does Retrieval-Augmented Generation improve factual accuracy compared with a standalone LLM (measured by faithfulness, groundedness, and BERTScore)? |
| **RQ3** | Can Generative AI automatically produce incident reports comparable to those written by DevOps engineers (measured by BLEU, ROUGE, and human evaluation)? |
| **RQ4** | How much does RAG reduce hallucinations compared to standard LLM generation? |

---

## 3. Literature Review

### 3.1 Retrieval-Augmented Generation
RAG (Lewis et al., 2020) introduced a paradigm combining parametric memory (LLM) with non-parametric memory (retriever). This hybrid approach grounds LLM output in retrieved documents, significantly reducing factual errors. In the domain of incident management, RAG has been explored for IT operations (AIOps), but systematic evaluation for Kubernetes incident reports remains under-explored.

### 3.2 Sentence Transformers for Incident Retrieval
Reimers & Gurevych (2019) demonstrated that Sentence-BERT (SBERT) embeddings outperform traditional TF-IDF and BM25 for semantic textual similarity tasks. For incident logs, which exhibit high lexical variability but semantic similarity (e.g., "OOMKilled" vs "memory limit exceeded"), SBERT embeddings are particularly promising.

### 3.3 LLMs for Incident Analysis
Recent work (Ahmed et al., 2024) has shown that LLMs can analyze log data and assist in root cause analysis. However, standalone LLMs are prone to hallucination, especially in specialized domains like Kubernetes where training data may be sparse.

### 3.4 Research Gap
No existing work comprehensively evaluates the end-to-end pipeline of embedding-based retrieval → RAG → LLM for automated Kubernetes incident report generation, including rigorous hallucination metrics.

---

## 4. Research Methodology (CRISP-DM)

This project follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** methodology, adapted for deep learning and generative AI research.

### Phase 1: Business Understanding
- **Objective:** Automate Kubernetes incident investigation and report generation
- **Success Criteria:** Reports must be accurate (≥ 90% factual), explainable, and reduce hallucination vs standalone LLM
- **Stakeholders:** DevOps/SRE teams, platform engineers

### Phase 2: Data Understanding
- **Data Sources:** Public Kubernetes incident datasets; synthetic generation if unavailable
- **Data Types:** Logs, Prometheus metrics, Grafana alerts, pod events, deployment failures
- **Exploratory Analysis:** Distribution of incident types, severity levels, root cause categories

### Phase 3: Data Preparation
- Text preprocessing (log parsing, normalization, deduplication)
- Feature engineering (severity encoding, temporal features)
- Train/validation/test split (70/15/15)
- Incident-to-embedding mapping

### Phase 4: Modeling
- **Embedding Model:** Sentence Transformers (`all-MiniLM-L6-v2`, `bge-base-en-v1.5`, `e5-base`)
- **Vector Database:** ChromaDB with cosine similarity indexing
- **LLM:** Llama 3 / Mistral / Gemma with RAG prompt templates
- **Pipeline:** Embed → Index → Retrieve → Augment → Generate

### Phase 5: Evaluation
- **Embedding Quality:** Precision@K, Recall@K, MRR, NDCG
- **Root Cause Accuracy:** Precision, Recall, F1
- **Report Quality:** BLEU, ROUGE, BERTScore
- **Hallucination:** Faithfulness, Groundedness
- **Human Evaluation:** DevOps engineer blind comparison

### Phase 6: Deployment
- FastAPI backend with async endpoints
- Streamlit dashboard with dark mode
- Docker containerization
- PostgreSQL for persistent storage

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA INGESTION                          │
│  Prometheus │ Grafana │ K8s Events │ App Logs │ System Logs │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                     │
│  Log Parsing → Normalization → Dedup → Feature Engineering  │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SENTENCE TRANSFORMER EMBEDDINGS                 │
│  Model: all-MiniLM-L6-v2 / bge-base-en-v1.5 / e5-base      │
│  Output: 384/768-dim dense vector embeddings                │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  VECTOR DATABASE (ChromaDB)                  │
│  Metadata: ID, Root Cause, Resolution, Severity, Timestamp  │
│  Index: Cosine similarity, HNSW                             │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   SEMANTIC SEARCH                            │
│  Query Embedding → Top-K Similar Incidents (K=5)            │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROMPT CONSTRUCTION                        │
│  System: "You are an experienced Kubernetes SRE..."         │
│  User: Current Incident + Retrieved Incidents + KB           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                            │
│  Model: Llama 3 / Mistral / Gemma                           │
│  Output: Structured Incident Report                         │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 INCIDENT REPORT OUTPUT                       │
│  ID │ Severity │ Root Cause │ Evidence │ Timeline │ Fixes   │
│  Confidence │ Alternative Causes │ Retrieved Incidents       │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Technology Justification

| Technology | Justification |
|------------|---------------|
| **Sentence Transformers** | SOTA for semantic textual similarity; efficient inference; proven on domain-specific text |
| **ChromaDB** | Open-source; native embedding support; cosine similarity; HNSW indexing; Python-native |
| **LangChain** | Modular RAG pipeline; prompt templates; document loaders; evaluation framework |
| **Llama 3 / Mistral** | Open-source; strong reasoning; locally deployable; reproducible research |
| **FastAPI** | Async Python; automatic OpenAPI docs; production-grade; typed endpoints |
| **Streamlit** | Rapid ML dashboard development; native Plotly integration; dark mode support |
| **PostgreSQL** | ACID compliance; JSON support for metadata; well-established ecosystem |

---

## 7. Experimental Design

### Experiment 1: Embedding Model Comparison
- Models: `all-MiniLM-L6-v2`, `bge-base-en-v1.5`, `e5-base`
- Metrics: Precision@K, Recall@K, MRR@K, NDCG@K
- Goal: Identify optimal embedding model for K8s incident retrieval

### Experiment 2: Top-K Retrieval Optimization
- K values: 1, 3, 5, 10, 20
- Metrics: Relevance, diversity, RAG output quality
- Goal: Find optimal number of retrieved incidents for prompt augmentation

### Experiment 3: Prompt Engineering
- Strategies: Zero-shot, Few-shot (2-example), Few-shot (5-example), RAG
- Metrics: Report completeness, format adherence, factual accuracy
- Goal: Quantify the contribution of each prompt component

### Experiment 4: RAG vs LLM Only (Main Experiment)
- Condition A: LLM without retrieval
- Condition B: LLM with RAG (Top-5 retrieval)
- Metrics: BERTScore, Faithfulness, Groundedness, Human evaluation
- Goal: Demonstrate RAG superiority for factual accuracy

### Experiment 5: Hallucination Analysis
- Measure hallucination rate via faithfulness metric
- Categorize hallucination types (entity, temporal, causal)
- Compare RAG vs non-RAG hallucination rates

### Experiment 6: Report Quality Human Evaluation
- 5 DevOps engineers rate 20 reports each (10 RAG, 10 non-RAG)
- Likert scale: 1-5 on accuracy, clarity, completeness, actionability
- Inter-rater reliability: Fleiss' Kappa

---

## 8. Evaluation Metrics

### Retrieval Metrics
| Metric | Formula / Description |
|--------|----------------------|
| Precision@K | TP@K / K |
| Recall@K | TP@K / Total Relevant |
| MRR | Mean Reciprocal Rank of first relevant result |
| NDCG@K | Normalized Discounted Cumulative Gain |

### Generation Metrics
| Metric | Description |
|--------|-------------|
| BLEU | N-gram overlap with reference reports |
| ROUGE-L | Longest common subsequence |
| BERTScore | Semantic similarity using BERT embeddings |
| Faithfulness | % of claims supported by retrieved evidence |
| Groundedness | % of generated content traceable to input context |

---

## 9. Deliverables

1. Complete source code (well-documented, typed, tested)
2. Synthetic K8s incident dataset (500+ incidents)
3. Pre-trained embedding models & generated embeddings
4. ChromaDB vector index
5. RAG pipeline implementation
6. FastAPI backend with REST API
7. Streamlit dashboard (dark mode, analytics)
8. Evaluation scripts and experiment results
9. IEEE-format research paper with publication-quality figures
10. Presentation slides

---

## 10. Timeline (CRISP-DM Phased)

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Business Understanding | Week 1 | Research proposal, architecture design |
| Data Understanding | Week 2 | Dataset exploration, EDA |
| Data Preparation | Week 2-3 | Preprocessing pipeline, synthetic data |
| Modeling | Week 3-5 | Embeddings, vector DB, RAG, LLM |
| Evaluation | Week 5-6 | All 6 experiments, metrics |
| Deployment | Week 6-7 | FastAPI, Streamlit, Docker |
| Paper & Presentation | Week 7-8 | IEEE paper, slides |

---

## 11. Ethical Considerations

- All data is synthetic or publicly available — no PII or sensitive production data
- LLM outputs are clearly labeled as AI-generated
- Confidence scores are provided for all generated reports
- Human-in-the-loop validation recommended before production use

---

## 12. References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP*.
3. Ahmed, T., et al. (2024). Log-based Anomaly Detection with Large Language Models. *ICSE*.
4. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
5. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
6. Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR*.
7. Lin, C-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop*.
8. Papineni, K., et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. *ACL*.
