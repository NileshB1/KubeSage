"""
KubeSage Presentation Generator
================================
Generates presentation slides (markdown) for the KubeSage MSc research project.
Can be converted to PowerPoint/Google Slides/Reveal.js.

Sections map to the IEEE paper structure.
"""

SLIDES = """---
marp: true
theme: default
paginate: true
size: 16:9
---

# KubeSage
## A Retrieval-Augmented Generative AI Framework for Automated Kubernetes Incident Investigation and Explainable Incident Report Generation

**Nilesh** | MSc Deep Learning & Generative AI | National College of Ireland | July 2026

---

# Problem & Research Questions

## The Problem
- Kubernetes generates terabytes of telemetry daily — logs, metrics, alerts, events
- SREs manually correlate 5+ tools; MTTR: **hours to days**
- Standalone LLMs hallucinate facts in specialized domains — up to 35% rate

## Research Questions

| ID | Question | Metric |
|----|----------|--------|
| **RQ1** | Can Sentence Transformers retrieve semantically similar K8s incidents? | Precision@K, Recall@K, MRR, NDCG |
| **RQ2** | Does RAG improve factual accuracy over standalone LLM? | Faithfulness |
| **RQ3** | Can AI generate incident reports comparable to DevOps engineers? | BLEU, ROUGE-L, Completeness |
| **RQ4** | How much does RAG reduce hallucinations? | Hallucination Rate |

---

# CRISP-DM Methodology & System Architecture

## CRISP-DM Phases

| Phase | Key Activity |
|-------|-------------|
| **Business Understanding** | Define: automate investigation, reduce MTTR & hallucinations |
| **Data Understanding** | 500 synthetic incidents, 7 types, realistic failure patterns |
| **Data Preparation** | Log parsing, normalization, SBERT embeddings (384-dim) |
| **Modeling** | RAG pipeline: Embed → Retrieve → Augment → Generate |
| **Evaluation** | 6 experiments across 4 research questions |
| **Deployment** | FastAPI + Streamlit + ChromaDB + Docker |

## 8-Stage Pipeline
```
Data → Preprocess → Embed (SBERT) → ChromaDB → Semantic Search → RAG Prompt → LLM → JSON Report
```

---

# Deep Learning + Generative AI Components

## Sentence Transformer Embeddings (DL)
- **Model:** `all-MiniLM-L6-v2` (384-dim)
- 500 incidents encoded in ~11s (CPU)
- Semantic similarity: "OOMKilled" ≈ "memory limit exceeded"
- **Result:** Precision@5 = **1.000**, MRR = **1.000**, NDCG@5 = **1.000**

## RAG Pipeline (GenAI)

| Stage | Action |
|-------|--------|
| **RETRIEVE** | Embed query → ChromaDB cosine search → Top-5 similar incidents |
| **AUGMENT** | System role + current incident + retrieved evidence + domain KB |
| **GENERATE** | SmolLM2-1.7B-Instruct → structured 14-field JSON report |

**Key constraint:** LLM MUST only use retrieved evidence — explicit fallback for insufficient evidence.

---

# Experimental Design & Key Results

## 6 Experiments

| # | Focus | Key Result |
|---|-------|-----------|
| E1 | Embedding retrieval (RQ1) | Precision@5 = **1.000**, MRR = **1.000** |
| E2 | Top-K optimization (RQ1) | **K=5** optimal: recall 0.071, projected completeness 94% (real: 100%) |
| E3 | Prompt strategies (RQ3) | RAG > Few-shot > Zero-shot for completeness |
| E4 | **RAG vs LLM Only (main)** | Faithfulness: **0.809** (RAG) — 1.7x better than 1.5B baseline |
| E5 | Hallucination analysis (RQ4) | Hallucination rate: **19.1%** (63% reduction vs baseline) |
| E6 | Human evaluation (RQ3) | Planned future work |

## Generation Quality (Real LLM, n=5, SmolLM2-1.7B, CPU)
| BLEU | ROUGE-L | Completeness | Faithfulness |
|:---:|:---:|:---:|:---:|
| 0.135 | 0.272 | **1.000** | **0.809** |

---

# Conclusion & Key Contributions

## Answers to Research Questions

| RQ | Answer |
|----|--------|
| **RQ1** | Sentence Transformers achieve Precision@5 = **1.000** and MRR = **1.000** |
| **RQ2** | RAG achieves **80.9%** faithfulness with SmolLM2-1.7B on CPU |
| **RQ3** | RAG produces **100%** structurally complete reports; semantic quality improves with model size |
| **RQ4** | Hallucination rate of **19.1%** — 63% reduction from 1.5B baseline |

## Key Contributions
1. Novel RAG architecture for K8s incident investigation
2. Comprehensive evaluation — 6 experiments, 4 research questions
3. Production-ready: FastAPI + Streamlit + Docker + ChromaDB
4. 500-incident synthetic dataset, 7 types, reproducible
5. Empirical evidence: RAG grounds LLM outputs in retrieved evidence

---

# Future Work & Thank You

## Future Directions
1. **Production datasets** — Validate on real-world K8s incident data
2. **Multi-modal embeddings** — Text + metrics + service topology graphs
3. **LLM fine-tuning** — Domain-specific training of Llama 3 / Mistral on incident reports
4. **Human evaluation** — Planned panel of 5 DevOps engineers rating RAG vs LLM-only reports

## KubeSage at a Glance
- 500-incident synthetic dataset | 384-dim SBERT embeddings
- ChromaDB vector database | RAG pipeline (80.9% faithful)
- FastAPI + Streamlit + Docker deployment
- CRISP-DM methodology | IEEE-format research paper

**Contact: Nilesh | NCI MSc Deep Learning & Generative AI | July 2026**
"""

# Write the slides to a markdown file
from pathlib import Path

output_path = Path(__file__).resolve().parent.parent / "paper" / "presentation_slides.md"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(SLIDES.strip())

print(f"[OK] Presentation slides written to: {output_path}")
print(f"     {len(SLIDES.split(chr(10)))} lines ({len(SLIDES)} characters)")
print(f"\nTip: Convert to presentation format with:")
print(f"  - Marp: npx @marp-team/marp-cli paper/presentation_slides.md -o paper/slides.pdf")
print(f"  - Pandoc: pandoc paper/presentation_slides.md -t beamer -o paper/slides.pdf")
print(f"  - Reveal.js: Pasting into slides.com or reveal-md")
