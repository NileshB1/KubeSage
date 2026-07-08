"""
KubeSage Real Evaluation Script
===============================
Runs the full RAG pipeline with real LLM on sample incidents
and computes actual generation metrics (BLEU, ROUGE-L, Faithfulness).

This replaces the (Projected) mock-LLM estimates in the IEEE paper
with real computed values from the downloaded SmolLM2-1.7B-Instruct model.

Usage:
    python paper/run_real_eval.py --num-samples 5

Note: CPU inference is ~230s per sample with SmolLM2. 5 samples = ~20 minutes.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from backend.logging_config import get_logger
from evaluation.metrics import GenerationMetrics, HallucinationMetrics
from models.rag_pipeline import RAGPipeline, ReportParser

logger = get_logger(__name__)


def load_sample_incidents(data_path: Path, n: int = 3) -> list[dict[str, Any]]:
    """Load N diverse incidents from the dataset."""
    with open(data_path, "r", encoding="utf-8") as f:
        incidents = json.load(f)

    # Pick one from each of N different incident types for diversity
    seen_types: set[str] = set()
    samples: list[dict[str, Any]] = []
    for inc in incidents:
        itype = inc.get("incident_type", "Unknown")
        if itype not in seen_types:
            seen_types.add(itype)
            samples.append(inc)
        if len(samples) >= n:
            break

    return samples


def build_reference_text(incident: dict[str, Any]) -> str:
    """Build ground-truth reference text from an incident's metadata."""
    parts = [
        f"Type: {incident.get('incident_type', 'N/A')}",
        f"Root Cause: {incident.get('root_cause', 'N/A')}",
        f"Resolution: {incident.get('resolution', 'N/A')}",
        f"Severity: {incident.get('severity', 'N/A')}",
    ]
    return " ".join(parts)


def build_generated_text(report: dict[str, Any]) -> str:
    """Build a text representation of the generated report for metric comparison."""
    parts = [
        f"Root Cause: {report.get('root_cause', 'N/A')}",
        f"Severity: {report.get('severity', 'N/A')}",
        f"Summary: {report.get('generated_summary', 'N/A')}",
    ]
    # Flatten evidence
    for ev in report.get("evidence", []):
        if ev and ev != "..." and ev != "N/A":
            parts.append(f"Evidence: {ev}")
    for fix in report.get("recommended_fixes", []):
        if fix and fix != "..." and fix != "N/A":
            parts.append(f"Fix: {fix}")
    return " ".join(parts)


def compute_report_completeness(report: dict[str, Any]) -> float:
    """Compute what fraction of required fields have non-placeholder values."""
    required = ReportParser.REQUIRED_FIELDS
    filled = 0
    for field in required:
        val = report.get(field)
        if val is None or val == "N/A" or val == "..." or val == "" or val == "INC-XXXX":
            continue
        if isinstance(val, list) and (len(val) == 0 or all(v in ("...", "N/A", "") for v in val)):
            continue
        filled += 1
    return filled / len(required)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--incidents", type=str, default="data/preprocessed_incidents.json")
    args = parser.parse_args()

    data_path = _PROJECT_ROOT / args.incidents
    output_dir = _PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("KubeSage Real LLM Evaluation")
    print(f"Model: HuggingFaceTB/SmolLM2-1.7B-Instruct (CPU)")
    print(f"Samples: {args.num_samples}")
    print("=" * 60)

    # Load samples
    incidents = load_sample_incidents(data_path, n=args.num_samples)
    print(f"\nLoaded {len(incidents)} diverse incidents:")
    for inc in incidents:
        print(f"  {inc['incident_id']}: {inc['incident_type']} ({inc['severity']})")

    # Initialize pipeline
    print("\nInitializing RAG pipeline (real LLM mode)...")
    pipeline = RAGPipeline(llm_mode="local")
    print(f"Vector DB: {pipeline.vector_db.count()} docs")

    # Run evaluation
    candidates: list[str] = []
    references: list[list[str]] = []
    generated_texts: list[str] = []
    source_docs: list[list[str]] = []
    completeness_scores: list[float] = []
    all_results: list[dict[str, Any]] = []

    total_start = time.time()

    for i, incident in enumerate(incidents):
        desc = incident.get("description", incident.get("title", ""))
        print(f"\n{'='*40}")
        print(f"Sample {i+1}/{len(incidents)}: {incident['incident_id']} ({incident['incident_type']})")
        print(f"Description: {desc[:120]}...")

        # Run investigation
        gen_start = time.time()
        results = pipeline.investigate(desc, top_k=5, rag_enabled=True)
        gen_time = time.time() - gen_start

        report = results.get("report", {})

        # Compute completeness
        completeness = compute_report_completeness(report)
        completeness_scores.append(completeness)

        # Build texts for metrics
        ref = build_reference_text(incident)
        gen = build_generated_text(report)

        candidates.append(gen)
        references.append([ref])
        generated_texts.append(gen)

        # Build source documents from retrieved incidents
        retrieved = results.get("retrieved_incidents", [])
        sd = []
        for r in retrieved[:3]:
            meta = r.get("metadata", {})
            sd.append(
                f"Type: {meta.get('incident_type', '')} "
                f"Root Cause: {meta.get('root_cause', '')} "
                f"Resolution: {meta.get('resolution', '')}"
            )
        source_docs.append(sd if sd else [ref])

        all_results.append({
            "incident_id": incident["incident_id"],
            "incident_type": incident["incident_type"],
            "severity": incident["severity"],
            "gen_time_s": round(gen_time, 1),
            "completeness": round(completeness, 3),
            "report_id": report.get("incident_id", "N/A"),
            "report_severity": report.get("severity", "N/A"),
            "report_confidence": report.get("confidence_score", 0),
            "report_root_cause": str(report.get("root_cause", "N/A"))[:200],
            "retrieved_count": results.get("retrieval_count", 0),
        })

        print(f"  Generated in {gen_time:.1f}s | Completeness: {completeness:.1%}")
        print(f"  Report ID: {report.get('incident_id', 'N/A')} | "
              f"Severity: {report.get('severity', 'N/A')} | "
              f"Confidence: {report.get('confidence_score', 0)}")

    total_time = time.time() - total_start

    # Compute generation metrics
    print(f"\n{'='*60}")
    print("Computing generation metrics...")

    gen_metrics = GenerationMetrics.compute_all(candidates, references)

    # Compute faithfulness for each generated text
    faith_scores = []
    for gen_text, sdocs in zip(generated_texts, source_docs):
        faith = HallucinationMetrics.compute_faithfulness(gen_text, sdocs)
        faith_scores.append(faith)

    avg_faith = float(np.mean([f["faithfulness"] for f in faith_scores]))
    avg_completeness = float(np.mean(completeness_scores))
    hallucination_rate = 1.0 - avg_faith

    # Print results
    print(f"\n{'='*60}")
    print("REAL COMPUTED EVALUATION RESULTS")
    print(f"(SmolLM2-1.7B-Instruct, CPU, {args.num_samples} samples, {total_time:.0f}s total)")
    print(f"{'='*60}")
    print(f"\n  Generation Quality:")
    print(f"    BLEU:    {gen_metrics['bleu']:.4f}")
    print(f"    ROUGE-L: {gen_metrics['rouge_l']:.4f}")
    print(f"\n  Hallucination:")
    print(f"    Avg Faithfulness:   {avg_faith:.4f}")
    print(f"    Hallucination Rate: {hallucination_rate:.4f} ({hallucination_rate*100:.1f}%)")
    print(f"\n  Report Quality:")
    print(f"    Avg Completeness:   {avg_completeness:.4f} ({avg_completeness*100:.0f}%)")
    print(f"    Avg Gen Time:       {total_time/len(incidents):.0f}s per incident")

    # Save results
    output = {
        "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "device": "CPU",
        "num_samples": len(incidents),
        "total_time_s": round(total_time, 1),
        "generation_metrics": gen_metrics,
        "hallucination": {
            "avg_faithfulness": round(avg_faith, 4),
            "hallucination_rate": round(hallucination_rate, 4),
        },
        "report_quality": {
            "avg_completeness": round(avg_completeness, 4),
        },
        "per_sample": all_results,
    }

    outpath = output_dir / "real_eval_results.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Results saved: {outpath}")

    # Print paper-ready summary
    print(f"\n{'='*60}")
    print("PAPER-READY SUMMARY")
    print(f"{'='*60}")
    print(f"BLEU: {gen_metrics['bleu']:.3f} | ROUGE-L: {gen_metrics['rouge_l']:.3f}")
    print(f"Faithfulness: {avg_faith:.3f} | Completeness: {avg_completeness:.3f}")
    print(f"Hallucination Rate: {hallucination_rate:.3f} ({hallucination_rate*100:.0f}%)")


if __name__ == "__main__":
    main()
