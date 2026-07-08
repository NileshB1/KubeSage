"""
KubeSage Real Retrieval Evaluation Script
==========================================
Computes actual retrieval metrics (Precision@K, Recall@K, MRR, NDCG)
from the existing dataset, embeddings, and ChromaDB index.

Uses leave-one-out methodology: for each incident, search the vector DB
and check if same-type incidents appear in top-K results.

Usage:
    python paper/run_retrieval_eval.py --k 5
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from evaluation.metrics import RetrievalMetrics
from backend.logging_config import get_logger

logger = get_logger(__name__)


def load_incidents(data_path: Path) -> list[dict[str, Any]]:
    """Load incident dataset."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(emb_path: Path) -> np.ndarray:
    """Load precomputed embeddings."""
    return np.load(emb_path)


def run_retrieval_evaluation(
    incidents: list[dict[str, Any]],
    embeddings: np.ndarray,
    vector_db,
    k_values: list[int] = [1, 3, 5, 10],
    sample_size: int | None = None,
) -> dict[str, Any]:
    """
    Run retrieval evaluation using leave-one-out methodology.

    For each incident, search the vector DB and measure if same-type
    incidents are retrieved in the top-K results.

    Args:
        incidents: List of incident dicts.
        embeddings: Precomputed embeddings array.
        vector_db: VectorDatabase instance.
        k_values: List of K values to evaluate.
        sample_size: Number of queries to evaluate (None = all).

    Returns:
        Dictionary of retrieval metrics.
    """
    queries = []
    total = len(incidents)
    eval_indices = list(range(total))

    if sample_size and sample_size < total:
        # Sample evenly across all incident types
        np.random.seed(42)
        eval_indices = sorted(np.random.choice(total, size=sample_size, replace=False))

    logger.info(f"Evaluating retrieval on {len(eval_indices)} queries...")
    start_time = time.time()

    # Group indices by incident type for relevance computation
    type_to_indices: dict[str, list[int]] = {}
    for i, inc in enumerate(incidents):
        itype = inc.get("incident_type", "Unknown")
        type_to_indices.setdefault(itype, []).append(i)

    type_to_ids: dict[str, set[str]] = {}
    for itype, idxs in type_to_indices.items():
        type_to_ids[itype] = {incidents[i]["incident_id"] for i in idxs}

    for idx in eval_indices:
        incident = incidents[idx]
        query_emb = embeddings[idx]
        incident_type = incident.get("incident_type", "Unknown")
        query_id = incident["incident_id"]

        # Search for K+1 to account for self-match filtering
        search_k = max(k_values) + 1
        results = vector_db.search(query_emb, top_k=search_k)

        # Filter out self-match (cosine similarity=1.0 with itself)
        all_results = results.get("results", [])
        filtered_results = [
            r for r in all_results
            if r.get("incident_id", "") != query_id
        ]
        # Keep only top max_k results after filtering
        retrieved_ids = [r.get("incident_id", "") for r in filtered_results[:max(k_values)]]

        # Relevant IDs: all incidents of the same type (excluding self)
        relevant_ids = type_to_ids.get(incident_type, set()) - {query_id}

        # Relevance scores for NDCG (1.0 for same type, 0.0 otherwise)
        relevance_scores = {}
        for rid in retrieved_ids:
            relevance_scores[rid] = 1.0 if rid in relevant_ids else 0.0

        queries.append({
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "relevance_scores": relevance_scores,
        })

    # Compute all metrics
    metrics = RetrievalMetrics.compute_all(queries, k_values=k_values)

    elapsed = time.time() - start_time
    logger.info(f"Evaluation complete in {elapsed:.1f}s")

    return {
        "num_queries": len(queries),
        "k_values": k_values,
        "metrics": metrics,
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    """Run retrieval evaluation and print results."""
    import argparse

    parser = argparse.ArgumentParser(description="Run real retrieval evaluation")
    parser.add_argument("--k", type=int, default=5, help="Primary K value")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (default: all)")
    parser.add_argument("--incidents", type=str, default="data/preprocessed_incidents.json")
    parser.add_argument("--embeddings", type=str, default="embeddings/incident_embeddings.npy")
    args = parser.parse_args()

    data_path = _PROJECT_ROOT / args.incidents
    emb_path = _PROJECT_ROOT / args.embeddings

    print("\n" + "=" * 60)
    print("KubeSage Real Retrieval Evaluation")
    print("=" * 60)

    # Load data
    print(f"\n[1] Loading incidents: {data_path}")
    incidents = load_incidents(data_path)
    print(f"    Loaded {len(incidents)} incidents")

    print(f"\n[2] Loading embeddings: {emb_path}")
    embeddings = load_embeddings(emb_path)
    print(f"    Shape: {embeddings.shape}")

    # Initialize ChromaDB
    from vector_db.build_index import VectorDatabase
    print("\n[3] Connecting to ChromaDB...")
    vdb = VectorDatabase()
    doc_count = vdb.count()
    print(f"    Indexed documents: {doc_count}")

    if doc_count == 0:
        print("\n[WARNING] Vector DB is empty. Building index...")
        vdb.build_index(embeddings, incidents)
        doc_count = vdb.count()
        print(f"    Built index with {doc_count} documents")

    # Run evaluation
    k_values = [1, 3, 5, 10]
    print(f"\n[4] Running retrieval evaluation (K = {k_values})...")
    results = run_retrieval_evaluation(
        incidents, embeddings, vdb,
        k_values=k_values,
        sample_size=args.sample,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nQueries evaluated: {results['num_queries']}")
    print(f"Time: {results['elapsed_seconds']:.1f}s")
    print()

    metrics = results["metrics"]
    for metric_name in ["precision", "recall", "ndcg"]:
        print(f"{metric_name.capitalize()}:")
        for k in k_values:
            key = f"@{k}"
            if key in metrics[metric_name]:
                print(f"  {metric_name.capitalize()}@{k}: {metrics[metric_name][key]:.4f}")

    if "mrr" in metrics and "mean" in metrics["mrr"]:
        print(f"\nMRR: {metrics['mrr']['mean']:.4f}")

    # Save results
    output_path = _PROJECT_ROOT / "results" / "retrieval_eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved: {output_path}")

    # Print summary for paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    for k in k_values:
        p = metrics["precision"].get(f"@{k}", 0)
        r = metrics["recall"].get(f"@{k}", 0)
        n = metrics["ndcg"].get(f"@{k}", 0)
        print(f"K={k:2d}  Precision={p:.3f}  Recall={r:.3f}  NDCG={n:.3f}")
    print(f"     MRR={metrics.get('mrr', {}).get('mean', 0):.3f}")


if __name__ == "__main__":
    main()
