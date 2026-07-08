"""
Calculates embedding similarity scoring 
"""

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# TODO
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from backend.logging_config import get_logger

logger = get_logger(__name__)


#embedding or retrieval metrics

class RetrievalMetrics:
    """
    Evaluation metrics for embedding-based incident retrieval
    """

    @staticmethod
    def precision_at_k(retrieved_ids: list[str],  relevant_ids: set[str], k: int,):
        """
        Fraction of top-K retrieved items that are relevant

        """
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        relevant_hits = sum(1 for rid in top_k if rid in relevant_ids)
        return relevant_hits / len(top_k)

    @staticmethod
    def recall_at_k(retrieved_ids: list[str],relevant_ids: set[str],k: int):
        """
        Recall@K: Fraction of all relevant items retrieved in top-K.

        """
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_hits = sum(1 for rid in top_k if rid in relevant_ids)
        return relevant_hits / len(relevant_ids)

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]):
        """
        MRR: Mean Reciprocal Rank of the first relevant item
        """
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str],  relevance_scores: dict[str, float], k: int):
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        """
        # DCG
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:k], 1):
            rel = relevance_scores.get(rid, 0.0)
            dcg += rel / math.log2(i + 1)

        # IDCG (ideal ranking)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, 1):
            idcg += rel / math.log2(i + 1)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def compute_all(
        queries: list[dict[str, Any]], k_values: list[int] = [1, 3, 5, 10]):
        """
        Compute all retrieval metrics across multiple K values
        """
        results: dict[str, dict[str, float]] = { "precision": {},  "recall": {},"mrr": {}, "ndcg": {}}

        mrr_scores = []

        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []

            for query in queries:
                retrieved = query.get("retrieved_ids", [])
                relevant = set(query.get("relevant_ids", []))
                relevance_scores = query.get("relevance_scores", {})

                precision_scores.append(RetrievalMetrics.precision_at_k(retrieved, relevant, k))


                recall_scores.append(RetrievalMetrics.recall_at_k(retrieved, relevant, k)
                )
                ndcg_scores.append(RetrievalMetrics.ndcg_at_k(retrieved,relevance_scores, k))

                # MRR only computed once
                if k == k_values[0]:
                    mrr_scores.append(RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant))

            results["precision"][f"@{k}"] = round(float(np.mean(precision_scores)), 4)
            results["recall"][f"@{k}"] = round(float(np.mean(recall_scores)), 4)


            results["ndcg"][f"@{k}"] = round(float(np.mean(ndcg_scores)), 4)

        results["mrr"] = {"mean": round(float(np.mean(mrr_scores)), 4)}

        return results


#root cause classification metrics

class ClassificationMetrics:
    """ Metrics for evaluating root cause prediction accuracy"""

    @staticmethod
    def compute_all(
        y_true: list[str], y_pred: list[str], labels: list[str] | None = None):
        """
        Compute classification metrics
        """
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),

            "precision_macro":round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),

            "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),

            "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),


            "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        }



# Generation Quality Metrics (RQ3)


class GenerationMetrics:
    """
    Metrics for evaluating generated report quality
    """

    @staticmethod
    def compute_bleu(candidate: str, references: list[str], max_n: int = 4):
        """
        Compute BLEU score
        Returns:
            BLEU score.
        """
        def get_ngrams(text: str, n: int) -> Counter:
            tokens = text.lower().split()
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

        candidate_tokens = candidate.lower().split()
        if len(candidate_tokens) < max_n:
            return 0.0

        #compute modified n-gram precision
        precisions = []
        for n in range(1, max_n + 1):
            candidate_ngrams = get_ngrams(candidate, n)

            #max counts across all references
            max_ref_counts: Counter = Counter()
            for ref in references:
                ref_ngrams = get_ngrams(ref, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

            #clipped counts
            clipped_count = sum(
                min(count, max_ref_counts.get(ngram, 0))
                for ngram, count in candidate_ngrams.items()
            )
            total_count = sum(candidate_ngrams.values())

            if total_count == 0:
                precisions.append(0.0)
            else:
                precisions.append(clipped_count / total_count)

        #brevity penalty
        ref_lengths = [len(ref.split()) for ref in references]
        closest_ref_len=min(ref_lengths, key=lambda x: abs(x - len(candidate_tokens)))
        brevity_penalty=min(1.0, math.exp(1 - closest_ref_len / max(1, len(candidate_tokens))))

        #geometric mean of precisions
        if any(p == 0 for p in precisions):
            geo_mean = 0.0
        else:
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

        return brevity_penalty * geo_mean

    @staticmethod
    def compute_rouge_l(candidate: str, reference: str) -> float:
        """
        Compute ROUGE-L (Longest Common Subsequence) F1 score
        Returns:
            ROUGE-L F1 score
        """
        cand_tokens = candidate.lower().split()

        ref_tokens = reference.lower().split()

        m, n = len(cand_tokens), len(ref_tokens)
        if m==0 or n==0:
            return 0.0

        #LCS using dynamic programming
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if cand_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]

        recall = lcs_length / n if n > 0 else 0.0
        precision = lcs_length / m if m > 0 else 0.0

        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    @staticmethod
    def compute_all(
        candidates: list[str],
        references: list[list[str]],
    ) -> dict[str, float]:
        """
        Compute all generation quality metrics
        """
        bleu_scores = []
        rouge_scores = []

        for i, candidate in enumerate(candidates):
            refs = references[i] if i < len(references) else []
            if not refs:
                continue

            bleu_scores.append(
                GenerationMetrics.compute_bleu(candidate, refs)
            )

            rouge_scores.append(
                GenerationMetrics.compute_rouge_l(candidate, refs[0])
            )

        return {
            "bleu": round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0,
            "rouge_l": round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0.0,
        }


# Hallucination Metrics 

class HallucinationMetrics:
    """
    Metrics for detecting and quantifying hallucinations in generated reports
        Numerical hallucination: Fabricated metrics or numbers
    """

    @staticmethod
    def compute_faithfulness(
        generated_text: str,  source_documents: list[str], use_sentence_transformer: bool = True,
    ):
        """
        Estimate faithfulness of generated text to source documents
        """
        # Split generated text into sentences/claims
        sentences = [s.strip() for s in generated_text.split('.') if s.strip()]

        if not sentences:
            return {"faithfulness": 0.0, "total_claims": 0, "supported_claims": 0}

        supported = 0
        total = len(sentences)

        if use_sentence_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity

                model = SentenceTransformer("all-MiniLM-L6-v2")
                source_embs = model.encode(source_documents, normalize_embeddings=True)

                for sentence in sentences:
                    sent_emb = model.encode([sentence], normalize_embeddings=True)
                    similarities = cosine_similarity(sent_emb, source_embs)[0]
                    if similarities.max() > 0.5:  #threshold as 0.5?? TODO Need to change?
                        supported += 1
            except ImportError:
                #fallback to keyword matching
                use_sentence_transformer = False

        if not use_sentence_transformer:
            for sentence in sentences:
                sent_lower = sentence.lower()
                if any(
                    any(word in sent_lower for word in doc.lower().split())
                    for doc in source_documents
                ):
                    supported += 1

        faithfulness = supported/total if total > 0 else 0.0

        return {
            "faithfulness": round(float(faithfulness), 4),"total_claims": total, "supported_claims": supported,
            "unsupported_claims": total - supported}

    @staticmethod
    def compute_groundedness(
        generated_text: str, context_texts: list[str]):
        """
        Groundedness: how well the generated text is grounded in
        the provided context
        """
        return HallucinationMetrics.compute_faithfulness(
            generated_text, context_texts, use_sentence_transformer=True,
        )



#comprehensive evaluation runner


class Evaluator:
    """
    Runs all evaluation metrics and produces a comprehensive report
    """

    @staticmethod
    def evaluate_all( retrieval_data: list[dict[str, Any]] | None = None,
        y_true: list[str] | None = None,   y_pred: list[str] | None = None,
        candidates: list[str] | None = None,   references: list[list[str]] | None = None,
        generated_texts: list[str] | None = None, source_documents: list[list[str]] | None = None):
        """
        run all evaluations and return comprehensive results

        """
        results: dict[str, Any] = {}

        if retrieval_data:
            results["retrieval"] = RetrievalMetrics.compute_all(retrieval_data)

        if y_true and y_pred:
            results["classification"] = ClassificationMetrics.compute_all(y_true, y_pred)

        if candidates and references:
            results["generation"] = GenerationMetrics.compute_all(candidates, references)

        if generated_texts and source_documents:
            faithfulness_scores = []
            for gen, sources in zip(generated_texts, source_documents):
                scores = HallucinationMetrics.compute_faithfulness(gen, sources)
                faithfulness_scores.append(scores["faithfulness"])
            results["hallucination"]={"avg_faithfulness": round(float(np.mean(faithfulness_scores)), 4),
                "std_faithfulness": round(float(np.std(faithfulness_scores)), 4) }

        return results



#CLI

def main() -> None:
    """Demo evaluation metrics."""
    print("\n" + "=" * 40)
    print("KubeSage evaluation metrics Demo")
    print("=" * 45)

    #Demo retrieval
    print("\n[1] Retrieval Metrics Demo")
    queries = [ {"retrieved_ids": ["INC-1", "INC-2", "INC-3"], "relevant_ids": {"INC-1", "INC-3"}},
        {"retrieved_ids": ["INC-4", "INC-5", "INC-6"], "relevant_ids": {"INC-6"}} ]
    retrieval = RetrievalMetrics.compute_all(queries)
    print(f"Precision@3: {retrieval['precision'].get('@3', 'N/A')}")
    print(f"Recall@3: {retrieval['recall'].get('@3', 'N/A')}")
    print(f"MRR: {retrieval['mrr'].get('mean', 'N/A')}")

    #demo classification
    print("\n[2] Classification Metrics Demo")
    y_true = ["OOMKilled", "CrashLoopBackOff", "NetworkFailure"]
    y_pred = ["OOMKilled", "NetworkFailure", "NetworkFailure"]
    cls = ClassificationMetrics.compute_all(y_true, y_pred)
    print(f"accuracy: {cls['accuracy']}")
    print(f"F1 Macro: {cls['f1_macro']}")

    #demo generation
    print("\n[3] Generation Metrics Demo")
    gen = GenerationMetrics.compute_all(
        candidates=["The pod crashed due to OOM", "Connection pool exhausted"],
        references=[["The OOM killed the pod"], ["DB connection pool was full"]],
    )
    print(f"BLEU: {gen['bleu']}")
    print(f" ROUGE-L: {gen['rouge_l']}")

    print("\nEvaluation demo completed.....")


if __name__ == "__main__":
    main()
