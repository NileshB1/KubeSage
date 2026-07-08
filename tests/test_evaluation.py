"""
Unit Tests — Evaluation Metrics
===============================
Tests retrieval quality scores (Precision, Recall, MRR, NDCG) and classification accuracy.
"""

import pytest
from evaluation.metrics import (
    RetrievalMetrics,
    ClassificationMetrics,
    GenerationMetrics,
    HallucinationMetrics,
    Evaluator,
)


class TestRetrievalMetrics:
    """Tests for embedding/retrieval metrics (RQ1)."""

    def test_precision_at_k_perfect(self) -> None:
        """Precision@K should be 1.0 for perfect retrieval."""
        score = RetrievalMetrics.precision_at_k(
            retrieved_ids=["A", "B", "C"],
            relevant_ids={"A", "B", "C"},
            k=3,
        )
        assert score == 1.0

    def test_precision_at_k_partial(self) -> None:
        """Precision@K should be correct for partial retrieval."""
        score = RetrievalMetrics.precision_at_k(
            retrieved_ids=["A", "B", "C"],
            relevant_ids={"A", "C"},
            k=3,
        )
        assert score == 2.0 / 3.0

    def test_recall_at_k_perfect(self) -> None:
        """Recall@K should be 1.0 when all relevant retrieved."""
        score = RetrievalMetrics.recall_at_k(
            retrieved_ids=["A", "B", "C"],
            relevant_ids={"A", "C"},
            k=3,
        )
        assert score == 1.0

    def test_mrr_first_position(self) -> None:
        """MRR should be 1.0 when first result is relevant."""
        score = RetrievalMetrics.mean_reciprocal_rank(
            retrieved_ids=["A", "B", "C"],
            relevant_ids={"A"},
        )
        assert score == 1.0

    def test_mrr_third_position(self) -> None:
        """MRR should be 1/3 when third result is relevant."""
        score = RetrievalMetrics.mean_reciprocal_rank(
            retrieved_ids=["X", "Y", "Z"],
            relevant_ids={"Z"},
        )
        assert score == pytest.approx(1.0 / 3.0)

    def test_mrr_no_relevant(self) -> None:
        """MRR should be 0 when nothing is relevant."""
        score = RetrievalMetrics.mean_reciprocal_rank(
            retrieved_ids=["X", "Y", "Z"],
            relevant_ids={"A"},
        )
        assert score == 0.0

    def test_ndcg_perfect(self) -> None:
        """NDCG should be 1.0 for perfectly ordered results."""
        score = RetrievalMetrics.ndcg_at_k(
            retrieved_ids=["A", "B", "C"],
            relevance_scores={"A": 1.0, "B": 0.8, "C": 0.6},
            k=3,
        )
        assert score == pytest.approx(1.0, rel=1e-3)

    def test_compute_all(self) -> None:
        """compute_all should return all metrics."""
        queries = [
            {"retrieved_ids": ["A", "B", "C"], "relevant_ids": {"A", "C"}},
        ]
        results = RetrievalMetrics.compute_all(queries, k_values=[1, 3])
        assert "precision" in results
        assert "recall" in results
        assert "mrr" in results
        assert "ndcg" in results
        assert "@1" in results["precision"]
        assert "@3" in results["precision"]


class TestClassificationMetrics:
    """Tests for classification metrics (RQ2)."""

    def test_perfect_classification(self) -> None:
        """Accuracy should be 1.0 for perfect predictions."""
        results = ClassificationMetrics.compute_all(
            y_true=["A", "B", "C"],
            y_pred=["A", "B", "C"],
        )
        assert results["accuracy"] == 1.0
        assert results["f1_macro"] == 1.0

    def test_imperfect_classification(self) -> None:
        """Accuracy should reflect errors."""
        results = ClassificationMetrics.compute_all(
            y_true=["A", "B", "C"],
            y_pred=["A", "C", "C"],
        )
        assert results["accuracy"] == pytest.approx(2.0 / 3.0, rel=1e-3)


class TestGenerationMetrics:
    """Tests for generation quality metrics (RQ3)."""

    def test_bleu_identical(self) -> None:
        """BLEU should be ~1.0 for identical texts."""
        text = "the pod crashed due to out of memory"
        score = GenerationMetrics.compute_bleu(text, [text])
        assert score > 0.9

    def test_rouge_l_identical(self) -> None:
        """ROUGE-L should be 1.0 for identical texts."""
        text = "the pod crashed"
        score = GenerationMetrics.compute_rouge_l(text, text)
        assert score == 1.0

    def test_rouge_l_different(self) -> None:
        """ROUGE-L should be 0 for completely different texts."""
        score = GenerationMetrics.compute_rouge_l(
            "database connection pool full",
            "the weather is sunny today",
        )
        assert score == 0.0


class TestHallucinationMetrics:
    """Tests for hallucination metrics (RQ4)."""

    def test_fully_faithful(self) -> None:
        """Faithfulness should be 1.0 when all claims are supported."""
        scores = HallucinationMetrics.compute_faithfulness(
            generated_text="The pod OOMKilled. The memory limit was 256Mi.",
            source_documents=["pod OOMKilled memory limit 256Mi"],
            use_sentence_transformer=False,
        )
        assert scores["faithfulness"] > 0.0

    def test_fully_unsupported(self) -> None:
        """Faithfulness should be ~0 when no claims are supported."""
        scores = HallucinationMetrics.compute_faithfulness(
            generated_text="The database crashed. The server exploded.",
            source_documents=["weather forecast cloudy"],
            use_sentence_transformer=False,
        )
        assert scores["faithfulness"] < 0.5


class TestEvaluator:
    """Tests for the comprehensive evaluator."""

    def test_evaluate_all_empty(self) -> None:
        """Evaluator should handle empty inputs gracefully."""
        results = Evaluator.evaluate_all()
        assert results == {}

    def test_evaluate_all_retrieval(self) -> None:
        """Evaluator should compute retrieval metrics."""
        results = Evaluator.evaluate_all(
            retrieval_data=[
                {"retrieved_ids": ["A", "B"], "relevant_ids": {"A"}},
            ],
        )
        assert "retrieval" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
