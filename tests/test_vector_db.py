"""
Unit Tests — ChromaDB Vector Database
=====================================
Validates ChromaDB collection initialization, document index building, 
and similarity query matching.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vector_db.build_index import VectorDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir() -> str:
    """Create a temporary directory for ChromaDB storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vdb(temp_dir: str) -> VectorDatabase:
    """Create a VectorDatabase with lazy init disabled (immediate init)."""
    return VectorDatabase(
        persist_directory=temp_dir,
        collection_name="test_collection",
        enable_lazy_init=False,
    )


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create 10 random 384-dim embeddings."""
    return np.random.randn(10, 384).astype(np.float32)


@pytest.fixture
def sample_incidents() -> list[dict]:
    """Create 10 sample incident dictionaries."""
    return [
        {
            "incident_id": f"INC-TEST-{i:04d}",
            "title": f"Test incident {i}",
            "description": f"Description of test incident {i}",
            "incident_type": "CrashLoopBackOff",
            "severity": "Critical",
            "root_cause": "Test root cause",
            "resolution": "Test resolution",
            "timestamp": "2024-06-15T14:30:00Z",
            "affected_services": ["test-service"],
            "source": "synthetic",
            "cleaned_text": f"cleaned description of incident {i}",
        }
        for i in range(10)
    ]


@pytest.fixture
def populated_vdb(
    vdb: VectorDatabase,
    sample_embeddings: np.ndarray,
    sample_incidents: list[dict],
) -> VectorDatabase:
    """Create a VectorDatabase pre-populated with 10 embeddings."""
    vdb.build_index(sample_embeddings, sample_incidents, model_name="test-model")
    return vdb


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------

class TestInitialization:
    """Tests for VectorDatabase construction and lazy init."""

    def test_lazy_init_does_not_create_client(self, temp_dir: str) -> None:
        """Lazy init should not create ChromaDB client immediately."""
        vdb = VectorDatabase(
            persist_directory=temp_dir,
            enable_lazy_init=True,
        )
        assert vdb._client is None
        assert vdb._collection is None
        assert not vdb._initialized

    def test_eager_init_creates_client(self, temp_dir: str) -> None:
        """Eager init should create ChromaDB client immediately."""
        vdb = VectorDatabase(
            persist_directory=temp_dir,
            enable_lazy_init=False,
        )
        assert vdb._initialized
        assert vdb._client is not None
        assert vdb._collection is not None

    def test_count_initializes_lazily(self, temp_dir: str) -> None:
        """count() should trigger lazy init."""
        vdb = VectorDatabase(persist_directory=temp_dir, enable_lazy_init=True)
        assert not vdb._initialized
        count = vdb.count()
        assert vdb._initialized
        assert count == 0

    def test_default_collection_name(self, temp_dir: str) -> None:
        """Should use config default collection name."""
        vdb = VectorDatabase(persist_directory=temp_dir, enable_lazy_init=False)
        assert "k8s_incidents" in vdb.collection_name

    def test_custom_collection_name(self, temp_dir: str) -> None:
        """Should accept custom collection names."""
        vdb = VectorDatabase(
            persist_directory=temp_dir,
            collection_name="my_custom_collection",
            enable_lazy_init=False,
        )
        assert vdb.collection_name == "my_custom_collection"


# ---------------------------------------------------------------------------
# Build Index Tests
# ---------------------------------------------------------------------------

class TestBuildIndex:
    """Tests for vector index construction."""

    def test_builds_index_successfully(
        self, vdb: VectorDatabase, sample_embeddings: np.ndarray, sample_incidents: list[dict],
    ) -> None:
        """Building index should succeed and return stats."""
        stats = vdb.build_index(sample_embeddings, sample_incidents, model_name="test")
        assert stats["num_documents"] == 10
        assert stats["embedding_dimension"] == 384
        assert stats["model_name"] == "test"
        assert stats["build_time_seconds"] > 0

    def test_count_matches_after_build(self, populated_vdb: VectorDatabase) -> None:
        """Count should match number of inserted embeddings."""
        assert populated_vdb.count() == 10

    def test_mismatched_lengths_raises(self, vdb: VectorDatabase) -> None:
        """Should raise ValueError when embedding count != incident count."""
        embeddings = np.random.randn(5, 384).astype(np.float32)
        incidents = [{"incident_id": f"INC-{i}"} for i in range(10)]
        with pytest.raises(ValueError, match="Embedding count"):
            vdb.build_index(embeddings, incidents)


# ---------------------------------------------------------------------------
# Search Tests
# ---------------------------------------------------------------------------

class TestSearch:
    """Tests for semantic search functionality."""

    def test_search_returns_results(
        self, populated_vdb: VectorDatabase, sample_embeddings: np.ndarray,
    ) -> None:
        """Search should return top-k results with scores."""
        query = sample_embeddings[0]  # Search with first embedding
        results = populated_vdb.search(query, top_k=3)
        assert results["count"] == 3
        assert len(results["results"]) == 3
        assert results["top_k"] == 3

    def test_search_returns_metadata(
        self, populated_vdb: VectorDatabase, sample_embeddings: np.ndarray,
    ) -> None:
        """Search results should include metadata."""
        query = sample_embeddings[5]
        results = populated_vdb.search(query, top_k=1)
        r = results["results"][0]
        assert "metadata" in r
        assert "incident_id" in r
        assert "similarity_score" in r
        assert "incident_type" in r["metadata"]

    def test_search_self_match_is_high_similarity(
        self, populated_vdb: VectorDatabase, sample_embeddings: np.ndarray,
    ) -> None:
        """Search with an indexed embedding should find it with high similarity."""
        query = sample_embeddings[0]
        results = populated_vdb.search(query, top_k=10)
        # The first embedding should be in top results
        top_ids = [r["incident_id"] for r in results["results"]]
        assert "INC-TEST-0000" in top_ids

    def test_search_respects_top_k(
        self, populated_vdb: VectorDatabase, sample_embeddings: np.ndarray,
    ) -> None:
        """Search should respect the top_k parameter."""
        for k in [1, 5]:
            results = populated_vdb.search(sample_embeddings[0], top_k=k)
            assert results["count"] == k

    def test_search_accepts_2d_query(
        self, populated_vdb: VectorDatabase,
    ) -> None:
        """Search should accept 2D query arrays."""
        query = np.random.randn(1, 384).astype(np.float32)
        results = populated_vdb.search(query, top_k=3)
        assert results["count"] == 3

    def test_empty_search_results(
        self, vdb: VectorDatabase,
    ) -> None:
        """Search on empty collection should return empty results."""
        query = np.random.randn(384).astype(np.float32)
        results = vdb.search(query, top_k=3)
        assert results["count"] == 0
        assert results["results"] == []

    def test_similarity_score_range(
        self, populated_vdb: VectorDatabase, sample_embeddings: np.ndarray,
    ) -> None:
        """Similarity scores should be between 0 and 1."""
        query = sample_embeddings[0]
        results = populated_vdb.search(query, top_k=5)
        for r in results["results"]:
            assert 0.0 <= r["similarity_score"] <= 1.0


# ---------------------------------------------------------------------------
# CRUD Tests
# ---------------------------------------------------------------------------

class TestCRUD:
    """Tests for get, delete operations."""

    def test_get_existing_incident(self, populated_vdb: VectorDatabase) -> None:
        """Should retrieve an incident by ID."""
        incident = populated_vdb.get_incident("INC-TEST-0000")
        assert incident is not None
        assert incident["incident_id"] == "INC-TEST-0000"
        assert "metadata" in incident

    def test_get_nonexistent_incident(self, populated_vdb: VectorDatabase) -> None:
        """Should return None for missing IDs."""
        incident = populated_vdb.get_incident("NONEXISTENT")
        assert incident is None

    def test_delete_existing_incident(self, populated_vdb: VectorDatabase) -> None:
        """Should delete an incident and return True."""
        count_before = populated_vdb.count()
        result = populated_vdb.delete_incident("INC-TEST-0000")
        assert result is True
        assert populated_vdb.count() == count_before - 1
        assert populated_vdb.get_incident("INC-TEST-0000") is None

    def test_delete_nonexistent_incident(self, populated_vdb: VectorDatabase) -> None:
        """Should return False for deleting missing incident."""
        result = populated_vdb.delete_incident("DOES_NOT_EXIST")
        assert result is False


# ---------------------------------------------------------------------------
# Stats Tests
# ---------------------------------------------------------------------------

class TestStats:
    """Tests for get_stats."""

    def test_stats_contains_all_keys(self, vdb: VectorDatabase) -> None:
        """get_stats should return all expected keys."""
        stats = vdb.get_stats()
        assert "collection_name" in stats
        assert "document_count" in stats
        assert "embedding_dimension" in stats
        assert "distance_metric" in stats
        assert "persist_directory" in stats

    def test_stats_document_count(self, populated_vdb: VectorDatabase) -> None:
        """Document count in stats should match count()."""
        stats = populated_vdb.get_stats()
        assert stats["document_count"] == populated_vdb.count()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
