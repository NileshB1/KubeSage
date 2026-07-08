"""
Configures and indexes the persistent vector database using ChromaDB
Handles document ingestion, similarity search using cosine distance and metadata management for Kubernetes incidents.
"""

import json
import sys
import time
from pathlib import Path

from typing import Any

# Ensure project root is on the path for cross-module imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.logging_config import get_logger

# Monkeypatch ChromaDB telemetry to prevent console warnings/errors
try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

logger = get_logger(__name__)


class VectorDatabase:
    """
    ChromaDB vector database for K8s incident retrieval
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None, enable_lazy_init: bool = True,
    ) -> None:
        """
        Initialize the vector database

        """
        self.persist_directory = str(
            persist_directory or settings.VECTOR_DB_DIR
        )
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_dim = settings.EMBEDDING_DIMENSION
        self._client = None
        self._collection = None
        self._initialized = False

        if not enable_lazy_init:
            self._init_client()
        else:
            logger.info(f"VectorDB configured (lazy) | Collection: '{self.collection_name}'")

    def _init_client(self) -> None:
        """Lazily initialize the ChromaDB client and collection."""
        if self._initialized:
            return

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_directory, settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": settings.CHROMA_DISTANCE_METRIC,
                "description": "KubeSage Kubernetes Incident Embeddings",
            },
        )

        self._initialized = True
        count = self._collection.count()
        logger.info(
            f"VectorDB initialized | Collection: '{self.collection_name}' | "
            f"Documents: {count} | Path: {self.persist_directory}"
        )

    @property
    def client(self):
        self._init_client()

        return self._client

    @property
    def collection(self):
        self._init_client()
        return self._collection

    def count(self) -> int:
        """Return number of stored embeddings."""
        self._init_client()
        return self._collection.count()

    # Build / Index


    def build_index(
        self,
        embeddings: np.ndarray,
        incidents: list[dict[str, Any]],
        model_name: str = "",
    ) -> dict[str, Any]:
        """
        Build the vector index from embeddings and incident metadata
        """
        if len(embeddings) != len(incidents):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) != incident count ({len(incidents)})"
            )

        # Ensure client is initialized
        self._init_client()

        # Clear existing collection and rebuild
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": settings.CHROMA_DISTANCE_METRIC,
                "description": "KubeSage Kubernetes Incident Embeddings",
                "model_name": model_name, "dimension": str(embeddings.shape[1]),
            },
        )

        logger.info(f"Building index for {len(embeddings)} embeddings....")
        start = time.time()

        # Prepare batch data
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for i, incident in enumerate(incidents):
            inc_id = incident.get("incident_id", f"INC-{i:04d}")
            ids.append(inc_id)
            documents.append(incident.get("cleaned_text", incident.get("description", "")))
            metadatas.append({
                "incident_id": inc_id,  "title": incident.get("title", "")[:200],
                "incident_type": incident.get("incident_type", "Unknown"),
                "severity": incident.get("severity", "Medium"),
                "root_cause": incident.get("root_cause", "")[:500],   "resolution": incident.get("resolution", "")[:500],
                "timestamp": incident.get("timestamp", ""), "affected_services": ", ".join(incident.get("affected_services", [])),
                "source": incident.get("source", "synthetic"),
            })

        #Split into batches of 500 for ChromaDB
        batch_size = 500
        total = len(embeddings)

        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)

            self.collection.add(
                ids=ids[start_idx:end_idx],
                embeddings=embeddings[start_idx:end_idx].tolist(),

                documents=documents[start_idx:end_idx],
                metadatas=metadatas[start_idx:end_idx],
            )

        build_time = time.time() - start
        self.embedding_dim = embeddings.shape[1]

        stats = {
            "num_documents": self.collection.count(),  "embedding_dimension": int(self.embedding_dim),
            "build_time_seconds": round(build_time, 2),"model_name": model_name,
        }

        logger.info(
            f"Index built in {build_time:.1f}s | Documents: {self.collection.count()} | "
            f"Dimension: {self.embedding_dim}"
        )

        return stats

  
    # Search / Query

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,  where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Perform semantic search over the vector database
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        start = time.time()

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,  where=where,
            include=["distances", "metadatas", "documents"],
        )

        query_time = time.time() - start

        # Format results
        formatted: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                #convert cosine distance to similarity, clamped to [0, 1]
                similarity = max(0.0, min(1.0, 1.0 - distance)) if distance is not None else 0.0

                formatted.append({
                    "incident_id": results["ids"][0][i],  "similarity_score": round(float(similarity), 4),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "document": results["documents"][0][i][:500] if results["documents"] else "",
                })

        return {
            "results": formatted,  "count": len(formatted),
            "query_time_ms": round(query_time * 1000, 2),
            "top_k": top_k,
        }


    #CRUD
    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        """
        Get a single incident from the vector database by ID
        """
        result = self.collection.get(
            ids=[incident_id],
            include=["embeddings", "metadatas", "documents"],
        )

        if result["ids"]:
            return {
                "incident_id": result["ids"][0],  "metadata": result["metadatas"][0] if result["metadatas"] else {},
                "document": result["documents"][0] if result["documents"] else "",
            }
        return None

    def delete_incident(self, incident_id: str) -> bool:
        """
        Delete an incident from the vector database
        """
        existing = self.get_incident(incident_id)
        if existing is None:
            return False

        self.collection.delete(ids=[incident_id])
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics"""
        return {
            "collection_name": self.collection_name, "document_count": self.collection.count(),
            "embedding_dimension": self.embedding_dim,
            "distance_metric": settings.CHROMA_DISTANCE_METRIC, "persist_directory": self.persist_directory,
        }


#cli
def main() -> None:
    """Build vector index from command line"""
    import argparse

    parser = argparse.ArgumentParser(description="building chromaDB vector index")

    parser.add_argument("--embeddings", type=str, required=True, help="Path to .npy embeddings file")
    parser.add_argument("--incidents", type=str, required=True, help="Path to incidents JSON file")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Model name used")


    parser.add_argument("--query", type=str, default=None, help="Optional test query")

    args = parser.parse_args()

    # Load data
    embeddings = np.load(args.embeddings)
    with open(args.incidents, "r") as f:
        incidents = json.load(f)

    logger.info(f"Loaded embeddings now: {embeddings.shape}")
    logger.info(f"Loaded incidents: {len(incidents)}")

    #building index
    vdb = VectorDatabase()
    stats = vdb.build_index(embeddings, incidents, args.model)

    print(f"vector index built:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Optional test query
    if args.query:
        print(f"Searching for: \"{args.query}\"")

        from embeddings.generate_embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator(model_name=args.model)
        query_emb = gen.generate_single_embedding(args.query)

        search_results = vdb.search(query_emb, top_k=3)
        print(f"Found {search_results['count']} results in {search_results['query_time_ms']}ms")
        for r in search_results["results"]:
            print(f"{r['incident_id']}: {r['similarity_score']:.3f} | {r['metadata'].get('incident_type', 'N/A')}")

#main method for tetsing
if __name__ == "__main__":
    main()
