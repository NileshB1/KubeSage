"""

Encodes cleaned Kubernetes incident descriptions and log contents into 
dense numerical vector spaces using Sentence Transformers
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# TODO need to check
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for incident text using Sentence Transformers
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize the embedding generator
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE

        logger.info(f"Loading embedding model: {self.model_name}")
        start = time.time()

        self.model = SentenceTransformer(self.model_name, device=self.device)

        load_time = time.time() - start
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded in {load_time:.1f}s | "
            f"Dimension: {self.dimension} | Device: {self.device}" )

    def generate_embeddings(
        self,  texts: list[str],
        batch_size: int | None = None, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        """
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
        start = time.time()

        embeddings = self.model.encode(
            texts,  batch_size=batch_size,    show_progress_bar=show_progress,
            normalize_embeddings=True,  # Normalize for cosine similarity
            convert_to_numpy=True,
        )

        elapsed = time.time() - start
        embeds_per_sec = len(texts) / elapsed
        logger.info(
            f"Generated {len(embeddings)} embeddings in {elapsed:.1f}s "
            f"({embeds_per_sec:.0f} texts/sec) | Shape: {embeddings.shape}"
        )

        return embeddings

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text query
        """
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0]


def prepare_texts_from_incidents(incidents: list[dict[str, Any]]) -> list[str]:
    """
    Extract text representations from incidents for embedding
    """
    from data.preprocess import IncidentPreprocessor

    texts = []
    preprocessor = IncidentPreprocessor()

    for incident in incidents:
        # Use preprocessed text if available
        if "cleaned_text" in incident and incident["cleaned_text"]:
            texts.append(incident["cleaned_text"])
        else:
            # Build from raw fields
            text = preprocessor.build_incident_text(incident)
            texts.append(text)

    return texts


#cli
def main() -> None:
    """Generate embeddings from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate incident embeddings")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")

    parser.add_argument("--output", type=str, required=True, help="Output .npy file")

    parser.add_argument("--model", type=str, default=None, help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load incidents
    with open(input_path, "r") as f:
        incidents = json.load(f)
    logger.info(f"Loaded {len(incidents)} incidents from {input_path}")

    # Prepare texts
    texts = prepare_texts_from_incidents(incidents)
    logger.info(f"Prepared {len(texts)} text representations....")

    # Generate embeddings
    generator = EmbeddingGenerator(model_name=args.model)
    embeddings = generator.generate_embeddings(texts, batch_size=args.batch_size)

    # Save embeddings (numpy format)
    np.save(output_path, embeddings)

    # Also save metadata
    meta_path = output_path.with_suffix(".json")
    meta: dict[str, Any] = {
        "model_name": generator.model_name, "dimension": int(generator.dimension),
        "num_embeddings": int(len(embeddings)),  "incident_ids": [inc["incident_id"] for inc in incidents],
        "shape": list(embeddings.shape),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Embeddings saved: {output_path}")
    logger.info(f"Metadata saved: {meta_path}")
    print(f"\nGenerated {len(embeddings)} embeddings ({generator.dimension}-dim)")
    print(f" Embeddings: {output_path}")
    print(f"Metadata:   {meta_path}")

#main method
if __name__ == "__main__":
    main()
