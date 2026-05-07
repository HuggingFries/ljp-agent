#!/usr/bin/env python3
"""
Retriever implementation for LJP-RAG unified historical case retrieval.
Loads a single index (unified KB) with L1 legal elements embedding.

Usage:
    from retriever import LJPRetriever
    retriever = LJPRetriever(config_path="config.yaml")
    results = retriever.retrieve(target_fact, target_elements, top_k=3)
"""

import json
import yaml
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent


class LJPRetriever:
    """
    Retriever for LJP-RAG unified historical case knowledge base.
    Loads a single index (unified KB with both correct + error analysis).
    """

    def __init__(
        self,
        config_path: str = None,
        index_root: Optional[str] = None,
        embedding_model: Optional[str] = None,
        device: str = "cpu",
    ):
        if config_path is None:
            config_path = ROOT_DIR / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.index_root = index_root or self.config["retriever"]["index_root"]
        self._load_index()

        model_name = embedding_model or self.config["index"]["embedding_model"]
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded, dimension: {self.embedding_dim}")

    def _load_index(self) -> None:
        """Load the unified index from disk."""
        cases_file = os.path.join(self.index_root, "unified_hierarchical_cases.json")
        embeddings_file = os.path.join(self.index_root, "unified_l1_embeddings.npy")
        meta_file = os.path.join(self.index_root, "unified_metadata.json")

        for path in [cases_file, embeddings_file, meta_file]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Unified index file not found: {path}")

        with open(meta_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        with open(cases_file, 'r', encoding='utf-8') as f:
            self.cases = json.load(f)
        self.embeddings = np.load(embeddings_file)

        logger.info(f"Loaded unified index: {len(self.cases)} cases, embeddings shape {self.embeddings.shape}")

    def _compute_target_embedding(
        self,
        target_elements: Dict[str, Any],
        target_fact: str,
    ) -> np.ndarray:
        """
        Compute weighted target embedding: elements (weight) + fact (weight).
        Weights are read from config.
        """
        cfg = self.config["retriever"]
        elements_weight = cfg.get("elements_weight", 0.7)
        fact_weight = cfg.get("fact_weight", 0.3)

        parts = []
        for name, value in target_elements.items():
            if value and str(value).strip():
                parts.append(f"{name}：{value}")
        elements_text = "\n".join(parts)

        elements_emb = np.zeros(self.embedding_dim)
        if elements_text:
            elements_emb = self.embedding_model.encode([elements_text], normalize_embeddings=True)[0]

        fact_emb = np.zeros(self.embedding_dim)
        if target_fact and target_fact.strip():
            fact_emb = self.embedding_model.encode([target_fact], normalize_embeddings=True)[0]

        combined = elements_weight * elements_emb + fact_weight * fact_emb
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined = combined / norm

        return combined

    def _cosine_search(self, embeddings: np.ndarray, target_emb: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Cosine similarity search."""
        similarities = np.dot(embeddings, target_emb)
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        return list(zip(top_indices, top_similarities))

    def retrieve(
        self,
        target_fact: str,
        target_elements: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar cases from the unified index.

        Args:
            target_fact: Original fact text
            target_elements: Extracted legal elements
            top_k: Number of results

        Returns:
            List of retrieved cases with similarity score
        """
        target_emb = self._compute_target_embedding(target_elements, target_fact)
        top_results = self._cosine_search(self.embeddings, target_emb, top_k)

        output = []
        for idx, sim in top_results:
            output.append({
                **self.cases[idx],
                "similarity": float(sim),
            })

        if output:
            logger.info(f"Retrieved {len(output)} cases, max similarity: {output[0]['similarity']:.3f}")
        else:
            logger.info("No cases retrieved")
        return output


def main():
    """Quick test for retriever initialization."""
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT_DIR / "config" / "config.yaml", help="Config file path")
    parser.add_argument("--index-root", help="Index root directory")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    try:
        retriever = LJPRetriever(
            config_path=args.config,
            index_root=args.index_root,
            device=args.device,
        )
        logger.info("Retriever initialized successfully (unified index)")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        exit(1)


if __name__ == "__main__":
    main()
