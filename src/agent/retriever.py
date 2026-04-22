#!/usr/bin/env python3
"""
Retriever implementation for LJP-RAG negative example enhancement.
Only implements current recommended approach: elements-weighted retrieval.
Negative KB: only L1 layer (fact elements) is embedded for retrieval.
Target: weighted combination of extracted legal elements (high weight) + original fact (low weight).

Usage:
  Import as module:
    from retriever import LJPRetriever
    retriever = LJPRetriever(config_path="config.json")
    results = retriever.retrieve_negative(target_fact, target_elements, top_k=k)
"""

import json
from tkinter import CURRENT
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
    Main retriever class for LJP-RAG negative knowledge base.
    Implements elements-weighted retrieval: only L1 layer embedded, weighted target combination.
    """
    
    def __init__(
        self,
        config_path: str = None,
        index_root: Optional[str] = None,
        embedding_model: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize retriever from config or explicit parameters.
        
        Args:
            config_path: Path to config json file
            index_root: Override index root directory from config
            embedding_model: Override embedding model name from config
            device: Device to run embedding model on
        """
        if config_path is None:
            config_path = ROOT_DIR / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.index_root = index_root or self.config["retriever"]["index_root"]
        self._load_negative_index()
        
        # Initialize embedding model
        model_name = embedding_model or self.config["index"]["embedding_model"]
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded, dimension: {self.embedding_dim}")
        
        # Element extraction is done by caller (agent/test script) using independent tool
        # We don't initialize extractor here, caller passes extracted elements to us
    
    def _load_negative_index(self) -> None:
        """Load pre-built negative hierarchical index from disk."""
        cases_path = os.path.join(self.index_root, "neg_hierarchical_cases.json")
        embeddings_path = os.path.join(self.index_root, "neg_l1_embeddings.npy")
        meta_path = os.path.join(self.index_root, "neg_metadata.json")
        
        for path in [cases_path, embeddings_path, meta_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Negative index file not found: {path}\n"
                    f"Did you run build_hierarchical_index.py first?"
                )
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.neg_meta = json.load(f)
        
        with open(cases_path, 'r', encoding='utf-8') as f:
            self.neg_cases = json.load(f)
        
        self.neg_l1_embeddings = np.load(embeddings_path)
        
        logger.info(f"Loaded negative index from {self.index_root}:")
        logger.info(f"  Number of negative cases: {len(self.neg_cases)}")
        logger.info(f"  Embedding shape: {self.neg_l1_embeddings.shape}")
    
    def _compute_target_embedding(
        self,
        target_elements: Dict[str, Any],
        target_fact: str,
    ) -> np.ndarray:
        """
        Compute weighted target embedding: elements (0.7) + fact (0.3).
        
        Args:
            target_elements: Extracted 7 legal elements from target case
            target_fact: Original fact text of target case
        
        Returns:
            Normalized weighted combined embedding
        """
        cfg = self.config["retriever"]["negative"]
        elements_weight = cfg.get("elements_weight", 0.7)
        fact_weight = cfg.get("fact_weight", 0.3)
        
        # Format elements text consistently with index building
        parts = []
        for name, value in target_elements.items():
            if value and str(value).strip():
                parts.append(f"{name}：{value}")
        elements_text = "\n".join(parts)
        
        # Compute separate embeddings
        elements_emb = np.zeros(self.embedding_dim)
        if elements_text:
            elements_emb = self.embedding_model.encode([elements_text], normalize_embeddings=True)[0]
        
        fact_emb = np.zeros(self.embedding_dim)
        if target_fact and target_fact.strip():
            fact_emb = self.embedding_model.encode([target_fact], normalize_embeddings=True)[0]
        
        # Weighted combination and normalization
        combined = elements_weight * elements_emb + fact_weight * fact_emb
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined = combined / norm
        
        return combined
    
    def _cosine_search(self, target_emb: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """
        Fast cosine similarity search against precomputed embeddings.
        All embeddings are normalized, so cosine similarity equals dot product.
        
        Args:
            target_emb: Target embedding (normalized)
            top_k: Number of top results to return
        
        Returns:
            List of (index, similarity_score) sorted by descending similarity
        """
        similarities = np.dot(self.neg_l1_embeddings, target_emb)
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        return list(zip(top_indices, top_similarities))
    
    def retrieve_negative(
        self,
        target_fact: str,
        target_elements: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar negative error cases for target case.
        Caller must extract elements first using LegalElementExtractor (independent tool).
        
        Args:
            target_fact: Original fact text of target case
            target_elements: Extracted legal elements from target case
            top_k: Fixed number of results to return
        
        Returns:
            List of retrieved negative cases with full L0-L3 layers and similarity score
        """
        target_emb = self._compute_target_embedding(target_elements, target_fact)
        top_results = self._cosine_search(target_emb, top_k)
        
        # Prepare output with full case data
        output = []
        for idx, sim in top_results:
            output.append({
                **self.neg_cases[idx],
                "similarity": float(sim),
            })
        
        logger.info(f"Retrieved {len(output)} negative cases, max similarity: {output[0]['similarity']:.3f}")
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
        logger.info("✅ Retriever initialized successfully")
        logger.info(f"Negative cases loaded: {len(retriever.neg_cases)}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize retriever: {e}")
        exit(1)


if __name__ == "__main__":
    main()
