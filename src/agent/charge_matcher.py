#!/usr/bin/env python3
"""
Embedding-based charge name mapper.
Maps free-text LLM output to standard CAIL2018 charge names via cosine similarity.

Usage:
    from charge_matcher import ChargeMatcher
    matcher = ChargeMatcher("data/accu.txt")
    standard_name, score = matcher.map_charge("诈骗罪")
    standard_names = matcher.map_charges(["诈骗罪", "盗窃"])
"""

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ChargeMatcher:
    """Map free-text charge names to standard CAIL2018 charge names."""

    def __init__(self, accu_path: str, model_name: str = "uer/sbert-base-chinese-nli"):
        with open(accu_path, 'r', encoding='utf-8') as f:
            self.standard_charges = [line.strip() for line in f if line.strip()]

        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._model = SentenceTransformer(model_name)
        self._charge_embs = self._model.encode(
            self.standard_charges, normalize_embeddings=True
        )
        logger.info(
            f"ChargeMatcher initialized: {len(self.standard_charges)} charges, model={model_name}"
        )

    def map_charge(self, raw_charge: str) -> Tuple[str, float]:
        """Map a single free-text charge to the closest standard name.

        Returns:
            (standard_name, similarity_score)
        """
        cleaned = raw_charge.strip().removesuffix("罪").strip()

        if not cleaned:
            return self.standard_charges[0], 0.0

        # Exact match shortcut
        for std in self.standard_charges:
            if cleaned == std:
                return std, 1.0

        # Embedding similarity fallback
        import numpy as np
        emb = self._model.encode([cleaned], normalize_embeddings=True)[0]
        sims = np.dot(self._charge_embs, emb)
        best = int(sims.argmax())
        return self.standard_charges[best], float(sims[best])

    def map_charges(self, raw_charges: List[str]) -> List[str]:
        """Map a list of free-text charges to standard names."""
        return [self.map_charge(c)[0] for c in raw_charges]
