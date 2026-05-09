"""Article number validator and mapper for CAIL2018.

Validates free-text article numbers against the known list.
Unlike ChargeMatcher (which uses Sentence-BERT similarity),
ArticleMatcher uses direct digit matching plus a feedback loop
for invalid predictions.

Now also maintains a charge→article mapping from training data
to narrow the feedback loop to charge-relevant articles only.
"""

import json
import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class ArticleMatcher:
    """Validate article numbers against the known list, with charge-aware narrowing."""

    def __init__(self, law_path: str, charge_article_data: str = None):
        with open(law_path, 'r', encoding='utf-8') as f:
            self.valid_articles: Set[str] = set(
                line.strip() for line in f if line.strip()
            )
        self.valid_list_text = "、".join(sorted(self.valid_articles,
                                                 key=lambda x: int(x)))
        self.max_iterations = 3

        # Optional charge→article mapping from training data
        self.charge_to_articles: Dict[str, Set[str]] = {}
        if charge_article_data:
            self._build_charge_article_mapping(charge_article_data)

        logger.info(f"ArticleMatcher initialized: {len(self.valid_articles)} valid articles"
                    f"{', with charge→article mapping' if self.charge_to_articles else ''}")

    def _build_charge_article_mapping(self, data_path: str):
        """Load pre-built charge→articles mapping from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self.charge_to_articles = {k: set(v) for k, v in raw.items()}
        logger.info(f"Loaded charge→article mapping from {data_path}: "
                    f"{len(self.charge_to_articles)} charges, "
                    f"avg {sum(len(v) for v in self.charge_to_articles.values()) / max(len(self.charge_to_articles), 1):.1f} articles per charge")

    def get_articles_for_charges(self, charges: List[str]) -> str:
        """Return narrowed article list for the given charges.
        Multiple charges use intersection (article must be valid for ALL charges).
        Falls back to full list if mapping is unavailable, charge unknown, or intersection empty.
        """
        if not self.charge_to_articles or not charges:
            return self.valid_list_text

        # All charges must be known
        for charge in charges:
            if charge not in self.charge_to_articles:
                return self.valid_list_text

        if len(charges) == 1:
            articles = self.charge_to_articles[charges[0]]
        else:
            article_sets = [self.charge_to_articles[c] for c in charges]
            articles = set.intersection(*article_sets)

        if not articles:
            return self.valid_list_text
        return "、".join(sorted(articles, key=lambda x: int(x)))

    def validate(self, articles: List[str]) -> Tuple[List[str], List[str]]:
        """Return (valid_articles, invalid_articles) after cleaning."""
        valid: List[str] = []
        invalid: List[str] = []
        seen = set()
        for a in articles:
            cleaned = self._clean(a)
            if not cleaned:
                invalid.append(str(a))
                continue
            if cleaned in self.valid_articles:
                if cleaned not in seen:
                    valid.append(cleaned)
                    seen.add(cleaned)
            else:
                invalid.append(cleaned)
        return valid, invalid

    def map_articles(self, articles: List[str]) -> List[str]:
        """Keep only valid articles, discard invalid ones."""
        valid, _ = self.validate(articles)
        return valid

    @staticmethod
    def _clean(article) -> str:
        """Extract digits from an article string."""
        if isinstance(article, (int, float)):
            return str(int(article))
        return "".join(c for c in str(article) if c.isdigit())
