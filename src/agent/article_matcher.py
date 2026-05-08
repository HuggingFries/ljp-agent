"""Article number validator and mapper for CAIL2018.

Validates free-text article numbers against the known list.
Unlike ChargeMatcher (which uses Sentence-BERT similarity),
ArticleMatcher uses direct digit matching plus a feedback loop
for invalid predictions.
"""

import logging
from typing import List, Tuple, Set

logger = logging.getLogger(__name__)


class ArticleMatcher:
    """Validate article numbers against the known list."""

    def __init__(self, law_path: str):
        with open(law_path, 'r', encoding='utf-8') as f:
            self.valid_articles: Set[str] = set(
                line.strip() for line in f if line.strip()
            )
        self.valid_list_text = "、".join(sorted(self.valid_articles,
                                                 key=lambda x: int(x)))
        self.max_iterations = 3
        logger.info(f"ArticleMatcher initialized: {len(self.valid_articles)} valid articles")

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
