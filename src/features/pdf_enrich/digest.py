"""Digest job placeholder for downstream enrichment workflows."""

from __future__ import annotations

from typing import List

from src.core.models import Article, Enrichment


def summarize_articles(articles: List[Article]) -> List[Enrichment]:
    """Placeholder summarization pipeline."""

    # Future work: call LLMs or rules-based engines to populate Enrichment payloads.
    return [article.enrichment or Enrichment() for article in articles]
