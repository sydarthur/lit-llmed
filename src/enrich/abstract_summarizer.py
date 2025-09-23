"""Generate structured summaries from article abstracts using the local LLM."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.core.log import get_logger
from src.core.models import Article
from src.enrich.llm import OllamaLLM
from src.enrich.prompts import get_abstract_template
from src.enrich.settings import EnrichmentPaths, resolve_paths

LOGGER = get_logger(__name__)


class AbstractSummarizer:
    """Summarise article abstracts into a structured JSON representation."""

    def __init__(
        self,
        *,
        paths: Optional[EnrichmentPaths] = None,
        llm: Optional[OllamaLLM] = None,
    ) -> None:
        self.paths = paths or resolve_paths()
        self.llm = llm or OllamaLLM()
        self.paths.ensure()
        self.template = get_abstract_template()["prompt"]

    def summarise_article(self, article: Article) -> Optional[Dict[str, object]]:
        if not article.abstract:
            LOGGER.debug(f"abstract_missing for {article.title}")
            return None
        prompt = self._build_prompt(article)
        response = self.llm.chat(prompt)
        if not response:
            LOGGER.warning(f"abstract_summary_failed for {article.title}")
            return None
        summary = self._parse_response(response)
        if not summary:
            LOGGER.error(f"abstract_summary_parse_failed for {article.title}")
            return None
        summary.setdefault("title", article.title)
        summary.setdefault("doi", article.doi)
        summary.setdefault("journal", article.journal)
        summary.setdefault("published_date", _format_date(article))
        return summary

    def summarise_articles(
        self,
        articles: Iterable[Article],
        *,
        label: Optional[str] = None,
    ) -> Path:
        results: List[Dict[str, object]] = []
        for article in articles:
            summary = self.summarise_article(article)
            if summary:
                results.append(summary)
        identifier = label or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = self.paths.abstracts_root / f"{identifier}_abstracts.json"
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info(f"abstract_summaries_written: {len(results)} summaries to {output_path}")
        return output_path

    def summarise_json_file(self, json_path: Path, *, label: Optional[str] = None) -> Path:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        articles = [Article.model_validate(obj) for obj in payload]
        label = label or json_path.stem
        return self.summarise_articles(articles, label=label)

    def _build_prompt(self, article: Article) -> str:
        return self.template.format(
            title=article.title,
            doi="null",
            journal="null", 
            published="null",
            abstract=article.abstract,
        )

    @staticmethod
    def _parse_response(response: str) -> Optional[Dict[str, object]]:
        response = response.strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Attempt to extract the first JSON object if the model returned extra text
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                snippet = response[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
            return None


def _format_date(article: Article) -> Optional[str]:
    if article.published:
        return article.published.date().isoformat()
    return None
