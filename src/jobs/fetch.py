"""Job entrypoints for fetching journal metadata."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.core.log import get_logger
from src.core.models import Article, Journal
from src.core.store import ConfigStore, ContentStore
from src.ingest.crossref import CrossrefClient
from src.integrate.zotero import ZoteroClient

LOGGER = get_logger(__name__)


def _slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


class FetchJob:
    """Coordinates fetching, persistence, and downstream exports."""

    def __init__(
        self,
        email: str,
        *,
        rate_limit_delay: float = 1.0,
        max_workers: int = 5,
        config_store: Optional[ConfigStore] = None,
        content_store: Optional[ContentStore] = None,
    ):
        self.email = email
        self.rate_limit_delay = rate_limit_delay
        self.max_workers = max_workers
        self.config_store = config_store or ConfigStore()
        self.content_store = content_store or ContentStore()
        self.client = CrossrefClient(email, rate_limit_delay=rate_limit_delay)

    def fetch_single(self, journal: Journal) -> List[Article]:
        LOGGER.info("fetch_single_start", issn=journal.issn)
        articles = self.client.fetch_latest(journal)
        LOGGER.info("fetch_single_complete", issn=journal.issn, count=len(articles))
        return articles

    def fetch_all(self, journals: Iterable[Journal]) -> Dict[str, List[Article]]:
        journals = [journal for journal in journals if journal.active]
        results: Dict[str, List[Article]] = {}
        LOGGER.info("fetch_all_start", count=len(journals))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetch_single, journal): journal for journal in journals}
            for future in as_completed(futures):
                journal = futures[future]
                try:
                    results[journal.issn] = future.result()
                except Exception as exc:  # pragma: no cover - defensive guardrail
                    LOGGER.error("fetch_all_error", issn=journal.issn, error=str(exc))
                    results[journal.issn] = []
        LOGGER.info("fetch_all_complete", journals=len(results))
        return results

    def run_full_fetch(
        self,
        *,
        export_csv: bool = True,
        export_ris: bool = True,
        export_zotero: bool = False,
        zotero_config: Optional[dict] = None,
    ) -> Dict[str, object]:
        journals = self.config_store.load_journals()
        active_journals = [journal for journal in journals if journal.active]
        journal_map = {journal.issn: journal for journal in active_journals}
        results = self.fetch_all(active_journals)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        summary = self._persist_results(results, journal_map, timestamp)

        all_articles: List[Article] = []
        for articles in results.values():
            all_articles.extend(articles)

        if export_csv and all_articles:
            csv_path = self.content_store.export_csv(
                all_articles,
                path=self.content_store.paths.csv / f"all_journals_{timestamp}.csv",
            )
            summary["csv_file"] = str(csv_path)

        if export_ris and all_articles:
            ris_path = self.content_store.export_ris(
                all_articles,
                path=self.content_store.paths.ris / f"all_journals_{timestamp}.ris",
            )
            summary["ris_file"] = str(ris_path)

        if export_zotero and all_articles:
            zotero_payload = zotero_config or self.config_store.load_zotero()
            summary["zotero_import"] = self._import_to_zotero(zotero_payload, results, journal_map)

        LOGGER.info("run_full_fetch_complete", total=summary["total_articles"])
        return summary

    def _persist_results(
        self,
        results: Dict[str, List[Article]],
        journal_map: Dict[str, Journal],
        timestamp: str,
    ) -> Dict[str, object]:
        summary = {
            "fetch_timestamp": timestamp,
            "total_articles": 0,
            "journals_fetched": len(results),
            "articles_per_journal": {},
        }
        for issn, articles in results.items():
            journal = journal_map.get(issn)
            name = journal.name if journal else issn
            summary["total_articles"] += len(articles)
            summary["articles_per_journal"][name] = len(articles)
            if not articles:
                continue
            filename = f"{_slugify(name)}_{timestamp}.json"
            self.content_store.write_json(articles, filename=filename, use_content_hash=False)
        combined_filename = f"all_journals_{timestamp}.json"
        self.content_store.write_json(
            [article for articles in results.values() for article in articles],
            filename=combined_filename,
            use_content_hash=False,
        )
        return summary

    def _import_to_zotero(
        self,
        config: Optional[dict],
        results: Dict[str, List[Article]],
        journal_map: Dict[str, Journal],
    ) -> str:
        if not config:
            LOGGER.warning("zotero_skipped", reason="missing_config")
            return "not configured"
        try:
            client = ZoteroClient(**config)
        except TypeError as exc:
            LOGGER.error("zotero_config_invalid", error=str(exc))
            return f"invalid config: {exc}"
        imported_collections = []
        for issn, articles in results.items():
            if not articles:
                continue
            journal = journal_map.get(issn)
            name = journal.name if journal else issn
            collection_name = (journal.zotero_collection if journal else None) or f"{name} Articles"
            key = client.get_or_create_collection(collection_name)
            if not key:
                LOGGER.error("zotero_collection_failed", journal=name)
                continue
            if client.import_articles(articles, key):
                imported_collections.append(collection_name)
        status = f"imported {len(imported_collections)} collections"
        LOGGER.info("zotero_import_summary", status=status)
        return status


def fetch_single_journal(email: str, issn: str) -> List[Article]:
    job = FetchJob(email)
    journals = job.config_store.load_journals()
    target = next((journal for journal in journals if journal.issn == issn), None)
    if not target:
        raise ValueError(f"Journal with ISSN {issn} not found in configuration")
    return job.fetch_single(target)


def fetch_all_journals(email: str, **kwargs) -> Dict[str, object]:
    job = FetchJob(email, **kwargs)
    return job.run_full_fetch()
