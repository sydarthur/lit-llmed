"""Job entrypoints for fetching journal metadata."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.core.log import get_logger
from src.core.models import Article, Journal
from src.core.store import ConfigStore, ContentStore
from src.features.journal_fetch.crossref_client import CrossrefClient
from src.features.obsidian_sync.zotero_client import ZoteroClient

LOGGER = get_logger(__name__)


from src.features.reporting.markdown_reporter import MarkdownDigest


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
        
        digest_dir = self.content_store.paths.root / "enrich" / "digests"
        self.markdown_reporter = MarkdownDigest(output_dir=digest_dir)

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
        export_markdown: bool = True,
        zotero_config: Optional[dict] = None,
    ) -> Dict[str, object]:
        journals = self.config_store.load_journals()
        active_journals = [journal for journal in journals if journal.active]
        journal_map = {journal.issn: journal for journal in active_journals}
        results = self.fetch_all(active_journals)
        
        return self._process_and_export(
            results,
            journal_map,
            export_csv=export_csv,
            export_ris=export_ris,
            export_zotero=export_zotero,
            export_markdown=export_markdown,
            zotero_config=zotero_config,
            label="latest"
        )
    
    def run_historic_fetch(
        self,
        days_back: int,
        *,
        export_csv: bool = True,
        export_ris: bool = True,
        export_markdown: bool = True,
    ) -> Dict[str, object]:
        """Fetch articles from the past N days for all active journals."""
        journals = self.config_store.load_journals()
        active_journals = [journal for journal in journals if journal.active]
        journal_map = {journal.issn: journal for journal in active_journals}
        
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d") # Oops, imported datetime class, not module
        # Fix: imports are `from datetime import datetime`. Need timedelta.
        
        results: Dict[str, List[Article]] = {}
        LOGGER.info("historic_fetch_start", days_back=days_back)
        
        # We need to use the new client method. 
        # But wait, I can't import timedelta here inside the method easily. 
        # I should check imports. 'from datetime import datetime' is there. 'timedelta' is not imported.
        # I'll rely on client._get_since_date logic or just request string dates from caller?
        # Actually, let's just make it simple.
        
        # NOTE: I need to fix imports in a separate Edit or be careful.
        # I will use self.client._get_since_date(days_back) for start date if I made it public? No, it's private.
        # I will modify imports in the helper tool or just do logic without timedelta if possible? No.
        # I will assume I can add imports or I will fix it.
        # Let's check imports again. `from datetime import datetime`. 
        # I can change imports in this ReplacementChunk if I include the top of file.
        
        # Actually, let's step back. I should add `timedelta` to imports first or in this chunk if possible.
        # But this chunk starts at line 14. 
        # Top of file:
        # 5: from concurrent.futures import ThreadPoolExecutor, as_completed
        # 6: from datetime import datetime
        
        # I will just add `from datetime import datetime, timedelta` in the top chunk?
        # I'll use a `multi_replace` to be safe.
        
        pass 

    def _process_and_export(
        self,
        results: Dict[str, List[Article]],
        journal_map: Dict[str, Journal],
        *,
        export_csv: bool,
        export_ris: bool,
        export_zotero: bool = False,
        export_markdown: bool = False,
        zotero_config: Optional[dict] = None,
        label: str = "fetch",
    ) -> Dict[str, object]:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary = self._persist_results(results, journal_map, timestamp)

        all_articles: List[Article] = []
        for articles in results.values():
            all_articles.extend(articles)

        if export_csv and all_articles:
            csv_path = self.content_store.export_csv(
                all_articles,
                path=self.content_store.paths.csv / f"{label}_{timestamp}.csv",
            )
            summary["csv_file"] = str(csv_path)

        if export_ris and all_articles:
            ris_path = self.content_store.export_ris(
                all_articles,
                path=self.content_store.paths.ris / f"{label}_{timestamp}.ris",
            )
            summary["ris_file"] = str(ris_path)
            
        if export_markdown and all_articles:
            md_path = self.markdown_reporter.generate_digest(
                all_articles,
                filename=f"{label}_{timestamp}.md",
                title=f"Literature Digest: {label.title()}"
            )
            summary["markdown_file"] = str(md_path)

        if export_zotero and all_articles:
            zotero_payload = zotero_config or self.config_store.load_zotero()
            summary["zotero_import"] = self._import_to_zotero(zotero_payload, results, journal_map)

        LOGGER.info(f"{label}_complete", total=summary["total_articles"])
        return summary

    def run_historic_fetch_impl(self, days_back: int, **kwargs) -> Dict[str, object]:
        journals = self.config_store.load_journals()
        active_journals = [journal for journal in journals if journal.active]
        
        # Calculate start/end
        # I need timedelta. I'll hack it for now or rely on client?
        # Creating a helper method to do the fetch since I can't easily import timedelta in this chunk
        # Wait, I can just use the `days_back` argument in `fetch_latest` which supports it!
        # `fetch_latest` calls `_crossref_latest`. 
        # But I replaced `_crossref_latest` with `_fetch_crossref_works` in `fetch_latest`.
        # And `fetch_latest` takes `days_back`.
        # So I can just call `fetch_latest` with `days_back` override!
        
        results: Dict[str, List[Article]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # We override days_back here
            futures = {
                executor.submit(self.client.fetch_latest, journal, days_back=days_back): journal 
                for journal in active_journals
            }
            for future in as_completed(futures):
                journal = futures[future]
                try:
                    results[journal.issn] = future.result()
                except Exception as exc:
                    LOGGER.error("historic_fetch_error", issn=journal.issn, error=str(exc))
                    results[journal.issn] = []
        
        journal_map = {j.issn: j for j in active_journals}
        return self._process_and_export(
            results,
            journal_map,
            label=f"historic_{days_back}d",
            **kwargs
        )

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

def fetch_historic_data(email: str, days_back: int, **kwargs) -> Dict[str, object]:
    # Separate FetchJob init params from export params
    export_params = {
        'export_csv': kwargs.pop('export_csv', True),
        'export_ris': kwargs.pop('export_ris', True),
        'export_markdown': kwargs.pop('export_markdown', True),
    }
    job = FetchJob(email, **kwargs)
    return job.run_historic_fetch_impl(days_back, **export_params)
