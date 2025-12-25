"""Crossref ingestion client and helpers."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from src.core.log import get_logger
from src.core.models import Article, Author, Journal, OpenAccess

LOGGER = get_logger(__name__)


class CrossrefClient:
    """Fetch journal articles from Crossref with optional enrichments."""

    CROSSREF_URL = "https://api.crossref.org/journals/{issn}/works"
    OPENALEX_URL = "https://api.openalex.org/works/https://doi.org/{doi}"
    UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"

    def __init__(self, email: str, rate_limit_delay: float = 1.0, session: Optional[requests.Session] = None):
        self.email = email
        self.rate_limit_delay = rate_limit_delay
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", f"lit-llmed/0.1 (mailto:{email})")

    def fetch_latest(
        self,
        journal: Journal,
        *,
        rows: Optional[int] = None,
        days_back: Optional[int] = None,
        include_abstracts: Optional[bool] = None,
        include_open_access: Optional[bool] = None,
    ) -> List[Article]:
        """Fetch and optionally enrich the latest articles for a journal."""

        rows = rows or journal.max_articles_per_fetch
        days_back = days_back or journal.days_back
        include_abstracts = journal.fetch_abstracts if include_abstracts is None else include_abstracts
        include_open_access = journal.fetch_oa_links if include_open_access is None else include_open_access

        raw_items = self._fetch_crossref_works(
            filter_str=f"issn:{journal.issn},type:journal-article,from-pub-date:{self._get_since_date(days_back)}",
            rows=rows,
        )
        return self._process_items(raw_items, journal, include_abstracts, include_open_access)

    def fetch_by_date_range(
        self,
        journal: Journal,
        start_date: str,
        end_date: str,
        *,
        rows: int = 100,
        include_abstracts: bool = False,
        include_open_access: bool = True,
    ) -> List[Article]:
        """Fetch articles for a specific date range (YYYY-MM-DD)."""
        raw_items = self._fetch_crossref_works(
            filter_str=f"issn:{journal.issn},type:journal-article,from-pub-date:{start_date},until-pub-date:{end_date}",
            rows=rows,
        )
        return self._process_items(raw_items, journal, include_abstracts, include_open_access)

    def _get_since_date(self, days_back: int) -> str:
        return (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    def _fetch_crossref_works(
        self,
        filter_str: str,
        rows: int,
        extra_params: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generic fetcher for Crossref works."""
        params = {
            "filter": filter_str,
            "sort": "published",
            "order": "desc",
            "rows": rows,
        }
        if extra_params:
            params.update(extra_params)
            
        # If ISSN is not involved, the URL is just /works
        # But wait, our base URL has {issn}. We need to adjust.
        # If we are searching generally (e.g. author), we shouldn't target a specific journal endpoint.
        # But _crossref_latest was using self.CROSSREF_URL which had {issn}.
        
        # Let's fix the URL handling.
        base_url = "https://api.crossref.org/works"
        
        try:
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            payload = response.json()
            return payload.get("message", {}).get("items", [])
        except requests.RequestException as exc:
            LOGGER.error("crossref_fetch_failed", error=str(exc))
            return []

    def _process_items(
        self,
        raw_items: List[Dict],
        journal: Journal,
        include_abstracts: bool,
        include_open_access: bool,
    ) -> List[Article]:
        articles: List[Article] = []
        for item in raw_items:
            # For author search, the ISSN might be in the item
            # We can try to extract the primary ISSN from the item if available
            item_journal = journal
            if journal.issn == "N/A" and "ISSN" in item:
                # Create a specific journal context if we discovered one
                issn_list = item.get("ISSN", [])
                primary_issn = issn_list[0] if issn_list else "Unknown"
                container_title = item.get("container-title", [])
                name = container_title[0] if container_title else "Unknown Journal"
                item_journal = Journal(name=name, issn=primary_issn, active=True)

            article = self._parse_crossref_item(item, item_journal)
            if not article:
                continue
            if include_abstracts and not article.abstract and article.doi:
                abstract = self._openalex_abstract(article.doi)
                if abstract:
                    article.abstract = abstract
            if include_open_access and article.doi:
                oa_url = self._unpaywall_oa_link(article.doi)
                if oa_url:
                    article.open_access = OpenAccess(url=oa_url)
            articles.append(article)
        LOGGER.info("fetched_articles", count=len(articles), context=journal.name)
        return articles

    def _openalex_abstract(self, doi: str) -> Optional[str]:
        url = self.OPENALEX_URL.format(doi=doi)
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:  # pragma: no cover
            LOGGER.warning("openalex_failed", doi=doi, error=str(exc))
            return None
        inverted = payload.get("abstract_inverted_index") or {}
        if not inverted:
            return None
        words: List[Optional[str]] = []
        for token, positions in inverted.items():
            for position in positions:
                while len(words) <= position:
                    words.append(None)
                words[position] = token
        resolved = " ".join(word for word in words if word)
        time.sleep(self.rate_limit_delay)
        return resolved or None

    def _unpaywall_oa_link(self, doi: str) -> Optional[str]:
        url = self.UNPAYWALL_URL.format(doi=doi)
        try:
            response = self.session.get(url, params={"email": self.email}, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:  # pragma: no cover
            LOGGER.warning("unpaywall_failed", doi=doi, error=str(exc))
            return None
        location = payload.get("best_oa_location") or {}
        time.sleep(self.rate_limit_delay)
        return location.get("url_for_pdf") or location.get("url")

    @staticmethod
    def _parse_crossref_item(item: Dict, journal: Journal) -> Optional[Article]:
        title_list = item.get("title") or []
        title = title_list[0] if title_list else ""
        if not title:
            return None

        authors: List[Author] = []
        for author in item.get("author", []):
            authors.append(Author.from_parts(author.get("given"), author.get("family")))

        published_date = _extract_publication_date(item)
        article = Article(
            title=title,
            doi=item.get("DOI"),
            url=item.get("URL"),
            journal=journal.name,
            issn=journal.issn,
            published=published_date,
            volume=item.get("volume"),
            issue=item.get("issue"),
            pages=item.get("page"),
            abstract=item.get("abstract"),
            keywords=item.get("subject") or [],
            citation_count=item.get("is-referenced-by-count"),
            authors=authors,
        )
        return article


def _extract_publication_date(item: Dict) -> Optional[datetime]:
    for key in ("published-print", "published-online", "issued"):
        container = item.get(key)
        if not container:
            continue
        parts = container.get("date-parts")
        if not parts:
            continue
        date_parts = parts[0]
        if not date_parts:
            continue
        year = date_parts[0]
        month = date_parts[1] if len(date_parts) > 1 else 1
        day = date_parts[2] if len(date_parts) > 2 else 1
        try:
            return datetime(year=int(year), month=int(month), day=int(day))
        except ValueError:
            continue
    return None
