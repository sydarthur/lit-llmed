"""GCP-compatible literature fetching without structured logging dependencies."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

from src.core.models import Article, Author, Journal, OpenAccess
from src.gcp.simple_logger import get_logger

logger = get_logger(__name__)


class GCPCrossrefClient:
    """Crossref client compatible with GCP environment."""

    CROSSREF_URL = "https://api.crossref.org/journals/{issn}/works"
    OPENALEX_URL = "https://api.openalex.org/works/https://doi.org/{doi}"
    UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"

    def __init__(self, email: str, rate_limit_delay: float = 1.0):
        self.email = email
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.setdefault("User-Agent", f"lit-llmed-gcp/1.0 (mailto:{email})")

    def fetch_latest(self, journal: Journal) -> List[Article]:
        """Fetch latest articles for a journal."""
        try:
            logger.info(f"Fetching articles for journal: {journal.name}")
            
            raw_items = self._crossref_latest(
                journal.issn, 
                rows=journal.max_articles_per_fetch,
                days_back=journal.days_back
            )
            
            articles: List[Article] = []
            for item in raw_items:
                article = self._parse_crossref_item(item, journal)
                if not article:
                    continue
                    
                if journal.fetch_abstracts and not article.abstract and article.doi:
                    abstract = self._openalex_abstract(article.doi)
                    if abstract:
                        article.abstract = abstract
                        
                if journal.fetch_oa_links and article.doi:
                    oa_url = self._unpaywall_oa_link(article.doi)
                    if oa_url:
                        article.open_access = OpenAccess(url=oa_url)
                        
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles for {journal.issn}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch articles for {journal.issn}: {str(e)}")
            return []

    def _crossref_latest(self, issn: str, rows: int, days_back: int) -> List[Dict]:
        """Fetch latest articles from Crossref."""
        try:
            since_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            params = {
                "filter": f"type:journal-article,from-pub-date:{since_date}",
                "sort": "published",
                "order": "desc",
                "rows": rows,
            }
            url = self.CROSSREF_URL.format(issn=issn)
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            
            payload = response.json()
            items = payload.get("message", {}).get("items", [])
            logger.info(f"Crossref returned {len(items)} items for {issn}")
            return items
            
        except requests.RequestException as e:
            logger.error(f"Crossref fetch failed for {issn}: {str(e)}")
            return []

    def _openalex_abstract(self, doi: str) -> Optional[str]:
        """Get abstract from OpenAlex."""
        try:
            url = self.OPENALEX_URL.format(doi=doi)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            
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
            
        except requests.RequestException as e:
            logger.warning(f"OpenAlex failed for {doi}: {str(e)}")
            return None

    def _unpaywall_oa_link(self, doi: str) -> Optional[str]:
        """Get open access link from Unpaywall."""
        try:
            url = self.UNPAYWALL_URL.format(doi=doi)
            response = self.session.get(url, params={"email": self.email}, timeout=30)
            response.raise_for_status()
            payload = response.json()
            
            location = payload.get("best_oa_location") or {}
            time.sleep(self.rate_limit_delay)
            return location.get("url_for_pdf") or location.get("url")
            
        except requests.RequestException as e:
            logger.warning(f"Unpaywall failed for {doi}: {str(e)}")
            return None

    @staticmethod
    def _parse_crossref_item(item: Dict, journal: Journal) -> Optional[Article]:
        """Parse Crossref item into Article model."""
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
    """Extract publication date from Crossref item."""
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