import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Article:
    title: str
    doi: str
    authors: List[str]
    published_date: str
    journal_name: str
    issn: str
    abstract: Optional[str] = None
    keywords: List[str] = None
    open_access_url: Optional[str] = None
    citation_count: Optional[int] = None
    url: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

class LiteratureFetcher:
    def __init__(self, email: str, rate_limit_delay: float = 1.0):
        self.email = email
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        
    def crossref_latest_articles(self, issn: str, rows: int = 50, days_back: int = 30) -> List[Dict]:
        """Fetch latest articles from CrossRef API with date filtering."""
        url = f"https://api.crossref.org/journals/{issn}/works"
        
        # Calculate date filter for recent articles
        since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            "filter": f"type:journal-article,from-pub-date:{since_date}",
            "sort": "published",
            "order": "desc",
            "rows": rows
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return response.json()["message"]["items"]
        except requests.RequestException as e:
            logger.error(f"Error fetching articles for ISSN {issn}: {e}")
            return []

    def openalex_abstract(self, doi: str) -> Optional[str]:
        """Fetch abstract from OpenAlex API."""
        try:
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            inv = data.get("abstract_inverted_index") or {}
            if not inv:
                return None
                
            # Reconstruct plaintext abstract
            words = []
            for word, positions in inv.items():
                for pos in positions:
                    while len(words) <= pos: 
                        words.append("")
                    words[pos] = word
                    
            txt = " ".join(w for w in words if w)
            time.sleep(self.rate_limit_delay)
            return txt or None
        except requests.RequestException as e:
            logger.warning(f"Could not fetch abstract for DOI {doi}: {e}")
            return None

    def unpaywall_oa_link(self, doi: str) -> Optional[str]:
        """Fetch open access link from Unpaywall API."""
        try:
            url = f"https://api.unpaywall.org/v2/{doi}"
            params = {"email": self.email}
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            loc = data.get("best_oa_location") or {}
            time.sleep(self.rate_limit_delay)
            return loc.get("url_for_pdf") or loc.get("url")
        except requests.RequestException as e:
            logger.warning(f"Could not fetch OA link for DOI {doi}: {e}")
            return None

    def parse_crossref_article(self, item: Dict, issn: str) -> Article:
        """Parse a CrossRef article item into an Article object."""
        # Extract title
        title = ""
        if "title" in item and item["title"]:
            title = item["title"][0]
            
        # Extract authors
        authors = []
        if "author" in item:
            for author in item["author"]:
                given = author.get("given", "")
                family = author.get("family", "")
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
        
        # Extract publication date
        published_date = ""
        if "published-print" in item and "date-parts" in item["published-print"]:
            date_parts = item["published-print"]["date-parts"][0]
            if len(date_parts) >= 3:
                published_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
            elif len(date_parts) >= 2:
                published_date = f"{date_parts[0]}-{date_parts[1]:02d}"
            elif len(date_parts) >= 1:
                published_date = str(date_parts[0])
        elif "published-online" in item and "date-parts" in item["published-online"]:
            date_parts = item["published-online"]["date-parts"][0]
            if len(date_parts) >= 3:
                published_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
        
        # Extract other metadata
        journal_name = item.get("container-title", [""])[0] if item.get("container-title") else ""
        doi = item.get("DOI", "")
        url = item.get("URL", "")
        volume = item.get("volume", "")
        issue = item.get("issue", "")
        pages = item.get("page", "")
        
        # Extract abstract if available
        abstract = None
        if "abstract" in item:
            abstract = item["abstract"]
        
        return Article(
            title=title,
            doi=doi,
            authors=authors,
            published_date=published_date,
            journal_name=journal_name,
            issn=issn,
            abstract=abstract,
            url=url,
            volume=volume,
            issue=issue,
            pages=pages
        )

    def fetch_journal_articles(self, issn: str, rows: int = 50, days_back: int = 30, 
                             fetch_abstracts: bool = True, fetch_oa_links: bool = True) -> List[Article]:
        """Fetch and enrich articles for a single journal."""
        logger.info(f"Fetching articles for journal ISSN: {issn}")
        
        # Get articles from CrossRef
        raw_articles = self.crossref_latest_articles(issn, rows, days_back)
        articles = []
        
        for item in raw_articles:
            try:
                article = self.parse_crossref_article(item, issn)
                
                # Enrich with abstract if needed and not already present
                if fetch_abstracts and not article.abstract and article.doi:
                    article.abstract = self.openalex_abstract(article.doi)
                
                # Enrich with open access link
                if fetch_oa_links and article.doi:
                    article.open_access_url = self.unpaywall_oa_link(article.doi)
                
                articles.append(article)
                logger.info(f"Processed: {article.title[:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(articles)} articles for ISSN {issn}")
        return articles

    def save_articles_json(self, articles: List[Article], filename: str):
        """Save articles to JSON file."""
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        articles_dict = [asdict(article) for article in articles]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} articles to {filename}")

    def load_articles_json(self, filename: str) -> List[Article]:
        """Load articles from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            articles_dict = json.load(f)
        
        articles = []
        for item in articles_dict:
            articles.append(Article(**item))
        
        logger.info(f"Loaded {len(articles)} articles from {filename}")
        return articles