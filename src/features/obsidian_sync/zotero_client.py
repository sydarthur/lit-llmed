"""Integration helpers for interacting with the Zotero API."""

from __future__ import annotations

from typing import Iterable, List, Optional

import requests

from src.core.log import get_logger
from src.core.models import Article, Author

LOGGER = get_logger(__name__)


class ZoteroClient:
    """Thin wrapper around the Zotero REST API with pragmatic defaults."""

    def __init__(self, api_key: str, user_id: str, library_type: str = "user"):
        self.api_key = api_key
        self.user_id = user_id
        self.library_type = library_type
        self.base_url = f"https://api.zotero.org/{library_type}s/{user_id}"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Zotero-API-Key": api_key,
                "User-Agent": "lit-llmed/0.1",
            }
        )

    def test_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/items?limit=1", timeout=10)
            return response.status_code == 200
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            LOGGER.error("zotero_connection_failed", error=str(exc))
            return False

    def get_or_create_collection(self, name: str) -> Optional[str]:
        """Get or create a collection, supporting nested paths like 'Parent/Child'."""
        # Handle nested collections
        if "/" in name:
            parts = name.split("/")
            parent_key = None

            # Create/find each level of the hierarchy
            for i, part in enumerate(parts):
                full_path = "/".join(parts[:i+1])
                key = self._find_collection_by_path(full_path, parent_key)
                if not key:
                    key = self._create_collection(part, parent_key)
                    if not key:
                        return None
                parent_key = key

            return parent_key
        else:
            # Simple collection (no nesting)
            key = self._find_collection(name)
            if key:
                return key
            return self._create_collection(name)

    def import_articles(self, articles: Iterable[Article], collection_key: Optional[str] = None) -> bool:
        payload = []
        for article in articles:
            item = self._to_zotero_item(article)
            if not item:
                continue
            if collection_key:
                item["collections"] = [collection_key]
            payload.append(item)
        if not payload:
            LOGGER.warning("zotero_import_skipped", reason="no_payload")
            return False
        try:
            response = self.session.post(
                f"{self.base_url}/items",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
        except requests.RequestException as exc:  # pragma: no cover
            LOGGER.error("zotero_import_failed", error=str(exc))
            return False
        if response.status_code != 200:
            LOGGER.error("zotero_import_error", status=response.status_code, body=response.text)
            return False
        result = response.json()
        success = len(result.get("successful", {}))
        failures = len(result.get("failed", {}))
        if failures:
            LOGGER.warning("zotero_partial_failure", failed=failures, successful=success)
        return success > 0

    def _find_collection(self, name: str) -> Optional[str]:
        """Find a collection by exact name match (no parent filtering)."""
        try:
            response = self.session.get(f"{self.base_url}/collections", timeout=30)
            response.raise_for_status()
            for collection in response.json():
                if collection.get("data", {}).get("name") == name:
                    return collection.get("key")
        except requests.RequestException as exc:  # pragma: no cover
            LOGGER.error("zotero_collection_lookup_failed", error=str(exc))
        return None

    def _find_collection_by_path(self, full_path: str, expected_parent_key: Optional[str]) -> Optional[str]:
        """Find a collection by name and parent key."""
        try:
            response = self.session.get(f"{self.base_url}/collections", timeout=30)
            response.raise_for_status()
            # Extract the last part of the path (the actual collection name)
            name = full_path.split("/")[-1]

            for collection in response.json():
                data = collection.get("data", {})
                coll_name = data.get("name")
                parent_coll = data.get("parentCollection")

                # Match by name and parent
                if coll_name == name:
                    if expected_parent_key is None and not parent_coll:
                        return collection.get("key")
                    elif expected_parent_key and parent_coll == expected_parent_key:
                        return collection.get("key")
        except requests.RequestException as exc:  # pragma: no cover
            LOGGER.error("zotero_collection_lookup_failed", error=str(exc))
        return None

    def _create_collection(self, name: str, parent_key: Optional[str] = None) -> Optional[str]:
        """Create a collection, optionally under a parent collection."""
        try:
            payload = {
                "name": name,
                "parentCollection": parent_key if parent_key else False
            }
            response = self.session.post(
                f"{self.base_url}/collections",
                json=[payload],
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:  # pragma: no cover
            LOGGER.error("zotero_collection_create_failed", error=str(exc))
            return None
        successful = result.get("successful", {})
        if "0" in successful:
            return successful["0"].get("key")
        LOGGER.error("zotero_collection_create_missing_key", response=result)
        return None

    @staticmethod
    def _to_zotero_item(article: Article) -> Optional[dict]:
        if not article.title or article.title == "Issue Information":
            return None
        creators: List[dict] = []
        for author in article.authors:
            creators.append(_to_creator(author))
        item = {
            "itemType": "journalArticle",
            "title": article.title,
            "creators": creators,
            "publicationTitle": article.journal,
            "volume": article.volume,
            "issue": article.issue,
            "pages": article.pages,
            "date": article.published.date().isoformat() if article.published else None,
            "DOI": article.doi,
            "ISSN": article.issn,
            "url": (article.open_access.url if article.open_access else article.url),
            "abstractNote": article.abstract,
        }
        return {key: value for key, value in item.items() if value}


def _to_creator(author: Author) -> dict:
    parts = author.name.split()
    if len(parts) >= 2:
        return {
            "creatorType": "author",
            "firstName": " ".join(parts[:-1]),
            "lastName": parts[-1],
        }
    return {"creatorType": "author", "name": author.name}
