"""Storage utilities for persisting configuration and article data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[assignment, import-not-found]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

from .models import Article, Journal, JournalStore, OpenAccess


class PathRegistry:
    """Resolve canonical output paths and provide convenience helpers."""

    def __init__(self, root: Path):
        self.root = root
        self.data = root / "data"
        self.exports = root / "exports"
        self.ris = self.exports / "ris"
        self.csv = self.exports / "csv"
        self.files = root / "files" / "pdf"
        self.cache = root / "cache"
        self.logs = root / "logs"

    def ensure(self) -> None:
        for path in (self.data, self.ris, self.csv, self.files, self.cache, self.logs):
            path.mkdir(parents=True, exist_ok=True)


class ConfigStore:
    """Load configuration from the repository config directory."""

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def journals_path(self) -> Path:
        return self.config_dir / "journals.json"

    def zotero_path(self) -> Path:
        return self.config_dir / "zotero.json"

    def settings_path(self) -> Path:
        return self.config_dir / "settings.toml"

    def load_journals(self) -> List[Journal]:
        path = self.journals_path()
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        store = JournalStore(**data)
        return store.journals

    def save_journals(self, journals: Iterable[Journal]) -> None:
        store = JournalStore(journals=list(journals))
        payload = json.dumps(
            json.loads(store.model_dump_json()),
            indent=2,
            ensure_ascii=False,
        )
        self.journals_path().write_text(payload, encoding="utf-8")

    def load_settings(self) -> dict:
        path = self.settings_path()
        if not path.exists() or tomllib is None:
            return {}
        with path.open("rb") as stream:
            return tomllib.load(stream)

    def load_zotero(self) -> Optional[dict]:
        path = self.zotero_path()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


class ContentStore:
    """Persist and export articles in multiple formats with stable hashing."""

    def __init__(self, output_root: Path = Path("output")):
        self.paths = PathRegistry(output_root)
        self.paths.ensure()

    @staticmethod
    def _hash_payload(articles: Iterable[Article]) -> str:
        digest = hashlib.sha256()
        for article in articles:
            digest.update(article.model_dump_json(sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def write_json(
        self,
        articles: List[Article],
        *,
        filename: Optional[str] = None,
        use_content_hash: bool = True,
    ) -> Path:
        if not filename:
            base = self._hash_payload(articles) if use_content_hash else "articles"
            filename = f"{base}.json"
        path = self.paths.data / filename
        payload = [json.loads(article.model_dump_json(by_alias=True)) for article in articles]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_json(self, path: Path) -> List[Article]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [Article.model_validate(obj) for obj in payload]

    def export_csv(self, articles: Iterable[Article], path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.paths.csv / "articles.csv"
        fieldnames = [
            "title",
            "doi",
            "authors",
            "published",
            "journal",
            "issn",
            "abstract",
            "open_access_url",
            "url",
            "volume",
            "issue",
            "pages",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for article in articles:
                open_access_url = None
                if isinstance(article.open_access, OpenAccess):
                    open_access_url = article.open_access.url
                writer.writerow(
                    {
                        "title": article.title,
                        "doi": article.doi or "",
                        "authors": "; ".join(author.name for author in article.authors),
                        "published": article.published.isoformat() if article.published else "",
                        "journal": article.journal or "",
                        "issn": article.issn or "",
                        "abstract": article.abstract or "",
                        "open_access_url": open_access_url or "",
                        "url": article.url or "",
                        "volume": article.volume or "",
                        "issue": article.issue or "",
                        "pages": article.pages or "",
                    }
                )
        return path

    def export_ris(self, articles: Iterable[Article], path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.paths.ris / "articles.ris"
        lines: List[str] = []
        for article in articles:
            entry = self._ris_entry(article)
            if not entry:
                continue
            lines.extend(entry)
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def _clean_text(value: Optional[str]) -> str:
        if not value:
            return ""
        replacements = {
            "<jats:title>ABSTRACT</jats:title>": "",
            "<jats:p>": "",
            "</jats:p>": "",
            "<scp>": "",
            "</scp>": "",
            "<i>": "",
            "</i>": "",
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
        return " ".join(value.split()).strip()

    def _ris_entry(self, article: Article) -> List[str]:
        if not article.title or article.title == "Issue Information":
            return []
        ris: List[str] = ["TY  - JOUR"]
        ris.append(f"TI  - {article.title}")
        for author in article.authors:
            ris.append(f"AU  - {author.name}")
        if article.journal:
            ris.append(f"JO  - {article.journal}")
            ris.append(f"T2  - {article.journal}")
        if article.published:
            ris.append(f"PY  - {article.published.year}")
            ris.append(f"DA  - {article.published.date().isoformat()}")
        if article.volume:
            ris.append(f"VL  - {article.volume}")
        if article.issue:
            ris.append(f"IS  - {article.issue}")
        if article.pages:
            ris.append(f"SP  - {article.pages}")
        if article.issn:
            ris.append(f"SN  - {article.issn}")
        if article.doi:
            ris.append(f"DO  - {article.doi}")
        if article.url:
            ris.append(f"UR  - {article.url}")
        if article.open_access and article.open_access.url:
            if article.open_access.url != article.url:
                ris.append(f"L1  - {article.open_access.url}")
        abstract = self._clean_text(article.abstract)
        if abstract:
            ris.append(f"AB  - {abstract}")
        for keyword in article.keywords:
            ris.append(f"KW  - {keyword}")
        ris.append("M1  - lit-llmed import")
        ris.append("ER  - ")
        return ris
