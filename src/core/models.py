"""Core data models for literature ingestion and enrichment."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Author(BaseModel):
    """Represents an article author."""

    name: str = Field(..., description="Display-friendly author name")
    given: Optional[str] = Field(None, description="Given name from upstream API")
    family: Optional[str] = Field(None, description="Family name from upstream API")

    @classmethod
    def from_parts(cls, given: Optional[str], family: Optional[str]) -> "Author":
        """Create an author from Crossref-style components."""
        parts = [p for p in (given, family) if p]
        name = " ".join(parts) if parts else (family or given or "Unknown Author")
        return cls(name=name, given=given or None, family=family or None)


class OpenAccess(BaseModel):
    """Metadata describing open-access availability of an article."""

    url: Optional[str] = Field(None, description="Landing page or PDF URL")
    license: Optional[str] = Field(None, description="License string if supplied")
    version: Optional[str] = Field(None, description="Publisher version information")


class Enrichment(BaseModel):
    """Placeholder for downstream LLM or rules-based enrichment."""

    tags: List[str] = Field(default_factory=list, description="Normalized topical tags")
    summary: Optional[str] = Field(None, description="Short machine generated summary")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score for enrichment payload"
    )


class Article(BaseModel):
    """Canonical article representation returned by ingestion pipelines."""

    title: str
    doi: Optional[str] = None
    url: Optional[str] = None
    journal: Optional[str] = Field(None, alias="journal_name")
    issn: Optional[str] = None
    published: Optional[datetime] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    citation_count: Optional[int] = Field(None, ge=0)

    authors: List[Author] = Field(default_factory=list)
    open_access: Optional[OpenAccess] = None
    enrichment: Optional[Enrichment] = None

    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp when the record was produced"
    )

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class Journal(BaseModel):
    """Journal configuration describing a source to ingest from."""

    name: str
    issn: str
    publisher: Optional[str] = None
    subject_area: Optional[str] = None
    zotero_collection: Optional[str] = None
    fetch_abstracts: bool = True
    fetch_oa_links: bool = True
    max_articles_per_fetch: int = Field(50, ge=1, le=1000)
    days_back: int = Field(30, ge=1, le=365)
    active: bool = True


class JournalStore(BaseModel):
    """Wrapper model for persisted journal configuration files."""

    journals: List[Journal] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
