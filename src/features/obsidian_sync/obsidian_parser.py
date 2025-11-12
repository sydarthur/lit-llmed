"""Parser for Obsidian markdown notes with Zotero metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.core.log import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ObsidianNote:
    """Parsed Obsidian note with metadata and content."""

    file_path: Path
    title: str
    doi: Optional[str] = None
    zotero_key: Optional[str] = None
    zotero_uri: Optional[str] = None
    citation_key: Optional[str] = None
    content: str = ""
    notes_section: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ObsidianNoteParser:
    """Parse Obsidian markdown files to extract Zotero metadata and notes."""

    # Common frontmatter patterns for Zotero metadata
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)
    DOI_PATTERN = re.compile(r"doi:\s*([^\n]+)", re.IGNORECASE)
    ZOTERO_KEY_PATTERN = re.compile(r"zotero[-_]?key:\s*([^\n]+)", re.IGNORECASE)
    ZOTERO_URI_PATTERN = re.compile(r"zotero[-_]?uri:\s*(zotero://[^\n]+)", re.IGNORECASE)
    CITATION_KEY_PATTERN = re.compile(r"(?:citekey|citation[-_]?key):\s*([^\n]+)", re.IGNORECASE)

    # Metadata section patterns (for DOI in body)
    METADATA_DOI_PATTERN = re.compile(r'\*\*DOI\*\*::\s*\[([^\]]+)\]', re.IGNORECASE)

    # Inline metadata patterns (for notes without frontmatter)
    INLINE_DOI_PATTERN = re.compile(r"DOI:\s*\[?([^\]\n]+)\]?", re.IGNORECASE)
    INLINE_ZOTERO_LINK = re.compile(r"\[Zotero\]\((zotero://[^\)]+)\)", re.IGNORECASE)

    # Notes section patterns
    NOTES_HEADER_PATTERN = re.compile(
        r"(?:^|\n)##?\s*(?:Critical Notes?|Notes?|My Notes?|Annotations?|Comments?|Annotations & Highlights)\s*\n",
        re.IGNORECASE
    )

    def parse_file(self, file_path: Path) -> Optional[ObsidianNote]:
        """Parse an Obsidian markdown file and extract metadata."""
        if not file_path.exists() or file_path.suffix != ".md":
            LOGGER.warning("invalid_file", extra={"path": str(file_path)})
            return None

        try:
            content = file_path.read_text(encoding="utf-8")
            return self.parse_content(content, file_path)
        except Exception as exc:
            LOGGER.error("parse_failed", extra={"path": str(file_path), "error": str(exc)})
            return None

    def parse_content(self, content: str, file_path: Path) -> ObsidianNote:
        """Parse markdown content and extract structured data."""
        title = file_path.stem
        doi = None
        zotero_key = None
        zotero_uri = None
        citation_key = None
        metadata = {}
        notes_section = ""

        # Extract frontmatter if present
        frontmatter_match = self.FRONTMATTER_PATTERN.match(content)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            metadata = self._parse_frontmatter(frontmatter)
            content_body = content[frontmatter_match.end() :]

            # Extract from frontmatter
            doi = metadata.get("doi")
            zotero_key = metadata.get("zotero_key") or metadata.get("zoterokey") or metadata.get("zoteroKey")
            zotero_uri = metadata.get("zotero_uri") or metadata.get("zoteroUri")
            citation_key = metadata.get("citekey") or metadata.get("citation_key")
            title = metadata.get("title", title)
        else:
            content_body = content

        # Fallback: search for inline metadata if not in frontmatter
        if not doi:
            # Try metadata section DOI pattern first
            doi_match = self.METADATA_DOI_PATTERN.search(content)
            if doi_match:
                doi = doi_match.group(1).strip()
            else:
                # Try other DOI patterns
                doi_match = self.DOI_PATTERN.search(content) or self.INLINE_DOI_PATTERN.search(content)
                if doi_match:
                    doi = doi_match.group(1).strip()

            # Clean DOI URL if present
            if doi and 'doi.org/' in doi:
                doi = doi.split('doi.org/')[-1].rstrip(')')

        if not zotero_key:
            key_match = self.ZOTERO_KEY_PATTERN.search(content)
            if key_match:
                zotero_key = key_match.group(1).strip()

        if not zotero_uri:
            uri_match = self.ZOTERO_URI_PATTERN.search(content) or self.INLINE_ZOTERO_LINK.search(
                content
            )
            if uri_match:
                zotero_uri = uri_match.group(1).strip()

        if not citation_key:
            cite_match = self.CITATION_KEY_PATTERN.search(content)
            if cite_match:
                citation_key = cite_match.group(1).strip()

        # Extract notes section
        notes_section = self._extract_notes_section(content_body)

        note = ObsidianNote(
            file_path=file_path,
            title=title,
            doi=doi,
            zotero_key=zotero_key,
            zotero_uri=zotero_uri,
            citation_key=citation_key,
            content=content,
            notes_section=notes_section,
            metadata=metadata,
        )

        LOGGER.info(
            "note_parsed",
            extra={
                "path": str(file_path),
                "doi": doi,
                "zotero_key": zotero_key,
                "has_notes": bool(notes_section),
            },
        )
        return note

    def _parse_frontmatter(self, frontmatter: str) -> dict:
        """Parse YAML-style frontmatter into a dictionary."""
        metadata = {}
        for line in frontmatter.split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace("-", "_")
                value = value.strip().strip('"').strip("'")
                metadata[key] = value
        return metadata

    def _extract_notes_section(self, content: str) -> str:
        """Extract the notes/annotations section from the content."""
        match = self.NOTES_HEADER_PATTERN.search(content)
        if not match:
            return ""

        # Get content after the notes header
        start_pos = match.end()
        rest = content[start_pos:]

        # Find the next major heading (# or ##) to determine end of notes
        next_header = re.search(r"\n##?\s+[^\n]+\n", rest)
        if next_header:
            notes = rest[: next_header.start()].strip()
        else:
            notes = rest.strip()

        return notes

    def extract_zotero_key_from_uri(self, zotero_uri: str) -> Optional[str]:
        """Extract item key from a Zotero URI.

        Example: zotero://select/library/items/ABCD1234 -> ABCD1234
        """
        if not zotero_uri:
            return None

        match = re.search(r"/items/([A-Z0-9]+)", zotero_uri)
        if match:
            return match.group(1)
        return None
