"""Zotero note synchronization client."""

from __future__ import annotations

from typing import List, Optional

import requests

from src.core.log import get_logger
from src.sync.obsidian_parser import ObsidianNote

LOGGER = get_logger(__name__)


class ZoteroNoteSyncer:
    """Sync notes from Obsidian back to Zotero items."""

    def __init__(self, api_key: str, user_id: str, library_type: str = "user"):
        """Initialize Zotero sync client.

        Args:
            api_key: Zotero API key
            user_id: Zotero user ID
            library_type: 'user' or 'group'
        """
        self.api_key = api_key
        self.user_id = user_id
        self.library_type = library_type
        self.base_url = f"https://api.zotero.org/{library_type}s/{user_id}"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Zotero-API-Key": api_key,
                "User-Agent": "lit-llmed-sync/0.1",
            }
        )

    def sync_note(self, obsidian_note: ObsidianNote, dry_run: bool = False) -> bool:
        """Sync an Obsidian note to Zotero.

        Args:
            obsidian_note: Parsed Obsidian note with metadata
            dry_run: If True, don't actually update Zotero

        Returns:
            True if sync succeeded, False otherwise
        """
        if not obsidian_note.notes_section:
            LOGGER.info("no_notes_to_sync", extra={"path": str(obsidian_note.file_path)})
            return False

        # Try to find the Zotero item
        item_key = self._find_item_key(obsidian_note)
        if not item_key:
            LOGGER.warning(
                "zotero_item_not_found",
                extra={
                    "path": str(obsidian_note.file_path),
                    "doi": obsidian_note.doi,
                    "zotero_key": obsidian_note.zotero_key,
                },
            )
            return False

        if dry_run:
            LOGGER.info(
                "sync_dry_run",
                extra={
                    "item_key": item_key,
                    "notes_length": len(obsidian_note.notes_section),
                },
            )
            return True

        # Update or create note in Zotero
        return self._update_item_note(item_key, obsidian_note.notes_section)

    def _find_item_key(self, obsidian_note: ObsidianNote) -> Optional[str]:
        """Find the Zotero item key for an Obsidian note.

        Tries in order:
        1. Explicit zotero_key from metadata
        2. Extract from zotero_uri
        3. Search by DOI
        4. Search by title
        """
        # 1. Check explicit key
        if obsidian_note.zotero_key:
            return obsidian_note.zotero_key

        # 2. Extract from URI
        if obsidian_note.zotero_uri:
            from src.sync.obsidian_parser import ObsidianNoteParser

            parser = ObsidianNoteParser()
            key = parser.extract_zotero_key_from_uri(obsidian_note.zotero_uri)
            if key:
                return key

        # 3. Search by DOI
        if obsidian_note.doi:
            key = self._search_by_doi(obsidian_note.doi)
            if key:
                return key

        # 4. Search by title
        return self._search_by_title(obsidian_note.title)

    def _search_by_doi(self, doi: str) -> Optional[str]:
        """Search for item by DOI."""
        try:
            # Clean DOI
            doi = doi.strip().replace("https://doi.org/", "").replace("http://doi.org/", "")

            params = {"q": doi, "qmode": "everything", "limit": 5}
            response = self.session.get(f"{self.base_url}/items", params=params, timeout=30)
            response.raise_for_status()

            items = response.json()
            for item in items:
                item_doi = item.get("data", {}).get("DOI", "")
                if item_doi and item_doi.lower() == doi.lower():
                    key = item.get("key")
                    LOGGER.info("found_by_doi", extra={"doi": doi, "key": key})
                    return key
        except requests.RequestException as exc:
            LOGGER.error("search_by_doi_failed", extra={"doi": doi, "error": str(exc)})

        return None

    def _search_by_title(self, title: str) -> Optional[str]:
        """Search for item by title."""
        try:
            params = {"q": title, "qmode": "titleCreatorYear", "limit": 3}
            response = self.session.get(f"{self.base_url}/items", params=params, timeout=30)
            response.raise_for_status()

            items = response.json()
            if items:
                # Return first match (best effort)
                key = items[0].get("key")
                item_title = items[0].get("data", {}).get("title", "")
                LOGGER.info("found_by_title", extra={"title": title, "key": key, "matched_title": item_title})
                return key
        except requests.RequestException as exc:
            LOGGER.error("search_by_title_failed", extra={"title": title, "error": str(exc)})

        return None

    def _update_item_note(self, item_key: str, note_content: str) -> bool:
        """Update or create a note for a Zotero item.

        Args:
            item_key: Zotero item key
            note_content: Note content in HTML or plain text

        Returns:
            True if update succeeded
        """
        try:
            # Get existing notes for this item
            response = self.session.get(
                f"{self.base_url}/items/{item_key}/children", timeout=30
            )
            response.raise_for_status()
            children = response.json()

            # Find existing note created by lit-llmed
            existing_note_key = None
            for child in children:
                if child.get("data", {}).get("itemType") == "note":
                    note_text = child.get("data", {}).get("note", "")
                    if "lit-llmed-sync" in note_text or "Obsidian Sync" in note_text:
                        existing_note_key = child.get("key")
                        break

            # Format note content
            formatted_note = self._format_note_html(note_content)

            if existing_note_key:
                # Update existing note
                return self._update_note(existing_note_key, formatted_note)
            else:
                # Create new note
                return self._create_note(item_key, formatted_note)

        except requests.RequestException as exc:
            LOGGER.error("update_item_note_failed", extra={"item_key": item_key, "error": str(exc)})
            return False

    def _create_note(self, parent_key: str, note_html: str) -> bool:
        """Create a new note attached to an item."""
        try:
            payload = [
                {
                    "itemType": "note",
                    "parentItem": parent_key,
                    "note": note_html,
                }
            ]
            response = self.session.post(
                f"{self.base_url}/items",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            success = len(result.get("successful", {})) > 0
            if success:
                LOGGER.info("note_created", extra={"parent_key": parent_key})
            else:
                LOGGER.error("note_create_failed", extra={"parent_key": parent_key, "response": result})
            return success
        except requests.RequestException as exc:
            LOGGER.error("create_note_failed", extra={"parent_key": parent_key, "error": str(exc)})
            return False

    def _update_note(self, note_key: str, note_html: str) -> bool:
        """Update an existing note."""
        try:
            # Get current note version
            response = self.session.get(f"{self.base_url}/items/{note_key}", timeout=30)
            response.raise_for_status()
            current_note = response.json()
            version = current_note.get("version")

            # Update note
            current_note["data"]["note"] = note_html

            response = self.session.put(
                f"{self.base_url}/items/{note_key}",
                json=current_note["data"],
                headers={
                    "Content-Type": "application/json",
                    "If-Unmodified-Since-Version": str(version),
                },
                timeout=30,
            )
            response.raise_for_status()
            LOGGER.info("note_updated", extra={"note_key": note_key})
            return True
        except requests.RequestException as exc:
            LOGGER.error("update_note_failed", extra={"note_key": note_key, "error": str(exc)})
            return False

    def _format_note_html(self, markdown_content: str) -> str:
        """Convert markdown note to HTML for Zotero.

        Args:
            markdown_content: Markdown formatted note content

        Returns:
            HTML formatted note
        """
        # Convert markdown to HTML (basic conversion)
        html = markdown_content

        # Convert markdown formatting to HTML
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)  # Bold
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)  # Italic
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)  # List items
        html = html.replace("\n\n", "</p><p>")  # Paragraphs

        # Wrap in paragraphs
        if not html.startswith("<"):
            html = f"<p>{html}</p>"

        # Add sync marker
        timestamp = __import__("datetime").datetime.utcnow().isoformat()
        header = f'<p><em>Synced from Obsidian via lit-llmed-sync on {timestamp}</em></p><hr/>'

        return header + html


# Import regex for HTML formatting
import re
