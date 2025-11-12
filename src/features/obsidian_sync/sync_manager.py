"""Sync manager to orchestrate Obsidian-Zotero synchronization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.core.log import get_logger
from src.features.obsidian_sync.obsidian_parser import ObsidianNoteParser
from src.features.obsidian_sync.zotero_sync import ZoteroNoteSyncer

LOGGER = get_logger(__name__)


class ObsidianChangeHandler(FileSystemEventHandler):
    """Handle file system changes in Obsidian vault."""

    def __init__(self, sync_manager: SyncManager):
        self.sync_manager = sync_manager
        self.pending_files = set()
        self.last_sync_time = {}

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if event.is_directory or not event.src_path.endswith(".md"):
            return

        file_path = Path(event.src_path)

        # Debounce: only sync if file hasn't been modified in last 2 seconds
        current_time = time.time()
        last_time = self.last_sync_time.get(file_path, 0)
        if current_time - last_time < 2:
            return

        self.last_sync_time[file_path] = current_time
        LOGGER.info("file_modified", extra={"path": str(file_path)})
        self.sync_manager.sync_file(file_path)

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if event.is_directory or not event.src_path.endswith(".md"):
            return

        file_path = Path(event.src_path)
        LOGGER.info("file_created", extra={"path": str(file_path)})
        # Wait a bit for file to be fully written
        time.sleep(0.5)
        self.sync_manager.sync_file(file_path)


class SyncManager:
    """Manage synchronization between Obsidian vault and Zotero library."""

    def __init__(
        self,
        vault_path: Path,
        api_key: str,
        user_id: str,
        library_type: str = "user",
        auto_sync: bool = False,
        dry_run: bool = False,
    ):
        """Initialize sync manager.

        Args:
            vault_path: Path to Obsidian vault
            api_key: Zotero API key
            user_id: Zotero user ID
            library_type: 'user' or 'group'
            auto_sync: If True, watch for file changes and auto-sync
            dry_run: If True, don't actually update Zotero
        """
        self.vault_path = Path(vault_path)
        self.auto_sync = auto_sync
        self.dry_run = dry_run

        self.parser = ObsidianNoteParser()
        self.syncer = ZoteroNoteSyncer(api_key, user_id, library_type)
        self.observer: Optional[Observer] = None

        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

    def sync_file(self, file_path: Path) -> bool:
        """Sync a single Obsidian note to Zotero.

        Args:
            file_path: Path to Obsidian markdown file

        Returns:
            True if sync succeeded
        """
        LOGGER.info("sync_file_start", extra={"path": str(file_path), "dry_run": self.dry_run})

        # Parse note
        note = self.parser.parse_file(file_path)
        if not note:
            LOGGER.warning("parse_failed", extra={"path": str(file_path)})
            return False

        # Check if note has Zotero metadata
        if not any([note.doi, note.zotero_key, note.zotero_uri]):
            LOGGER.info("no_zotero_metadata", extra={"path": str(file_path)})
            return False

        # Sync to Zotero
        success = self.syncer.sync_note(note, dry_run=self.dry_run)

        if success:
            LOGGER.info("sync_success", extra={"path": str(file_path)})
        else:
            LOGGER.warning("sync_failed", extra={"path": str(file_path)})

        return success

    def sync_vault(self, folder: Optional[str] = None) -> dict:
        """Sync all notes in vault (or specific folder) to Zotero.

        Args:
            folder: Optional subfolder to sync (relative to vault_path)

        Returns:
            Summary dict with sync statistics
        """
        if folder:
            search_path = self.vault_path / folder
        else:
            search_path = self.vault_path

        if not search_path.exists():
            raise ValueError(f"Path does not exist: {search_path}")

        LOGGER.info("sync_vault_start", extra={"path": str(search_path), "dry_run": self.dry_run})

        # Find all markdown files
        md_files = list(search_path.rglob("*.md"))
        LOGGER.info("found_markdown_files", extra={"count": len(md_files)})

        # Sync each file
        results = {
            "total": len(md_files),
            "synced": 0,
            "skipped": 0,
            "failed": 0,
        }

        for md_file in md_files:
            try:
                success = self.sync_file(md_file)
                if success:
                    results["synced"] += 1
                else:
                    results["skipped"] += 1
            except Exception as exc:
                LOGGER.error("sync_exception", extra={"path": str(md_file), "error": str(exc)})
                results["failed"] += 1

        LOGGER.info("sync_vault_complete", extra=results)
        return results

    def start_watch(self, folder: Optional[str] = None):
        """Start watching vault for changes and auto-sync.

        Args:
            folder: Optional subfolder to watch (relative to vault_path)
        """
        if not self.auto_sync:
            LOGGER.warning("auto_sync_disabled", extra={})
            return

        if folder:
            watch_path = self.vault_path / folder
        else:
            watch_path = self.vault_path

        if not watch_path.exists():
            raise ValueError(f"Path does not exist: {watch_path}")

        LOGGER.info("starting_watch", extra={"path": str(watch_path), "dry_run": self.dry_run})

        event_handler = ObsidianChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(watch_path), recursive=True)
        self.observer.start()

        LOGGER.info("watch_started", extra={"path": str(watch_path)})

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_watch()

    def stop_watch(self):
        """Stop watching vault."""
        if self.observer:
            LOGGER.info("stopping_watch", extra={})
            self.observer.stop()
            self.observer.join()
            self.observer = None
            LOGGER.info("watch_stopped", extra={})

    def sync_specific_notes(self, file_paths: List[Path]) -> dict:
        """Sync a specific list of note files.

        Args:
            file_paths: List of paths to Obsidian markdown files

        Returns:
            Summary dict with sync statistics
        """
        LOGGER.info("sync_specific_start", extra={"count": len(file_paths), "dry_run": self.dry_run})

        results = {
            "total": len(file_paths),
            "synced": 0,
            "skipped": 0,
            "failed": 0,
        }

        for file_path in file_paths:
            try:
                success = self.sync_file(file_path)
                if success:
                    results["synced"] += 1
                else:
                    results["skipped"] += 1
            except Exception as exc:
                LOGGER.error("sync_exception", extra={"path": str(file_path), "error": str(exc)})
                results["failed"] += 1

        LOGGER.info("sync_specific_complete", extra=results)
        return results
