"""Placeholder job for syncing existing content into Zotero."""

from __future__ import annotations

from typing import Optional

from src.core.store import ConfigStore, ContentStore
from src.integrate.zotero import ZoteroClient


def sync_existing_exports(
    *,
    config_store: Optional[ConfigStore] = None,
    content_store: Optional[ContentStore] = None,
) -> str:
    """Push already-exported JSON payloads to Zotero collections.

    This placeholder demonstrates where future synchronization logic would live.
    """

    config_store = config_store or ConfigStore()
    content_store = content_store or ContentStore()
    zotero_config = config_store.load_zotero()
    if not zotero_config:
        return "zotero not configured"
    client = ZoteroClient(**zotero_config)
    if not client.test_connection():
        return "failed to connect to zotero"
    # Future implementation: enumerate content_store.paths.data for JSON files and import.
    return f"ready to sync from {content_store.paths.data}"
