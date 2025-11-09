#!/usr/bin/env python3
"""Test script for Obsidian-Zotero synchronization."""

import json
from pathlib import Path

from src.core.store import ConfigStore
from src.sync import ObsidianNoteParser, SyncManager, ZoteroNoteSyncer

# Example usage paths
EXAMPLE_NOTE = Path("src/sync/EXAMPLE_NOTE.md")
VAULT_PATH = Path("/Users/syamalakonda/Documents/-nalanda")
LITERATURE_FOLDER = "Labs/Literature"


def test_parser():
    """Test parsing an Obsidian note."""
    print("=" * 70)
    print("  TEST 1: PARSING OBSIDIAN NOTE")
    print("=" * 70)

    parser = ObsidianNoteParser()

    if EXAMPLE_NOTE.exists():
        print(f"\n📄 Parsing: {EXAMPLE_NOTE}")
        note = parser.parse_file(EXAMPLE_NOTE)

        if note:
            print(f"✅ Parsed successfully!")
            print(f"   Title: {note.title}")
            print(f"   DOI: {note.doi}")
            print(f"   Citekey: {note.citation_key}")
            print(f"   Zotero Key: {note.zotero_key}")
            print(f"   Has Notes: {bool(note.notes_section)}")
            if note.notes_section:
                print(f"   Notes Length: {len(note.notes_section)} chars")
                print(f"\n   Notes Preview:")
                print("   " + "-" * 66)
                preview = note.notes_section[:200]
                print(f"   {preview}...")
                print("   " + "-" * 66)
        else:
            print("❌ Failed to parse note")
    else:
        print(f"⚠️  Example note not found at {EXAMPLE_NOTE}")

    print("\n")


def test_vault_scan():
    """Test scanning a real vault for notes."""
    print("=" * 70)
    print("  TEST 2: SCANNING VAULT FOR LITERATURE NOTES")
    print("=" * 70)

    if not VAULT_PATH.exists():
        print(f"⚠️  Vault not found at {VAULT_PATH}")
        print("   Update VAULT_PATH in this script to point to your Obsidian vault")
        return

    lit_path = VAULT_PATH / LITERATURE_FOLDER
    if not lit_path.exists():
        print(f"⚠️  Literature folder not found at {lit_path}")
        return

    print(f"\n📁 Scanning: {lit_path}")
    md_files = list(lit_path.glob("*.md"))
    print(f"   Found {len(md_files)} markdown files")

    parser = ObsidianNoteParser()
    notes_with_doi = []
    notes_with_notes = []

    print("\n   Parsing notes...")
    for md_file in md_files[:5]:  # Sample first 5
        note = parser.parse_file(md_file)
        if note:
            print(f"   ✓ {md_file.name}")
            if note.doi:
                notes_with_doi.append(note)
            if note.notes_section:
                notes_with_notes.append(note)

    print(f"\n📊 Results:")
    print(f"   Notes with DOI: {len(notes_with_doi)}")
    print(f"   Notes with annotations: {len(notes_with_notes)}")
    print(f"   Ready to sync: {len(notes_with_notes)}")

    print("\n")


def test_sync_dry_run():
    """Test sync with dry run (no actual Zotero updates)."""
    print("=" * 70)
    print("  TEST 3: DRY RUN SYNC TEST")
    print("=" * 70)

    # Load Zotero config
    config_path = Path("config/zotero.json")
    if not config_path.exists():
        print("❌ config/zotero.json not found")
        return

    with open(config_path) as f:
        config = json.load(f)

    if config.get("api_key") == "YOUR_ZOTERO_API_KEY":
        print("❌ Please configure config/zotero.json first")
        return

    if not VAULT_PATH.exists():
        print(f"⚠️  Vault not found at {VAULT_PATH}")
        return

    print(f"\n🔄 Initializing sync manager (DRY RUN mode)...")
    print(f"   Vault: {VAULT_PATH}")
    print(f"   Zotero User: {config['user_id']}")

    manager = SyncManager(
        vault_path=VAULT_PATH,
        api_key=config["api_key"],
        user_id=config["user_id"],
        library_type=config.get("library_type", "user"),
        dry_run=True,  # Don't actually update Zotero
    )

    print(f"\n📝 Testing sync for Literature folder...")
    results = manager.sync_vault(folder=LITERATURE_FOLDER)

    print(f"\n📊 Sync Results (DRY RUN):")
    print(f"   Total files: {results['total']}")
    print(f"   Would sync: {results['synced']}")
    print(f"   Skipped: {results['skipped']}")
    print(f"   Failed: {results['failed']}")

    print("\n✅ Dry run complete! No changes made to Zotero.")
    print("   To actually sync, set dry_run=False")

    print("\n")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "OBSIDIAN-ZOTERO SYNC TESTS" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    test_parser()
    test_vault_scan()
    test_sync_dry_run()

    print("=" * 70)
    print("  ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Review the sync results above")
    print("   2. If everything looks good, run a real sync:")
    print("      python -m src.cli sync-obsidian --vault /path/to/vault")
    print("\n")


if __name__ == "__main__":
    main()
