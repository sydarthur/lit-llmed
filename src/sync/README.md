# Obsidian-Zotero Sync Module

Bidirectional synchronization between Obsidian literature notes and Zotero library.

## Features

- **Parse Obsidian Notes**: Extract metadata (DOI, citation keys, Zotero URIs) from markdown files
- **Sync to Zotero**: Push notes and annotations from Obsidian back to Zotero items
- **Auto-Watch**: Monitor Obsidian vault for changes and auto-sync
- **Smart Matching**: Find Zotero items by DOI, Zotero key, or title
- **Dry Run Mode**: Preview sync operations without making changes

## Quick Start

### 1. Configure Zotero Credentials

Ensure your `config/zotero.json` has valid credentials:
```json
{
  "api_key": "your-zotero-api-key",
  "user_id": "12345678",
  "library_type": "user"
}
```

### 2. Test Sync

Run the test script to verify everything works:
```bash
.venv/bin/python test_obsidian_sync.py
```

### 3. Sync Your Vault

```bash
# Dry run (preview only)
.venv/bin/python -m src.cli sync vault /path/to/obsidian/vault --folder Labs/Literature --dry-run

# Actual sync
.venv/bin/python -m src.cli sync vault /path/to/obsidian/vault --folder Labs/Literature

# Watch for changes and auto-sync
.venv/bin/python -m src.cli sync vault /path/to/obsidian/vault --folder Labs/Literature --watch
```

### 4. Sync Single File

```bash
.venv/bin/python -m src.cli sync file /path/to/note.md
```

## Obsidian Note Format

The parser recognizes several metadata formats:

### Option 1: Frontmatter (Recommended)
```yaml
---
type: literature
citekey: smith2024telemedicine
doi: 10.1234/example.2024.001
zotero_key: ABCD1234
---
```

### Option 2: Metadata Section
```markdown
## Metadata
**DOI**:: [10.1234/example.2024.001](https://doi.org/10.1234/example.2024.001)
**Authors**:: Smith, J.; Jones, A.
```

### Option 3: Inline Links
```markdown
[Zotero](zotero://select/library/items/ABCD1234)
DOI: 10.1234/example.2024.001
```

## Notes Sections

The sync module looks for notes under these headers:
- `## Notes`
- `## My Notes`
- `## Annotations`
- `## Critical Notes`

Everything under these headers (until the next major heading) will be synced to Zotero.

### Example Note with Annotations

```markdown
---
citekey: leeTelemedicineReimbursementsHospital2025
---

# On Telemedicine Reimbursements and Hospital Adoption

## Metadata
**DOI**:: [10.1002/joom.70021](https://doi.org/10.1002/joom.70021)

## Abstract
There continue to be substantial differences in the quality and access...

## Critical Notes

**Strengths**:
- Rigorous causal inference approach using difference-in-differences
- Natural experiment from phased policy rollout

**Limitations**:
- Limited to 2014-2019 timeframe (pre-pandemic)
- Self-reported adoption data

**Key Insight:** The differential effect on inpatient vs outpatient services suggests that reimbursement alone is insufficient.

## Annotations & Highlights

This paper is crucial for understanding policy-driven technology adoption. The institutional isomorphism lens helps explain why hospitals adopt telemedicine even when economic incentives are unclear.
```

## How It Works

### 1. Note Parsing
The `ObsidianNoteParser` extracts:
- DOI (from frontmatter, metadata section, or inline)
- Zotero item key (from frontmatter or URI)
- Citation key
- Notes/annotations sections

### 2. Zotero Matching
The `ZoteroNoteSyncer` finds items using:
1. Explicit `zotero_key` from metadata
2. Key extracted from `zotero://` URI
3. Search by DOI (exact match)
4. Search by title (fuzzy match)

### 3. Note Syncing
- Converts markdown to HTML
- Creates or updates note attached to Zotero item
- Marks notes with sync timestamp
- Preserves existing Zotero metadata

### 4. Auto-Watch (Optional)
The `SyncManager` can monitor your vault:
- Watches for file modifications and creations
- Debounces rapid changes (2-second delay)
- Auto-syncs when notes are saved

## Architecture

```
src/sync/
├── __init__.py                 # Module exports
├── obsidian_parser.py          # Parse Obsidian markdown
├── zotero_sync.py             # Sync to Zotero API
├── sync_manager.py            # Orchestrate sync operations
├── EXAMPLE_NOTE.md            # Example note format
└── README.md                  # This file
```

## API Reference

### ObsidianNoteParser

```python
from src.sync import ObsidianNoteParser

parser = ObsidianNoteParser()
note = parser.parse_file(Path("note.md"))

# Access parsed data
print(note.doi)           # DOI if found
print(note.zotero_key)    # Zotero item key if found
print(note.notes_section) # Extracted notes content
```

### ZoteroNoteSyncer

```python
from src.sync import ZoteroNoteSyncer

syncer = ZoteroNoteSyncer(
    api_key="your-key",
    user_id="12345678",
    library_type="user"
)

success = syncer.sync_note(note, dry_run=False)
```

### SyncManager

```python
from src.sync import SyncManager
from pathlib import Path

manager = SyncManager(
    vault_path=Path("/path/to/vault"),
    api_key="your-key",
    user_id="12345678",
    dry_run=False
)

# Sync entire vault
results = manager.sync_vault(folder="Labs/Literature")

# Sync single file
manager.sync_file(Path("/path/to/note.md"))

# Watch for changes
manager.start_watch(folder="Labs/Literature")
```

## CLI Commands

```bash
# Sync entire vault or folder
python -m src.cli sync vault <vault-path> [--folder FOLDER] [--dry-run] [--watch]

# Sync single file
python -m src.cli sync file <file-path> [--dry-run]
```

## Configuration

### Zotero API Permissions

Your Zotero API key needs:
- ✅ Read library access
- ✅ Write library access (to create/update notes)

### Vault Structure

Works with any Obsidian vault structure. Common patterns:
```
Vault/
├── Literature/           # All literature notes
├── Projects/
│   └── Research/
│       └── Literature/   # Project-specific notes
└── Zettelkasten/
    └── Literature/
```

## Troubleshooting

### "No Zotero metadata" - Skipped

The note doesn't have any identifiable Zotero metadata (DOI, Zotero key, or URI). Add one of:
- `doi: 10.xxxx/xxxx` in frontmatter
- `**DOI**:: [10.xxxx/xxxx]` in metadata section
- Zotero link: `[Zotero](zotero://select/library/items/ABCD1234)`

### "No notes to sync" - Skipped

The note doesn't have a recognized notes section. Add content under:
- `## Notes`
- `## My Notes`
- `## Annotations`
- `## Critical Notes`

### "Zotero item not found"

The parser found metadata but couldn't locate the item in Zotero. Causes:
- DOI mismatch (check formatting)
- Item not in your library
- Wrong library type (user vs group)

### Connection Issues

Check your Zotero credentials:
```bash
.venv/bin/python test_zotero.py
```

## Advanced Usage

### Custom Note Formats

Extend `ObsidianNoteParser` to recognize custom patterns:

```python
from src.sync import ObsidianNoteParser
import re

class CustomParser(ObsidianNoteParser):
    # Add custom DOI pattern
    CUSTOM_DOI = re.compile(r"Reference:\s*doi:([^\n]+)")

    def parse_content(self, content, file_path):
        note = super().parse_content(content, file_path)

        # Try custom pattern if DOI not found
        if not note.doi:
            match = self.CUSTOM_DOI.search(content)
            if match:
                note.doi = match.group(1).strip()

        return note
```

### Selective Sync

Only sync notes with specific tags:

```python
from pathlib import Path
from src.sync import SyncManager, ObsidianNoteParser

manager = SyncManager(...)
parser = ObsidianNoteParser()

notes_to_sync = []
for md_file in Path("vault").rglob("*.md"):
    note = parser.parse_file(md_file)
    if note and "sync-to-zotero" in note.metadata.get("tags", ""):
        notes_to_sync.append(md_file)

results = manager.sync_specific_notes(notes_to_sync)
```

## Development

Run tests:
```bash
.venv/bin/python test_obsidian_sync.py
```

Enable debug logging:
```bash
python -m src.cli sync vault /path --log-level DEBUG
```

## License

Part of lit-llmed project.
