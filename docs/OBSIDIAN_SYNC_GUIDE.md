# Obsidian-Zotero Sync Setup Guide

Complete guide to syncing your Obsidian literature notes back to Zotero.

## What This Does

Syncs your notes and annotations from Obsidian markdown files back to the corresponding items in your Zotero library. This allows you to:

- Take notes in Obsidian (better writing experience)
- Have those notes appear in Zotero (for reference management)
- Keep everything synchronized automatically

## Quick Start

### 1. Install & Setup

```bash
# Activate virtual environment
source .venv/bin/activate  # or: .venv/bin/activate on Unix

# Verify Zotero connection
python test_zotero.py
```

### 2. Test Sync (Dry Run)

```bash
# Test without making changes
python -m src.cli sync vault /Users/syamalakonda/Documents/-nalanda \
  --folder Labs/Literature \
  --dry-run
```

### 3. Run Actual Sync

```bash
# Sync your literature notes
python -m src.cli sync vault /Users/syamalakonda/Documents/-nalanda \
  --folder Labs/Literature
```

### 4. Enable Auto-Sync (Optional)

```bash
# Watch for changes and auto-sync
python -m src.cli sync vault /Users/syamalakonda/Documents/-nalanda \
  --folder Labs/Literature \
  --watch
```

Press Ctrl+C to stop watching.

## Your Current Setup

Based on your vault structure:

- **Vault Path**: `/Users/syamalakonda/Documents/-nalanda`
- **Literature Folder**: `Labs/Literature`
- **Zotero User**: `14208742`
- **Current Notes**: 1 note found (`leeTelemedicineReimbursementsHospital2025.md`)

## Adding Notes Sections to Sync

Your note currently doesn't have a notes section. To add one:

### Option 1: Use Existing Sections

Add content under any of these headers:
```markdown
## Critical Notes

**Strengths**:
- Your thoughts here

**Limitations**:
- Your analysis here
```

### Option 2: Add New Section

```markdown
## Notes

My key takeaways from this paper:
- Important finding #1
- Methodological insight #2
- Connection to my research #3
```

### Option 3: Add Annotations Section

```markdown
## Annotations & Highlights

%% begin annotations %%

**Reading note 2025-11-08:**
This paper is crucial for understanding policy-driven technology adoption.

**Key insight:**
The institutional isomorphism lens explains technology adoption patterns.

%% end annotations %%
```

## Example Workflow

1. **Read paper in Zotero**
2. **Create/open Obsidian note** (auto-populated with metadata)
3. **Add your notes** under `## Critical Notes` or `## Annotations`
4. **Save the file**
5. **Sync to Zotero**:
   ```bash
   python -m src.cli sync file /Users/syamalakonda/Documents/-nalanda/Labs/Literature/your-note.md
   ```
6. **Notes appear in Zotero** as a child note attached to the article

## Commands Reference

### Test Connection
```bash
python test_zotero.py
```

### Test Sync System
```bash
python test_obsidian_sync.py
```

### Sync Entire Folder (Dry Run)
```bash
python -m src.cli sync vault /Users/syamalakonda/Documents/-nalanda \
  --folder Labs/Literature \
  --dry-run
```

### Sync Entire Folder (Actual)
```bash
python -m src.cli sync vault /Users/syamalakonda/Documents/-nalanda \
  --folder Labs/Literature
```

### Sync Single File
```bash
python -m src.cli sync file /Users/syamalakonda/Documents/-nalanda/Labs/Literature/yourfile.md
```

### Auto-Watch Mode
```bash
python -m src.cli sync vault /Users/syamalakonda/Documents/-nalanda \
  --folder Labs/Literature \
  --watch
```

## CLI Help

```bash
# See all sync commands
python -m src.cli sync --help

# Help for vault sync
python -m src.cli sync vault --help

# Help for file sync
python -m src.cli sync file --help
```

## What Gets Synced

The sync module extracts and syncs:

1. **Notes from these sections**:
   - `## Notes`
   - `## My Notes`
   - `## Annotations`
   - `## Critical Notes`
   - `## Annotations & Highlights`

2. **Everything under the header** until the next major heading (## or #)

3. **Markdown formatting** is converted to HTML for Zotero

## How Items Are Matched

The syncer finds your Zotero items by:

1. **DOI** (most reliable) - from frontmatter or metadata section
2. **Zotero Key** - from frontmatter or Zotero URI links
3. **Title** - fuzzy match as fallback

Your notes use DOI matching, which is the most reliable method.

## Verification

After syncing, check Zotero:

1. Open Zotero desktop app
2. Find the article (search by DOI or title)
3. Look for a child note titled "Synced from Obsidian..."
4. Your notes will be there!

## Troubleshooting

### "Skipped: No notes to sync"
→ Add content under a recognized notes header (see above)

### "Skipped: No Zotero metadata"
→ Ensure your note has DOI in metadata section:
```markdown
**DOI**:: [10.1002/joom.70021](https://doi.org/10.1002/joom.70021)
```

### "Failed to find Zotero item"
→ Check that the item exists in your Zotero library and DOI matches exactly

### Connection issues
→ Run `python test_zotero.py` to verify credentials

## Pro Tips

1. **Use dry-run first** to preview what will be synced
2. **Enable auto-watch** when actively taking notes
3. **Check sync logs** (JSON output) for detailed status
4. **Verify in Zotero** after first sync to ensure notes appear correctly

## Advanced Configuration

### Custom Note Headers

Edit `src/sync/obsidian_parser.py` line 53 to add custom headers:
```python
NOTES_HEADER_PATTERN = re.compile(
    r"(?:^|\n)##?\s*(?:Notes?|My Notes?|Annotations?|Comments?|Your Custom Header)\s*\n",
    re.IGNORECASE
)
```

### Filter by Tags

Only sync notes with specific tags - see [src/sync/README.md](src/sync/README.md) for examples.

## Next Steps

1. ✅ Tested Zotero connection
2. ✅ Created sync module
3. ✅ Tested sync (dry run)
4. 🔲 Add notes to your Obsidian files
5. 🔲 Run real sync
6. 🔲 Verify notes appear in Zotero
7. 🔲 (Optional) Set up auto-watch for continuous sync

## Files Created

- `src/sync/` - Sync module code
  - `obsidian_parser.py` - Parse Obsidian notes
  - `zotero_sync.py` - Sync to Zotero
  - `sync_manager.py` - Orchestrate operations
  - `README.md` - Technical documentation
- `test_obsidian_sync.py` - Test script
- `OBSIDIAN_SYNC_GUIDE.md` - This file

## Need Help?

- **Technical docs**: See [src/sync/README.md](src/sync/README.md)
- **Test sync**: Run `python test_obsidian_sync.py`
- **Check logs**: Add `--log-level DEBUG` to any command
