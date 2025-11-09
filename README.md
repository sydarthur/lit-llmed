# lit-llmed

Modernized tooling for fetching, enriching, and exporting scholarly articles from Crossref and related services.

## Project Layout

```
.
├─ config/
│  ├─ journals.json        # Journal sources & ingest settings
│  ├─ settings.toml        # Global knobs (paths, batching, limits)
│  └─ zotero.json          # Optional Zotero credentials (keep secrets out of Git)
├─ output/
│  ├─ cache/               # HTTP cache (future use)
│  ├─ data/                # Content-addressed JSON payloads
│  ├─ enrich/
│  │  ├─ notes/            # Obsidian-friendly markdown notes
│  │  └─ debug/            # Chunk inspection output
│  ├─ exports/
│  │  ├─ csv/              # Flattened exports
│  │  └─ ris/              # RIS exports for reference managers
│  ├─ files/pdf/           # Placeholder for future PDF storage
│  └─ logs/                # JSON logs per run
├─ pdfs_input/             # Drop PDFs here for LLM summarisation workflows
├─ src/
│  ├─ core/                # Models, persistence helpers, logging
│  │  ├─ log.py
│  │  ├─ models.py
│  │  └─ store.py
│  ├─ ingest/
│  │  └─ crossref.py       # Crossref + OpenAlex + Unpaywall ingestion
│  ├─ enrich/
│  │  ├─ chunked_processor.py
│  │  ├─ abstract_summarizer.py
│  │  ├─ llm.py
│  │  ├─ multi_folder_processor.py
│  │  ├─ pdf_processor.py
│  │  ├─ prompts.py
│  │  └─ settings.py
│  ├─ integrate/
│  │  └─ zotero.py         # Zotero API client
│  ├─ sync/                # Obsidian-Zotero bidirectional sync
│  │  ├─ obsidian_parser.py
│  │  ├─ zotero_sync.py
│  │  ├─ sync_manager.py
│  │  └─ README.md         # Sync documentation
│  ├─ jobs/
│  │  ├─ digest.py         # Placeholder for future enrichment workflows
│  │  ├─ fetch.py          # Fetch orchestration & exports
│  │  ├─ scheduler.py      # Lightweight scheduling loop
│  │  └─ sync_zotero.py    # Placeholder for Zotero sync job
│  └─ cli.py               # Typer-based CLI
├─ tests/                  # Pytest suite covering core utilities
├─ .env.example            # Suggested environment variables
├─ requirements.txt
└─ README.md
```

## Getting Started

1. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure runtime files** inside `config/`:
   - `journals.json` – define the journals you want to monitor. The CLI can add/remove entries for you.
   - `settings.toml` – optional global defaults (rate limits, worker counts, output paths).
   - `zotero.json` – only required when pushing directly to Zotero. Do not commit real credentials; see `.env.example` for guidance.

3. **Set your contact email** (required by Unpaywall/OpenAlex) via environment variable or CLI option.

## CLI Usage

The CLI is powered by [Typer](https://typer.tiangolo.com/). Run `python -m src.cli --help` to see all commands.

Common commands:

```bash
# Fetch a single journal by ISSN (must exist in config/journals.json)
python -m src.cli fetch-single --email you@example.com 0735-3766

# Fetch all active journals and generate CSV + RIS exports
python -m src.cli fetch-all --email you@example.com

# Toggle journal activation and manage config entries
python -m src.cli journals list
python -m src.cli journals add --name "New Journal" --issn 1234-5678
python -m src.cli journals toggle 1234-5678

# Run a one-off scheduled fetch or start a long-running schedule
python -m src.cli schedule --email you@example.com --run-once
python -m src.cli schedule --email you@example.com --interval-hours 12

# Sync Obsidian literature notes to Zotero
python -m src.cli sync vault /path/to/obsidian/vault --folder Labs/Literature
python -m src.cli sync file /path/to/note.md
python -m src.cli sync vault /path/to/vault --folder Literature --watch  # Auto-sync
```

All logs are emitted as structured JSON for easy piping into observability tooling.

## Obsidian-Zotero Sync

Bidirectional synchronization between Obsidian literature notes and Zotero library. Write notes in Obsidian's **Critical Notes** or **Annotations & Highlights** sections, then sync them back to Zotero.

**Quick start:**

```bash
# Test Zotero connection
python test_zotero.py

# Dry run (preview without changes)
python -m src.cli sync vault /path/to/vault --folder Literature --dry-run

# Sync notes to Zotero
python -m src.cli sync vault /path/to/vault --folder Literature

# Auto-sync on file changes
python -m src.cli sync vault /path/to/vault --folder Literature --watch
```

**Features:**
- Parse Obsidian notes for Zotero metadata (DOI, citation keys, Zotero URIs)
- Sync notes from Obsidian to Zotero items
- Smart matching by Zotero key, DOI, or title
- Auto-watch mode for continuous sync
- Notes synced back to Obsidian on re-import

See [OBSIDIAN_SYNC_GUIDE.md](OBSIDIAN_SYNC_GUIDE.md) for complete setup instructions and [src/sync/README.md](src/sync/README.md) for technical documentation.

## Abstract Summaries

To transform stored article JSON into structured abstract summaries, use the `AbstractSummarizer` helper. It reads any list of article payloads (such as `output/data/*.json`) and writes enriched summaries to `output/enrich/abstracts/`:

```bash
python - <<'PY'
from pathlib import Path
from src.enrich.abstract_summarizer import AbstractSummarizer

summarizer = AbstractSummarizer()
summarizer.summarise_json_file(Path("output/data/all_journals_20240101.json"))
PY
```

Each entry contains summary text, methodology, key findings, normalised tags, and copies of the journal metadata for downstream analytics.

## Tests

Pytests live in `tests/`. Run them locally with:

```bash
pytest
```

## Next Steps

- Expand `src/enrich` processors with persistence back into the article store.
- Extend `jobs/sync_zotero.py` to import existing JSON exports.
- Wire in caching (e.g., sqlite/httpcache) to populate `output/cache`.
