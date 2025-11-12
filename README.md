# lit-llmed

Modernized tooling for fetching, enriching, and exporting scholarly articles from Crossref and related services.

## Project Layout

```
.
├─ config/
│  ├─ journals.json        # Journal sources & ingest settings
│  ├─ settings.toml        # Global knobs (paths, batching, limits)
│  ├─ zotero.json          # Zotero credentials (keep secrets out of Git)
│  ├─ gcp/                 # GCP deployment configs
│  │  ├─ cloudrun.yaml
│  │  ├─ gcp-config.yaml
│  │  ├─ gcp-env.example
│  │  └─ bigquery-schema.sql
│  └─ docs/                # Config documentation
│      ├─ bigquery-api-guide.md
│      ├─ github-actions-setup.md
│      └─ deploy.sh
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
├─ src/
│  ├─ core/                # Shared utilities (models, storage, logging)
│  │  ├─ models.py
│  │  ├─ store.py
│  │  └─ log.py
│  ├─ features/            # Feature modules organized by domain
│  │  ├─ journal_fetch/    # Crossref/OpenAlex/Unpaywall ingestion
│  │  │  ├─ crossref_client.py
│  │  │  ├─ fetch_job.py
│  │  │  └─ scheduler.py
│  │  ├─ pdf_enrich/       # LLM-based PDF enrichment
│  │  │  ├─ abstract_summarizer.py
│  │  │  ├─ pdf_processor.py
│  │  │  ├─ chunked_processor.py
│  │  │  ├─ multi_folder_processor.py
│  │  │  ├─ llm.py
│  │  │  ├─ prompts.py
│  │  │  └─ settings.py
│  │  ├─ obsidian_sync/    # Obsidian-Zotero bidirectional sync
│  │  │  ├─ obsidian_parser.py
│  │  │  ├─ zotero_sync.py
│  │  │  ├─ sync_manager.py
│  │  │  ├─ zotero_client.py
│  │  │  └─ README.md
│  │  └─ gcp_cloud/        # GCP Cloud Run deployment
│  │      ├─ app.py
│  │      ├─ bq_client.py
│  │      ├─ gcs_storage.py
│  │      ├─ fetch_handler.py
│  │      └─ Dockerfile
│  └─ cli.py               # Typer-based CLI entry
├─ scripts/                # Helper and test scripts
│  ├─ test_zotero.py
│  ├─ check_zotero_item.py
│  ├─ get_zotero_info.py
│  ├─ search_zotero.py
│  └─ test_obsidian_sync.py
├─ docs/                   # Documentation
│  ├─ OBSIDIAN_SYNC_GUIDE.md
│  └─ SYNC_SUMMARY.md
├─ tests/                  # Pytest suite
├─ .env.example
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
python scripts/test_zotero.py

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

See [docs/OBSIDIAN_SYNC_GUIDE.md](docs/OBSIDIAN_SYNC_GUIDE.md) for complete setup instructions and [src/features/obsidian_sync/README.md](src/features/obsidian_sync/README.md) for technical documentation.

## Abstract Summaries

To transform stored article JSON into structured abstract summaries, use the `AbstractSummarizer` helper. It reads any list of article payloads (such as `output/data/*.json`) and writes enriched summaries to `output/enrich/abstracts/`:

```bash
python - <<'PY'
from pathlib import Path
from src.features.pdf_enrich.abstract_summarizer import AbstractSummarizer

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
