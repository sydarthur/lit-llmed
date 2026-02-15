# lit-llmed - Project Timeline

## Current Status (December 2025)

**lit-llmed** is a literature management and enrichment system for academic researchers. The project has evolved from scattered scripts to a production-ready pipeline with automated journal monitoring, LLM enrichment capabilities, and bidirectional Obsidian-Zotero synchronization.

**Latest Update (December 25, 2025)**: Completed major feature additions including markdown digest generation, historic data fetching, and enhanced Zotero integration with nested collections. Successfully merged development branch to main with comprehensive testing. Created detailed build plan for future abstract enrichment pipeline.

**Previous Update (September 24, 2025)**: Deployed GCP Cloud Run service with BigQuery integration. Restructured project with feature-based architecture. Added comprehensive Obsidian-Zotero bidirectional sync.

---

## Project Evolution

### Phase 1: Initial Scripts (Early 2024)

**Approach**: Scattered Python scripts for literature management
- Basic Crossref API integration
- Manual RIS export for Zotero
- Simple CSV outputs
- No automation or scheduling

**Location**: `lit-api/` directory (later removed)

**Key Learnings**:
- Need for structured data models
- Importance of content-addressable storage
- Desire for automated monitoring

### Phase 2: LLM Experimentation (Mid 2024)

**Focus**: Exploring LLM capabilities for literature enrichment

**Experiments**:
- Local Ollama integration for abstract summarization
- PDF text extraction with pdfplumber
- Chunked processing for large documents
- Custom prompts for OSCM domain

**Location**: `llm-space/` directory (later removed)

**Outcomes**:
- Proof of concept for LLM-based summarization
- Identified challenges with PDF processing (cost, quality)
- Decided to focus on abstract-only enrichment

**Technical Debt**:
- No integration with fetching pipeline
- Inconsistent output formats
- Hard to maintain separate codebases

### Phase 3: Feature-Based Restructuring (September 2025)

**Major Pivot**: Complete codebase reorganization

**Motivation**:
- Unify scattered scripts into cohesive pipeline
- Enable proper dependency injection and testing
- Support multiple export formats
- Add cloud deployment capabilities

**New Architecture**:
```
src/
├── core/           # Shared infrastructure (models, storage, logging)
├── features/       # Feature modules
│   ├── journal_fetch/      # Crossref/OpenAlex integration
│   ├── pdf_enrich/         # LLM enrichment (tabled)
│   ├── obsidian_sync/      # Bidirectional sync
│   ├── gcp_cloud/          # Cloud deployment
│   └── reporting/          # Output generation
└── cli.py          # Unified CLI interface
```

**Key Decisions**:
- Pydantic v2 for data validation
- Typer for CLI framework
- Feature-based organization (not layer-based)
- Content-addressed storage with SHA256
- Structured JSON logging

**Commit**: `3740576` - "Restructure project with feature-based organization"

### Phase 4: Production Features (September - December 2025)

**Focus**: Building production-ready features

**Completed Features**:

1. **Journal Monitoring Pipeline**
   - Multi-threaded parallel fetching
   - Rate-limited API clients
   - Metadata enrichment (abstracts, OA links)
   - Multi-format export (JSON, CSV, RIS)

2. **Obsidian-Zotero Sync** (Complete)
   - Parse Obsidian markdown for metadata
   - Extract notes from specific sections
   - Smart item matching (Zotero key → DOI → Title)
   - File watcher for auto-sync
   - Dry-run mode for testing
   - **Status**: Fully working and tested

3. **GCP Cloud Deployment**
   - Flask API on Cloud Run
   - BigQuery data warehouse
   - Cloud Storage for exports
   - Automated GitHub Actions CI/CD
   - **Status**: Deployed, partial functionality

4. **Nested Zotero Collections**
   - Support for hierarchical collections (Auto/JBL, Auto/POM)
   - Parent/child collection management
   - Enhanced collection finding/creation logic
   - **Commit**: `b556ee5` - "Fix logging bug and add nested Zotero collection support"

5. **Markdown Digest Generation**
   - Professional literature summaries
   - Collapsible abstracts (HTML details)
   - BibTeX citations
   - Table of contents
   - **Commit**: `fba2f9a` - "Add markdown digests and historic fetch features"

6. **Historic Data Fetching**
   - Fetch articles from past N days
   - Separate from latest fetch
   - All export formats supported
   - CLI: `fetch-historic --days-back 90`

---

## Technical Architecture

### Core Infrastructure

**Data Models** (`src/core/models.py`):
- `Article`: Journal article with metadata
- `Author`: Author with given/family names
- `Journal`: Journal configuration
- `OpenAccess`: OA link information
- `Enrichment`: LLM enrichment data (future)

**Storage** (`src/core/store.py`):
- Content-addressed JSON files (SHA256)
- CSV export with flattened schema
- RIS export for reference managers
- ConfigStore for journal/Zotero settings

**Logging** (`src/core/log.py`):
- Structured JSON logging
- StructuredLogger wrapper for kwargs support
- Timestamp and context in all logs
- **Fixed Bug**: Python logger kwargs compatibility

### Feature Modules

#### 1. Journal Fetch (`src/features/journal_fetch/`)

**Components**:
- `CrossrefClient`: Crossref API integration
- `FetchJob`: Orchestration and parallel execution
- `LiteratureScheduler`: Scheduled fetching

**APIs Integrated**:
- Crossref (primary metadata source)
- OpenAlex (abstract enrichment)
- Unpaywall (OA link discovery)

**Configuration**: `config/journals.json`
- 3 journals: JBL, POM, JOM
- Per-journal settings (days_back, max_articles, etc.)
- Zotero collection mapping

#### 2. Obsidian Sync (`src/features/obsidian_sync/`)

**Components**:
- `ObsidianParser`: Extract metadata from markdown
- `ZoteroClient`: Zotero API wrapper
- `SyncManager`: Orchestration and file watching
- `zotero_sync.py`: Core sync logic

**Capabilities**:
- Parse frontmatter and content
- Extract notes from specific sections
- Smart item matching (multiple strategies)
- Markdown → HTML conversion
- Create/update Zotero notes
- Watch vault for auto-sync

**Status**: Production-ready, fully tested

#### 3. GCP Cloud (`src/features/gcp_cloud/`)

**Infrastructure**:
- Flask REST API (`app.py`)
- BigQuery client (`bq_client.py`)
- Cloud Storage client (`gcs_storage.py`)
- Dockerized deployment

**Endpoints**:
- `GET /` - Health check
- `GET /api/status` - Service status
- `POST /api/fetch` - Trigger fetch
- `GET /api/storage/list` - List GCS files
- `POST /api/bigquery/sync` - Sync to BigQuery
- `GET /api/bigquery/stats` - Processing stats

**Deployment**: Cloud Run in `us-central1`
- URL: `https://lit-llmed-gcp-ommilbgjsa-uc.a.run.app`
- **Issue**: Missing `GCP_PROJECT_ID` env var (BigQuery endpoints fail)

#### 4. Reporting (`src/features/reporting/`)

**Components**:
- `MarkdownDigest`: Professional markdown generation

**Features**:
- Grouped by journal
- Article metadata and citations
- Collapsible abstracts
- BibTeX code blocks
- Auto-generated citekeys

**Output**: `output/enrich/digests/`

#### 5. PDF Enrich (`src/features/pdf_enrich/`)

**Status**: Tabled for future work

**Existing Components**:
- Abstract summarizer (Ollama)
- PDF processor (pdfplumber)
- Chunked processor
- Prompt templates

**Future Plan**: See `docs/BUILD_PLAN_ABSTRACT_ENRICHMENT.md`

---

## Recent Milestones (December 25, 2025)

### 1. Logging Bug Fix

**Problem**: Structured logging with kwargs failed
- Code: `LOGGER.info("message", key=value)`
- Error: `_log() got unexpected keyword argument`

**Solution**: `StructuredLogger` wrapper class
- Converts kwargs to `extra` dict
- Maintains structured logging benefits
- Backwards compatible

**Impact**: Unblocked all journal fetching functionality

### 2. Nested Zotero Collections

**Enhancement**: Support hierarchical collections

**Implementation**:
- Enhanced `get_or_create_collection()` to handle paths
- Added `_find_collection_by_path()` for parent/child lookup
- Modified `_create_collection()` to accept parent key

**Usage**: `Auto/Journal of Business Logistics`
- Creates "Auto" if missing
- Creates "Journal of Business Logistics" under "Auto"
- Returns child collection key

**Testing**: Successfully imported 31 articles to nested collections

### 3. Markdown Digest Generation

**Feature**: Professional literature summaries

**Schema**:
```markdown
# Literature Digest: {Title}
**Date:** {timestamp}
**Total Articles:** {count}

[TOC]

## {Journal Name}
*{count} new articles*

### {Article Title}
**Authors:** {authors} ({year})
**DOI:** [{doi}](https://doi.org/{doi})

<details>
<summary><strong>Abstract</strong></summary>
{abstract}
</details>

#### Citation
```bibtex
{bibtex}
```
```

**Testing**: Generated digests for:
- Latest fetch: 31 articles
- Historic 90-day: 74 articles

**Known Issue**: Journal names showing as "Unknown Journal"
- Root cause: `article.journal` field not populated in parsing
- Workaround: Group by ISSN or fetch separately

### 4. Historic Data Fetching

**Feature**: Fetch articles from past N days

**Implementation**:
- New CLI command: `fetch-historic --days-back N`
- Uses existing `fetch_latest()` with `days_back` override
- Separate from daily/weekly fetch runs
- Labeled output: `historic_90d_YYYYMMDD_HHMMSS.*`

**Use Case**: Backfilling when setting up new journals

**Testing**: 90-day fetch retrieved 74 articles across 3 journals

### 5. Code Cleanup

**Removed**:
- Experimental author search feature (incomplete)
- Old `lit-api/` directory (4,548 deletions)
- Old `llm-space/` directory

**Added**:
- Comprehensive documentation
- Test scripts
- Build plan for future enhancements

**Net Changes**: +7,016 insertions, -4,548 deletions

---

## Current Focus

### Production System

**Stable Features**:
- ✅ Journal monitoring (3 journals: JBL, POM, JOM)
- ✅ Multi-format export (JSON, CSV, RIS, Markdown)
- ✅ Obsidian-Zotero bidirectional sync
- ✅ Nested Zotero collections
- ✅ Historic data fetching
- ✅ Markdown digest generation

**Deployed Services**:
- ✅ GCP Cloud Run API (partial functionality)
- ⚠️ BigQuery integration (needs env var fix)

### Immediate Priorities

1. **Fix GCP BigQuery**
   - Add `GCP_PROJECT_ID` to Cloud Run environment
   - Test BigQuery sync endpoints
   - Verify data warehouse pipeline

2. **Journal Field Bug**
   - Fix markdown digest showing "Unknown Journal"
   - Ensure `article.journal` populated during parsing
   - Update Crossref client if needed

3. **Documentation**
   - Update README with new features
   - Add markdown digest examples
   - Document nested collection setup

### Future Enhancements (Planned)

See `docs/BUILD_PLAN_ABSTRACT_ENRICHMENT.md` for detailed plan:

**Phase 1: Abstract Enrichment**
- LLM analysis on abstracts (not PDFs)
- Categorize by topic, theory, method
- Extract key constructs and contributions
- Generate structured metadata
- Enhanced markdown notes with tags
- Grouped digests (by topic, method, theory)

**Phase 2: Deep PDF Analysis**
- Watch Zotero "Deep Review" collection
- Targeted extraction (sections, not full PDF)
- Detailed methodology extraction
- Research gap identification
- Richer literature review notes

**Estimated Timeline**: Q1 2026 for Phase 1

---

## Key Files

### Core Infrastructure
- **Main CLI**: `src/cli.py` (296 lines)
- **Data Models**: `src/core/models.py` (92 lines)
- **Storage**: `src/core/store.py` (230 lines)
- **Logging**: `src/core/log.py` (82 lines)

### Feature Modules
- **Crossref Client**: `src/features/journal_fetch/crossref_client.py`
- **Fetch Orchestration**: `src/features/journal_fetch/fetch_job.py`
- **Zotero Client**: `src/features/obsidian_sync/zotero_client.py`
- **Sync Manager**: `src/features/obsidian_sync/sync_manager.py`
- **Markdown Reporter**: `src/features/reporting/markdown_reporter.py`
- **GCP API**: `src/features/gcp_cloud/app.py` (588 lines)
- **BigQuery Client**: `src/features/gcp_cloud/bq_client.py` (442 lines)

### Configuration
- **Journals**: `config/journals.json`
- **Zotero**: `config/zotero.json` (gitignored)
- **Settings**: `config/settings.toml`
- **GCP Config**: `config/gcp/gcp-config.yaml`

### Documentation
- **Main README**: `README.md`
- **Sync Guide**: `docs/OBSIDIAN_SYNC_GUIDE.md`
- **Sync Summary**: `docs/SYNC_SUMMARY.md`
- **BigQuery API**: `config/docs/bigquery-api-guide.md`
- **Build Plan**: `docs/BUILD_PLAN_ABSTRACT_ENRICHMENT.md`
- **This Timeline**: `docs/PROJECT_TIMELINE.md`

### Scripts
- **Test Zotero**: `scripts/test_zotero.py`
- **Test Sync**: `scripts/test_obsidian_sync.py`
- **Check Item**: `scripts/check_zotero_item.py`
- **Search Zotero**: `scripts/search_zotero.py`

### Tests
- **Models**: `tests/test_models.py`
- **Storage**: `tests/test_store.py`
- **Enrichment**: `tests/test_abstract_summarizer.py`
- **Prompts**: `tests/test_enrich_prompts.py`

### Output
- **Data**: `output/data/` (JSON exports)
- **CSV**: `output/exports/csv/`
- **RIS**: `output/exports/ris/`
- **Digests**: `output/enrich/digests/` (markdown)
- **Logs**: `output/logs/` (JSON logs per run)

---

## Technical Notes

### CLI Commands

**Journal Management**:
```bash
python -m src.cli journals list
python -m src.cli journals add --name "..." --issn "..."
python -m src.cli journals toggle {issn}
```

**Fetching**:
```bash
# Latest articles
python -m src.cli fetch-all --email you@example.com

# Historic data
python -m src.cli fetch-historic --days-back 90 --email you@example.com

# With exports
python -m src.cli fetch-all --email you@example.com --csv --ris --markdown --zotero

# Scheduled
python -m src.cli schedule --email you@example.com --interval-hours 24
```

**Obsidian Sync**:
```bash
# Vault/folder sync
python -m src.cli sync vault /path/to/vault --folder Literature

# With auto-watch
python -m src.cli sync vault /path/to/vault --folder Literature --watch

# Single file
python -m src.cli sync file /path/to/note.md

# Dry run
python -m src.cli sync vault /path/to/vault --dry-run
```

### Data Pipeline

**Flow**:
1. **Fetch**: Crossref API → Article objects
2. **Enrich**: OpenAlex (abstracts) + Unpaywall (OA links)
3. **Store**: Content-addressed JSON files
4. **Export**: CSV, RIS, Markdown
5. **Sync**: Push to Zotero collections

**Content Addressing**:
- SHA256 hash of article DOI → filename
- Prevents duplicates
- Enables version tracking (future)

**Export Formats**:
- **JSON**: Full metadata, nested structure
- **CSV**: Flattened, spreadsheet-ready
- **RIS**: Reference manager import
- **Markdown**: Human-readable digest

### Cloud Architecture

**GCP Services**:
- **Cloud Run**: Serverless container (Flask API)
- **Cloud Storage**: File storage (bucket: `lit-llmed-prod`)
- **BigQuery**: Data warehouse
  - `articles` table (partitioned by date)
  - `file_processing_log` table (sync tracking)
  - MERGE/upsert to prevent duplicates

**Deployment**:
- Docker container (Python 3.11-slim)
- Gunicorn (2 workers, 300s timeout)
- Non-root user for security
- GitHub Actions CI/CD

**Current Issue**: BigQuery endpoints fail due to missing `GCP_PROJECT_ID` env var

### Development Workflow

**Environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies**:
- `requests` - HTTP client
- `pydantic>=2.5` - Data validation
- `typer` - CLI framework
- `pdfplumber` - PDF text extraction
- `watchdog` - File system monitoring
- `flask` - Web framework (GCP)
- `google-cloud-storage` - GCS client
- `google-cloud-bigquery` - BigQuery client

**Version Control**:
- Main branch: `main` (stable releases)
- Development: `dev-gcp-connect` (active work)
- Feature branches: As needed

**Testing**:
- Manual testing with real APIs
- Test scripts in `scripts/`
- Pytest for unit tests (limited coverage)

---

## Repository Structure Snapshot

```
lit-llmed/
├── src/                           # Source code
│   ├── core/                      # Shared infrastructure
│   │   ├── models.py             # Pydantic data models
│   │   ├── store.py              # Storage and export
│   │   └── log.py                # Structured logging
│   ├── features/                  # Feature modules
│   │   ├── journal_fetch/        # Crossref/OpenAlex
│   │   │   ├── crossref_client.py
│   │   │   ├── fetch_job.py
│   │   │   └── scheduler.py
│   │   ├── obsidian_sync/        # Obsidian-Zotero sync
│   │   │   ├── obsidian_parser.py
│   │   │   ├── zotero_client.py
│   │   │   ├── sync_manager.py
│   │   │   └── zotero_sync.py
│   │   ├── pdf_enrich/           # LLM enrichment (tabled)
│   │   │   ├── abstract_summarizer.py
│   │   │   ├── pdf_processor.py
│   │   │   ├── llm.py
│   │   │   └── prompts.py
│   │   ├── reporting/            # Output generation
│   │   │   └── markdown_reporter.py
│   │   └── gcp_cloud/            # Cloud deployment
│   │       ├── app.py            # Flask API
│   │       ├── bq_client.py      # BigQuery
│   │       ├── gcs_storage.py    # Cloud Storage
│   │       └── Dockerfile
│   └── cli.py                     # CLI entrypoint
├── config/                        # Configuration
│   ├── journals.json             # Journal configs
│   ├── zotero.json               # Zotero credentials
│   ├── settings.toml             # Global settings
│   ├── gcp/                      # GCP configs
│   └── docs/                     # Config documentation
├── scripts/                       # Helper scripts
│   ├── test_zotero.py
│   ├── test_obsidian_sync.py
│   └── check_zotero_item.py
├── tests/                         # Unit tests
│   ├── test_models.py
│   ├── test_store.py
│   └── test_abstract_summarizer.py
├── docs/                          # Documentation
│   ├── README.md
│   ├── OBSIDIAN_SYNC_GUIDE.md
│   ├── SYNC_SUMMARY.md
│   ├── BUILD_PLAN_ABSTRACT_ENRICHMENT.md
│   └── PROJECT_TIMELINE.md       # This document
├── output/                        # Generated outputs
│   ├── data/                     # JSON exports
│   ├── exports/
│   │   ├── csv/
│   │   └── ris/
│   ├── enrich/
│   │   └── digests/              # Markdown digests
│   └── logs/                     # Run logs
├── .github/
│   └── workflows/                # GitHub Actions
│       ├── deploy-gcp.yml
│       └── test-only.yml
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md                      # Main documentation
```

---

## Commit History Highlights

| Commit | Date | Description |
|--------|------|-------------|
| `9acbb4f` | Dec 25, 2025 | Add comprehensive build plan for abstract enrichment |
| `fba2f9a` | Dec 25, 2025 | Add markdown digests and historic fetch features |
| `b556ee5` | Dec 25, 2025 | Fix logging bug and add nested Zotero collection support |
| `3740576` | Sep 24, 2025 | Restructure project with feature-based organization |
| `5361432` | Sep 23, 2025 | Update gitignore to exclude venv and Zotero credentials |
| `784e932` | Sep 23, 2025 | Add comprehensive sync implementation summary |
| `3394a26` | Sep 22, 2025 | Add Obsidian-Zotero bidirectional sync module |
| `5dddb46` | Sep 21, 2025 | Add BigQuery integration with upsert and file tracking |
| `186d08e` | Sep 20, 2025 | Fix logging compatibility for GCP deployment |
| `87cf7e2` | Sep 19, 2025 | Adding lit fetcher |

---

## Reflection

The lit-llmed project has matured from a collection of experimental scripts into a production-ready literature management system. Key evolution highlights:

**Technical Maturation**:
- From scattered scripts → Feature-based architecture
- From manual execution → Scheduled automation
- From single format → Multi-format exports
- From local-only → Cloud-deployed with API

**Feature Completeness**:
- Core pipeline is stable and tested
- Obsidian-Zotero sync is production-ready
- Export formats meet immediate needs
- Cloud deployment enables remote access

**Strategic Decisions**:
- Tabled PDF enrichment (too expensive, focus on abstracts)
- Removed author search (incomplete, not critical)
- Prioritized data quality over feature breadth
- Built extensible architecture for future enhancements

**Next Phase**:
The build plan for abstract enrichment positions us for the next major enhancement: automated categorization and structured metadata extraction using LLMs on abstracts. This will transform lit-llmed from a monitoring tool to an intelligent research assistant.

**Lessons Learned**:
1. Feature-based organization scales better than layer-based
2. Content-addressed storage prevents duplicates and aids debugging
3. Nested Zotero collections require careful API handling
4. Markdown digests are more useful than expected
5. Structured logging is essential for debugging remote services
6. Start simple (abstracts) before tackling hard problems (PDFs)

The project is now positioned for sustainable growth with a solid foundation, clear roadmap, and working production system.
