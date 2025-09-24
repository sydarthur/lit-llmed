# BigQuery Integration API Guide

## New BigQuery Endpoints

### 1. Setup BigQuery Tables
```bash
POST /api/bigquery/setup
```
Creates the `literature_data` dataset and required tables:
- `articles` - Main literature data with upsert capability
- `file_processing_log` - Tracks processed files

### 2. Sync All Files to BigQuery
```bash
POST /api/bigquery/sync
Content-Type: application/json

{
  "force_reprocess": false,  # Optional: reprocess already processed files
  "folder": "data"          # Optional: GCS folder to sync (default: "data")
}
```

### 3. Sync Single File
```bash
POST /api/bigquery/sync/data/filename.json
Content-Type: application/json

{
  "force_reprocess": false  # Optional: reprocess if already processed
}
```

### 4. Get Processing Statistics
```bash
GET /api/bigquery/stats?days=7
```
Returns processing stats for the last N days.

### 5. Execute Custom Query
```bash
POST /api/bigquery/query
Content-Type: application/json

{
  "query": "SELECT title, journal_name, published_date FROM `literature_data.articles` WHERE published_date >= '2025-09-01'",
  "limit": 100  # Optional: default 100
}
```
**Note:** Only SELECT queries allowed for security.

## BigQuery Tables Schema

### Articles Table
- **Primary Key:** `doi` (upsert based on DOI)
- **Partitioned by:** `published_date`
- **Clustered by:** `issn`, `journal_name`

**Key Fields:**
- `doi`, `title`, `journal_name`, `issn`
- `published_date`, `abstract`, `keywords`
- `authors[]` (nested array of author objects)
- `open_access` (nested object with URL, license)
- `citation_count`, `fetch_timestamp`

### File Processing Log
- **Partitioned by:** `processing_timestamp`
- **Clustered by:** `processing_status`, `file_name`

**Key Fields:**
- `gcs_path`, `file_name`, `processing_status`
- `articles_processed`, `articles_inserted`, `articles_updated`
- `error_message`, `retry_count`

## Usage Workflow

1. **Initial Setup:**
   ```bash
   curl -X POST https://your-service-url/api/bigquery/setup
   ```

2. **Fetch Literature + Auto-sync:**
   ```bash
   # Fetch new articles (already stores in GCS)
   curl -X POST https://your-service-url/api/fetch
   
   # Sync to BigQuery
   curl -X POST https://your-service-url/api/bigquery/sync
   ```

3. **Query Data:**
   ```bash
   curl -X POST https://your-service-url/api/bigquery/query \
     -H "Content-Type: application/json" \
     -d '{"query": "SELECT journal_name, COUNT(*) as article_count FROM `literature_data.articles` GROUP BY journal_name"}'
   ```

## Upsert Logic

- **Insert:** New articles (based on DOI) are inserted
- **Update:** Existing articles update: title, abstract, citation_count, updated_at, fetch_timestamp
- **Deduplication:** Files already processed successfully are skipped (unless `force_reprocess=true`)

## Views Available

- `literature_data.latest_articles` - Articles from last 90 days
- `literature_data.processing_stats` - Daily processing statistics

## Error Handling

- Failed file processing is logged with error messages
- Retry logic can be implemented using the `retry_count` field
- Use `force_reprocess=true` to reprocess failed files