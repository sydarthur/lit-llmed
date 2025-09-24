-- BigQuery schema for lit-llmed literature data

-- Dataset creation
CREATE SCHEMA IF NOT EXISTS `literature_data`
OPTIONS(
  description="Literature review and academic articles data",
  location="US"
);

-- Articles table with comprehensive schema
CREATE OR REPLACE TABLE `literature_data.articles` (
  -- Primary identifiers
  doi STRING NOT NULL,
  title STRING NOT NULL,
  
  -- Journal information
  journal_name STRING,
  issn STRING,
  publisher STRING,
  subject_area STRING,
  
  -- Publication details
  published_date DATE,
  volume STRING,
  issue STRING,
  pages STRING,
  url STRING,
  
  -- Content
  abstract TEXT,
  keywords ARRAY<STRING>,
  
  -- Metrics
  citation_count INT64,
  
  -- Authors (nested/repeated field)
  authors ARRAY<STRUCT<
    given STRING,
    family STRING,
    name STRING
  >>,
  
  -- Open access information
  open_access STRUCT<
    url STRING,
    license STRING,
    version STRING
  >,
  
  -- Metadata
  fetch_timestamp TIMESTAMP,
  source_file STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(published_date)
CLUSTER BY issn, journal_name
OPTIONS(
  description="Academic articles and literature data with upsert capability"
);

-- File processing log table
CREATE OR REPLACE TABLE `literature_data.file_processing_log` (
  -- File identification
  file_path STRING NOT NULL,
  file_name STRING NOT NULL,
  gcs_path STRING NOT NULL,
  
  -- Processing details
  processing_timestamp TIMESTAMP NOT NULL,
  processing_status STRING NOT NULL, -- SUCCESS, FAILED, SKIPPED
  articles_processed INT64 DEFAULT 0,
  articles_inserted INT64 DEFAULT 0,
  articles_updated INT64 DEFAULT 0,
  
  -- Error handling
  error_message STRING,
  retry_count INT64 DEFAULT 0,
  
  -- Metadata
  fetch_timestamp STRING, -- From source filename
  processor_version STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(processing_timestamp)
CLUSTER BY processing_status, file_name
OPTIONS(
  description="Log of processed literature data files for tracking and deduplication"
);

-- View for latest articles by journal
CREATE OR REPLACE VIEW `literature_data.latest_articles` AS
SELECT 
  doi,
  title,
  journal_name,
  issn,
  published_date,
  abstract,
  citation_count,
  ARRAY_LENGTH(authors) as author_count,
  open_access.url as open_access_url,
  fetch_timestamp,
  created_at
FROM `literature_data.articles`
WHERE published_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
ORDER BY published_date DESC, journal_name;

-- View for processing statistics
CREATE OR REPLACE VIEW `literature_data.processing_stats` AS
SELECT 
  DATE(processing_timestamp) as processing_date,
  processing_status,
  COUNT(*) as files_count,
  SUM(articles_processed) as total_articles_processed,
  SUM(articles_inserted) as total_articles_inserted,
  SUM(articles_updated) as total_articles_updated,
  AVG(articles_processed) as avg_articles_per_file
FROM `literature_data.file_processing_log`
GROUP BY DATE(processing_timestamp), processing_status
ORDER BY processing_date DESC;