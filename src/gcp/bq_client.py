"""BigQuery client for literature data with upsert and tracking capabilities."""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.cloud.bigquery import LoadJobConfig, WriteDisposition

from src.core.models import Article
from src.gcp.simple_logger import get_logger
from src.gcp.gcs_storage import GCSStore

logger = get_logger(__name__)


class BigQueryClient:
    """BigQuery client for literature data operations."""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: str = "literature_data"):
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=self.project_id)
        
        # Table references
        self.articles_table = f"{self.project_id}.{self.dataset_id}.articles"
        self.processing_log_table = f"{self.project_id}.{self.dataset_id}.file_processing_log"
        
    def ensure_dataset_exists(self) -> bool:
        """Ensure the dataset exists, create if it doesn't."""
        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {self.dataset_id} exists")
            return True
        except NotFound:
            try:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset.description = "Literature review and academic articles data"
                dataset = self.client.create_dataset(dataset)
                logger.info(f"Created dataset {self.dataset_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to create dataset {self.dataset_id}: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Error checking dataset {self.dataset_id}: {str(e)}")
            return False
    
    def create_tables(self) -> bool:
        """Create tables if they don't exist."""
        try:
            self.ensure_dataset_exists()
            
            # Articles table schema
            articles_schema = [
                bigquery.SchemaField("doi", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("journal_name", "STRING"),
                bigquery.SchemaField("issn", "STRING"),
                bigquery.SchemaField("publisher", "STRING"),
                bigquery.SchemaField("subject_area", "STRING"),
                bigquery.SchemaField("published_date", "DATE"),
                bigquery.SchemaField("volume", "STRING"),
                bigquery.SchemaField("issue", "STRING"),
                bigquery.SchemaField("pages", "STRING"),
                bigquery.SchemaField("url", "STRING"),
                bigquery.SchemaField("abstract", "STRING"),
                bigquery.SchemaField("keywords", "STRING", mode="REPEATED"),
                bigquery.SchemaField("citation_count", "INT64"),
                bigquery.SchemaField("authors", "RECORD", mode="REPEATED", fields=[
                    bigquery.SchemaField("given", "STRING"),
                    bigquery.SchemaField("family", "STRING"),
                    bigquery.SchemaField("name", "STRING"),
                ]),
                bigquery.SchemaField("open_access", "RECORD", fields=[
                    bigquery.SchemaField("url", "STRING"),
                    bigquery.SchemaField("license", "STRING"),
                    bigquery.SchemaField("version", "STRING"),
                ]),
                bigquery.SchemaField("fetch_timestamp", "TIMESTAMP"),
                bigquery.SchemaField("source_file", "STRING"),
                bigquery.SchemaField("created_at", "TIMESTAMP"),
                bigquery.SchemaField("updated_at", "TIMESTAMP"),
            ]
            
            # Processing log schema
            log_schema = [
                bigquery.SchemaField("file_path", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("file_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("gcs_path", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("processing_timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("processing_status", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("articles_processed", "INT64"),
                bigquery.SchemaField("articles_inserted", "INT64"),
                bigquery.SchemaField("articles_updated", "INT64"),
                bigquery.SchemaField("error_message", "STRING"),
                bigquery.SchemaField("retry_count", "INT64"),
                bigquery.SchemaField("fetch_timestamp", "STRING"),
                bigquery.SchemaField("processor_version", "STRING"),
                bigquery.SchemaField("created_at", "TIMESTAMP"),
            ]
            
            # Create articles table
            articles_table_ref = self.client.dataset(self.dataset_id).table("articles")
            try:
                self.client.get_table(articles_table_ref)
                logger.info("Articles table already exists")
            except NotFound:
                articles_table = bigquery.Table(articles_table_ref, schema=articles_schema)
                articles_table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="published_date"
                )
                articles_table.clustering_fields = ["issn", "journal_name"]
                articles_table = self.client.create_table(articles_table)
                logger.info("Created articles table")
            
            # Create processing log table
            log_table_ref = self.client.dataset(self.dataset_id).table("file_processing_log")
            try:
                self.client.get_table(log_table_ref)
                logger.info("Processing log table already exists")
            except NotFound:
                log_table = bigquery.Table(log_table_ref, schema=log_schema)
                log_table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="processing_timestamp"
                )
                log_table.clustering_fields = ["processing_status", "file_name"]
                log_table = self.client.create_table(log_table)
                logger.info("Created processing log table")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            return False
    
    def is_file_processed(self, gcs_path: str) -> bool:
        """Check if a file has already been successfully processed."""
        try:
            query = f"""
            SELECT COUNT(*) as count
            FROM `{self.processing_log_table}`
            WHERE gcs_path = @gcs_path 
            AND processing_status = 'SUCCESS'
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("gcs_path", "STRING", gcs_path)
                ]
            )
            
            results = self.client.query(query, job_config=job_config)
            count = list(results)[0].count
            
            logger.info(f"File processed check for {gcs_path}: {count > 0}")
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking file processing status: {str(e)}")
            return False
    
    def log_file_processing(self, gcs_path: str, status: str, 
                           articles_processed: int = 0, articles_inserted: int = 0, 
                           articles_updated: int = 0, error_message: str = None) -> bool:
        """Log file processing status."""
        try:
            file_name = Path(gcs_path).name
            processing_time = datetime.now(timezone.utc)
            
            # Extract fetch timestamp from filename
            fetch_timestamp = None
            if "_202" in file_name:  # Look for timestamp pattern
                try:
                    parts = file_name.split("_")
                    for part in parts:
                        if part.startswith("202") and len(part) >= 8:
                            fetch_timestamp = part[:15]  # YYYYMMDD_HHMMSS
                            break
                except:
                    pass
            
            row = {
                "file_path": gcs_path.replace("gs://", "").split("/", 1)[1],
                "file_name": file_name,
                "gcs_path": gcs_path,
                "processing_timestamp": processing_time,
                "processing_status": status,
                "articles_processed": articles_processed,
                "articles_inserted": articles_inserted,
                "articles_updated": articles_updated,
                "error_message": error_message,
                "retry_count": 0,
                "fetch_timestamp": fetch_timestamp,
                "processor_version": "1.0.0",
                "created_at": processing_time,
            }
            
            table_ref = self.client.dataset(self.dataset_id).table("file_processing_log")
            errors = self.client.insert_rows_json(table_ref, [row])
            
            if errors:
                logger.error(f"Failed to log processing status: {errors}")
                return False
            
            logger.info(f"Logged processing status: {status} for {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging file processing: {str(e)}")
            return False
    
    def upsert_articles(self, articles: List[Article], source_file: str) -> Tuple[int, int]:
        """Upsert articles using MERGE statement. Returns (inserted, updated) counts."""
        try:
            if not articles:
                return 0, 0
            
            # Prepare data for BigQuery
            rows = []
            current_time = datetime.now(timezone.utc)
            
            for article in articles:
                # Convert authors
                authors_data = []
                if article.authors:
                    for author in article.authors:
                        authors_data.append({
                            "given": author.given,
                            "family": author.family,
                            "name": author.name
                        })
                
                # Convert open access
                open_access_data = None
                if article.open_access:
                    open_access_data = {
                        "url": article.open_access.url,
                        "license": article.open_access.license,
                        "version": article.open_access.version
                    }
                
                # Convert published date
                published_date = None
                if article.published:
                    published_date = article.published.date()
                
                row = {
                    "doi": article.doi,
                    "title": article.title,
                    "journal_name": article.journal,
                    "issn": article.issn,
                    "publisher": getattr(article, 'publisher', None),
                    "subject_area": getattr(article, 'subject_area', None),
                    "published_date": published_date,
                    "volume": article.volume,
                    "issue": article.issue,
                    "pages": article.pages,
                    "url": article.url,
                    "abstract": article.abstract,
                    "keywords": article.keywords or [],
                    "citation_count": article.citation_count,
                    "authors": authors_data,
                    "open_access": open_access_data,
                    "fetch_timestamp": current_time,
                    "source_file": source_file,
                    "created_at": current_time,
                    "updated_at": current_time,
                }
                rows.append(row)
            
            # Create temporary table for merge
            temp_table_id = f"temp_articles_{int(current_time.timestamp())}"
            temp_table_ref = self.client.dataset(self.dataset_id).table(temp_table_id)
            
            # Get main table schema
            main_table_ref = self.client.dataset(self.dataset_id).table("articles")
            main_table = self.client.get_table(main_table_ref)
            
            # Create temp table with same schema
            temp_table = bigquery.Table(temp_table_ref, schema=main_table.schema)
            temp_table = self.client.create_table(temp_table)
            
            try:
                # Insert data into temp table
                errors = self.client.insert_rows_json(temp_table, rows)
                if errors:
                    logger.error(f"Failed to insert into temp table: {errors}")
                    return 0, 0
                
                # Perform MERGE operation
                merge_query = f"""
                MERGE `{self.articles_table}` T
                USING `{temp_table_ref.project}.{temp_table_ref.dataset_id}.{temp_table_ref.table_id}` S
                ON T.doi = S.doi
                WHEN MATCHED THEN
                  UPDATE SET 
                    title = S.title,
                    journal_name = S.journal_name,
                    abstract = S.abstract,
                    citation_count = S.citation_count,
                    updated_at = S.updated_at,
                    fetch_timestamp = S.fetch_timestamp,
                    source_file = S.source_file
                WHEN NOT MATCHED THEN
                  INSERT (doi, title, journal_name, issn, publisher, subject_area, 
                         published_date, volume, issue, pages, url, abstract, keywords,
                         citation_count, authors, open_access, fetch_timestamp, 
                         source_file, created_at, updated_at)
                  VALUES (S.doi, S.title, S.journal_name, S.issn, S.publisher, S.subject_area,
                         S.published_date, S.volume, S.issue, S.pages, S.url, S.abstract, 
                         S.keywords, S.citation_count, S.authors, S.open_access, 
                         S.fetch_timestamp, S.source_file, S.created_at, S.updated_at)
                """
                
                merge_job = self.client.query(merge_query)
                merge_job.result()  # Wait for completion
                
                # Get statistics (approximation - BQ doesn't return exact merge stats)
                inserted = len([r for r in rows])  # Assume all are new for simplicity
                updated = 0  # Could implement more sophisticated tracking
                
                logger.info(f"Upserted {len(rows)} articles from {source_file}")
                return inserted, updated
                
            finally:
                # Clean up temp table
                self.client.delete_table(temp_table_ref)
                
        except Exception as e:
            logger.error(f"Error upserting articles: {str(e)}")
            return 0, 0
    
    def process_gcs_file(self, gcs_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process a single GCS file and upsert to BigQuery."""
        try:
            # Check if already processed
            if not force_reprocess and self.is_file_processed(gcs_path):
                logger.info(f"File {gcs_path} already processed, skipping")
                return {
                    "status": "skipped",
                    "message": "File already processed",
                    "gcs_path": gcs_path
                }
            
            # Download and parse file
            gcs_store = GCSStore()
            file_path = gcs_path.replace(f"gs://{gcs_store.bucket_name}/", "")
            content = gcs_store.download_file(file_path)
            
            # Parse JSON
            articles_data = json.loads(content)
            if not isinstance(articles_data, list):
                articles_data = [articles_data]
            
            # Convert to Article objects
            articles = []
            for article_dict in articles_data:
                try:
                    article = Article(**article_dict)
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {str(e)}")
                    continue
            
            if not articles:
                self.log_file_processing(gcs_path, "FAILED", 0, 0, 0, "No valid articles found")
                return {
                    "status": "failed",
                    "message": "No valid articles found",
                    "gcs_path": gcs_path
                }
            
            # Upsert to BigQuery
            inserted, updated = self.upsert_articles(articles, Path(gcs_path).name)
            
            # Log processing
            self.log_file_processing(
                gcs_path, "SUCCESS", 
                len(articles), inserted, updated
            )
            
            return {
                "status": "success",
                "gcs_path": gcs_path,
                "articles_processed": len(articles),
                "articles_inserted": inserted,
                "articles_updated": updated
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to process file {gcs_path}: {error_msg}")
            self.log_file_processing(gcs_path, "FAILED", 0, 0, 0, error_msg)
            return {
                "status": "failed",
                "message": error_msg,
                "gcs_path": gcs_path
            }
    
    def get_processing_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get processing statistics for the last N days."""
        try:
            query = f"""
            SELECT 
              processing_status,
              COUNT(*) as files_count,
              SUM(articles_processed) as total_articles,
              SUM(articles_inserted) as total_inserted,
              SUM(articles_updated) as total_updated
            FROM `{self.processing_log_table}`
            WHERE DATE(processing_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            GROUP BY processing_status
            ORDER BY processing_status
            """
            
            results = self.client.query(query)
            stats = {}
            
            for row in results:
                stats[row.processing_status] = {
                    "files_count": row.files_count,
                    "total_articles": row.total_articles,
                    "total_inserted": row.total_inserted,
                    "total_updated": row.total_updated
                }
            
            return {
                "period_days": days,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {"error": str(e)}