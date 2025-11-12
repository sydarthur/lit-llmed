"""Google Cloud Storage integration for literature data."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from google.cloud import storage
from google.cloud.exceptions import NotFound

from src.features.gcp_cloud.simple_logger import get_logger
from src.core.models import Article

LOGGER = get_logger(__name__)


class GCSStore:
    """Handles uploading and downloading from Google Cloud Storage."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable is required")
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        
        # Folder structure
        self.folders = {
            'data': 'data',
            'exports': 'exports', 
            'logs': 'logs',
            'cache': 'cache'
        }
    
    def upload_articles(self, articles: List[Article], filename: str, folder: str = 'data') -> str:
        """Upload articles as JSON to GCS."""
        try:
            # Convert articles to dict for JSON serialization
            articles_data = [article.model_dump() for article in articles]
            
            # Create blob path
            blob_path = f"{self.folders[folder]}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            # Upload JSON data
            blob.upload_from_string(
                json.dumps(articles_data, indent=2, default=str),
                content_type='application/json'
            )
            
            gs_path = f"gs://{self.bucket_name}/{blob_path}"
            LOGGER.info("gcs_upload_success", path=gs_path, articles_count=len(articles))
            return gs_path
            
        except Exception as e:
            LOGGER.error("gcs_upload_failed", filename=filename, error=str(e))
            raise
    
    def upload_text_file(self, content: str, filename: str, folder: str = 'exports', content_type: str = 'text/plain') -> str:
        """Upload text content (CSV, RIS, etc.) to GCS."""
        try:
            blob_path = f"{self.folders[folder]}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            blob.upload_from_string(content, content_type=content_type)
            
            gs_path = f"gs://{self.bucket_name}/{blob_path}"
            LOGGER.info("gcs_text_upload_success", path=gs_path, size=len(content))
            return gs_path
            
        except Exception as e:
            LOGGER.error("gcs_text_upload_failed", filename=filename, error=str(e))
            raise
    
    def upload_summary(self, summary: Dict[str, Any], timestamp: str) -> str:
        """Upload fetch summary/metadata."""
        try:
            filename = f"fetch_summary_{timestamp}.json"
            blob_path = f"{self.folders['logs']}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            blob.upload_from_string(
                json.dumps(summary, indent=2, default=str),
                content_type='application/json'
            )
            
            gs_path = f"gs://{self.bucket_name}/{blob_path}"
            LOGGER.info("gcs_summary_upload_success", path=gs_path)
            return gs_path
            
        except Exception as e:
            LOGGER.error("gcs_summary_upload_failed", error=str(e))
            raise
    
    def list_files(self, folder: str = 'data', prefix: str = '') -> List[str]:
        """List files in a GCS folder."""
        try:
            folder_path = f"{self.folders[folder]}/"
            if prefix:
                folder_path += prefix
                
            blobs = self.bucket.list_blobs(prefix=folder_path)
            files = [blob.name for blob in blobs if not blob.name.endswith('/')]
            
            LOGGER.info("gcs_list_success", folder=folder, count=len(files))
            return files
            
        except Exception as e:
            LOGGER.error("gcs_list_failed", folder=folder, error=str(e))
            return []
    
    def download_file(self, file_path: str) -> str:
        """Download file content as string."""
        try:
            blob = self.bucket.blob(file_path)
            content = blob.download_as_text()
            
            LOGGER.info("gcs_download_success", path=file_path)
            return content
            
        except NotFound:
            LOGGER.warning("gcs_file_not_found", path=file_path)
            raise FileNotFoundError(f"File not found: gs://{self.bucket_name}/{file_path}")
        except Exception as e:
            LOGGER.error("gcs_download_failed", path=file_path, error=str(e))
            raise
    
    def ensure_bucket_exists(self) -> bool:
        """Ensure the bucket exists, create if it doesn't."""
        try:
            self.bucket.reload()
            LOGGER.info("gcs_bucket_exists", bucket=self.bucket_name)
            return True
        except NotFound:
            try:
                # Try to create bucket
                bucket = self.client.create_bucket(self.bucket_name)
                LOGGER.info("gcs_bucket_created", bucket=self.bucket_name)
                return True
            except Exception as e:
                LOGGER.error("gcs_bucket_create_failed", bucket=self.bucket_name, error=str(e))
                return False
        except Exception as e:
            LOGGER.error("gcs_bucket_check_failed", bucket=self.bucket_name, error=str(e))
            return False