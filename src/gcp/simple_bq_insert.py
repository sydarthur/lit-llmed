"""Simple BigQuery insertion for debugging."""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any
from google.cloud import bigquery
from src.core.models import Article
from src.gcp.simple_logger import get_logger

logger = get_logger(__name__)


def simple_insert_articles(articles: List[Article], project_id: str = None) -> bool:
    """Simple direct insert to test BigQuery connectivity."""
    try:
        project_id = project_id or os.getenv('GCP_PROJECT_ID', 'lit-llmed-prod')  # fallback
        client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.literature_data.articles"
        
        # Get table reference
        table = client.get_table(table_id)
        
        # Prepare simple rows
        rows = []
        current_time = datetime.now(timezone.utc)
        
        for article in articles:
            # Simple row with only required fields
            row = {
                "doi": article.doi or f"no-doi-{current_time.timestamp()}",
                "title": article.title or "Unknown Title",
                "journal_name": article.journal,
                "issn": article.issn,
                "published_date": article.published.date() if article.published else None,
                "abstract": article.abstract,
                "citation_count": article.citation_count or 0,
                "created_at": current_time,
                "updated_at": current_time,
                "fetch_timestamp": current_time,
                "source_file": "debug_insert"
            }
            rows.append(row)
        
        # Insert rows
        errors = client.insert_rows_json(table, rows)
        
        if errors:
            logger.error(f"Insert errors: {errors}")
            return False
        else:
            logger.info(f"Successfully inserted {len(rows)} articles")
            return True
            
    except Exception as e:
        logger.error(f"Simple insert failed: {str(e)}")
        return False