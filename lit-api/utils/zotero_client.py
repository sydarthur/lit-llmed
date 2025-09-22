#!/usr/bin/env python3
"""
Professional Zotero API client for literature management.
"""

import json
import requests
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ZoteroClient:
    """Professional Zotero API client with comprehensive error handling."""
    
    def __init__(self, api_key: str, user_id: str, library_type: str = "user"):
        """
        Initialize Zotero client.
        
        Args:
            api_key: Zotero API key
            user_id: Zotero user ID  
            library_type: "user" or "group"
        """
        self.api_key = api_key
        self.user_id = user_id
        self.library_type = library_type
        self.base_url = f"https://api.zotero.org/{library_type}s/{user_id}"
        
        self.session = requests.Session()
        self.session.headers.update({
            'Zotero-API-Key': api_key,
            'User-Agent': 'Literature Fetcher/1.0'
        })
    
    def test_connection(self) -> bool:
        """Test API connection and permissions."""
        try:
            response = self.session.get(f"{self.base_url}/items?limit=1", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_user_info(self) -> Optional[Dict]:
        """Get user information from API key."""
        try:
            response = self.session.get("https://api.zotero.org/keys/current", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
        return None
    
    def create_collection(self, name: str, parent_key: Optional[str] = None) -> Optional[str]:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            parent_key: Parent collection key (for subcollections)
            
        Returns:
            Collection key if successful, None otherwise
        """
        collection_data = [{
            "name": name,
            "parentCollection": parent_key or False
        }]
        
        try:
            response = self.session.post(
                f"{self.base_url}/collections",
                headers={'Content-Type': 'application/json'},
                json=collection_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'successful' in result and '0' in result['successful']:
                    collection_key = result['successful']['0']['key']
                    logger.info(f"Created collection '{name}' with key: {collection_key}")
                    return collection_key
            
            logger.error(f"Collection creation failed: {response.status_code} - {response.text}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
        
        return None
    
    def find_collection_by_name(self, name: str) -> Optional[str]:
        """Find collection key by name."""
        try:
            response = self.session.get(f"{self.base_url}/collections", timeout=30)
            
            if response.status_code == 200:
                collections = response.json()
                for collection in collections:
                    if collection.get('data', {}).get('name') == name:
                        return collection['key']
            
        except Exception as e:
            logger.error(f"Error finding collection: {e}")
        
        return None
    
    def get_or_create_collection(self, name: str) -> Optional[str]:
        """Get existing collection or create new one."""
        # Try to find existing collection
        collection_key = self.find_collection_by_name(name)
        
        if collection_key:
            logger.info(f"Found existing collection '{name}': {collection_key}")
            return collection_key
        
        # Create new collection
        return self.create_collection(name)
    
    def import_articles(self, articles: List[Dict], collection_key: Optional[str] = None) -> bool:
        """
        Import articles to Zotero.
        
        Args:
            articles: List of article dictionaries
            collection_key: Collection to add articles to
            
        Returns:
            True if successful, False otherwise
        """
        if not articles:
            logger.warning("No articles to import")
            return False
        
        items = []
        for article in articles:
            if not article.get('title') or article.get('title') == "Issue Information":
                continue
            
            item = self._convert_article_to_zotero_item(article)
            
            if collection_key:
                item['collections'] = [collection_key]
            
            items.append(item)
        
        if not items:
            logger.warning("No valid articles to import")
            return False
        
        try:
            response = self.session.post(
                f"{self.base_url}/items",
                headers={'Content-Type': 'application/json'},
                json=items,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                successful_count = len(result.get('successful', {}))
                failed_count = len(result.get('failed', {}))
                
                logger.info(f"Import completed: {successful_count} successful, {failed_count} failed")
                
                if failed_count > 0:
                    logger.warning(f"Failed items: {result.get('failed', {})}")
                
                return successful_count > 0
            else:
                logger.error(f"Import failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during import: {e}")
            return False
    
    def _convert_article_to_zotero_item(self, article: Dict) -> Dict:
        """Convert article dictionary to Zotero item format."""
        # Clean abstract
        abstract = self._clean_abstract(article.get('abstract', ''))
        
        # Create creators
        creators = []
        for author in article.get('authors', []):
            parts = author.strip().split()
            if len(parts) >= 2:
                creators.append({
                    "creatorType": "author",
                    "firstName": " ".join(parts[:-1]),
                    "lastName": parts[-1]
                })
            else:
                creators.append({
                    "creatorType": "author",
                    "name": author
                })
        
        item = {
            "itemType": "journalArticle",
            "title": article.get('title', ''),
            "creators": creators,
            "publicationTitle": article.get('journal_name', ''),
            "volume": article.get('volume', ''),
            "issue": article.get('issue', ''),
            "pages": article.get('pages', ''),
            "date": article.get('published_date', ''),
            "DOI": article.get('doi', ''),
            "ISSN": article.get('issn', ''),
            "url": article.get('open_access_url') or article.get('url', ''),
            "abstractNote": abstract
        }
        
        # Remove empty fields
        return {k: v for k, v in item.items() if v}
    
    def _clean_abstract(self, abstract: str) -> str:
        """Clean JATS markup and other formatting from abstract."""
        if not abstract:
            return ""
        
        # Remove JATS markup
        abstract = abstract.replace('<jats:title>ABSTRACT</jats:title>', '')
        abstract = abstract.replace('<jats:p>', '').replace('</jats:p>', '')
        abstract = abstract.replace('<scp>', '').replace('</scp>', '')
        abstract = abstract.replace('<i>', '').replace('</i>', '')
        abstract = abstract.replace('&amp;', '&')
        abstract = abstract.replace('&lt;', '<').replace('&gt;', '>')
        
        # Remove extra whitespace and limit length
        abstract = ' '.join(abstract.split())
        return abstract[:10000]  # Zotero limit
    
    def get_collection_items(self, collection_key: str) -> List[Dict]:
        """Get all items in a collection."""
        try:
            response = self.session.get(f"{self.base_url}/collections/{collection_key}/items", timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get collection items: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting collection items: {e}")
            return []